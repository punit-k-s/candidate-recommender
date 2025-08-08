# src/preprocess.py
"""
Resume parser + light cleaner.

- Parses PDF, DOCX, TXT (from bytes)
- Returns raw-ish text
- Optional minimal cleanup:
    - collapse whitespace
    - strip a few boilerplate phrases
    - optional stopword removal (very small list; off by default)
"""

from __future__ import annotations
import io
import re
from typing import Optional

# ---------- Optional light stopwords (tiny set; tweak as needed) ----------
STOPWORDS = {
    "the","a","an","and","or","of","to","in","for","on","with","by","at",
    "from","is","are","was","were","be","been","being","that","this","it",
    "as","i","we","you","they","he","she","their","our","my","your"
}

# ---------- Boilerplate patterns we can safely drop ----------
BOILERPLATE_PATTERNS = [
    r"references available upon request",
    r"i hereby consent",
    r"confidentiality statement",
    r"declaration[: ]",
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)

# Collapse runs of spaces/tabs, and long runs of newlines
WS_RUN_RE   = re.compile(r"[ \t]+")
NL_RUN_RE   = re.compile(r"\n{3,}")

def _clean_minimal(text: str, *, remove_stopwords: bool=False) -> str:
    """Minimal cleanup: trim, drop boilerplate, collapse whitespace."""
    if not text:
        return ""

    # Normalize newlines/tabs
    text = text.replace("\r", "\n").replace("\t", " ")
    # Drop obvious boilerplate
    text = BOILERPLATE_RE.sub(" ", text)
    # Strip trailing spaces on each line (keeps line structure)
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(lines)

    # Collapse whitespace
    text = WS_RUN_RE.sub(" ", text)
    text = NL_RUN_RE.sub("\n\n", text)
    text = text.strip()

    if remove_stopwords:
        # Very light stopword removal; maintain punctuation and numbers.
        tokens = re.findall(r"\w+|\S", text)
        kept = []
        for tok in tokens:
            if tok.isalpha() and tok.lower() in STOPWORDS:
                continue
            kept.append(tok)
        text = "".join(kept)
        # quick tidy after join
        text = WS_RUN_RE.sub(" ", text).strip()

    return text

# ---------- Parsers (from bytes) ----------

def _parse_pdf_bytes(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        # pypdf not installed; tell caller clearly
        raise RuntimeError("pypdf is required to parse PDFs. pip install pypdf")

    reader = PdfReader(io.BytesIO(data))
    chunks = []
    for page in reader.pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(chunks).strip()

def _parse_docx_bytes(data: bytes) -> str:
    try:
        import docx  # python-docx
    except Exception:
        raise RuntimeError("python-docx is required to parse .docx. pip install python-docx")

    f = io.BytesIO(data)
    doc = docx.Document(f)
    return "\n".join(p.text for p in doc.paragraphs).strip()

def _parse_txt_bytes(data: bytes) -> str:
    # Best-effort decode
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    # last resort: ignore errors
    return data.decode("utf-8", errors="ignore")

# ---------- Public API ----------

def parse_resume_bytes(
    filename: str,
    data: bytes,
    *,
    clean: bool = True,
    remove_stopwords: bool = False,
) -> str:
    """
    Parse resume from bytes (PDF/DOCX/TXT). Optionally apply minimal cleanup.

    Args:
        filename: original file name (used for extension)
        data: file bytes
        clean: apply minimal whitespace/boilerplate cleanup
        remove_stopwords: tiny optional stopword removal (off by default)

    Returns:
        text (str)
    """
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        raw = _parse_pdf_bytes(data)
    elif name.endswith(".docx"):
        raw = _parse_docx_bytes(data)
    elif name.endswith(".txt"):
        raw = _parse_txt_bytes(data)
    else:
        # Try to guess: default to txt decode and hope for the best
        raw = _parse_txt_bytes(data)

    if clean:
        return _clean_minimal(raw, remove_stopwords=remove_stopwords)
    return raw
