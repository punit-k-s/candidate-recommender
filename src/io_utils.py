import io

try:
    from pypdf import PdfReader
    _HAS_PDF = True
except Exception:
    _HAS_PDF = False

try:
    import docx  # python-docx
    _HAS_DOCX = True
except Exception:
    _HAS_DOCX = False


def read_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return file_bytes.decode("latin-1", errors="ignore")


def read_pdf(file_bytes: bytes) -> str:
    if not _HAS_PDF:
        return ""
    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(texts).strip()


def read_docx(file_bytes: bytes) -> str:
    if not _HAS_DOCX:
        return ""
    buf = io.BytesIO(file_bytes)
    document = docx.Document(buf)
    return "\n".join(p.text for p in document.paragraphs).strip()


def read_any(filename: str, file_bytes: bytes) -> str:
    name = filename.lower()
    if name.endswith(".txt"):
        return read_txt(file_bytes)
    if name.endswith(".pdf"):
        return read_pdf(file_bytes)
    if name.endswith(".docx"):
        return read_docx(file_bytes)
    return read_txt(file_bytes)


def normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split())
