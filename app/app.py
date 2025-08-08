# app.py
from dotenv import load_dotenv
import os

# --- Env setup ---------------------------------------------------------------
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# --- Stdlib / third-party ----------------------------------------------------
import sys
from pathlib import Path
import streamlit as st
import numpy as np

# --- Local imports -----------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.state import init_state, add_resume, reset_all
from src.io_utils import read_any
from src.parsing import extract_name
from src.embeddings.openai_client import OpenAIEmbeddingClient
from src.ranking import rank_by_cosine
from src.utils.batching import embed_in_batches
from src.utils.tokens import count_tokens


# --- Core logic --------------------------------------------------------------
def compute(jd_text, resumes, top_k=5):
    """
    Compute embeddings for JD + resumes (no NLP preprocessing),
    then rank top_k by cosine similarity.
    """

    if not jd_text.strip():
        st.warning("Enter a job description first.")
        return None
    if not resumes:
        st.warning("Upload or paste at least one resume.")
        return None

    try:
        backend = OpenAIEmbeddingClient()
        api_key = getattr(backend, "api_key", "<none>")
        print(f"Using API key starting with: {api_key[:8]}...")

        # 1) JD as-is (trim only)
        jd_raw = jd_text.strip()
        if not jd_raw:
            st.error("Job description is empty.")
            return None

        # 2) Resumes as-is (skip empties)
        kept_names = []
        texts = [jd_raw]  # JD first
        for r in resumes:
            name = r["name"]
            parsed = (r.get("text") or "").strip()
            if not parsed:
                st.warning(f"{name}: skipped (empty or unreadable resume).")
                continue
            kept_names.append(name)
            texts.append(parsed)

        if len(texts) <= 1:
            st.error("No usable resumes provided.")
            return None

        # 3) Log size summary
        labels = ["JD"] + kept_names
        tok_counts = [count_tokens(t, "text-embedding-3-small") for t in texts]
        print("\n=== Embedding batch summary ===")
        print(f"Total texts to embed: {len(texts)}")
        for i, (lbl, tks) in enumerate(zip(labels, tok_counts)):
            print(f"{i:02d}. {lbl:>12} | ~{tks} tokens | {len(texts[i])} chars")

        # 4) Embed in safe batches (item & token caps)
        try:
            embs = embed_in_batches(
                backend,
                texts,
                model="text-embedding-3-small",
                max_text_tokens=8191,    
                max_batch_items=128,     
                max_batch_tokens=60000,  
                truncate_long=True,
                verbose=True,
            )  
        except Exception as embed_err:
            print("Embed call failed:", type(embed_err), repr(embed_err))
            st.error(f"Embed call failed: {embed_err}")
            return None

        print(f"\nEmbeddings shape: {getattr(embs, 'shape', None)}")

        # 5) Rank JD vs resumes
        q_vec = embs[:1]    # (1, D)
        doc_vecs = embs[1:] # (R, D)
        if doc_vecs.size == 0:
            st.error("No vectors produced for resumes.")
            return None

        return rank_by_cosine(q_vec, doc_vecs, kept_names, top_k=top_k, normalize=False)

    except Exception as e:
        print("Embedding failed:", type(e), repr(e))
        st.error(f"Embedding request failed: {e}")
        return None





# --- UI ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Candidate Recommender", page_icon="🧭", layout="centered")
    init_state()

    st.title("Candidate Recommendation Engine")
    st.caption("Enter a JD, add resumes (upload or paste), then compute similarity.")

    # Sidebar: reset only
    with st.sidebar:
        st.subheader("Run")
        st.button("Compute again", help="Clear JD and resumes and start fresh.", on_click=reset_all)

    # 1) Job description
    st.subheader("1) Job description")
    st.session_state.jd_text = st.text_area(
        "Paste job description",
        value=st.session_state.jd_text,
        height=160,
    )

    # 2) Add resumes
    st.subheader("2) Add resumes")

    # File uploads (store both text + original bytes/filename for later ZIP)
    uploaded = st.file_uploader(
        "Upload PDF/DOCX/TXT resumes (multiple)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    if uploaded:
        for f in uploaded:
            file_bytes = f.read()
            text = read_any(f.name, file_bytes)
            name = extract_name(f.name, len(st.session_state.resumes) + 1)
            # dedupe by name only (user can re-upload with a different name if needed)
            if not any(r["name"].lower() == name.lower() for r in st.session_state.resumes):
                # NOTE: add_resume now supports data/filename (as per your updated state.py)
                add_resume(name, text, data=file_bytes, filename=f.name)
            else:
                # quietly skip duplicate by name
                pass

    # Paste resume (form clears on submit)
    st.markdown("**Or paste a resume (plain text)**")
    with st.form("paste_resume_form", clear_on_submit=True):
        pasted = st.text_area("Paste resume text", key="paste_resume", height=160)
        submitted = st.form_submit_button("Add pasted resume")
        if submitted:
            content = (pasted or "").strip()
            if not content:
                st.warning("Paste some resume text first.")
            else:
                pasted_count = sum(1 for r in st.session_state.resumes if r["name"].startswith("Pasted_"))
                name = f"Pasted_{pasted_count + 1}"
                is_dup = any(
                    r["name"].lower() == name.lower() or r.get("text", "").strip() == content
                    for r in st.session_state.resumes
                )
                if is_dup:
                    st.info(f"{name} already added — skipped")
                else:
                    add_resume(name, content)  # text-only pasted resumes

    # List currently added resumes
    if st.session_state.resumes:
        st.write("**Uploaded/Pasted resumes:**")
        for r in st.session_state.resumes:
            st.write(f"- {r['name']}")

    # 3) Compute similarity (inline top-k selection)
    st.subheader("3) Compute similarity")
    top_k = st.radio("Show top results", [5, 10], index=1, horizontal=True)
    if st.button("Compute similarity"):
        with st.spinner("Computing embeddings and ranking…"):
            st.session_state.results = compute(
                st.session_state.jd_text,
                st.session_state.resumes,
                top_k=top_k,
            )

    # Results + ZIP download of top-K original files
    if st.session_state.results is not None:
        st.success("Done. Top matches below.")
        df = st.session_state.results
        st.dataframe(df, use_container_width=True)

        # Build ZIP for top-K original files
        import io, zipfile, re

        by_name = {r["name"]: r for r in st.session_state.resumes}
        top_names = df["Candidate"].tolist()

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for nm in top_names:
                r = by_name.get(nm)
                if not r:
                    continue

                data = r.get("data")
                fname = r.get("filename")
                if data and fname:
                    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", fname)
                    zf.writestr(safe, data)
                else:
                    # fallback: write .txt with the parsed/plain text
                    txt = r.get("text", "")
                    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{nm}.txt")
                    zf.writestr(safe, txt)

        zip_buf.seek(0)
        st.download_button(
            f"Download top {top_k} resumes (original formats, ZIP)",
            data=zip_buf,
            file_name="top_resumes.zip",
            mime="application/zip",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
