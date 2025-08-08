import streamlit as st
from src.io_utils import normalize_whitespace

def init_state():
    st.session_state.setdefault("resumes", [])  # list[dict]: {name, text}
    st.session_state.setdefault("jd_text", "")
    st.session_state.setdefault("results", None)
    st.session_state.setdefault("paste_resume", "")

def add_resume(name: str, text: str, data: bytes | None = None, filename: str | None = None):
    text = normalize_whitespace(text)
    # de-dupe by exact name
    if any(r["name"].lower() == name.lower() for r in st.session_state.resumes):
        return
    st.session_state.resumes.append({
        "name": name,
        "text": text,
        "data": data,         
        "filename": filename,  
    })

def reset_all():
    st.session_state["resumes"] = []
    st.session_state["jd_text"] = ""
    st.session_state["results"] = None
    st.session_state["paste_resume"] = ""
    st.rerun()
