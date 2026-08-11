"""
Streamlit frontend for the Smart Document QA system.
Run with: streamlit run frontend/app.py
Requires the FastAPI backend to be running (see backend/main.py).
"""
import streamlit as st
import requests
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
from config import BACKEND_URL  # noqa: E402

st.set_page_config(page_title="Smart Document QA", page_icon="📄", layout="wide")
st.title("📄 Smart Document QA & Knowledge Query System")
st.caption("Upload a PDF, then ask questions in plain English. Answers come with page-level references.")

# ---------- Sidebar: upload + document management ----------
with st.sidebar:
    st.header("📤 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file and st.button("Index Document", use_container_width=True):
        with st.spinner("Extracting text, chunking, and embedding..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=300)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Indexed '{data['filename']}' — {data['chunks_indexed']} chunks")
                else:
                    st.error(f"Error: {resp.json().get('detail', resp.text)}")
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the backend. Is FastAPI running on port 8000?")

    st.divider()
    st.header("📚 Indexed Documents")
    try:
        docs = requests.get(f"{BACKEND_URL}/documents", timeout=10).json()["documents"]
        if docs:
            for d in docs:
                col1, col2 = st.columns([4, 1])
                col1.write(f"• {d}")
                if col2.button("🗑️", key=f"del_{d}"):
                    requests.delete(f"{BACKEND_URL}/documents/{d}")
                    st.rerun()
        else:
            st.info("No documents indexed yet.")
    except requests.exceptions.ConnectionError:
        st.warning("Backend not reachable.")
        docs = []

# ---------- Main: question answering ----------
st.subheader("💬 Ask a Question")

source_filter = st.selectbox(
    "Search within",
    options=["All documents"] + (docs if docs else []),
)

question = st.text_input("Your question", placeholder="e.g. What are the key findings in section 3?")

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Searching and generating answer..."):
        payload = {
            "question": question,
            "source_filter": None if source_filter == "All documents" else source_filter,
        }
        try:
            resp = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=120)
            if resp.status_code == 200:
                result = resp.json()

                st.markdown("### Answer")
                if result["mode"] == "extractive":
                    st.info("Extractive mode (no LLM key set) — showing best-matching passages.")
                st.write(result["answer"])

                st.markdown("### 🔍 Sources")
                for i, src in enumerate(result["sources"], start=1):
                    with st.expander(f"[{i}] {src['source']} — Page {src['page']} (score: {src['score']})"):
                        st.write(src["text"])
            else:
                st.error(f"Error: {resp.json().get('detail', resp.text)}")
        except requests.exceptions.ConnectionError:
            st.error("Can't reach the backend. Is FastAPI running on port 8000?")
