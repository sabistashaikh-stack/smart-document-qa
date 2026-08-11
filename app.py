import streamlit as st
import sys
import os
import tempfile

# Add backend folder to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from ingest import process_pdf
from vectorstore import vector_store
from rag import answer_question


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Smart Document QA",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Smart Document QA & Knowledge Query System")

st.caption(
    "Upload a PDF and ask questions in plain English. "
    "Answers are generated using RAG with page-level references."
)


# -------------------------------------------------
# Sidebar - Upload Document
# -------------------------------------------------

with st.sidebar:

    st.header("📤 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        if st.button(
            "Index Document",
            use_container_width=True
        ):

            with st.spinner(
                "Extracting text, creating chunks and embeddings..."
            ):

                try:

                    # Create temporary PDF file
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(
                            uploaded_file.getvalue()
                        )

                        temp_path = temp_file.name

                    # Process PDF
                    records = process_pdf(
                        temp_path,
                        source_name=uploaded_file.name
                    )

                    # Remove temporary file
                    os.remove(temp_path)

                    if not records:

                        st.error(
                            "No extractable text found. "
                            "This may be a scanned/image-only PDF."
                        )

                    else:

                        # Store embeddings
                        vector_store.add_documents(records)

                        st.success(
                            f"Successfully indexed "
                            f"'{uploaded_file.name}' — "
                            f"{len(records)} chunks"
                        )

                except Exception as e:

                    st.error(
                        f"Error while processing PDF: {e}"
                    )


# -------------------------------------------------
# Indexed Documents
# -------------------------------------------------

with st.sidebar:

    st.divider()

    st.header("📚 Indexed Documents")

    try:

        docs = vector_store.list_sources()

        if docs:

            for document in docs:

                st.write(f"• {document}")

        else:

            st.info("No documents indexed yet.")

    except Exception as e:

        st.warning(
            f"Unable to load documents: {e}"
        )

        docs = []


# -------------------------------------------------
# Question Answering
# -------------------------------------------------

st.subheader("💬 Ask a Question")


if docs:

    source_filter = st.selectbox(
        "Search within",
        options=["All documents"] + docs
    )

else:

    source_filter = "All documents"


question = st.text_input(
    "Your question",
    placeholder="e.g. What are the key findings of this research paper?"
)


if st.button(
    "Ask",
    type="primary"
) and question.strip():

    with st.spinner(
        "Searching documents and generating answer..."
    ):

        try:

            result = answer_question(
                question=question,
                source_filter=(
                    None
                    if source_filter == "All documents"
                    else source_filter
                )
            )

            # -------------------------------
            # Answer
            # -------------------------------

            st.markdown("### 🤖 Answer")

            if result["mode"] == "generative":

                st.success(
                    "Answer generated using RAG + Claude"
                )

            elif result["mode"] == "extractive":

                st.info(
                    "Extractive mode: showing the most "
                    "relevant passages."
                )

            st.write(result["answer"])


            # -------------------------------
            # Sources
            # -------------------------------

            st.markdown("### 🔍 Sources")

            if result["sources"]:

                for i, source in enumerate(
                    result["sources"],
                    start=1
                ):

                    with st.expander(
                        f"[{i}] "
                        f"{source['source']} — "
                        f"Page {source['page']} "
                        f"(score: {source['score']})"
                    ):

                        st.write(
                            source["text"]
                        )

            else:

                st.info(
                    "No sources found."
                )

        except Exception as e:

            st.error(
                f"Error while answering question: {e}"
            )


# -------------------------------------------------
# Footer
# -------------------------------------------------

st.divider()

st.caption(
    "Smart Document QA | "
    "PDF + Embeddings + ChromaDB + RAG + Claude"
)
