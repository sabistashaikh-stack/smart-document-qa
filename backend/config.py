"""
Central configuration for the Document QA system.
Change values here instead of hunting through the codebase.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Where uploaded PDFs are stored temporarily
UPLOAD_DIR = BASE_DIR / "uploaded_docs"
UPLOAD_DIR.mkdir(exist_ok=True)

# Where ChromaDB persists its vector index (survives restarts)
CHROMA_DIR = BASE_DIR / "chroma_db"
CHROMA_DIR.mkdir(exist_ok=True)

CHROMA_COLLECTION_NAME = "documents"

# Embedding model — small & fast; swap for "BAAI/bge-large-en-v1.5" for better accuracy
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 150     # overlap between consecutive chunks

# Retrieval
TOP_K = 4               # how many chunks to retrieve per query

# LLM answer generation (optional — falls back to extractive mode if no key)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = "claude-sonnet-4-5"

# FastAPI backend URL, used by the Streamlit frontend
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
