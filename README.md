# Smart Document QA & Knowledge Query System

Upload large PDFs (research papers, enterprise docs) and ask natural-language
questions. Answers come back with exact page-level references, powered by
semantic search (RAG).

## Architecture

```
PDF Upload → Text Extraction (pdfplumber) → Chunking (with overlap)
  → Embeddings (HuggingFace, local) → ChromaDB (vector store)
  → Query → Semantic Retrieval (top-k) → LLM Answer Generation (optional)
  → Answer + Page References
```

## Project Structure

```
docqa/
├── backend/
│   ├── config.py       # all settings in one place
│   ├── ingest.py        # PDF → text → chunks
│   ├── vectorstore.py   # ChromaDB + embeddings wrapper
│   ├── rag.py           # retrieval + answer generation
│   └── main.py           # FastAPI app (upload/query endpoints)
├── frontend/
│   └── app.py            # Streamlit UI
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Optional: enable LLM-generated answers

Without this, the system runs in **extractive mode** — it still retrieves and
shows the most relevant passages, just without a generated summary.

```bash
export ANTHROPIC_API_KEY=your_key_here   # Windows: set ANTHROPIC_API_KEY=...
```

## Running

**1. Start the backend (from the `backend/` folder):**
```bash
cd backend
uvicorn main:app --reload --port 8000
```

**2. Start the frontend (in a new terminal, from the project root):**
```bash
streamlit run frontend/app.py
```

Open the Streamlit URL it prints (usually http://localhost:8501), upload a
PDF, and start asking questions.

## API Reference (for React/other frontends)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload` | multipart file upload (PDF), returns chunks indexed |
| POST | `/query` | `{"question": "...", "source_filter": null, "top_k": 4}` |
| GET | `/documents` | list indexed document names |
| DELETE | `/documents/{name}` | remove a document from the index |
| GET | `/health` | healthcheck |

## Where to Extend

- **Chunking**: swap the naive sliding-window chunker in `ingest.py` for
  LangChain's `RecursiveCharacterTextSplitter` or a semantic chunker.
- **Hybrid search**: add BM25 keyword scoring alongside vector search and
  merge/re-rank results (improves accuracy on exact-term queries).
- **Multi-file formats**: extend `ingest.py` to handle `.docx`, `.txt`, `.md`
  using the same chunk → embed → store pattern.
- **Streaming answers**: switch the LLM call in `rag.py` to
  `client.messages.stream(...)` and expose a `/query/stream` SSE endpoint.
- **Scale to FAISS**: swap `vectorstore.py`'s backend from ChromaDB to FAISS
  if you need to index millions of chunks — you'll need to manage metadata
  storage separately (e.g. SQLite) since FAISS only stores vectors.
