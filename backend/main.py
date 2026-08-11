"""
FastAPI backend for the Smart Document QA system.

Endpoints:
  POST /upload         -> upload a PDF, extract + chunk + embed + index it
  POST /query          -> ask a natural language question, get answer + sources
  GET  /documents      -> list all indexed documents
  DELETE /documents/{name} -> remove a document from the index
  GET  /health         -> simple healthcheck
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import shutil
import os

from config import UPLOAD_DIR, TOP_K
from ingest import process_pdf
from vectorstore import vector_store
from rag import answer_question

app = FastAPI(title="Smart Document QA System")

# allow the Streamlit / React frontend to call this API from another port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    source_filter: Optional[str] = None
    top_k: int = TOP_K


class QueryResponse(BaseModel):
    answer: str
    mode: str
    sources: List[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        records = process_pdf(str(save_path), source_name=file.filename)
        if not records:
            raise HTTPException(
                status_code=422,
                detail="No extractable text found in this PDF (it may be scanned/image-only)."
            )
        vector_store.add_documents(records)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    return {
        "filename": file.filename,
        "chunks_indexed": len(records),
        "status": "indexed",
    }


@app.post("/query", response_model=QueryResponse)
def query_documents(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = answer_question(
        question=req.question,
        source_filter=req.source_filter,
        top_k=req.top_k,
    )
    return result


@app.get("/documents")
def list_documents():
    return {"documents": vector_store.list_sources()}


@app.delete("/documents/{name}")
def delete_document(name: str):
    vector_store.delete_source(name)
    upload_path = UPLOAD_DIR / name
    if upload_path.exists():
        os.remove(upload_path)
    return {"status": "deleted", "document": name}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
