"""
Handles turning a raw PDF into clean, page-tagged text chunks
ready for embedding.
"""
from typing import List, Dict
import pdfplumber
import uuid

from config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_pages(pdf_path: str) -> List[Dict]:
    """
    Extracts text from each page of a PDF.
    Returns: [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"page": i, "text": text})
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Simple sliding-window character chunker with overlap.
    Splits on sentence-ish boundaries where possible to avoid
    cutting words in half mid-sentence.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # try to end on a sentence boundary near the chunk edge
        if end < text_len:
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + chunk_size * 0.5:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break
        start = end - overlap  # step back for overlap

    return chunks


def process_pdf(pdf_path: str, source_name: str) -> List[Dict]:
    """
    Full pipeline: PDF -> pages -> chunks -> metadata-tagged records.

    Returns a list of dicts, each ready to be embedded and stored:
    {
        "id": unique chunk id,
        "text": chunk text,
        "metadata": {"source": filename, "page": page_number}
    }
    """
    pages = extract_pages(pdf_path)
    records = []

    for page_data in pages:
        page_num = page_data["page"]
        page_text = page_data["text"]
        chunks = chunk_text(page_text)

        for chunk in chunks:
            records.append({
                "id": str(uuid.uuid4()),
                "text": chunk,
                "metadata": {
                    "source": source_name,
                    "page": page_num,
                }
            })

    return records
