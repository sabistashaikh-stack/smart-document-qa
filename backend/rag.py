"""
Retrieval-Augmented Generation engine.

Flow:
  1. Embed the user's question
  2. Retrieve top-k relevant chunks from ChromaDB
  3. If an Anthropic API key is set -> ask the LLM to answer using ONLY
     the retrieved context, with citations.
  4. If no key is set -> fall back to "extractive mode": just return the
     most relevant chunks directly (still useful, no LLM required).
"""
from typing import Dict
from vectorstore import vector_store
from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, TOP_K


def _build_context_block(hits) -> str:
    """Formats retrieved chunks into a labeled context block for the LLM."""
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(
            f"[Chunk {i} | Source: {h['source']} | Page: {h['page']}]\n{h['text']}"
        )
    return "\n\n".join(blocks)


def _generate_with_llm(question: str, hits) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    context = _build_context_block(hits)

    system_prompt = (
        "You are a precise document QA assistant. Answer the user's question "
        "using ONLY the information in the provided context chunks. "
        "If the answer isn't in the context, say so clearly — do not make things up. "
        "Always cite which chunk number(s) support your answer, e.g. (Chunk 2, Page 5)."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=800,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return "".join(block.text for block in response.content if block.type == "text")


def answer_question(question: str, source_filter: str = None, top_k: int = TOP_K) -> Dict:
    """
    Main entry point. Returns:
    {
        "answer": str,
        "mode": "generative" | "extractive",
        "sources": [{"source": ..., "page": ..., "score": ..., "text": ...}, ...]
    }
    """
    hits = vector_store.query(question, top_k=top_k, source_filter=source_filter)

    if not hits:
        return {
            "answer": "No relevant content found. Try uploading a document first, "
                      "or rephrase your question.",
            "mode": "none",
            "sources": [],
        }

    if ANTHROPIC_API_KEY:
        try:
            answer_text = _generate_with_llm(question, hits)
            mode = "generative"
        except Exception as e:
            # graceful fallback if the API call fails for any reason
            answer_text = (
                f"[LLM call failed: {e}] Showing top matching passages instead:\n\n"
                + "\n\n---\n\n".join(h["text"] for h in hits[:2])
            )
            mode = "extractive"
    else:
        # No API key configured -> extractive mode: just surface best chunks
        answer_text = (
            "No LLM API key configured — showing the most relevant passages found:\n\n"
            + "\n\n---\n\n".join(h["text"] for h in hits[:2])
        )
        mode = "extractive"

    return {
        "answer": answer_text,
        "mode": mode,
        "sources": hits,
    }
