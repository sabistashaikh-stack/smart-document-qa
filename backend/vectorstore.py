"""
Wraps ChromaDB + a HuggingFace sentence-transformer model
into a simple add/query interface.
"""
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME, TOP_K


class VectorStore:
    def __init__(self):
        # Persistent client -> index survives server restarts
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        # HuggingFace embedding function (runs locally, no API calls)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )

        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, records: List[Dict]):
        """
        records: [{"id": ..., "text": ..., "metadata": {...}}, ...]
        Embeds and upserts them into the collection.
        """
        if not records:
            return

        self.collection.upsert(
            ids=[r["id"] for r in records],
            documents=[r["text"] for r in records],
            metadatas=[r["metadata"] for r in records],
        )

    def query(self, query_text: str, top_k: int = TOP_K, source_filter: str = None) -> List[Dict]:
        """
        Returns top_k most relevant chunks for the query, with metadata + distance score.
        """
        where = {"source": source_filter} if source_filter else None

        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where,
        )

        hits = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            hits.append({
                "text": doc,
                "source": meta.get("source"),
                "page": meta.get("page"),
                "score": round(1 - dist, 4),  # convert distance -> similarity
            })

        return hits

    def list_sources(self) -> List[str]:
        """Returns distinct document names currently indexed."""
        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted(set(m["source"] for m in all_meta)) if all_meta else []

    def delete_source(self, source_name: str):
        self.collection.delete(where={"source": source_name})


# single shared instance used across the app
vector_store = VectorStore()
