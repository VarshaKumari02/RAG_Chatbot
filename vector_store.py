"""
vector_store.py
───────────────
Manages:
  1. Generating embeddings via SentenceTransformers
  2. Storing/searching vectors with ChromaDB (persistent)
  3. Per-document deletion by UUID doc_id

Document metadata (filenames, upload time, chunk counts) is intentionally
stored in doc_registry.py (SQLite) rather than here. Fetching that info
from ChromaDB required scanning every chunk's metadata — O(N) per listing.
"""

import os
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, VECTOR_STORE_DIR, TOP_K_RESULTS


class VectorStore:
    """
    ChromaDB-backed persistent vector store with SentenceTransformer embeddings.

    Usage:
        store = VectorStore()
        store.add_chunks(chunks)
        results = store.search("What is RAG?")
    """

    COLLECTION_NAME = "rag_documents"

    def __init__(self):
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

        print(f"[VectorStore] Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        # Persistent ChromaDB client — data saved to disk automatically
        self.client = chromadb.PersistentClient(
            path=VECTOR_STORE_DIR,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )

        print(f"[VectorStore] Ready — {self.total_chunks} chunks in store.")

    # ── Indexing ──────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Dict[str, Any]], doc_id: str) -> int:
        """
        Embed and index a list of chunk dicts.

        Args:
            chunks: Output of document_processor.process_document()
            doc_id: UUID from doc_registry — stored in each chunk's metadata
                    so we can do O(1) targeted deletions later.

        Returns:
            Number of new chunks added.
        """
        if not chunks:
            return 0

        texts     = [c["text"]     for c in chunks]
        sources   = [c["source"]   for c in chunks]   # original filename (display)
        chunk_ids = [c["chunk_id"] for c in chunks]

        print(f"[VectorStore] Embedding {len(texts)} chunks…")
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        embeddings = [e.tolist() for e in embeddings]

        # Unique Chroma IDs: “<doc_id>__<chunk_id>”
        ids = [f"{doc_id}__{cid}" for cid in chunk_ids]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {"source": s, "chunk_id": c, "doc_id": doc_id}
                for s, c in zip(sources, chunk_ids)
            ],
        )

        print(f"[VectorStore] Indexed {len(chunks)} chunks. Total: {self.total_chunks}")
        return len(chunks)

    def remove_by_doc_id(self, doc_id: str) -> int:
        """
        Remove all chunks belonging to a particular document UUID.

        Returns:
            Number of chunks deleted (0 if doc_id not found).
        """
        results = self.collection.get(where={"doc_id": doc_id})
        if results and results["ids"]:
            self.collection.delete(ids=results["ids"])
            count = len(results["ids"])
            print(f"[VectorStore] Removed {count} chunks for doc_id='{doc_id}'.")
            return count
        return 0

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """
        Semantic search over the vector store.

        Returns:
            List of top-k matching chunk dicts, each with a 'score' key.
        """
        if self.total_chunks == 0:
            return []

        query_embedding = self.model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.total_chunks),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text":     doc,
                "source":   meta.get("source", "unknown"),
                "chunk_id": meta.get("chunk_id", 0),
                "score":    round(1 - dist, 4),  # convert cosine distance → similarity
            })

        return chunks

    # ── Info ──────────────────────────────────────────────────────────

    @property
    def total_chunks(self) -> int:
        return self.collection.count()
