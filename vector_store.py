"""
vector_store.py
───────────────
Manages:
  1. Generating embeddings via SentenceTransformers
  2. Storing/searching vectors with ChromaDB (persistent)
  3. Per-document deletion support
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

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Embed and index a list of chunk dicts.

        Args:
            chunks: Output of document_processor.process_document()

        Returns:
            Number of new chunks added.
        """
        if not chunks:
            return 0

        texts      = [c["text"]      for c in chunks]
        sources    = [c["source"]    for c in chunks]
        chunk_ids  = [c["chunk_id"]  for c in chunks]

        print(f"[VectorStore] Embedding {len(texts)} chunks…")
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        embeddings = [e.tolist() for e in embeddings]

        # Build unique IDs: "<source>__<chunk_id>"
        ids = [f"{src}__{cid}" for src, cid in zip(sources, chunk_ids)]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": s, "chunk_id": c} for s, c in zip(sources, chunk_ids)],
        )

        print(f"[VectorStore] Indexed {len(chunks)} chunks. Total: {self.total_chunks}")
        return len(chunks)

    def remove_source(self, source_name: str) -> bool:
        """
        Remove all chunks belonging to a particular source document.

        Returns:
            True if source was found and removed, False otherwise.
        """
        if source_name not in self.sources:
            return False

        # Query all IDs for this source then delete them
        results = self.collection.get(where={"source": source_name})
        if results and results["ids"]:
            self.collection.delete(ids=results["ids"])
            print(f"[VectorStore] Removed {len(results['ids'])} chunks for '{source_name}'.")
            return True

        return False

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

    # ── Info ──────────────────────────────────────────────────────────────────

    @property
    def total_chunks(self) -> int:
        return self.collection.count()

    @property
    def sources(self) -> List[str]:
        """Return unique list of indexed source document names."""
        if self.total_chunks == 0:
            return []
        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        seen = []
        for m in all_meta:
            src = m.get("source", "")
            if src and src not in seen:
                seen.append(src)
        return seen
