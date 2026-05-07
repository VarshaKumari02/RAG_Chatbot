"""
config.py — Central configuration loaded from .env
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", 5))

# ── Storage ───────────────────────────────────────────────────────────────────
VECTOR_STORE_DIR: str = "vector_store"
UPLOAD_DIR: str = "uploads"
