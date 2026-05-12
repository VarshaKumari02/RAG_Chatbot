"""
config.py
─────────
Central configuration using Pydantic BaseSettings.

Why BaseSettings?
  - os.getenv() silently returns an empty string for missing keys. The app
    then starts fine but crashes at the first LLM call with a cryptic auth
    error.
  - BaseSettings validates required fields *at import time*. If GROQ_API_KEY
    is absent the process exits immediately with a clear error message, not
    a runtime 500 somewhere deep in a request handler.
  - All environment variables are documented in one place with types and
    defaults, making the config self-describing.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings — populated from environment variables / .env file.
    Fields with no default are *required*; the app refuses to start without them.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM (required) ────────────────────────────────────────────────────────
    groq_api_key: str = Field(..., description="Groq API key — REQUIRED")
    groq_model:   str = Field("llama-3.3-70b-versatile", description="Groq model ID")

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = Field("all-MiniLM-L6-v2", description="SentenceTransformer model name")

    # ── Chunking ──────────────────────────────────────────────────────────────
    # Values are now CHARACTER counts (not word counts) to match
    # LangChain's RecursiveCharacterTextSplitter.
    chunk_size:    int = Field(1500, ge=200,  description="Max characters per chunk")
    chunk_overlap: int = Field(200,  ge=0,   description="Overlap between chunks in characters")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k_results: int = Field(5, ge=1, le=20, description="Top-K chunks to retrieve per query")

    # ── Storage ───────────────────────────────────────────────────────────────
    vector_store_dir: str = Field("vector_store", description="Directory for ChromaDB persistence")
    upload_dir:       str = Field("uploads",      description="Directory for uploaded files")

    @field_validator("groq_api_key")
    @classmethod
    def groq_api_key_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "GROQ_API_KEY is missing or empty. "
                "Add it to your .env file: GROQ_API_KEY=gsk_..."
            )
        return v


# ── Module-level singleton ────────────────────────────────────────────────────
# Instantiated once at import time — validation fires here, not at request time.
settings = Settings()

# ── Backward-compatible aliases ───────────────────────────────────────────────
# Other modules already import these names directly (e.g. `from config import GROQ_API_KEY`).
# Keeping them avoids touching every file that imports config.
GROQ_API_KEY:     str = settings.groq_api_key
GROQ_MODEL:       str = settings.groq_model
EMBEDDING_MODEL:  str = settings.embedding_model
CHUNK_SIZE:       int = settings.chunk_size
CHUNK_OVERLAP:    int = settings.chunk_overlap
TOP_K_RESULTS:    int = settings.top_k_results
VECTOR_STORE_DIR: str = settings.vector_store_dir
UPLOAD_DIR:       str = settings.upload_dir
