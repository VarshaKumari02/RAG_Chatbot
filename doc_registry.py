"""
doc_registry.py
───────────────
A tiny SQLite-backed registry that stores document-level metadata.

Why not use ChromaDB for this?
  The naive approach (vector_store.sources) fetches ALL chunk metadata from
  ChromaDB to build a document list — that is O(N) in chunks and causes memory
  spikes as the database grows.  A dedicated SQLite table is O(1) for listing
  documents and adds zero heavyweight dependencies.

Schema
──────
  documents(
      doc_id       TEXT PRIMARY KEY,   -- UUID assigned at upload time
      original_name TEXT NOT NULL,     -- original filename the user uploaded
      stored_name   TEXT NOT NULL,     -- actual filename on disk (UUID-prefixed)
      chunk_count   INTEGER NOT NULL,
      uploaded_at   TEXT NOT NULL      -- ISO-8601 UTC timestamp
  )
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

from config import UPLOAD_DIR

_DB_PATH = str(Path(UPLOAD_DIR) / "doc_registry.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the documents table if it doesn't exist yet."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id        TEXT PRIMARY KEY,
                original_name TEXT NOT NULL,
                stored_name   TEXT NOT NULL,
                chunk_count   INTEGER NOT NULL DEFAULT 0,
                uploaded_at   TEXT NOT NULL
            )
            """
        )
        conn.commit()


# ── Write operations ──────────────────────────────────────────────────────────

def register_document(
    original_name: str,
    stored_name: str,
    chunk_count: int,
    doc_id: str,
) -> str:
    """
    Add a new document record.  The caller must supply the doc_id (UUID)
    that was already used when indexing chunks into ChromaDB, so both
    systems share the exact same identifier.

    Returns the doc_id for convenience.
    """
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO documents (doc_id, original_name, stored_name, chunk_count, uploaded_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (doc_id, original_name, stored_name, chunk_count, now),
        )
        conn.commit()
    return doc_id


def delete_document(doc_id: str) -> bool:
    """Remove a document record by doc_id. Returns True if a row was deleted."""
    with _connect() as conn:
        cursor = conn.execute(
            "DELETE FROM documents WHERE doc_id = ?", (doc_id,)
        )
        conn.commit()
        return cursor.rowcount > 0


# ── Read operations ───────────────────────────────────────────────────────────

def get_document(doc_id: str) -> Optional[Dict]:
    """Fetch a single document record by doc_id, or None if not found."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
    return dict(row) if row else None


def find_by_original_name(original_name: str) -> Optional[Dict]:
    """Return the first document record whose original_name matches, or None."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE original_name = ?", (original_name,)
        ).fetchone()
    return dict(row) if row else None


def list_documents() -> List[Dict]:
    """Return all document records ordered by upload time (newest first)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY uploaded_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]
