"""
document_processor.py
─────────────────────
Responsible for:
  1. Accepting uploaded files (PDF, DOCX, TXT, PPTX, XLSX)
  2. Extracting raw text
  3. Splitting text into overlapping chunks
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any

import pypdf
import docx
from pptx import Presentation
import openpyxl

from config import CHUNK_SIZE, CHUNK_OVERLAP


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Remove excessive whitespace and non-printable characters."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    return text.strip()


# ── Extractors ────────────────────────────────────────────────────────────────

def extract_text_from_pdf(path: str) -> str:
    text_parts = []
    with open(path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return _clean_text("\n".join(text_parts))


def extract_text_from_docx(path: str) -> str:
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return _clean_text("\n".join(paragraphs))


def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return _clean_text(f.read())


def extract_text_from_pptx(path: str) -> str:
    prs = Presentation(path)
    text_parts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                text_parts.append(shape.text)
    return _clean_text("\n".join(text_parts))


def extract_text_from_xlsx(path: str) -> str:
    wb = openpyxl.load_workbook(path, data_only=True)
    text_parts = []
    for sheet in wb.worksheets:
        for row in sheet.iter_rows(values_only=True):
            row_text = " | ".join(str(cell) for cell in row if cell is not None)
            if row_text.strip():
                text_parts.append(row_text)
    return _clean_text("\n".join(text_parts))


EXTRACTOR_MAP = {
    ".pdf":  extract_text_from_pdf,
    ".docx": extract_text_from_docx,
    ".txt":  extract_text_from_txt,
    ".pptx": extract_text_from_pptx,
    ".xlsx": extract_text_from_xlsx,
}


def extract_text(file_path: str) -> str:
    """Route to the correct extractor based on file extension."""
    ext = Path(file_path).suffix.lower()
    extractor = EXTRACTOR_MAP.get(ext)
    if not extractor:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(EXTRACTOR_MAP.keys())}")
    return extractor(file_path)


# ── Chunker ───────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    source: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping word-based chunks.

    Returns a list of dicts:
      { "text": <str>, "source": <filename>, "chunk_id": <int> }
    """
    words = text.split()
    chunks: List[Dict[str, Any]] = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text_str = " ".join(words[start:end])
        chunks.append({
            "text": chunk_text_str,
            "source": source,
            "chunk_id": chunk_id,
        })
        chunk_id += 1
        start += chunk_size - overlap

    return chunks


# ── Main entry ────────────────────────────────────────────────────────────────

def process_document(file_path: str) -> List[Dict[str, Any]]:
    """
    Full pipeline: extract → clean → chunk.

    Args:
        file_path: Absolute path to the uploaded document.

    Returns:
        List of chunk dicts ready to be embedded.
    """
    filename = Path(file_path).name
    raw_text = extract_text(file_path)

    if not raw_text:
        raise ValueError(f"Could not extract any text from '{filename}'. The file may be empty or image-based.")

    chunks = chunk_text(raw_text, source=filename)
    return chunks
