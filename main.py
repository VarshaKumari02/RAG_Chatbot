"""
main.py
───────
FastAPI application exposing the RAG chatbot as a REST API.

Endpoints:
  POST /upload        — Upload and index a document
  POST /ask           — Ask a question (RAG pipeline)
  GET  /documents     — List indexed documents
  DELETE /documents/{name} — Remove a document from the index
  POST /clear-history — Reset conversation memory
  GET  /health        — Health check
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import aiofiles

from config import UPLOAD_DIR, TOP_K_RESULTS
from document_processor import process_document
from vector_store import VectorStore
from rag_engine import RAGEngine


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieval-Augmented Generation chatbot — upload documents, ask questions.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (the frontend)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Singletons ────────────────────────────────────────────────────────────────
os.makedirs(UPLOAD_DIR, exist_ok=True)
vector_store = VectorStore()
rag_engine   = RAGEngine(vector_store)


# ── Pydantic models ───────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = TOP_K_RESULTS


class AskResponse(BaseModel):
    answer:   str
    sources:  List[str]
    model:    str


class DocumentInfo(BaseModel):
    name:         str
    total_chunks: int


class StatusResponse(BaseModel):
    status:  str
    message: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return JSONResponse({"message": "RAG Chatbot API is running. Visit /docs for the API."})


@app.get("/health", tags=["System"])
async def health():
    return {
        "status":       "ok",
        "total_chunks": vector_store.total_chunks,
        "indexed_docs": vector_store.sources,
        "model":        rag_engine.client._base_url if hasattr(rag_engine.client, "_base_url") else "groq",
    }


@app.post("/upload", response_model=StatusResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, DOCX, TXT, PPTX, XLSX).
    The document is processed, chunked, embedded, and indexed automatically.
    """
    allowed_extensions = {".pdf", ".docx", ".txt", ".pptx", ".xlsx"}
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(allowed_extensions)}",
        )

    # Check if already indexed
    if file.filename in vector_store.sources:
        raise HTTPException(
            status_code=409,
            detail=f"'{file.filename}' is already indexed. Delete it first to re-index.",
        )

    # Save uploaded file
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    async with aiofiles.open(save_path, "wb") as out:
        content = await file.read()
        await out.write(content)

    # Process & index
    try:
        chunks = process_document(save_path)
        count  = vector_store.add_chunks(chunks)
    except ValueError as e:
        os.remove(save_path)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    return StatusResponse(
        status="success",
        message=f"'{file.filename}' indexed successfully ({count} chunks added).",
    )


@app.post("/ask", response_model=AskResponse, tags=["Chat"])
async def ask_question(body: AskRequest):
    """
    Ask a question. The RAG pipeline retrieves relevant chunks and generates an answer.
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if vector_store.total_chunks == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Please upload at least one document first.",
        )

    try:
        result = rag_engine.generate_answer(body.question, top_k=body.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        model=result["model"],
    )


@app.get("/documents", response_model=List[str], tags=["Documents"])
async def list_documents():
    """Return a list of all indexed document names."""
    return vector_store.sources


@app.delete("/documents/{document_name}", response_model=StatusResponse, tags=["Documents"])
async def delete_document(document_name: str):
    """Remove a document and all its chunks from the index."""
    removed = vector_store.remove_source(document_name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"'{document_name}' not found in index.")

    # Also remove the file from uploads if present
    file_path = os.path.join(UPLOAD_DIR, document_name)
    if os.path.exists(file_path):
        os.remove(file_path)

    return StatusResponse(
        status="success",
        message=f"'{document_name}' removed from index.",
    )


@app.post("/clear-history", response_model=StatusResponse, tags=["Chat"])
async def clear_conversation_history():
    """Clear the conversation memory so the next question starts fresh."""
    rag_engine.clear_history()
    return StatusResponse(status="success", message="Conversation history cleared.")


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
