"""
main.py
───────
FastAPI application — RAG chatbot REST + streaming API.

Optimizations in this version
──────────────────────────────
1. SSE Streaming        → GET /ask/stream — AsyncGroq stream=True, tokens
                          arrive in real-time via Server-Sent Events.
2. Semantic Chunking    → document_processor.py uses LangChain's
                          RecursiveCharacterTextSplitter.
3. Session Memory       → session_id in every /ask request, managed in
                          RAGEngine per-session dict.
4. Background Uploads   → POST /upload returns 202 immediately; embedding
                          runs in a BackgroundTask thread.
5. Polished UI          → static/index.html (glassmorphism, Tailwind CDN,
                          marked.js markdown rendering, drag-and-drop).

Endpoints:
  POST   /upload                — Upload document (202 + background embed)
  GET    /upload/status/{doc_id}— Poll indexing status
  GET    /ask/stream            — Stream answer (SSE)
  POST   /ask                   — Non-streaming answer (fallback)
  GET    /documents             — List indexed documents
  DELETE /documents/{doc_id}    — Remove a document
  POST   /clear-history         — Reset session memory
  GET    /health                — Health check
"""

import os
import uuid
import logging
from pathlib import Path
from typing import Annotated, Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from config import UPLOAD_DIR, TOP_K_RESULTS
import doc_registry
from document_processor import process_document
from vector_store import VectorStore
from rag_engine import RAGEngine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieval-Augmented Generation chatbot — upload documents, ask questions.",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Singletons ────────────────────────────────────────────────────────────────
os.makedirs(UPLOAD_DIR, exist_ok=True)
doc_registry.init_db()
_vector_store = VectorStore()
_rag_engine   = RAGEngine(_vector_store)

# ── In-memory upload status tracker ──────────────────────────────────────────
# Maps doc_id -> {"status": "processing"|"ready"|"error", "message": str}
_upload_status: Dict[str, Dict] = {}


# ── Dependency providers ──────────────────────────────────────────────────────

def get_vector_store() -> VectorStore:
    return _vector_store

def get_rag_engine() -> RAGEngine:
    return _rag_engine

VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
RAGEngineDep   = Annotated[RAGEngine,   Depends(get_rag_engine)]


# ── Pydantic models ───────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question:   str
    top_k:      Optional[int] = TOP_K_RESULTS
    session_id: Optional[str] = None


class AskResponse(BaseModel):
    answer:  str
    sources: List[str]
    model:   str


class DocumentInfo(BaseModel):
    doc_id:        str
    original_name: str
    chunk_count:   int
    uploaded_at:   str


class UploadAccepted(BaseModel):
    doc_id:  str
    message: str


class StatusResponse(BaseModel):
    status:  str
    message: str


class ClearHistoryRequest(BaseModel):
    session_id: Optional[str] = None


# ── Background task ───────────────────────────────────────────────────────────

def _index_document_background(
    doc_id: str,
    save_path: str,
    original_name: str,
    stored_name: str,
    vs: VectorStore,
) -> None:
    """
    Runs in a FastAPI BackgroundTask thread.
    Processes and indexes the document, then updates _upload_status and registry.
    """
    try:
        _upload_status[doc_id] = {"status": "processing", "message": "Extracting and embedding…"}
        chunks = process_document(save_path, display_name=original_name)
        count  = vs.add_chunks(chunks, doc_id=doc_id)
        doc_registry.register_document(
            original_name=original_name,
            stored_name=stored_name,
            chunk_count=count,
            doc_id=doc_id,
        )
        _upload_status[doc_id] = {
            "status": "ready",
            "message": f"Indexed {count} chunks successfully.",
        }
        logger.info("[upload] doc_id=%s ready (%d chunks)", doc_id, count)
    except ValueError as exc:
        _upload_status[doc_id] = {"status": "error", "message": str(exc)}
        if os.path.exists(save_path):
            os.remove(save_path)
        logger.warning("[upload] doc_id=%s ValueError: %s", doc_id, exc)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return JSONResponse({"message": "RAG Chatbot API v4. Visit /docs for the API."})


@app.get("/health", tags=["System"])
async def health(vs: VectorStoreDep, engine: RAGEngineDep):
    docs = doc_registry.list_documents()
    return {
        "status":          "ok",
        "total_chunks":    vs.total_chunks,
        "total_documents": len(docs),
        "active_sessions": engine.active_session_count(),
        "model":           "groq",
    }


@app.post("/upload", response_model=UploadAccepted, status_code=202, tags=["Documents"])
def upload_document(
    background_tasks: BackgroundTasks,
    vs: VectorStoreDep,
    file: UploadFile = File(...),
):
    """
    Upload a document. Returns **202 Accepted** immediately.
    Embedding and indexing happen in a background thread.
    Poll `GET /upload/status/{doc_id}` to know when the document is ready.
    """
    allowed_extensions = {".pdf", ".docx", ".txt", ".pptx", ".xlsx"}
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(allowed_extensions)}",
        )

    doc_id      = str(uuid.uuid4())
    stored_name = f"{doc_id}{ext}"
    save_path   = os.path.join(UPLOAD_DIR, stored_name)

    content = file.file.read()
    with open(save_path, "wb") as out:
        out.write(content)

    # Mark as queued and hand off to background
    _upload_status[doc_id] = {"status": "processing", "message": "Queued for indexing…"}
    background_tasks.add_task(
        _index_document_background,
        doc_id, save_path, file.filename, stored_name, vs,
    )

    return UploadAccepted(
        doc_id=doc_id,
        message=f"'{file.filename}' accepted. Indexing in background. Poll /upload/status/{doc_id}.",
    )


@app.get("/upload/status/{doc_id}", response_model=StatusResponse, tags=["Documents"])
async def upload_status(doc_id: str):
    """Poll indexing status for a recently uploaded document."""
    info = _upload_status.get(doc_id)
    if not info:
        record = doc_registry.get_document(doc_id)
        if record:
            return StatusResponse(status="ready", message=f"Indexed {record['chunk_count']} chunks.")
        raise HTTPException(status_code=404, detail=f"No upload job found for doc_id='{doc_id}'.")
    return StatusResponse(status=info["status"], message=info["message"])


@app.get("/ask/stream", tags=["Chat"])
async def ask_stream(
    engine: RAGEngineDep,
    vs: VectorStoreDep,
    question: str = Query(..., description="The question to ask"),
    session_id: Optional[str] = Query(None, description="Session UUID for isolated history"),
    top_k: int = Query(TOP_K_RESULTS, ge=1, le=20),
):
    """
    Stream the answer as Server-Sent Events (SSE).
    The UI connects once and receives token-by-token output — matching ChatGPT UX.

    SSE event format:
      data: {"type": "sources", "sources": ["file.pdf"]}
      data: {"type": "token",   "text": "Hello"}
      data: {"type": "done"}
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if vs.total_chunks == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload a document first.",
        )

    sid = session_id or str(uuid.uuid4())

    return StreamingResponse(
        engine.stream_answer(question, top_k=top_k, session_id=sid),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Non-streaming fallback ────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse, tags=["Chat"])
def ask_question(body: AskRequest, vs: VectorStoreDep, engine: RAGEngineDep):
    """
    Non-streaming answer endpoint (for API clients / testing).
    Use GET /ask/stream for real-time token streaming in the UI.
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if vs.total_chunks == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload a document first.",
        )

    from groq import RateLimitError, AuthenticationError, APIConnectionError

    session_id = body.session_id or str(uuid.uuid4())

    try:
        result = engine.generate_answer(
            body.question,
            top_k=body.top_k,
            session_id=session_id,
        )
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Groq authentication failed. Check GROQ_API_KEY.")
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Groq rate limit reached. Please wait and retry.")
    except APIConnectionError:
        raise HTTPException(status_code=503, detail="Could not reach Groq API. Check internet connection.")

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        model=result["model"],
    )


@app.get("/documents", response_model=List[DocumentInfo], tags=["Documents"])
async def list_documents():
    """Return metadata for all indexed documents (O(1) SQLite lookup)."""
    rows = doc_registry.list_documents()
    return [
        DocumentInfo(
            doc_id=r["doc_id"],
            original_name=r["original_name"],
            chunk_count=r["chunk_count"],
            uploaded_at=r["uploaded_at"],
        )
        for r in rows
    ]


@app.delete("/documents/{doc_id}", response_model=StatusResponse, tags=["Documents"])
def delete_document(doc_id: str, vs: VectorStoreDep):
    """Remove a document and all its chunks by UUID."""
    record = doc_registry.get_document(doc_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    vs.remove_by_doc_id(doc_id)

    file_path = os.path.join(UPLOAD_DIR, record["stored_name"])
    if os.path.exists(file_path):
        os.remove(file_path)

    doc_registry.delete_document(doc_id)
    _upload_status.pop(doc_id, None)

    return StatusResponse(
        status="success",
        message=f"'{record['original_name']}' (id={doc_id}) removed.",
    )


@app.post("/clear-history", response_model=StatusResponse, tags=["Chat"])
async def clear_conversation_history(
    engine: RAGEngineDep,
    body: ClearHistoryRequest = ClearHistoryRequest(),
):
    """Clear conversation memory for a specific session."""
    session_id = body.session_id or "default"
    engine.clear_history(session_id=session_id)
    return StatusResponse(
        status="success",
        message=f"Conversation history cleared for session '{session_id}'.",
    )


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
