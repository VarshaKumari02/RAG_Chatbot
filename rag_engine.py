"""
rag_engine.py
─────────────
The core RAG pipeline:
  1. Retrieve relevant chunks from the vector store
  2. Build a prompt with the retrieved context
  3. Call the Groq LLM and return the answer

Session Management:
  - Each client sends a unique session_id (UUID generated in the browser).
  - Conversation history is stored per session_id in self._sessions.
  - Sessions idle for more than SESSION_TTL_MINUTES are automatically purged.

Retry / Resilience:
  - The Groq API call is wrapped with tenacity: up to 4 attempts,
    exponential back-off (1s → 8s), retrying only on RateLimitError
    and APIConnectionError so real errors surface immediately.
"""

import time
from dataclasses import dataclass, field
from groq import Groq, AsyncGroq
from groq import RateLimitError, APIConnectionError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from typing import AsyncIterator, List, Dict, Any, Tuple
import logging
import json

logger = logging.getLogger(__name__)

from config import GROQ_API_KEY, GROQ_MODEL, TOP_K_RESULTS
from vector_store import VectorStore

# Sessions idle longer than this (in minutes) are purged automatically
SESSION_TTL_MINUTES: int = 30


@dataclass
class _Session:
    """Holds per-user conversation history and last-activity timestamp."""
    history: List[Dict[str, str]] = field(default_factory=list)
    last_active: float = field(default_factory=time.monotonic)


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an intelligent, helpful assistant that answers questions \
strictly based on the provided document context.

Rules:
- Answer ONLY from the context provided. Do NOT use outside knowledge.
- If the answer is not in the context, say: \
  "I couldn't find relevant information in the uploaded documents."
- Be concise, clear, and structured. Use bullet points or numbered lists where helpful.
- Always cite the source document name(s) at the end of your answer.
- Never make up information or hallucinate facts."""


class RAGEngine:
    """
    Retrieval-Augmented Generation engine.

    Args:
        vector_store: An initialised VectorStore instance.
    """

    def __init__(self, vector_store: VectorStore):
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Please add it to your .env file."
            )
        self.client       = Groq(api_key=GROQ_API_KEY)
        self.async_client = AsyncGroq(api_key=GROQ_API_KEY)   # for SSE streaming
        self.store = vector_store
        # Per-session history: { session_id -> _Session }
        self._sessions: Dict[str, _Session] = {}

    # ── Session helpers ───────────────────────────────────────────────────────

    def _get_session(self, session_id: str) -> _Session:
        """Return existing session or create a new one, then touch its timestamp."""
        if session_id not in self._sessions:
            self._sessions[session_id] = _Session()
        session = self._sessions[session_id]
        session.last_active = time.monotonic()
        return session

    def _purge_expired_sessions(self) -> None:
        """Remove sessions that have been idle longer than SESSION_TTL_MINUTES."""
        cutoff = time.monotonic() - SESSION_TTL_MINUTES * 60
        expired = [sid for sid, s in self._sessions.items() if s.last_active < cutoff]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            print(f"[RAGEngine] Purged {len(expired)} expired session(s).")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """Return the most relevant chunks for a query."""
        return self.store.search(query, top_k=top_k)

    # ── Prompt builder ────────────────────────────────────────────────────────

    def _build_context_prompt(
        self, query: str, chunks: List[Dict[str, Any]]
    ) -> Tuple[str, List[str]]:
        """
        Build the user message that includes retrieved context.

        Returns:
            (user_message_str, list_of_source_names)
        """
        if not chunks:
            return (
                f"Question: {query}\n\n"
                "(No relevant context found in the uploaded documents.)",
                [],
            )

        context_blocks = []
        sources = []
        for i, chunk in enumerate(chunks, 1):
            src = chunk.get("source", "Unknown")
            if src not in sources:
                sources.append(src)
            context_blocks.append(
                f"[Context {i} — Source: {src}]\n{chunk['text']}"
            )

        context_str = "\n\n".join(context_blocks)
        user_message = (
            f"Use the following document excerpts to answer the question.\n\n"
            f"{context_str}\n\n"
            f"Question: {query}"
        )
        return user_message, sources

    # ── LLM call ──────────────────────────────────────────────────────────────

    def generate_answer(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        use_history: bool = True,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline: retrieve → build prompt → generate answer.

        Args:
            query:       User's question.
            top_k:       Number of chunks to retrieve.
            use_history: Whether to include conversation history.

        Returns:
            {
              "answer":   <str>,
              "sources":  <List[str]>,
              "chunks":   <List[dict]>,
              "model":    <str>,
            }
        """
        # Housekeeping — remove stale sessions on every call
        self._purge_expired_sessions()

        # Step 1 — Retrieve
        chunks = self.retrieve(query, top_k=top_k)

        # Step 2 — Build context prompt
        user_message, sources = self._build_context_prompt(query, chunks)

        # Step 3 — Compose messages list
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if use_history:
            session = self._get_session(session_id)
            # Include last 10 messages (5 user+assistant turn pairs)
            messages.extend(session.history[-10:])

        messages.append({"role": "user", "content": user_message})

        # Step 4 — Call Groq (with automatic retry on transient failures)
        response = self._call_groq_with_retry(messages)

        answer = response.choices[0].message.content.strip()

        # Step 5 — Update this session's conversation history
        session = self._get_session(session_id)
        session.history.append({"role": "user", "content": query})
        session.history.append({"role": "assistant", "content": answer})

        return {
            "answer":  answer,
            "sources": sources,
            "chunks":  chunks,
            "model":   GROQ_MODEL,
        }

    def clear_history(self, session_id: str = "default") -> None:
        """Clear the conversation memory for a specific session."""
        if session_id in self._sessions:
            self._sessions[session_id].history = []

    def active_session_count(self) -> int:
        """Return the number of currently tracked sessions (useful for /health)."""
        return len(self._sessions)

    # ── Streaming answer ────────────────────────────────────────────────

    async def stream_answer(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        session_id: str = "default",
    ) -> AsyncIterator[str]:
        """
        Streaming RAG pipeline: retrieve → build prompt → stream tokens.

        Yields Server-Sent Event strings:
          - Token chunks : 'data: {"type":"token",  "text":"..."}\n\n'
          - Sources done : 'data: {"type":"sources","sources":[...]}\n\n'
          - Stream end   : 'data: {"type":"done"}\n\n'
        """
        self._purge_expired_sessions()

        # Step 1 — Retrieve synchronously (embedding is CPU-bound, already fast)
        chunks = self.retrieve(query, top_k=top_k)
        user_message, sources = self._build_context_prompt(query, chunks)

        # Step 2 — Compose messages
        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        session = self._get_session(session_id)
        messages.extend(session.history[-10:])
        messages.append({"role": "user", "content": user_message})

        # Step 3 — Emit sources before the first token so the UI can show them
        yield f'data: {json.dumps({"type": "sources", "sources": sources})}\n\n'

        # Step 4 — Stream tokens from Groq
        full_answer: List[str] = []
        async with await self.async_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
            stream=True,
        ) as stream:
            async for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    full_answer.append(token)
                    yield f'data: {json.dumps({"type": "token", "text": token})}\n\n'

        # Step 5 — Persist assistant answer in session history
        answer_text = "".join(full_answer)
        session.history.append({"role": "user",      "content": query})
        session.history.append({"role": "assistant", "content": answer_text})

        yield f'data: {json.dumps({"type": "done"})}\n\n'

    # ── Groq call with retry ───────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_groq_with_retry(self, messages: List[Dict[str, str]]):
        """
        Call the Groq chat completions API.
        Automatically retried on RateLimitError or APIConnectionError
        (up to 4 attempts, exponential backoff: 1s → 2s → 4s → 8s).
        Any other exception propagates immediately.
        """
        return self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
