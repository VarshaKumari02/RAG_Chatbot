"""
rag_engine.py
─────────────
The core RAG pipeline:
  1. Retrieve relevant chunks from the vector store
  2. Build a prompt with the retrieved context
  3. Call the Groq LLM and return the answer
"""

from groq import Groq
from typing import List, Dict, Any, Tuple

from config import GROQ_API_KEY, GROQ_MODEL, TOP_K_RESULTS
from vector_store import VectorStore


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
        self.client = Groq(api_key=GROQ_API_KEY)
        self.store = vector_store
        self.conversation_history: List[Dict[str, str]] = []

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
        # Step 1 — Retrieve
        chunks = self.retrieve(query, top_k=top_k)

        # Step 2 — Build context prompt
        user_message, sources = self._build_context_prompt(query, chunks)

        # Step 3 — Compose messages list
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if use_history:
            messages.extend(self.conversation_history[-10:])  # last 5 turns

        messages.append({"role": "user", "content": user_message})

        # Step 4 — Call Groq
        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content.strip()

        # Step 5 — Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": answer})

        return {
            "answer":  answer,
            "sources": sources,
            "chunks":  chunks,
            "model":   GROQ_MODEL,
        }

    def clear_history(self):
        """Reset the conversation memory."""
        self.conversation_history = []
