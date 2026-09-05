"""
rag_engine.py
=============
Local vector storage (ChromaDB + HuggingFace embeddings) and a
multilingual Retrieval-Augmented Generation QA chain powered entirely
by a local Ollama daemon (`llama3.2`).

No network calls beyond http://127.0.0.1:11434 (Ollama) and the
one-time local HuggingFace model cache download.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ollama
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from ingestion import IngestedChunk

logger = logging.getLogger("rag_engine")

DEFAULT_PERSIST_DIR = "./chroma_local_store"
DEFAULT_COLLECTION_NAME = "multimodal_rag_collection"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class RAGEngineError(RuntimeError):
    """Raised for vector-store or local LLM call failures."""


# --------------------------------------------------------------------------- #
# Vector store management
# --------------------------------------------------------------------------- #


class VectorStoreManager:
    """Wraps a local, persisted ChromaDB collection using local HF embeddings."""

    def __init__(
        self,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        logger.info("Loading local embedding model '%s' (CPU/GPU, no API calls)...", embedding_model_name)
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        except Exception as exc:  # noqa: BLE001
            raise RAGEngineError(f"Failed to load local embedding model: {exc}") from exc

        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        try:
            self.store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
        except Exception as exc:  # noqa: BLE001
            raise RAGEngineError(f"Failed to initialize local Chroma store: {exc}") from exc

    def add_chunks(self, chunks: list[IngestedChunk]) -> int:
        """Embed and persist a batch of IngestedChunks. Returns count added."""
        if not chunks:
            return 0

        documents = [
            Document(page_content=chunk.content, metadata=self._clean_metadata(chunk.metadata))
            for chunk in chunks
            if chunk.content and chunk.content.strip()
        ]
        if not documents:
            return 0

        try:
            self.store.add_documents(documents)
        except Exception as exc:  # noqa: BLE001
            raise RAGEngineError(f"Failed to add documents to local vector store: {exc}") from exc

        logger.info("Added %d chunks to collection '%s'", len(documents), self.collection_name)
        return len(documents)

    @staticmethod
    def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Chroma metadata values must be str/int/float/bool — coerce anything else."""
        cleaned: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                cleaned[key] = value if value is not None else ""
            else:
                cleaned[key] = str(value)
        return cleaned

    def similarity_search(self, query: str, k: int = 5, media_type_filter: str | None = None) -> list[Document]:
        try:
            if media_type_filter:
                return self.store.similarity_search(query, k=k, filter={"media_type": media_type_filter})
            return self.store.similarity_search(query, k=k)
        except Exception as exc:  # noqa: BLE001
            raise RAGEngineError(f"Local vector similarity search failed: {exc}") from exc

    def document_count(self) -> int:
        try:
            return self.store._collection.count()  # noqa: SLF001 — no public count() in this version
        except Exception:  # noqa: BLE001
            return -1


# --------------------------------------------------------------------------- #
# Multilingual RAG QA chain
# --------------------------------------------------------------------------- #

MULTILINGUAL_SYSTEM_PROMPT = """You are a helpful, precise, multilingual local AI assistant \
operating entirely offline with no internet access.

STRICT LANGUAGE RULE:
1. Detect the natural language of the user's most recent message (including script/dialect).
2. Your ENTIRE response must be written in that exact same language. Do not switch to \
English or any other language unless the user explicitly writes in English or explicitly \
asks you to respond in a different language.
3. This rule applies even if the retrieved CONTEXT below is in a different language than the \
user's query — translate/synthesize the relevant information into the user's query language.

GROUNDING RULE:
- Base your answer primarily on the CONTEXT provided below, which was retrieved from the \
user's own uploaded documents, spreadsheets, audio/video transcripts, and images.
- If the context does not contain enough information to answer confidently, say so honestly \
in the user's language rather than fabricating an answer.
- When useful, mention which source file(s) the information came from (the source filename \
is available in each context block).
"""


@dataclass
class RetrievedContext:
    content: str
    source: str
    media_type: str
    extra: dict[str, Any]


class RAGEngine:
    """Builds prompts from retrieved local context and queries a local Ollama model."""

    def __init__(self, vector_store: VectorStoreManager, llm_model: str = "llama3.2", top_k: int = 5) -> None:
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.top_k = top_k

    def _retrieve(self, query: str) -> list[RetrievedContext]:
        docs = self.vector_store.similarity_search(query, k=self.top_k)
        results: list[RetrievedContext] = []
        for doc in docs:
            meta = doc.metadata or {}
            results.append(
                RetrievedContext(
                    content=doc.page_content,
                    source=str(meta.get("source", "unknown")),
                    media_type=str(meta.get("media_type", "unknown")),
                    extra={k: v for k, v in meta.items() if k not in {"source", "media_type"}},
                )
            )
        return results

    @staticmethod
    def _format_context_block(contexts: list[RetrievedContext]) -> str:
        if not contexts:
            return "No relevant local context was found for this query."

        blocks = []
        for i, ctx in enumerate(contexts, start=1):
            extra_str = ", ".join(f"{k}={v}" for k, v in ctx.extra.items() if v not in (None, ""))
            header = f"[Context {i} | source: {ctx.source} | type: {ctx.media_type}" + (f" | {extra_str}" if extra_str else "") + "]"
            blocks.append(f"{header}\n{ctx.content}")
        return "\n\n".join(blocks)

    def query(
        self,
        user_query: str,
        chat_history: list[dict[str, str]] | None = None,
        media_type_filter: str | None = None,
    ) -> dict[str, Any]:
        """Run one RAG turn: retrieve local context, call local llama3.2, return answer + sources.

        `chat_history` is a list of {"role": "user"|"assistant", "content": str} dicts,
        used to keep short-term conversational memory in the local LLM call.
        """
        if not user_query or not user_query.strip():
            raise RAGEngineError("Empty query provided to RAG engine.")

        try:
            if media_type_filter:
                docs = self.vector_store.similarity_search(user_query, k=self.top_k, media_type_filter=media_type_filter)
                contexts = [
                    RetrievedContext(
                        content=d.page_content,
                        source=str((d.metadata or {}).get("source", "unknown")),
                        media_type=str((d.metadata or {}).get("media_type", "unknown")),
                        extra={k: v for k, v in (d.metadata or {}).items() if k not in {"source", "media_type"}},
                    )
                    for d in docs
                ]
            else:
                contexts = self._retrieve(user_query)
        except RAGEngineError as exc:
            logger.error("Retrieval failed: %s", exc)
            contexts = []

        context_block = self._format_context_block(contexts)

        messages: list[dict[str, str]] = [{"role": "system", "content": MULTILINGUAL_SYSTEM_PROMPT}]
        if chat_history:
            # keep only the last few turns to bound prompt size
            messages.extend(chat_history[-6:])

        user_turn = f"CONTEXT:\n{context_block}\n\nUSER QUESTION:\n{user_query}"
        messages.append({"role": "user", "content": user_turn})

        try:
            response = ollama.chat(model=self.llm_model, messages=messages)
            answer = response["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001
            raise RAGEngineError(
                f"Local Ollama call to model '{self.llm_model}' failed: {exc}. "
                "Confirm `ollama serve` is running and `ollama pull llama3.2` has completed."
            ) from exc

        return {
            "answer": answer,
            "sources": [{"source": c.source, "media_type": c.media_type, **c.extra} for c in contexts],
        }
