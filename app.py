"""
app.py
======
Streamlit frontend for the local multimodal & multilingual RAG chatbot.

Run with:
    streamlit run app.py

Everything downstream (embeddings, vector store, LLM, Whisper, YOLO,
vision model) runs locally — no cloud API keys are read or required.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import streamlit as st

from ingestion import IngestionError, IngestionPipeline
from rag_engine import RAGEngine, RAGEngineError, VectorStoreManager

logger = logging.getLogger("app")

SUPPORTED_EXTENSIONS = [
    "pdf", "docx", "txt",       # documents
    "xlsx", "csv",              # tabular
    "mp3", "wav", "m4a",        # audio
    "mp4", "mkv",               # video
    "jpg", "jpeg", "png",       # images
]

st.set_page_config(page_title="Local Multimodal RAG Chatbot", page_icon="🧠", layout="wide")


# --------------------------------------------------------------------------- #
# Cached local resources — loaded once per session, not on every rerun
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline()


@st.cache_resource(show_spinner=False)
def get_vector_store() -> VectorStoreManager:
    return VectorStoreManager()


def get_rag_engine(_vector_store: VectorStoreManager) -> RAGEngine:
    # Not cached: cheap to construct, and we want it to always see the live vector store.
    return RAGEngine(vector_store=_vector_store)


# --------------------------------------------------------------------------- #
# Session state initialization
# --------------------------------------------------------------------------- #

if "messages" not in st.session_state:
    st.session_state.messages: list[dict[str, str]] = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files: set[str] = set()

if "ingestion_log" not in st.session_state:
    st.session_state.ingestion_log: list[dict[str, str]] = []


def _file_signature(uploaded_file) -> str:
    """Cheap dedup key so re-rendering the uploader doesn't re-ingest the same file."""
    return f"{uploaded_file.name}:{uploaded_file.size}"


def ingest_uploaded_file(uploaded_file, pipeline: IngestionPipeline, vector_store: VectorStoreManager) -> None:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)

    try:
        chunks = pipeline.process_file(tmp_path)
        added = vector_store.add_chunks(chunks)
        st.session_state.ingestion_log.append(
            {"file": uploaded_file.name, "status": "success", "detail": f"{added} chunks embedded"}
        )
    except IngestionError as exc:
        st.session_state.ingestion_log.append({"file": uploaded_file.name, "status": "error", "detail": str(exc)})
    except RAGEngineError as exc:
        st.session_state.ingestion_log.append({"file": uploaded_file.name, "status": "error", "detail": str(exc)})
    except Exception as exc:  # noqa: BLE001 — never let an unexpected error crash the app
        st.session_state.ingestion_log.append(
            {"file": uploaded_file.name, "status": "error", "detail": f"Unexpected error: {exc}"}
        )
    finally:
        tmp_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Sidebar — multi-file uploader
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("📎 Upload local assets")
    st.caption(
        "Documents, spreadsheets, audio, video, and images — all processed "
        "entirely on this machine. Nothing is sent to a cloud API."
    )

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
        help="Supported: PDF, DOCX, TXT, XLSX, CSV, MP3, WAV, M4A, MP4, MKV, JPG, PNG",
    )

    if uploaded_files:
        pipeline = get_ingestion_pipeline()
        vector_store = get_vector_store()

        new_files = [f for f in uploaded_files if _file_signature(f) not in st.session_state.processed_files]

        if new_files:
            progress = st.progress(0.0, text="Starting local ingestion...")
            for i, f in enumerate(new_files):
                progress.progress((i) / len(new_files), text=f"Processing '{f.name}' locally...")
                ingest_uploaded_file(f, pipeline, vector_store)
                st.session_state.processed_files.add(_file_signature(f))
            progress.progress(1.0, text="Done.")
            progress.empty()

    if st.session_state.ingestion_log:
        st.divider()
        st.subheader("Ingestion status")
        for entry in reversed(st.session_state.ingestion_log[-15:]):
            if entry["status"] == "success":
                st.success(f"**{entry['file']}** — {entry['detail']}", icon="✅")
            else:
                st.error(f"**{entry['file']}** — {entry['detail']}", icon="⚠️")

    st.divider()
    try:
        doc_count = get_vector_store().document_count()
        if doc_count >= 0:
            st.metric("Chunks in local vector store", doc_count)
    except Exception:  # noqa: BLE001
        pass

    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------- #
# Main chat interface
# --------------------------------------------------------------------------- #

st.title("🧠 Local Multimodal & Multilingual RAG Chatbot")
st.caption(
    "Runs 100% locally via Ollama (llama3.2 / llama3.2-vision), Whisper, YOLOv8, "
    "and a local ChromaDB vector store. Ask in any language — it replies in kind."
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources used"):
                for src in msg["sources"]:
                    extras = ", ".join(f"{k}: {v}" for k, v in src.items() if k not in {"source", "media_type"} and v)
                    line = f"- **{src.get('source', 'unknown')}** ({src.get('media_type', 'unknown')})"
                    if extras:
                        line += f" — {extras}"
                    st.markdown(line)

user_query = st.chat_input("Ask a question about your uploaded files, in any language...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        vector_store = get_vector_store()
        rag_engine = get_rag_engine(vector_store)

        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]

        try:
            with st.spinner("Thinking locally..."):
                result = rag_engine.query(user_query, chat_history=history)
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("Sources used"):
                    for src in result["sources"]:
                        extras = ", ".join(f"{k}: {v}" for k, v in src.items() if k not in {"source", "media_type"} and v)
                        line = f"- **{src.get('source', 'unknown')}** ({src.get('media_type', 'unknown')})"
                        if extras:
                            line += f" — {extras}"
                        st.markdown(line)
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
            )
        except RAGEngineError as exc:
            error_msg = f"⚠️ Local inference error: {exc}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
        except Exception as exc:  # noqa: BLE001
            error_msg = f"⚠️ Unexpected error: {exc}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
