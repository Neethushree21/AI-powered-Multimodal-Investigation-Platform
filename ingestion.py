"""
ingestion.py
============
Local, offline ingestion pipeline for a multimodal RAG chatbot.

Handles four asset classes and normalizes them all into a common
`IngestedChunk` shape so downstream vector storage / retrieval code
stays agnostic to the original media type:

    - Documents      (.pdf, .docx, .txt)      -> DocumentProcessor
    - Tabular data    (.xlsx, .csv)            -> TabularProcessor
    - Audio / Video   (.mp3, .wav, .m4a,
                       .mp4, .mkv)             -> AudioVideoProcessor
    - Images          (.jpg, .png)             -> VisualProcessor

All model inference is local:
    - openai-whisper        -> offline speech-to-text
    - ultralytics YOLOv8    -> offline object detection
    - Ollama (llama3.2-vision) via http://127.0.0.1:11434 -> local VLM

No calls ever leave localhost.
"""

from __future__ import annotations

import base64
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import ollama
import pandas as pd
from docx import Document as DocxDocument
from pypdf import PdfReader
from ultralytics import YOLO

logger = logging.getLogger("ingestion")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# --------------------------------------------------------------------------- #
# Shared data model
# --------------------------------------------------------------------------- #


@dataclass
class IngestedChunk:
    """A normalized unit of content ready for embedding + vector storage."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class IngestionError(RuntimeError):
    """Raised when a file cannot be ingested (corrupt, unsupported, tool missing)."""


# --------------------------------------------------------------------------- #
# Shared text chunking utility
# --------------------------------------------------------------------------- #


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Simple sliding-window character chunker with overlap.

    Deliberately dependency-free (no external text-splitter needed) so this
    module has no hidden coupling to a specific LangChain splitter version.
    """
    text = text.strip()
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        # try to break on a sentence/paragraph boundary near the end for readability
        boundary = text.rfind("\n", start, end)
        if boundary == -1 or boundary <= start:
            boundary = text.rfind(". ", start, end)
        if boundary != -1 and boundary > start + (chunk_size // 2):
            end = boundary + 1
        chunks.append(text[start:end].strip())
        if end >= text_len:
            break
        start = end - overlap
    return [c for c in chunks if c]


def _run_ffmpeg(args: list[str]) -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise IngestionError(
            "ffmpeg binary not found on PATH. Install it (e.g. `apt-get install ffmpeg` "
            "or `brew install ffmpeg`) — it is required for audio/video ingestion."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise IngestionError(f"ffmpeg failed: {exc.stderr.strip()}") from exc


# --------------------------------------------------------------------------- #
# 1. Audio / Video Processor
# --------------------------------------------------------------------------- #


class AudioVideoProcessor:
    """Transcribes audio/video offline with Whisper and extracts video keyframes."""

    AUDIO_EXTS = {".mp3", ".wav", ".m4a"}
    VIDEO_EXTS = {".mp4", ".mkv"}

    def __init__(self, whisper_model_size: str = "base", keyframe_interval_sec: int = 5) -> None:
        self.keyframe_interval_sec = keyframe_interval_sec
        self._whisper_model_size = whisper_model_size
        self._whisper_model = None  # lazy-loaded to avoid paying the cost if unused

    def _get_whisper_model(self):
        if self._whisper_model is None:
            import whisper  # local import: heavy, only needed here

            logger.info("Loading local Whisper model '%s' (offline, cached after first run)...", self._whisper_model_size)
            self._whisper_model = whisper.load_model(self._whisper_model_size)
        return self._whisper_model

    def _extract_audio_to_wav(self, media_path: Path) -> Path:
        """Extract/convert to a 16kHz mono WAV for consistent Whisper input."""
        wav_path = Path(tempfile.gettempdir()) / f"{media_path.stem}_extracted.wav"
        _run_ffmpeg(["-i", str(media_path), "-ac", "1", "-ar", "16000", str(wav_path)])
        return wav_path

    def transcribe(self, file_path: str | Path) -> str:
        """Run fully offline Whisper transcription on an audio or video file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise IngestionError(f"File not found: {file_path}")

        try:
            wav_path = self._extract_audio_to_wav(file_path)
            model = self._get_whisper_model()
            logger.info("Transcribing '%s' locally with Whisper...", file_path.name)
            result = model.transcribe(str(wav_path))
            return result.get("text", "").strip()
        except IngestionError:
            raise
        except Exception as exc:  # noqa: BLE001 — surface as a clean ingestion error
            raise IngestionError(f"Whisper transcription failed for {file_path.name}: {exc}") from exc
        finally:
            if "wav_path" in locals() and wav_path.exists():
                wav_path.unlink(missing_ok=True)

    def extract_keyframes(self, video_path: str | Path, output_dir: str | Path) -> list[Path]:
        """Grab one frame every `keyframe_interval_sec` seconds using OpenCV."""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IngestionError(f"OpenCV could not open video file: {video_path.name}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = max(int(fps * self.keyframe_interval_sec), 1)

        saved_paths: list[Path] = []
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    timestamp_sec = frame_idx / fps
                    out_path = output_dir / f"{video_path.stem}_kf_{int(timestamp_sec)}s.jpg"
                    cv2.imwrite(str(out_path), frame)
                    saved_paths.append(out_path)
                frame_idx += 1
        finally:
            cap.release()

        logger.info("Extracted %d keyframes from '%s'", len(saved_paths), video_path.name)
        return saved_paths

    def process(self, file_path: str | Path) -> tuple[list[IngestedChunk], list[Path]]:
        """Full pipeline for an audio/video file.

        Returns (transcript_chunks, keyframe_image_paths). Keyframe paths are
        returned separately so the caller can route them through VisualProcessor.
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        transcript = self.transcribe(file_path)

        chunks: list[IngestedChunk] = []
        for i, piece in enumerate(chunk_text(transcript, chunk_size=1200, overlap=200)):
            chunks.append(
                IngestedChunk(
                    content=piece,
                    metadata={
                        "source": file_path.name,
                        "media_type": "video" if ext in self.VIDEO_EXTS else "audio",
                        "chunk_index": i,
                        "content_kind": "transcript",
                    },
                )
            )

        keyframes: list[Path] = []
        if ext in self.VIDEO_EXTS:
            kf_dir = Path(tempfile.gettempdir()) / f"{file_path.stem}_keyframes"
            keyframes = self.extract_keyframes(file_path, kf_dir)

        return chunks, keyframes


# --------------------------------------------------------------------------- #
# 2. Visual Processor — YOLOv8 object detection + local Ollama vision summary
# --------------------------------------------------------------------------- #


class VisualProcessor:
    """Local object detection (YOLOv8) + local VLM scene summarization (llama3.2-vision)."""

    def __init__(self, yolo_model_path: str = "yolov8n.pt", vision_model: str = "llama3.2-vision") -> None:
        self.vision_model = vision_model
        self._yolo_model_path = yolo_model_path
        self._yolo_model: YOLO | None = None

    def _get_yolo_model(self) -> YOLO:
        if self._yolo_model is None:
            logger.info("Loading local YOLOv8 model '%s'...", self._yolo_model_path)
            self._yolo_model = YOLO(self._yolo_model_path)
        return self._yolo_model

    def detect_objects(self, image_path: str | Path) -> list[str]:
        """Run local YOLOv8 inference and return a de-duplicated label list."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise IngestionError(f"Image not found: {image_path}")

        model = self._get_yolo_model()
        try:
            results = model(str(image_path), verbose=False)
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"YOLOv8 inference failed on {image_path.name}: {exc}") from exc

        labels: list[str] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = names.get(cls_id, str(cls_id))
                if label not in labels:
                    labels.append(label)
        return labels

    def summarize_image(self, image_path: str | Path, detected_objects: list[str]) -> str:
        """Ask the local llama3.2-vision model (via Ollama) to describe the image."""
        image_path = Path(image_path)
        objects_hint = ", ".join(detected_objects) if detected_objects else "no objects confidently detected"

        prompt = (
            "You are a precise visual assistant. Describe this image factually in 3-5 sentences: "
            "the scene, key subjects, and notable context. An offline object detector found these "
            f"candidate objects (use them as hints, but trust your own visual reading too): {objects_hint}."
        )

        try:
            response = ollama.chat(
                model=self.vision_model,
                messages=[{"role": "user", "content": prompt, "images": [str(image_path)]}],
            )
            return response["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(
                f"Local Ollama vision model '{self.vision_model}' call failed for {image_path.name}: {exc}. "
                "Confirm `ollama serve` is running and `ollama pull llama3.2-vision` has completed."
            ) from exc

    def process_image(self, image_path: str | Path, source_label: str | None = None, extra_metadata: dict[str, Any] | None = None) -> IngestedChunk:
        """Full single-image pipeline: detect -> summarize -> package as a chunk."""
        image_path = Path(image_path)
        detected_objects = self.detect_objects(image_path)
        summary = self.summarize_image(image_path, detected_objects)

        metadata: dict[str, Any] = {
            "source": source_label or image_path.name,
            "media_type": "image",
            "detected_objects": ", ".join(detected_objects) if detected_objects else "none",
            "content_kind": "visual_summary",
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return IngestedChunk(content=summary, metadata=metadata)

    def process_video_keyframes(self, keyframe_paths: list[Path], source_video_name: str) -> list[IngestedChunk]:
        """Run the image pipeline over every extracted keyframe of a video."""
        chunks: list[IngestedChunk] = []
        for kf_path in keyframe_paths:
            try:
                # keyframe filenames are formatted "<stem>_kf_<seconds>s.jpg"
                timestamp_token = kf_path.stem.split("_kf_")[-1]
                chunk = self.process_image(
                    kf_path,
                    source_label=source_video_name,
                    extra_metadata={"media_type": "video", "timestamp": timestamp_token, "content_kind": "visual_summary"},
                )
                chunks.append(chunk)
            except IngestionError as exc:
                logger.warning("Skipping keyframe %s: %s", kf_path.name, exc)
        return chunks


# --------------------------------------------------------------------------- #
# 3. Tabular Processor — .xlsx / .csv -> schema-aware text
# --------------------------------------------------------------------------- #


class TabularProcessor:
    """Converts spreadsheets/CSVs into schema descriptions + Markdown row batches."""

    def __init__(self, rows_per_chunk: int = 40) -> None:
        self.rows_per_chunk = rows_per_chunk

    def _describe_schema(self, sheet_name: str, df: pd.DataFrame) -> str:
        dtypes_desc = "\n".join(f"  - {col} ({dtype})" for col, dtype in df.dtypes.items())
        return (
            f"Sheet: {sheet_name}\n"
            f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
            f"Columns:\n{dtypes_desc}"
        )

    def _sheet_to_chunks(self, sheet_name: str, df: pd.DataFrame, source_name: str) -> list[IngestedChunk]:
        if df.empty:
            return []

        chunks: list[IngestedChunk] = []
        schema_desc = self._describe_schema(sheet_name, df)

        # First chunk: schema overview so retrieval can surface "what columns exist" queries
        chunks.append(
            IngestedChunk(
                content=schema_desc,
                metadata={
                    "source": source_name,
                    "media_type": "tabular",
                    "sheet": sheet_name,
                    "content_kind": "schema",
                },
            )
        )

        total_rows = len(df)
        for start in range(0, total_rows, self.rows_per_chunk):
            batch = df.iloc[start : start + self.rows_per_chunk]
            try:
                markdown_table = batch.to_markdown(index=False)
            except ImportError as exc:
                raise IngestionError("`tabulate` package is required for Markdown table rendering (pip install tabulate)") from exc

            content = f"{schema_desc}\n\nRows {start}-{start + len(batch) - 1}:\n{markdown_table}"
            chunks.append(
                IngestedChunk(
                    content=content,
                    metadata={
                        "source": source_name,
                        "media_type": "tabular",
                        "sheet": sheet_name,
                        "row_range": f"{start}-{start + len(batch) - 1}",
                        "content_kind": "row_batch",
                    },
                )
            )
        return chunks

    def process(self, file_path: str | Path) -> list[IngestedChunk]:
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        if not file_path.exists():
            raise IngestionError(f"File not found: {file_path}")

        chunks: list[IngestedChunk] = []
        try:
            if ext == ".csv":
                df = pd.read_csv(file_path)
                chunks.extend(self._sheet_to_chunks("default", df, file_path.name))
            elif ext == ".xlsx":
                excel_file = pd.ExcelFile(file_path)
                for sheet_name in excel_file.sheet_names:
                    df = excel_file.parse(sheet_name)
                    chunks.extend(self._sheet_to_chunks(sheet_name, df, file_path.name))
            else:
                raise IngestionError(f"Unsupported tabular extension: {ext}")
        except IngestionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"Failed to parse tabular file {file_path.name}: {exc}") from exc

        return chunks


# --------------------------------------------------------------------------- #
# 4. Document Processor — .pdf / .docx / .txt -> text -> chunks
# --------------------------------------------------------------------------- #


class DocumentProcessor:
    """Loads unstructured documents and chunks them for embedding."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 150) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _load_pdf(self, file_path: Path) -> str:
        try:
            reader = PdfReader(str(file_path))
            pages_text = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(f"[Page {page_num + 1}]\n{text}")
            return "\n\n".join(pages_text)
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"Failed to read PDF {file_path.name}: {exc}") from exc

    def _load_docx(self, file_path: Path) -> str:
        try:
            doc = DocxDocument(str(file_path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            # also capture table content, since many .docx reports embed tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells)
                    if row_text.strip(" |"):
                        paragraphs.append(row_text)
            return "\n".join(paragraphs)
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"Failed to read DOCX {file_path.name}: {exc}") from exc

    def _load_txt(self, file_path: Path) -> str:
        try:
            return file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"Failed to read TXT {file_path.name}: {exc}") from exc

    def process(self, file_path: str | Path) -> list[IngestedChunk]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise IngestionError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()
        loaders = {".pdf": self._load_pdf, ".docx": self._load_docx, ".txt": self._load_txt}
        loader = loaders.get(ext)
        if loader is None:
            raise IngestionError(f"Unsupported document extension: {ext}")

        raw_text = loader(file_path)
        if not raw_text.strip():
            logger.warning("No extractable text found in %s", file_path.name)
            return []

        pieces = chunk_text(raw_text, chunk_size=self.chunk_size, overlap=self.overlap)
        return [
            IngestedChunk(
                content=piece,
                metadata={
                    "source": file_path.name,
                    "media_type": "document",
                    "chunk_index": i,
                    "content_kind": "text",
                },
            )
            for i, piece in enumerate(pieces)
        ]


# --------------------------------------------------------------------------- #
# Top-level orchestrator — dispatches by extension
# --------------------------------------------------------------------------- #


class IngestionPipeline:
    """Single entry point: routes any supported file to the right processor(s)."""

    def __init__(
        self,
        whisper_model_size: str = "base",
        yolo_model_path: str = "yolov8n.pt",
        vision_model: str = "llama3.2-vision",
    ) -> None:
        self.audio_video = AudioVideoProcessor(whisper_model_size=whisper_model_size)
        self.visual = VisualProcessor(yolo_model_path=yolo_model_path, vision_model=vision_model)
        self.tabular = TabularProcessor()
        self.document = DocumentProcessor()

    def process_file(self, file_path: str | Path) -> list[IngestedChunk]:
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext in {".pdf", ".docx", ".txt"}:
            return self.document.process(file_path)

        if ext in {".xlsx", ".csv"}:
            return self.tabular.process(file_path)

        if ext in {".jpg", ".jpeg", ".png"}:
            return [self.visual.process_image(file_path)]

        if ext in AudioVideoProcessor.AUDIO_EXTS | AudioVideoProcessor.VIDEO_EXTS:
            transcript_chunks, keyframes = self.audio_video.process(file_path)
            visual_chunks: list[IngestedChunk] = []
            if keyframes:
                visual_chunks = self.visual.process_video_keyframes(keyframes, source_video_name=file_path.name)
            return transcript_chunks + visual_chunks

        raise IngestionError(
            f"Unsupported file type '{ext}' for {file_path.name}. "
            "Supported: .pdf .docx .txt .xlsx .csv .mp3 .wav .m4a .mp4 .mkv .jpg .png"
        )
