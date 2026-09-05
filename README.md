# Local Multimodal & Multilingual RAG Chatbot

A 100% local, air-gapped-capable Retrieval-Augmented Generation system.
No OpenAI/Anthropic/cloud embedding calls — everything resolves to
`localhost` (Ollama daemon) or runs in-process (Whisper, YOLOv8,
HuggingFace embeddings).

## 1. System prerequisites

```bash
# ffmpeg is required by whisper/pydub for audio extraction & decoding
# Debian/Ubuntu:
sudo apt-get update && sudo apt-get install -y ffmpeg
# macOS:
brew install ffmpeg
```

## 2. Install Ollama and pull local models

Install Ollama from https://ollama.com/download, then:

```bash
ollama pull llama3.2          # text reasoning / RAG QA chain
ollama pull llama3.2-vision   # image/frame understanding & visual summaries

# Start the local daemon (usually auto-starts as a service; if not):
ollama serve
```

Verify it's up locally:

```bash
curl http://127.0.0.1:11434/api/tags
```

## 3. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

The first run of `openai-whisper`, `sentence-transformers`, and
`ultralytics` will download model weights once from their respective
hubs and cache them locally under `~/.cache`. After that, all
inference is fully offline.

## 4. Run the app

```bash
streamlit run app.py
```

Open the printed local URL (default `http://localhost:8501`).

## 5. Project layout

```
.
├── ingestion.py      # audio/video, visual (YOLO+VLM), tabular, document processors
├── rag_engine.py      # ChromaDB vector store + multilingual RAG QA chain
├── app.py            # Streamlit frontend
├── requirements.txt
└── README.md
```

## 6. Supported uploads

| Category   | Extensions                 | Pipeline                                   |
|------------|-----------------------------|---------------------------------------------|
| Documents  | .pdf, .docx, .txt          | Text extraction + chunking                  |
| Tabular    | .xlsx, .csv                | Pandas → schema + Markdown chunks           |
| Audio      | .mp3, .wav, .m4a           | ffmpeg → Whisper transcription              |
| Video      | .mp4, .mkv                 | Whisper transcript + YOLOv8/vision keyframes|
| Images     | .jpg, .png                 | YOLOv8 detection + llama3.2-vision summary  |

## 7. Notes on scaling this prototype

- Swap the local Chroma persistence directory for a networked Qdrant
  instance if you need multi-user concurrent access.
- Whisper's `base` model is used by default for speed; switch to
  `small`/`medium` in `AudioVideoProcessor` for better transcription
  accuracy at the cost of latency.
- `yolov8n.pt` (nano) is the fastest YOLO checkpoint; swap to `yolov8s.pt`
  or larger for better detection accuracy.
