# DocFlow — Architecture Guide

## System Overview

DocFlow is a modular, event-driven document intelligence platform built around a clean pipeline architecture.

```
                    ┌─────────────────────────────────────────┐
                    │            Input Layer                   │
                    │  File Upload │ API │ Folder Watch │ URL  │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │         Format Dispatcher                │
                    │  PDF │ DOCX │ PPTX │ XLSX │ CSV │ HTML  │
                    │  TXT │ JSON │ XML │ IMG │ Audio │ ZIP   │
                    └──────────────────┬──────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
   ┌──────────▼──────┐    ┌────────────▼────────┐   ┌──────────▼──────┐
   │  Document Parse  │    │    OCR Engine        │   │ Audio/Video     │
   │  PyMuPDF         │    │    PaddleOCR (GPU)   │   │ Whisper         │
   │  pdfplumber      │    │    EasyOCR           │   │ yt-dlp          │
   │  python-docx     │    │    Tesseract         │   │                 │
   └──────────┬──────┘    └────────────┬────────┘   └──────────┬──────┘
              │                        │                        │
              └────────────────────────┴────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │         Content Normalisation            │
                    │  → Clean Markdown                        │
                    │  → Structured JSON                       │
                    │  → Table extraction                      │
                    │  → Metadata enrichment                   │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │           AI Summarisation               │
                    │  OpenAI │ Anthropic │ Ollama │ HuggingFace│
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │         RAG Chunking Pipeline            │
                    │  Recursive │ Heading │ Token │ Semantic  │
                    │  + Metadata: source, page, section, ID   │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │        Embedding Generation              │
                    │  SentenceTransformers │ OpenAI │ Ollama  │
                    │  Batch processing │ Async │ Cached       │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │          Vector Storage                  │
                    │  ChromaDB │ FAISS │ Qdrant │ Pinecone    │
                    │  + Metadata filtering + Hybrid search    │
                    └──────────────────┬──────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
   ┌──────────▼──────┐    ┌────────────▼────────┐   ┌──────────▼──────┐
   │  FastAPI REST    │    │  Streamlit UI        │   │  Desktop App    │
   │  + Swagger       │    │  + Dark mode         │   │  PySide6        │
   │  + Auth          │    │  + Search UI         │   │  Drag & Drop    │
   │  + Rate limit    │    │  + Chunk explorer    │   │                 │
   └─────────────────┘    └────────────────────  ┘   └─────────────────┘
```

## Design Principles

### 1. Lazy Loading
All parser modules are lazy-imported. Heavy ML models (Whisper, BLIP, PaddleOCR) only load when actually needed, keeping startup time fast.

### 2. Provider Abstraction
Every subsystem (OCR, embeddings, LLMs, vector stores) has a provider-agnostic interface. Swapping from OpenAI to Ollama requires only a config change, not code changes.

### 3. Graceful Degradation
Every component has fallbacks:
- PaddleOCR → EasyOCR → Tesseract
- faster-whisper → openai-whisper → OpenAI API
- GPU → CPU

### 4. Async-First
All I/O operations are async. CPU-bound tasks (model inference) run in thread pools via `asyncio.run_in_executor`, never blocking the event loop.

### 5. Metadata Preservation
Every chunk carries provenance: source file, page number, section title, chunk ID, token count. This enables precise attribution in RAG answers.

## Component Deep Dives

### PDF Pipeline
```
PDF File
  → PyMuPDF: fast text + font-size heading detection
  → pdfplumber: accurate table extraction
  → (if scanned) → render page to image → OCR
  → Output: Markdown + tables + images + hyperlinks + metadata
```

### OCR Pipeline
```
Image Input
  → PIL Image preprocessing (grayscale, resize)
  → PaddleOCR (GPU): layout-aware, multi-language
    ↳ EasyOCR (CPU fallback)
    ↳ Tesseract (final fallback)
  → Confidence filtering (threshold configurable)
  → Output: plain text + bounding boxes + confidence scores
```

### Chunking Strategies
| Strategy | Best For | How It Works |
|---|---|---|
| `recursive` | General documents | Split on paragraphs → sentences → words |
| `heading` | Structured docs | Split at Markdown headings, preserve section context |
| `token` | LLM input prep | Split by exact token count using tiktoken |
| `semantic` | High-quality RAG | Split by embedding similarity drop between sentences |

### Embedding Cache
Embeddings are cached in-memory by content hash. For a large re-index, cache prevents duplicate API calls for identical text chunks.

## Scaling Considerations

### Horizontal Scaling
- API workers: stateless, scale freely (`API_WORKERS=N`)
- Celery workers: scale independently for heavy batch jobs
- Vector stores: ChromaDB for single-node, Qdrant/Pinecone for distributed

### GPU Acceleration
Set `USE_GPU=true`, `OCR_USE_GPU=true`, `WHISPER_DEVICE=cuda` in `.env`.
The GPU Dockerfile installs CUDA-enabled PyTorch and PaddlePaddle.

### Memory Management
- Streaming PDF rendering (page-by-page) for large files
- Embedding batch size controls GPU memory usage (`EMBEDDING_BATCH_SIZE`)
- Temp files cleaned up after each request
