# DocFlow — Installation & Deployment Guide

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| Python | 3.10+ | Required |
| pip | 23+ | `pip install --upgrade pip` |
| Tesseract | 5.x | `apt install tesseract-ocr` |
| FFmpeg | 6.x | `apt install ffmpeg` (for audio) |
| Docker | 24+ | Optional, for containerised deployment |
| NVIDIA CUDA | 12.x | Optional, for GPU acceleration |

---

## 1. Local Development

### Clone and Install

```bash
git clone https://github.com/yourorg/docflow.git
cd docflow

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install with CPU extras
pip install -e ".[cpu,ui]"

# Or full install (GPU + Desktop)
pip install -e ".[all]"
```

### Configure Environment

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
```

### Run Services

```bash
# API server (port 8000)
uvicorn api.main:app --reload --port 8000

# Streamlit UI (port 8501) — separate terminal
streamlit run frontend/app.py

# Desktop app
python desktop/main.py

# Background worker (optional, requires Redis)
celery -A core.celery_app worker --loglevel=info
```

---

## 2. Docker Deployment

### CPU (Standard)

```bash
cd docker
docker-compose up -d

# Services started:
# - DocFlow API:       http://localhost:8000
# - Streamlit UI:      http://localhost:8501
# - PostgreSQL:        localhost:5432
# - Redis:             localhost:6379
```

### GPU (NVIDIA)

```bash
# Requires nvidia-container-toolkit
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

### With Monitoring Stack

```bash
# Includes Prometheus + Grafana
docker-compose --profile monitoring up -d

# Grafana: http://localhost:3000 (admin/docflow)
# Prometheus: http://localhost:9090
```

---

## 3. Production Deployment

### Environment Variables for Production

```bash
APP_ENV=production
DEBUG=false
SECRET_KEY=<32-char-random>
API_KEYS=key1,key2,key3         # Enable API key auth
DATABASE_URL=postgresql+asyncpg://user:pass@host/db
REDIS_URL=redis://redis:6379/0
OPENAI_API_KEY=sk-...
SENTRY_DSN=https://...@sentry.io/...
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name docflow.example.com;

    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 500M;
    }

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## 4. GPU Setup

### Install NVIDIA Container Toolkit

```bash
# Ubuntu
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### GPU .env Settings

```bash
USE_GPU=true
OCR_USE_GPU=true
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
CUDA_VISIBLE_DEVICES=0
```

---

## 5. Vector Store Setup

### ChromaDB (Default, no setup needed)
```bash
VECTOR_STORE=chroma
CHROMA_PERSIST_DIR=./data/chroma
```

### Qdrant
```bash
docker run -p 6333:6333 qdrant/qdrant
VECTOR_STORE=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=docflow
```

### FAISS
```bash
VECTOR_STORE=faiss
FAISS_INDEX_PATH=./data/faiss/index.bin
```

---

## 6. Python SDK Usage

```python
import asyncio
from pathlib import Path
from core.pipeline import DocFlowPipeline, ProcessingOptions

async def main():
    pipeline = DocFlowPipeline()

    opts = ProcessingOptions(
        enable_ocr=True,
        enable_summarization=True,
        summarization_provider="openai",
        enable_embeddings=True,
        embedding_provider="sentence_transformers",
        chunk_strategy="heading",
        chunk_size=512,
        store_in_vectordb=True,
    )

    # Single file
    result = await pipeline.process(Path("report.pdf"), opts)
    print(result.markdown[:500])
    print(f"Chunks: {len(result.chunks)}")
    print(f"Summary: {result.summary}")

    # Batch folder
    results = await pipeline.process_folder(Path("./documents/"), opts)
    print(f"Processed {len(results)} files")

asyncio.run(main())
```

---

## 7. LangChain RAG Pipeline

```python
from docflow.integrations.langchain import DocFlowLoader, DocFlowRetriever
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Index documents
loader = DocFlowLoader("./my_documents/", enable_ocr=True)
docs = loader.load()

# Build RAG chain
retriever = DocFlowRetriever(top_k=5)
llm = ChatOpenAI(model="gpt-4o-mini")
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Query
answer = chain.run("What are the quarterly revenue figures?")
print(answer)
```

---

## 8. Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=. --cov-report=html

# Specific module
pytest tests/test_suite.py::TestChunker -v

# Load tests (requires running server)
locust -f tests/locustfile.py --host=http://localhost:8000 \
  --users=50 --spawn-rate=5 --run-time=60s --headless
```
