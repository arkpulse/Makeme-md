# ============================================================
#  DocFlow — Dockerfile (CPU)
# ============================================================
FROM python:3.11-slim

LABEL maintainer="DocFlow Team"
LABEL description="Universal AI Document Intelligence Platform"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    ffmpeg \
    libmagic1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 docflow
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[cpu,ui]"

# Copy application code
COPY --chown=docflow:docflow . .

# Create data directories
RUN mkdir -p /app/data/{uploads,outputs,temp,models,chroma,faiss} && \
    chown -R docflow:docflow /app/data

USER docflow

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default: run API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
