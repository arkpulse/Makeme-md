"""
DocFlow — Global Configuration
Centralised settings via Pydantic BaseSettings.
All values are loaded from environment variables or .env file.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "DocFlow"
    app_version: str = "1.0.0"
    app_env: str = "development"
    debug: bool = False
    secret_key: str = "change-me-in-production"
    log_level: str = "INFO"

    # ── Server ───────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_reload: bool = False

    # ── Storage ──────────────────────────────────────────────────────────────
    upload_dir: Path = Path("./data/uploads")
    output_dir: Path = Path("./data/outputs")
    temp_dir: Path = Path("./data/temp")
    model_cache_dir: Path = Path("./data/models")

    # ── Upload Limits ────────────────────────────────────────────────────────
    max_upload_size_mb: int = 500
    allowed_extensions: str = "pdf,docx,pptx,xlsx,csv,json,xml,txt,html,md,png,jpg,jpeg,tiff,bmp,webp,mp3,wav,flac,m4a,zip"

    @property
    def allowed_extensions_set(self) -> set[str]:
        return {ext.strip().lower() for ext in self.allowed_extensions.split(",")}

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    # ── LLM Providers ────────────────────────────────────────────────────────
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-sonnet-4-20250514"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_embedding_model: str = "nomic-embed-text"

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_batch_size: int = 64

    # ── Vector Store ─────────────────────────────────────────────────────────
    vector_store: str = "chroma"
    chroma_persist_dir: Path = Path("./data/chroma")
    faiss_index_path: Path = Path("./data/faiss/index.bin")
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "docflow"
    pinecone_api_key: Optional[str] = None
    pinecone_index: str = "docflow"
    weaviate_url: str = "http://localhost:8080"

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./data/docflow.db"

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600

    # ── Celery ───────────────────────────────────────────────────────────────
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ── OCR ──────────────────────────────────────────────────────────────────
    ocr_engine: str = "auto"
    ocr_language: str = "en"
    ocr_use_gpu: bool = False
    ocr_confidence_threshold: float = 0.7
    tesseract_cmd: str = "/usr/bin/tesseract"

    # ── Audio ────────────────────────────────────────────────────────────────
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    enable_speaker_diarization: bool = False
    huggingface_token: Optional[str] = None

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunk_min_size: int = 100

    # ── Summarization ────────────────────────────────────────────────────────
    summarization_provider: str = "openai"
    enable_auto_summarize: bool = False

    # ── Security ─────────────────────────────────────────────────────────────
    api_key_header: str = "X-API-Key"
    api_keys: str = ""  # Comma-separated; empty = no auth required
    rate_limit_per_minute: int = 60
    enable_cors: bool = True
    cors_origins: str = "*"

    @property
    def valid_api_keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    # ── Monitoring ───────────────────────────────────────────────────────────
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    sentry_dsn: Optional[str] = None

    # ── GPU ──────────────────────────────────────────────────────────────────
    cuda_visible_devices: str = "0"
    use_gpu: bool = False

    def ensure_dirs(self) -> None:
        """Create required data directories if they don't exist."""
        for d in [self.upload_dir, self.output_dir, self.temp_dir, self.model_cache_dir]:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings singleton."""
    settings = Settings()
    settings.ensure_dirs()
    return settings


# Convenience alias used throughout the codebase
settings = get_settings()
