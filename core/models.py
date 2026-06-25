"""
DocFlow — Database Models
SQLAlchemy async ORM models for file metadata, processing logs, and search index.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, DateTime, Float, Integer, JSON, String, Text, func
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class FileRecord(Base):
    """Tracks every file uploaded and processed."""
    __tablename__ = "file_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    file_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    original_filename: Mapped[str] = mapped_column(String(512))
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    mime_type: Mapped[str] = mapped_column(String(128), default="")
    file_type: Mapped[str] = mapped_column(String(64), default="")

    # Processing status
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending|processing|done|failed
    processing_time_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Outputs
    markdown_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    json_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    table_count: Mapped[int] = mapped_column(Integer, default=0)
    page_count: Mapped[int] = mapped_column(Integer, default=0)

    # Metadata
    doc_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self) -> str:
        return f"<FileRecord {self.original_filename} [{self.status}]>"


class ChunkRecord(Base):
    """Stores document chunks for retrieval and audit."""
    __tablename__ = "chunk_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    chunk_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    file_id: Mapped[str] = mapped_column(String(64), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)

    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    char_count: Mapped[int] = mapped_column(Integer, default=0)

    section_title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    chunk_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ProcessingLog(Base):
    """Audit log for all processing events."""
    __tablename__ = "processing_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    level: Mapped[str] = mapped_column(String(16), default="INFO")
    event: Mapped[str] = mapped_column(String(256))
    details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    duration_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
