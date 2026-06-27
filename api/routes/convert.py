"""
DocFlow — Conversion Route
Main document processing endpoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from core.config import settings
from core.pipeline import DocFlowPipeline, ProcessingOptions

router = APIRouter()
_pipeline = DocFlowPipeline()


class ConvertRequest(BaseModel):
    file_path: str
    enable_ocr: bool = True
    enable_summarization: bool = False
    enable_embeddings: bool = True
    chunk_strategy: str = "recursive"
    chunk_size: int = 512


class ConvertResponse(BaseModel):
    file_id: str
    file_name: str
    markdown: str
    summary: str
    chunk_count: int
    tables_found: int
    processing_time_s: float
    success: bool
    errors: list[str]


@router.post("/convert", response_model=ConvertResponse)
async def convert_file(
    file: UploadFile = File(...),
    enable_ocr: bool = Form(True),
    enable_summarization: bool = Form(False),
    enable_embeddings: bool = Form(False),
    chunk_strategy: str = Form("recursive"),
    chunk_size: int = Form(512),
):
    """Convert an uploaded file to Markdown + structured outputs."""
    import uuid, tempfile

    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(413, "File too large")

    # Save to temp
    suffix = Path(file.filename or "file").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    opts = ProcessingOptions(
        enable_ocr=enable_ocr,
        enable_summarization=enable_summarization,
        enable_embeddings=enable_embeddings,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
    )

    try:
        result = await _pipeline.process(tmp_path, opts)
    finally:
        tmp_path.unlink(missing_ok=True)

    return ConvertResponse(
        file_id=result.file_id,
        file_name=result.file_name,
        markdown=result.markdown,
        summary=result.summary,
        chunk_count=len(result.chunks),
        tables_found=len(result.tables),
        processing_time_s=result.processing_time_s,
        success=result.success,
        errors=result.errors,
    )
