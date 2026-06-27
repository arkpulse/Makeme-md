"""
DocFlow — Upload Route
Handles file upload, validation, and storage.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from core.config import settings
from utils.file_utils import validate_upload

router = APIRouter()


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    size_bytes: int
    mime_type: str
    path: str


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a file for processing. Returns a file_id for subsequent API calls."""
    # Validate extension
    ext = Path(file.filename or "").suffix.lstrip(".").lower()
    if ext not in settings.allowed_extensions_set:
        raise HTTPException(400, f"Unsupported file type: .{ext}")

    # Check size
    contents = await file.read()
    if len(contents) > settings.max_upload_bytes:
        raise HTTPException(413, f"File exceeds {settings.max_upload_size_mb}MB limit")

    # Save to upload directory
    file_id = str(uuid.uuid4())
    dest_path = settings.upload_dir / f"{file_id}_{file.filename}"
    dest_path.write_bytes(contents)

    from utils.file_utils import get_mime_type
    mime = get_mime_type(dest_path)

    return UploadResponse(
        file_id=file_id,
        filename=file.filename or "",
        size_bytes=len(contents),
        mime_type=mime,
        path=str(dest_path),
    )
