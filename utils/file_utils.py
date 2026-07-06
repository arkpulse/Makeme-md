"""
DocFlow — File Utilities
"""
from __future__ import annotations

import mimetypes
from pathlib import Path

from core.config import settings


def get_extension(path: Path) -> str:
    return path.suffix.lstrip(".").lower()


def get_mime_type(path: Path) -> str:
    """Detect MIME type using python-magic with mimetypes fallback."""
    try:
        import magic
        return magic.from_file(str(path), mime=True)
    except (ImportError, Exception):
        mime, _ = mimetypes.guess_type(str(path))
        return mime or "application/octet-stream"


def validate_file(path: Path) -> None:
    """Raise ValueError if the file is invalid."""
    if not path.exists():
        raise ValueError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"File is empty: {path}")
    ext = get_extension(path)
    if ext not in settings.allowed_extensions_set:
        raise ValueError(f"Unsupported file type: .{ext}")


def validate_upload(filename: str, size_bytes: int) -> None:
    """Validate an incoming upload."""
    ext = Path(filename).suffix.lstrip(".").lower()
    if ext not in settings.allowed_extensions_set:
        raise ValueError(f"Unsupported extension: .{ext}")
    if size_bytes > settings.max_upload_bytes:
        raise ValueError(f"File size {size_bytes} exceeds limit {settings.max_upload_bytes}")
