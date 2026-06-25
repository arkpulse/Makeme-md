"""
DocFlow — Format Dispatcher
Routes incoming files to the appropriate parser based on MIME type / extension.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from utils.file_utils import get_extension, get_mime_type

# Lazy imports to avoid loading heavy dependencies at startup
EXTENSION_MAP: dict[str, str] = {
    # Documents
    "pdf": "pdf",
    "docx": "docx",
    "doc": "docx",
    "pptx": "pptx",
    "ppt": "pptx",
    "xlsx": "xlsx",
    "xls": "xlsx",
    "csv": "csv",
    # Text / Data
    "txt": "text",
    "md": "markdown",
    "markdown": "markdown",
    "html": "html",
    "htm": "html",
    "xml": "xml",
    "json": "json",
    # Images
    "png": "image",
    "jpg": "image",
    "jpeg": "image",
    "tiff": "image",
    "tif": "image",
    "bmp": "image",
    "webp": "image",
    # Audio
    "mp3": "audio",
    "wav": "audio",
    "flac": "audio",
    "m4a": "audio",
    "ogg": "audio",
    # Archives
    "zip": "archive",
}

MIME_MAP: dict[str, str] = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "text/csv": "csv",
    "text/plain": "text",
    "text/markdown": "markdown",
    "text/html": "html",
    "application/xml": "xml",
    "text/xml": "xml",
    "application/json": "json",
    "image/png": "image",
    "image/jpeg": "image",
    "image/tiff": "image",
    "image/bmp": "image",
    "image/webp": "image",
    "audio/mpeg": "audio",
    "audio/wav": "audio",
    "audio/flac": "audio",
    "audio/mp4": "audio",
    "application/zip": "archive",
}


def _load_parser(parser_type: str):
    """Lazy-import and return the correct parser class."""
    if parser_type == "pdf":
        from parsers.pdf_parser import PDFParser
        return PDFParser()
    elif parser_type == "docx":
        from parsers.docx_parser import DOCXParser
        return DOCXParser()
    elif parser_type == "pptx":
        from parsers.pptx_parser import PPTXParser
        return PPTXParser()
    elif parser_type == "xlsx":
        from parsers.xlsx_parser import XLSXParser
        return XLSXParser()
    elif parser_type == "csv":
        from parsers.csv_parser import CSVParser
        return CSVParser()
    elif parser_type in ("html", "xml"):
        from parsers.html_parser import HTMLParser
        return HTMLParser()
    elif parser_type in ("text", "markdown"):
        from parsers.text_parser import TextParser
        return TextParser()
    elif parser_type == "json":
        from parsers.json_parser import JSONParser
        return JSONParser()
    elif parser_type == "image":
        from parsers.image_parser import ImageParser
        return ImageParser()
    elif parser_type == "audio":
        from parsers.audio_parser import AudioParser
        return AudioParser()
    elif parser_type == "archive":
        from parsers.archive_parser import ArchiveParser
        return ArchiveParser()
    else:
        raise ValueError(f"No parser registered for type: {parser_type!r}")


class FormatDispatcher:
    """
    Determines the correct parser for a file and delegates processing.
    Resolution order: file extension → MIME type → fallback to plain text.
    """

    def resolve_parser_type(self, file_path: Path) -> str:
        ext = get_extension(file_path)
        if ext in EXTENSION_MAP:
            return EXTENSION_MAP[ext]

        mime = get_mime_type(file_path)
        if mime in MIME_MAP:
            return MIME_MAP[mime]

        logger.warning(f"[Dispatcher] Unknown format for {file_path.name}, falling back to text")
        return "text"

    async def dispatch(self, file_path: Path, options: Any) -> dict:
        """
        Route the file to the appropriate parser and return a normalised dict.

        Returns a dict with keys:
            markdown, text, structured, tables, images, metadata, hyperlinks
        """
        parser_type = self.resolve_parser_type(file_path)
        logger.debug(f"[Dispatcher] {file_path.name} → {parser_type} parser")

        parser = _load_parser(parser_type)
        result = await parser.parse(file_path, options)

        # Ensure all expected keys are present
        result.setdefault("markdown", "")
        result.setdefault("text", result.get("markdown", ""))
        result.setdefault("structured", {})
        result.setdefault("tables", [])
        result.setdefault("images", [])
        result.setdefault("metadata", {})
        result.setdefault("hyperlinks", [])

        return result
