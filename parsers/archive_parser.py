"""
DocFlow — Archive Parser
Recursively extracts and processes ZIP archives.
"""
from __future__ import annotations

import asyncio
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from loguru import logger

from core.config import settings


class ArchiveParser:
    async def parse(self, file_path: Path, options: Any) -> dict:
        logger.info(f"[ArchiveParser] Extracting: {file_path.name}")

        if not zipfile.is_zipfile(str(file_path)):
            return {"markdown": "<!-- Not a valid ZIP file -->", "text": "",
                    "structured": {}, "tables": [], "images": [], "metadata": {}, "hyperlinks": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(str(file_path), "r") as zf:
                zf.extractall(tmpdir)

            # Recursively process all extracted files
            from core.dispatcher import FormatDispatcher
            dispatcher = FormatDispatcher()

            extracted_path = Path(tmpdir)
            all_files = [
                p for p in extracted_path.rglob("*")
                if p.is_file()
                and p.suffix.lstrip(".").lower() in settings.allowed_extensions_set
                and not p.name.startswith(".")
            ]

            logger.info(f"[ArchiveParser] Found {len(all_files)} files inside archive")

            results = await asyncio.gather(
                *[dispatcher.dispatch(f, options) for f in all_files],
                return_exceptions=True
            )

            combined_md: list[str] = [f"# Archive: {file_path.name}\n"]
            all_tables: list[dict] = []

            for f, res in zip(all_files, results):
                if isinstance(res, Exception):
                    logger.warning(f"[ArchiveParser] Failed to parse {f.name}: {res}")
                    continue
                combined_md.append(f"\n---\n## File: {f.name}\n\n{res.get('markdown', '')}")
                all_tables.extend(res.get("tables", []))

        return {
            "markdown": "\n".join(combined_md),
            "text": "\n".join(combined_md),
            "structured": {"file_count": len(all_files)},
            "tables": all_tables,
            "images": [],
            "metadata": {"source": file_path.name, "file_count": len(all_files)},
            "hyperlinks": [],
        }
