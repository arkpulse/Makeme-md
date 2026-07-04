"""
DocFlow — PDF Parser
Advanced PDF parsing using PyMuPDF + pdfplumber with OCR fallback.
Preserves headings, tables, lists, images, hyperlinks, and multi-column layout.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from loguru import logger

from utils.markdown_utils import tables_to_markdown, clean_markdown


class PDFParser:
    """
    Multi-strategy PDF parser.

    Strategy selection:
        1. PyMuPDF (fitz) — fast text + metadata extraction
        2. pdfplumber — accurate table extraction
        3. OCR fallback — for scanned / image-only PDFs
    """

    async def parse(self, file_path: Path, options: Any) -> dict:
        logger.info(f"[PDFParser] Parsing: {file_path.name}")

        # Attempt text-based extraction first
        result = await self._parse_with_fitz(file_path, options)

        text_ratio = len(result.get("text", "").strip()) / max(result.get("page_count", 1), 1)
        is_scanned = text_ratio < 50  # Less than ~50 chars/page → likely scanned

        if is_scanned and options.enable_ocr:
            logger.info(f"[PDFParser] Scanned PDF detected — switching to OCR for {file_path.name}")
            ocr_result = await self._parse_with_ocr(file_path, options)
            result["text"] = ocr_result.get("text", result["text"])
            result["markdown"] = ocr_result.get("markdown", result["markdown"])
            result["ocr_used"] = True

        # Augment tables via pdfplumber
        if options.extract_tables:
            plumber_tables = await self._extract_tables_pdfplumber(file_path)
            result["tables"].extend(plumber_tables)

        return result

    async def _parse_with_fitz(self, file_path: Path, options: Any) -> dict:
        """Primary extraction using PyMuPDF (fitz)."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("[PDFParser] PyMuPDF not installed, falling back to pdfplumber")
            return await self._parse_with_pdfplumber(file_path, options)

        doc = fitz.open(str(file_path))
        pages_md: list[str] = []
        all_tables: list[dict] = []
        all_images: list[dict] = []
        all_links: list[str] = []
        metadata: dict = {}

        # Document-level metadata
        meta = doc.metadata or {}
        metadata = {
            "title": meta.get("title", ""),
            "author": meta.get("author", ""),
            "subject": meta.get("subject", ""),
            "keywords": meta.get("keywords", ""),
            "creator": meta.get("creator", ""),
            "producer": meta.get("producer", ""),
            "page_count": doc.page_count,
        }

        for page_num, page in enumerate(doc, start=1):
            page_md_parts: list[str] = []

            # Extract text blocks with font-size heuristics for heading detection
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            for block in blocks:
                if block["type"] != 0:  # 0 = text
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        if not text:
                            continue
                        size = span.get("size", 12)
                        flags = span.get("flags", 0)
                        bold = bool(flags & 2**4)

                        # Heuristic heading detection
                        if size >= 18 or (size >= 14 and bold):
                            page_md_parts.append(f"\n## {text}\n")
                        elif size >= 14 or bold:
                            page_md_parts.append(f"\n### {text}\n")
                        else:
                            page_md_parts.append(text + " ")

            # Hyperlinks
            for link in page.get_links():
                uri = link.get("uri", "")
                if uri:
                    all_links.append(uri)

            # Image extraction (optional)
            if options.extract_images:
                for img_ref in page.get_images(full=True):
                    xref = img_ref[0]
                    try:
                        base_img = doc.extract_image(xref)
                        all_images.append({
                            "page": page_num,
                            "ext": base_img["ext"],
                            "width": base_img.get("width", 0),
                            "height": base_img.get("height", 0),
                        })
                    except Exception:
                        pass

            page_text = "".join(page_md_parts)
            if page_text.strip():
                pages_md.append(f"\n---\n*Page {page_num}*\n\n{page_text}")

        doc.close()

        full_markdown = "\n".join(pages_md)
        full_text = "\n".join(
            p.replace("## ", "").replace("### ", "").replace("\n---\n*Page ", "")
            for p in pages_md
        )

        return {
            "markdown": clean_markdown(full_markdown),
            "text": full_text,
            "structured": {"pages": len(pages_md)},
            "tables": all_tables,
            "images": all_images,
            "metadata": metadata,
            "hyperlinks": list(set(all_links)),
            "page_count": metadata.get("page_count", len(pages_md)),
            "ocr_used": False,
        }

    async def _parse_with_pdfplumber(self, file_path: Path, options: Any) -> dict:
        """Fallback using pdfplumber for text extraction."""
        try:
            import pdfplumber
        except ImportError:
            return {"markdown": "", "text": "", "structured": {}, "tables": [],
                    "images": [], "metadata": {}, "hyperlinks": []}

        pages_text: list[str] = []
        all_tables: list[dict] = []

        with pdfplumber.open(str(file_path)) as pdf:
            metadata = {
                "page_count": len(pdf.pages),
                **(pdf.metadata or {}),
            }
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(f"*Page {i}*\n\n{text}")

                if options.extract_tables:
                    for tbl in page.extract_tables() or []:
                        all_tables.append({"page": i, "data": tbl})

        full_text = "\n\n".join(pages_text)
        return {
            "markdown": full_text,
            "text": full_text,
            "structured": {},
            "tables": all_tables,
            "images": [],
            "metadata": metadata,
            "hyperlinks": [],
        }

    async def _extract_tables_pdfplumber(self, file_path: Path) -> list[dict]:
        """Extract tables using pdfplumber for better accuracy than fitz."""
        tables: list[dict] = []
        try:
            import pdfplumber
            with pdfplumber.open(str(file_path)) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    for tbl in page.extract_tables() or []:
                        if tbl and len(tbl) > 1:
                            tables.append({
                                "page": i,
                                "data": tbl,
                                "markdown": tables_to_markdown(tbl),
                            })
        except Exception as exc:
            logger.warning(f"[PDFParser] pdfplumber table extraction failed: {exc}")
        return tables

    async def _parse_with_ocr(self, file_path: Path, options: Any) -> dict:
        """Convert each PDF page to image and run OCR."""
        try:
            import fitz
            from ocr.engine import OCREngine
        except ImportError:
            return {"markdown": "", "text": ""}

        ocr = OCREngine(engine=options.ocr_engine)
        doc = fitz.open(str(file_path))
        pages: list[str] = []

        for page_num, page in enumerate(doc, start=1):
            # Render page to PIL image at 2x resolution for accuracy
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")

            from PIL import Image
            img = Image.open(io.BytesIO(img_bytes))
            text = await ocr.extract_text(img)
            if text.strip():
                pages.append(f"*Page {page_num}*\n\n{text}")

        doc.close()
        full_text = "\n\n".join(pages)
        return {"markdown": full_text, "text": full_text}
