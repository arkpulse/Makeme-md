"""
DocFlow — HTML, Text, JSON, and XML Parsers
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger


class HTMLParser:
    async def parse(self, file_path: Path, options: Any) -> dict:
        logger.info(f"[HTMLParser] Parsing: {file_path.name}")
        try:
            from bs4 import BeautifulSoup
            import html2text
        except ImportError:
            raw = file_path.read_text(errors="replace")
            return {"markdown": raw, "text": raw, "structured": {},
                    "tables": [], "images": [], "metadata": {}, "hyperlinks": []}

        raw_html = file_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw_html, "lxml")

        # Metadata from <meta> tags
        metadata = {}
        for tag in soup.find_all("meta"):
            name = tag.get("name") or tag.get("property", "")
            content = tag.get("content", "")
            if name and content:
                metadata[name] = content

        # Convert to Markdown
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.body_width = 0
        md = h.handle(raw_html)

        # Hyperlinks
        links = [a["href"] for a in soup.find_all("a", href=True)]

        # Tables
        tables_data = []
        for i, table in enumerate(soup.find_all("table")):
            rows = []
            for tr in table.find_all("tr"):
                row = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if row:
                    rows.append(row)
            if rows:
                tables_data.append({"index": i, "data": rows})

        return {
            "markdown": md,
            "text": soup.get_text(separator="\n"),
            "structured": {"title": soup.title.string if soup.title else ""},
            "tables": tables_data,
            "images": [],
            "metadata": metadata,
            "hyperlinks": links,
        }


class TextParser:
    async def parse(self, file_path: Path, options: Any) -> dict:
        logger.info(f"[TextParser] Parsing: {file_path.name}")
        try:
            import chardet
            raw_bytes = file_path.read_bytes()
            detected = chardet.detect(raw_bytes)
            encoding = detected.get("encoding") or "utf-8"
            text = raw_bytes.decode(encoding, errors="replace")
        except Exception:
            text = file_path.read_text(errors="replace")

        return {
            "markdown": text,
            "text": text,
            "structured": {},
            "tables": [],
            "images": [],
            "metadata": {"encoding": encoding if "encoding" in dir() else "utf-8"},
            "hyperlinks": [],
        }


class JSONParser:
    async def parse(self, file_path: Path, options: Any) -> dict:
        import json
        logger.info(f"[JSONParser] Parsing: {file_path.name}")
        try:
            data = json.loads(file_path.read_text(errors="replace"))
        except json.JSONDecodeError as e:
            return {"markdown": f"<!-- JSON parse error: {e} -->", "text": "",
                    "structured": {}, "tables": [], "images": [], "metadata": {}, "hyperlinks": []}

        # Pretty markdown code block
        md = f"```json\n{json.dumps(data, indent=2, ensure_ascii=False)}\n```"
        return {
            "markdown": md,
            "text": json.dumps(data, ensure_ascii=False),
            "structured": data if isinstance(data, dict) else {"data": data},
            "tables": [],
            "images": [],
            "metadata": {"type": type(data).__name__},
            "hyperlinks": [],
        }
