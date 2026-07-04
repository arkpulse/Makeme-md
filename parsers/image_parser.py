"""
DocFlow — Image Parser
OCR extraction, EXIF metadata, BLIP captioning, and layout analysis for images.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger


class ImageParser:
    async def parse(self, file_path: Path, options: Any) -> dict:
        logger.info(f"[ImageParser] Processing: {file_path.name}")
        from PIL import Image

        img = Image.open(str(file_path))
        metadata = self._extract_exif(img)
        metadata["format"] = img.format
        metadata["mode"] = img.mode
        metadata["width"], metadata["height"] = img.size

        # OCR
        ocr_text = ""
        if options.enable_ocr:
            from ocr.engine import OCREngine
            engine = OCREngine()
            ocr_text = await engine.extract_text(img)

        # Caption (optional — requires transformers)
        caption = ""
        try:
            caption = await self._generate_caption(img)
        except Exception:
            pass

        md_parts = [f"## Image: {file_path.name}\n"]
        if caption:
            md_parts.append(f"**Caption:** {caption}\n")
        if ocr_text.strip():
            md_parts.append(f"\n### Extracted Text\n\n{ocr_text}")

        return {
            "markdown": "\n".join(md_parts),
            "text": ocr_text,
            "structured": {"caption": caption, "ocr_text": ocr_text},
            "tables": [],
            "images": [{"file": file_path.name, "caption": caption, **metadata}],
            "metadata": metadata,
            "hyperlinks": [],
        }

    def _extract_exif(self, img) -> dict:
        """Extract EXIF metadata from image."""
        try:
            from PIL.ExifTags import TAGS
            exif_raw = img._getexif() or {}
            return {TAGS.get(k, str(k)): str(v) for k, v in exif_raw.items()}
        except Exception:
            return {}

    async def _generate_caption(self, img) -> str:
        """Generate image caption using BLIP model."""
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch

        model_name = "Salesforce/blip-image-captioning-base"
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)

        inputs = processor(img, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=50)
        return processor.decode(out[0], skip_special_tokens=True)
