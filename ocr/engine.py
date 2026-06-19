"""
DocFlow — OCR Engine
Unified OCR interface supporting PaddleOCR (GPU), EasyOCR, and Tesseract.
Automatic engine selection with graceful fallbacks.
"""
from __future__ import annotations

import asyncio
import io
from typing import Any

from loguru import logger

from core.config import settings


class OCREngine:
    """
    Abstract OCR interface.

    Engine priority:
        auto → PaddleOCR (GPU if available) → EasyOCR → Tesseract
    """

    def __init__(self, engine: str = "auto", language: str | None = None):
        self.engine = engine or settings.ocr_engine
        self.language = language or settings.ocr_language
        self._paddle = None
        self._easy = None

    async def extract_text(self, image) -> str:
        """
        Run OCR on a PIL Image and return extracted text.
        Falls through engines until one succeeds.
        """
        if self.engine in ("auto", "paddle"):
            try:
                return await self._paddle_ocr(image)
            except Exception as e:
                if self.engine == "paddle":
                    raise
                logger.debug(f"[OCR] PaddleOCR unavailable: {e}")

        if self.engine in ("auto", "easyocr"):
            try:
                return await self._easyocr(image)
            except Exception as e:
                if self.engine == "easyocr":
                    raise
                logger.debug(f"[OCR] EasyOCR unavailable: {e}")

        # Always fall through to tesseract
        return await self._tesseract(image)

    async def extract_with_boxes(self, image) -> list[dict]:
        """
        Run OCR and return list of {'text', 'box', 'confidence'} dicts.
        Used for layout-aware extraction.
        """
        try:
            return await self._paddle_ocr_with_boxes(image)
        except Exception:
            # EasyOCR with detail
            return await self._easyocr_with_boxes(image)

    # ── PaddleOCR ────────────────────────────────────────────────────────────
    def _get_paddle(self):
        if self._paddle is None:
            from paddleocr import PaddleOCR
            self._paddle = PaddleOCR(
                use_angle_cls=True,
                lang=self._map_language_paddle(self.language),
                use_gpu=settings.ocr_use_gpu,
                show_log=False,
            )
        return self._paddle

    async def _paddle_ocr(self, image) -> str:
        import numpy as np
        paddle = self._get_paddle()
        img_array = np.array(image.convert("RGB"))
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: paddle.ocr(img_array, cls=True))
        lines = []
        for page in (result or []):
            for line in (page or []):
                if line and len(line) >= 2:
                    text_info = line[1]
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                        text = text_info[0]
                        conf = text_info[1] if len(text_info) > 1 else 1.0
                        if conf >= settings.ocr_confidence_threshold:
                            lines.append(text)
        return "\n".join(lines)

    async def _paddle_ocr_with_boxes(self, image) -> list[dict]:
        import numpy as np
        paddle = self._get_paddle()
        img_array = np.array(image.convert("RGB"))
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: paddle.ocr(img_array, cls=True))
        boxes = []
        for page in (result or []):
            for line in (page or []):
                if line and len(line) >= 2:
                    box = line[0]
                    text, conf = line[1] if len(line[1]) > 1 else (line[1][0], 1.0)
                    boxes.append({"text": text, "box": box, "confidence": float(conf)})
        return boxes

    # ── EasyOCR ──────────────────────────────────────────────────────────────
    def _get_easy(self):
        if self._easy is None:
            import easyocr
            self._easy = easyocr.Reader(
                [self._map_language_easy(self.language)],
                gpu=settings.ocr_use_gpu,
                model_storage_directory=str(settings.model_cache_dir),
            )
        return self._easy

    async def _easyocr(self, image) -> str:
        import numpy as np
        reader = self._get_easy()
        img_array = np.array(image.convert("RGB"))
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: reader.readtext(img_array))
        return "\n".join(
            item[1] for item in result
            if item[2] >= settings.ocr_confidence_threshold
        )

    async def _easyocr_with_boxes(self, image) -> list[dict]:
        import numpy as np
        reader = self._get_easy()
        img_array = np.array(image.convert("RGB"))
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: reader.readtext(img_array))
        return [{"box": r[0], "text": r[1], "confidence": r[2]} for r in result]

    # ── Tesseract ─────────────────────────────────────────────────────────────
    async def _tesseract(self, image) -> str:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
        # Preprocess: convert to grayscale
        img = image.convert("L")
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, lambda: pytesseract.image_to_string(img))
        return text.strip()

    # ── Language mapping helpers ──────────────────────────────────────────────
    @staticmethod
    def _map_language_paddle(lang: str) -> str:
        mapping = {"en": "en", "zh": "ch", "fr": "fr", "de": "german",
                   "ja": "japan", "ko": "korean", "ar": "arabic"}
        return mapping.get(lang[:2], "en")

    @staticmethod
    def _map_language_easy(lang: str) -> str:
        mapping = {"en": "en", "zh": "ch_sim", "fr": "fr", "de": "de",
                   "ja": "ja", "ko": "ko", "ar": "ar"}
        return mapping.get(lang[:2], "en")


class BatchOCRProcessor:
    """Process multiple images in parallel with GPU batching."""

    def __init__(self, engine: str = "auto", max_batch_size: int = 8):
        self.ocr = OCREngine(engine=engine)
        self.max_batch_size = max_batch_size

    async def process_batch(self, images: list) -> list[str]:
        """Process a list of PIL Images and return OCR text for each."""
        semaphore = asyncio.Semaphore(self.max_batch_size)

        async def _process_one(img) -> str:
            async with semaphore:
                return await self.ocr.extract_text(img)

        return await asyncio.gather(*[_process_one(img) for img in images])
