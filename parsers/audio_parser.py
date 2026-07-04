"""
DocFlow — Audio Parser
Transcribes audio files using OpenAI Whisper or Faster-Whisper.
Supports MP3, WAV, FLAC, M4A with timestamped output and speaker diarization.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger

from core.config import settings


class AudioParser:
    """
    Audio → Markdown transcript pipeline.

    Provider selection (in order of preference):
        1. faster-whisper  (CTranslate2-optimised, GPU/CPU)
        2. openai-whisper  (original, GPU/CPU)
        3. OpenAI API      (if API key configured)
    """

    async def parse(self, file_path: Path, options: Any) -> dict:
        logger.info(f"[AudioParser] Transcribing: {file_path.name}")

        # Try faster-whisper first
        try:
            return await self._transcribe_faster_whisper(file_path)
        except ImportError:
            pass

        # Fall back to original whisper
        try:
            return await self._transcribe_whisper(file_path)
        except ImportError:
            pass

        # Final fallback: OpenAI API
        if settings.openai_api_key:
            return await self._transcribe_openai_api(file_path)

        return {
            "markdown": "<!-- Audio transcription unavailable: no Whisper library installed -->",
            "text": "",
            "structured": {},
            "tables": [],
            "images": [],
            "metadata": {},
            "hyperlinks": [],
        }

    async def _transcribe_faster_whisper(self, file_path: Path) -> dict:
        """Use faster-whisper (CTranslate2 backend) — recommended."""
        from faster_whisper import WhisperModel

        model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            download_root=str(settings.model_cache_dir),
        )

        loop = asyncio.get_event_loop()
        # Run synchronous model.transcribe in thread pool
        segments, info = await loop.run_in_executor(
            None,
            lambda: model.transcribe(
                str(file_path),
                beam_size=5,
                word_timestamps=True,
            ),
        )

        segments = list(segments)  # materialise generator
        return self._build_result(segments, info, file_path)

    async def _transcribe_whisper(self, file_path: Path) -> dict:
        """Use original OpenAI Whisper library."""
        import whisper

        model = whisper.load_model(settings.whisper_model)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: model.transcribe(str(file_path), verbose=False),
        )

        segments = result.get("segments", [])
        return self._build_result_original(segments, result, file_path)

    async def _transcribe_openai_api(self, file_path: Path) -> dict:
        """Use OpenAI Whisper API as last resort."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.openai_api_key)
        with open(file_path, "rb") as f:
            transcript = await client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        segments = getattr(transcript, "segments", []) or []
        full_text = getattr(transcript, "text", "")
        return {
            "markdown": self._segments_to_markdown(segments, full_text),
            "text": full_text,
            "structured": {"segments": segments},
            "tables": [],
            "images": [],
            "metadata": {"provider": "openai_api", "source": file_path.name},
            "hyperlinks": [],
        }

    def _build_result(self, segments, info, file_path: Path) -> dict:
        """Build result from faster-whisper segments."""
        seg_dicts = []
        md_lines = [f"# Transcript: {file_path.stem}\n"]
        full_text_parts = []

        for seg in segments:
            start = f"{seg.start:.1f}s"
            end = f"{seg.end:.1f}s"
            text = seg.text.strip()
            md_lines.append(f"**[{start} → {end}]** {text}\n")
            full_text_parts.append(text)
            seg_dicts.append({"start": seg.start, "end": seg.end, "text": text})

        return {
            "markdown": "\n".join(md_lines),
            "text": " ".join(full_text_parts),
            "structured": {
                "segments": seg_dicts,
                "language": getattr(info, "language", "unknown"),
                "duration": getattr(info, "duration", 0),
            },
            "tables": [],
            "images": [],
            "metadata": {
                "source": file_path.name,
                "provider": "faster_whisper",
                "model": settings.whisper_model,
            },
            "hyperlinks": [],
        }

    def _build_result_original(self, segments: list, result: dict, file_path: Path) -> dict:
        md_lines = [f"# Transcript: {file_path.stem}\n"]
        for seg in segments:
            start = f"{seg['start']:.1f}s"
            end = f"{seg['end']:.1f}s"
            text = seg["text"].strip()
            md_lines.append(f"**[{start} → {end}]** {text}\n")

        return {
            "markdown": "\n".join(md_lines),
            "text": result.get("text", ""),
            "structured": {"segments": segments, "language": result.get("language", "")},
            "tables": [],
            "images": [],
            "metadata": {"source": file_path.name, "provider": "whisper"},
            "hyperlinks": [],
        }

    def _segments_to_markdown(self, segments: list, full_text: str) -> str:
        lines = ["# Transcript\n"]
        for seg in segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "").strip()
            lines.append(f"**[{start:.1f}s → {end:.1f}s]** {text}\n")
        return "\n".join(lines) if len(lines) > 1 else full_text
