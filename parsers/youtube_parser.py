"""
DocFlow — YouTube Parser
Downloads YouTube transcripts or falls back to audio download + Whisper.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger


# YouTube URL detection pattern
YT_PATTERN = re.compile(
    r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[\w\-]+"
)


def is_youtube_url(url: str) -> bool:
    return bool(YT_PATTERN.match(url.strip()))


class YouTubeParser:
    """
    YouTube ingestion pipeline.

    Strategy:
        1. Try youtube-transcript-api (fast, no download needed)
        2. Fall back to yt-dlp audio download + Whisper transcription
    """

    async def parse_url(self, url: str, options: Any) -> dict:
        logger.info(f"[YouTubeParser] Processing: {url}")

        # Get video metadata
        meta = await self._get_metadata(url)

        # Try transcript API first
        try:
            transcript_data = await self._get_transcript(meta.get("video_id", ""))
            if transcript_data:
                return self._build_result(transcript_data, meta, method="transcript_api")
        except Exception as exc:
            logger.warning(f"[YouTubeParser] Transcript API failed: {exc}")

        # Fall back to audio download + Whisper
        return await self._download_and_transcribe(url, meta, options)

    async def _get_metadata(self, url: str) -> dict:
        """Extract video metadata using yt-dlp."""
        try:
            import yt_dlp
        except ImportError:
            return {}

        ydl_opts = {"quiet": True, "no_warnings": True, "extract_flat": False}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                return {
                    "video_id": info.get("id", ""),
                    "title": info.get("title", ""),
                    "channel": info.get("uploader", ""),
                    "upload_date": info.get("upload_date", ""),
                    "duration": info.get("duration", 0),
                    "description": info.get("description", ""),
                    "view_count": info.get("view_count", 0),
                    "url": url,
                }
            except Exception as exc:
                logger.warning(f"[YouTubeParser] Metadata extraction failed: {exc}")
                return {"url": url}

    async def _get_transcript(self, video_id: str) -> list[dict] | None:
        """Try youtube-transcript-api for captions."""
        if not video_id:
            return None
        from youtube_transcript_api import YouTubeTranscriptApi
        transcripts = YouTubeTranscriptApi.get_transcript(video_id)
        return transcripts  # list of {"text", "start", "duration"}

    async def _download_and_transcribe(self, url: str, meta: dict, options: Any) -> dict:
        """Download audio and transcribe with Whisper."""
        import yt_dlp

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "audio.mp3"
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": str(audio_path.with_suffix("")),
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "128",
                }],
                "quiet": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # Find downloaded file
            audio_files = list(Path(tmpdir).glob("*.mp3"))
            if not audio_files:
                return {"markdown": "<!-- Audio download failed -->", "text": "",
                        "structured": {}, "tables": [], "images": [],
                        "metadata": meta, "hyperlinks": [url]}

            from parsers.audio_parser import AudioParser
            audio_result = await AudioParser().parse(audio_files[0], options)
            audio_result["metadata"].update(meta)
            return audio_result

    def _build_result(self, transcript: list[dict], meta: dict, method: str) -> dict:
        title = meta.get("title", "YouTube Video")
        channel = meta.get("channel", "")
        date = meta.get("upload_date", "")
        url = meta.get("url", "")

        md_lines = [
            f"# {title}\n",
            f"**Channel:** {channel}  ",
            f"**Date:** {date}  ",
            f"**URL:** {url}\n",
            "---\n",
            "## Transcript\n",
        ]

        full_text_parts = []
        for entry in transcript:
            text = entry.get("text", "").strip()
            start = entry.get("start", 0)
            md_lines.append(f"**[{start:.1f}s]** {text}\n")
            full_text_parts.append(text)

        return {
            "markdown": "\n".join(md_lines),
            "text": " ".join(full_text_parts),
            "structured": {"transcript": transcript, "metadata": meta},
            "tables": [],
            "images": [],
            "metadata": {**meta, "method": method},
            "hyperlinks": [url],
        }
