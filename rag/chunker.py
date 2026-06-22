"""
DocFlow — RAG Chunking System
Multi-strategy text chunker: recursive, semantic, heading-aware, token-aware.
All chunks include rich metadata for downstream retrieval.
"""
from __future__ import annotations

import re
import uuid
from typing import Any

from loguru import logger

from core.config import settings


class Chunker:
    """
    Unified chunking interface.

    Strategies:
        recursive  — splits on paragraphs, sentences, words (LangChain-style)
        heading    — splits on Markdown headings, preserving section context
        token      — splits by token count using tiktoken
        semantic   — splits based on embedding similarity (requires model)
    """

    def chunk(
        self,
        text: str,
        strategy: str = "recursive",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        metadata: dict | None = None,
    ) -> list[dict]:
        if not text or not text.strip():
            return []

        chunk_size = chunk_size or settings.chunk_size
        chunk_overlap = chunk_overlap or settings.chunk_overlap
        meta = metadata or {}

        if strategy == "heading":
            chunks = self._heading_chunk(text, chunk_size)
        elif strategy == "token":
            chunks = self._token_chunk(text, chunk_size, chunk_overlap)
        elif strategy == "semantic":
            chunks = self._semantic_chunk(text, chunk_size)
        else:
            chunks = self._recursive_chunk(text, chunk_size, chunk_overlap)

        # Attach metadata and IDs
        result = []
        for i, (chunk_text, chunk_meta) in enumerate(chunks):
            if len(chunk_text.strip()) < settings.chunk_min_size:
                continue
            token_count = self._rough_token_count(chunk_text)
            result.append({
                "chunk_id": str(uuid.uuid4()),
                "chunk_index": i,
                "text": chunk_text.strip(),
                "token_count": token_count,
                "char_count": len(chunk_text),
                **meta,
                **chunk_meta,
            })

        logger.debug(f"[Chunker] {strategy} strategy → {len(result)} chunks")
        return result

    # ── Recursive Chunker ────────────────────────────────────────────────────
    def _recursive_chunk(
        self, text: str, size: int, overlap: int
    ) -> list[tuple[str, dict]]:
        """
        Recursively split on: paragraphs → sentences → words.
        Mirrors LangChain's RecursiveCharacterTextSplitter.
        """
        separators = ["\n\n", "\n", ". ", " ", ""]
        return [(c, {}) for c in self._split_recursive(text, separators, size, overlap)]

    def _split_recursive(
        self, text: str, separators: list[str], size: int, overlap: int
    ) -> list[str]:
        if len(text) <= size:
            return [text]

        sep = separators[0] if separators else ""
        splits = text.split(sep) if sep else list(text)

        chunks: list[str] = []
        current = ""

        for split in splits:
            candidate = current + (sep if current else "") + split
            if len(candidate) <= size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                    # Overlap: carry last N chars
                    current = current[-overlap:] + (sep if sep else "") + split if overlap else split
                else:
                    # Single segment too large → recurse with next separator
                    if len(separators) > 1:
                        chunks.extend(self._split_recursive(split, separators[1:], size, overlap))
                    else:
                        chunks.append(split)
                    current = ""

        if current:
            chunks.append(current)

        return [c for c in chunks if c.strip()]

    # ── Heading-Aware Chunker ────────────────────────────────────────────────
    def _heading_chunk(self, text: str, size: int) -> list[tuple[str, dict]]:
        """Split on Markdown headings, keeping section title in metadata."""
        heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        matches = list(heading_re.finditer(text))

        if not matches:
            return self._recursive_chunk(text, size, size // 8)

        sections: list[tuple[str, str, int]] = []  # (title, content, level)
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            title = match.group(2).strip()
            level = len(match.group(1))
            content = text[start:end].strip()
            sections.append((title, content, level))

        chunks = []
        for title, content, level in sections:
            # If section is small enough, keep as single chunk
            if len(content) <= size:
                chunks.append((content, {"section_title": title, "heading_level": level}))
            else:
                # Recursively split large sections
                sub = self._recursive_chunk(content, size, size // 8)
                for j, (sub_text, _) in enumerate(sub):
                    chunks.append((sub_text, {"section_title": title, "heading_level": level, "sub_chunk": j}))

        return chunks

    # ── Token-Aware Chunker ──────────────────────────────────────────────────
    def _token_chunk(self, text: str, size: int, overlap: int) -> list[tuple[str, dict]]:
        """Split by token count using tiktoken."""
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(text)
            chunks = []
            start = 0
            while start < len(tokens):
                end = start + size
                chunk_tokens = tokens[start:end]
                chunk_text = enc.decode(chunk_tokens)
                chunks.append((chunk_text, {"token_start": start, "token_end": end}))
                start += size - overlap
            return chunks
        except ImportError:
            logger.warning("[Chunker] tiktoken not installed, falling back to character chunking")
            return self._recursive_chunk(text, size * 4, overlap * 4)

    # ── Semantic Chunker ─────────────────────────────────────────────────────
    def _semantic_chunk(self, text: str, size: int) -> list[tuple[str, dict]]:
        """
        Split based on sentence embedding similarity.
        Groups semantically similar consecutive sentences together.
        Falls back to recursive if sentence-transformers unavailable.
        """
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            sentences = re.split(r"(?<=[.!?])\s+", text)
            if len(sentences) < 3:
                return self._recursive_chunk(text, size, size // 8)

            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(sentences, show_progress_bar=False)

            # Compute cosine similarity between consecutive sentences
            chunks = []
            current_sents = [sentences[0]]
            current_text = sentences[0]

            for i in range(1, len(sentences)):
                sim = np.dot(embeddings[i - 1], embeddings[i]) / (
                    np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i]) + 1e-10
                )
                candidate = current_text + " " + sentences[i]
                # Split if similarity drops or chunk is getting large
                if sim < 0.5 or len(candidate) > size * 1.5:
                    chunks.append((current_text, {"semantic_break": True}))
                    current_text = sentences[i]
                else:
                    current_text = candidate

            if current_text:
                chunks.append((current_text, {}))

            return chunks

        except ImportError:
            logger.warning("[Chunker] sentence-transformers unavailable, using recursive fallback")
            return self._recursive_chunk(text, size, size // 8)

    # ── Utilities ────────────────────────────────────────────────────────────
    @staticmethod
    def _rough_token_count(text: str) -> int:
        """Rough token count (1 token ≈ 4 chars)."""
        return len(text) // 4
