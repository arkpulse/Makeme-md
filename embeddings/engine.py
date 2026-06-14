"""
DocFlow — Embedding Engine
Unified embedding interface: OpenAI, SentenceTransformers, Ollama, HuggingFace.
Includes batch processing, async pipelines, and result caching.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from loguru import logger

from core.config import settings


class EmbeddingEngine:
    """
    Provider-agnostic embedding generation.

    Providers:
        sentence_transformers  (default, local, no API key)
        openai                 (requires OPENAI_API_KEY)
        ollama                 (local Ollama server)
        huggingface            (HF Inference API)
    """

    def __init__(self):
        self._st_model = None
        self._cache: dict[str, list[float]] = {}

    async def embed(self, text: str, provider: str | None = None) -> list[float]:
        """Embed a single text string."""
        result = await self.embed_batch([text], provider)
        return result[0] if result else []

    async def embed_batch(
        self, texts: list[str], provider: str | None = None
    ) -> list[list[float]]:
        """Embed a batch of texts. Results are cached by content hash."""
        provider = provider or settings.embedding_provider

        # Check cache
        cache_keys = [self._cache_key(t) for t in texts]
        uncached_indices = [i for i, k in enumerate(cache_keys) if k not in self._cache]
        uncached_texts = [texts[i] for i in uncached_indices]

        if uncached_texts:
            logger.debug(f"[Embeddings] Computing {len(uncached_texts)} embeddings via {provider}")
            new_embeddings = await self._dispatch(uncached_texts, provider)
            for idx, emb in zip(uncached_indices, new_embeddings):
                self._cache[cache_keys[idx]] = emb

        return [self._cache[k] for k in cache_keys]

    async def _dispatch(self, texts: list[str], provider: str) -> list[list[float]]:
        if provider == "openai":
            return await self._openai_embed(texts)
        elif provider == "ollama":
            return await self._ollama_embed(texts)
        elif provider == "huggingface":
            return await self._hf_embed(texts)
        else:
            return await self._st_embed(texts)

    # ── SentenceTransformers ─────────────────────────────────────────────────
    def _get_st_model(self):
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(
                settings.embedding_model,
                cache_folder=str(settings.model_cache_dir),
            )
        return self._st_model

    async def _st_embed(self, texts: list[str]) -> list[list[float]]:
        model = self._get_st_model()
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(
                texts,
                batch_size=settings.embedding_batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            ),
        )
        return [emb.tolist() for emb in embeddings]

    # ── OpenAI ───────────────────────────────────────────────────────────────
    async def _openai_embed(self, texts: list[str]) -> list[list[float]]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)

        # Batch into chunks of 100 (API limit)
        all_embeddings = []
        for i in range(0, len(texts), 100):
            batch = texts[i:i + 100]
            response = await client.embeddings.create(
                model=settings.openai_embedding_model,
                input=batch,
            )
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings

    # ── Ollama ───────────────────────────────────────────────────────────────
    async def _ollama_embed(self, texts: list[str]) -> list[list[float]]:
        import httpx
        embeddings = []
        async with httpx.AsyncClient(timeout=60) as client:
            for text in texts:
                resp = await client.post(
                    f"{settings.ollama_base_url}/api/embeddings",
                    json={"model": settings.ollama_embedding_model, "prompt": text},
                )
                resp.raise_for_status()
                embeddings.append(resp.json()["embedding"])
        return embeddings

    # ── HuggingFace ──────────────────────────────────────────────────────────
    async def _hf_embed(self, texts: list[str]) -> list[list[float]]:
        import httpx
        headers = {"Authorization": f"Bearer {settings.huggingface_token}"}
        url = f"https://api-inference.huggingface.co/models/{settings.embedding_model}"
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json={"inputs": texts}, headers=headers)
            resp.raise_for_status()
            return resp.json()

    # ── Cache ────────────────────────────────────────────────────────────────
    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def clear_cache(self) -> None:
        self._cache.clear()
