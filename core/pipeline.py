"""
DocFlow — Core Processing Pipeline
Orchestrates the full document ingestion → parse → chunk → embed → store workflow.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from core.config import settings
from core.dispatcher import FormatDispatcher
from rag.chunker import Chunker
from embeddings.engine import EmbeddingEngine
from utils.file_utils import get_mime_type, validate_file


@dataclass
class ProcessingOptions:
    """User-configurable options for a single processing job."""
    # Parsing
    enable_ocr: bool = True
    ocr_engine: str = "auto"           # auto | paddle | easyocr | tesseract
    extract_images: bool = True
    extract_tables: bool = True
    extract_metadata: bool = True

    # Chunking
    chunk_strategy: str = "recursive"  # recursive | semantic | heading | token
    chunk_size: int = 512
    chunk_overlap: int = 64

    # AI features
    enable_summarization: bool = False
    summarization_provider: str = "openai"
    enable_embeddings: bool = True
    embedding_provider: str = "sentence_transformers"

    # Storage
    store_in_vectordb: bool = True
    vector_store: str = "chroma"

    # Output
    output_formats: list[str] = field(default_factory=lambda: ["markdown", "json"])


@dataclass
class ProcessingResult:
    """Complete result of processing a single document."""
    file_id: str
    source_path: str
    file_name: str
    file_type: str
    processing_time_s: float

    # Core outputs
    markdown: str = ""
    plain_text: str = ""
    structured_json: dict = field(default_factory=dict)

    # Extracted components
    tables: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    hyperlinks: list[str] = field(default_factory=list)

    # AI outputs
    summary: str = ""
    key_insights: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)

    # RAG outputs
    chunks: list[dict] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)

    # Status
    success: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


class DocFlowPipeline:
    """
    Main document processing pipeline.

    Workflow:
        1. Validate file
        2. Detect format → dispatch to correct parser
        3. Extract content (text, tables, images, metadata)
        4. OCR fallback if needed
        5. Clean and normalise Markdown output
        6. AI summarisation (optional)
        7. Semantic chunking
        8. Embedding generation (optional)
        9. Vector store ingestion (optional)
       10. Persist result to database
    """

    def __init__(self):
        self.dispatcher = FormatDispatcher()
        self.chunker = Chunker()
        self._embedding_engine: Optional[EmbeddingEngine] = None

    @property
    def embedding_engine(self) -> EmbeddingEngine:
        if self._embedding_engine is None:
            self._embedding_engine = EmbeddingEngine()
        return self._embedding_engine

    def _compute_file_id(self, path: Path) -> str:
        """Stable file ID based on content hash."""
        h = hashlib.xxh64()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    async def process(
        self,
        file_path: Path | str,
        options: Optional[ProcessingOptions] = None,
    ) -> ProcessingResult:
        """Process a single file end-to-end and return a ProcessingResult."""
        file_path = Path(file_path)
        options = options or ProcessingOptions()
        start = time.perf_counter()

        logger.info(f"[Pipeline] Starting: {file_path.name}")

        # ── 1. Validate ───────────────────────────────────────────────────
        try:
            validate_file(file_path)
        except ValueError as e:
            return ProcessingResult(
                file_id="",
                source_path=str(file_path),
                file_name=file_path.name,
                file_type="unknown",
                processing_time_s=0.0,
                success=False,
                errors=[str(e)],
            )

        file_id = self._compute_file_id(file_path)
        mime = get_mime_type(file_path)

        # ── 2. Parse ──────────────────────────────────────────────────────
        try:
            parse_result = await self.dispatcher.dispatch(file_path, options)
        except Exception as exc:
            logger.exception(f"[Pipeline] Parse failed for {file_path.name}")
            return ProcessingResult(
                file_id=file_id,
                source_path=str(file_path),
                file_name=file_path.name,
                file_type=mime,
                processing_time_s=time.perf_counter() - start,
                success=False,
                errors=[f"Parse error: {exc}"],
            )

        # ── 3. Summarise (optional) ────────────────────────────────────────
        summary = ""
        key_insights: list[str] = []
        if options.enable_summarization and parse_result.get("text"):
            try:
                from core.summarizer import Summarizer
                summarizer = Summarizer(provider=options.summarization_provider)
                ai_out = await summarizer.summarize(parse_result["text"])
                summary = ai_out.get("summary", "")
                key_insights = ai_out.get("key_insights", [])
            except Exception as exc:
                logger.warning(f"[Pipeline] Summarization failed: {exc}")

        # ── 4. Chunk ──────────────────────────────────────────────────────
        chunks = self.chunker.chunk(
            text=parse_result.get("markdown", "") or parse_result.get("text", ""),
            strategy=options.chunk_strategy,
            chunk_size=options.chunk_size,
            chunk_overlap=options.chunk_overlap,
            metadata={
                "source": file_path.name,
                "file_id": file_id,
                "mime_type": mime,
            },
        )

        # ── 5. Embed (optional) ───────────────────────────────────────────
        embeddings: list[list[float]] = []
        if options.enable_embeddings and chunks:
            try:
                texts = [c["text"] for c in chunks]
                embeddings = await self.embedding_engine.embed_batch(
                    texts, provider=options.embedding_provider
                )
                # Attach embedding to each chunk
                for i, emb in enumerate(embeddings):
                    if i < len(chunks):
                        chunks[i]["embedding"] = emb
            except Exception as exc:
                logger.warning(f"[Pipeline] Embedding failed: {exc}")

        # ── 6. Store in vector DB (optional) ─────────────────────────────
        if options.store_in_vectordb and chunks and embeddings:
            try:
                from vectorstores.base import get_vector_store
                vs = get_vector_store(options.vector_store)
                await vs.upsert(chunks)
            except Exception as exc:
                logger.warning(f"[Pipeline] Vector store insert failed: {exc}")

        elapsed = time.perf_counter() - start
        logger.success(f"[Pipeline] Done: {file_path.name} in {elapsed:.2f}s, {len(chunks)} chunks")

        return ProcessingResult(
            file_id=file_id,
            source_path=str(file_path),
            file_name=file_path.name,
            file_type=mime,
            processing_time_s=round(elapsed, 3),
            markdown=parse_result.get("markdown", ""),
            plain_text=parse_result.get("text", ""),
            structured_json=parse_result.get("structured", {}),
            tables=parse_result.get("tables", []),
            images=parse_result.get("images", []),
            metadata=parse_result.get("metadata", {}),
            hyperlinks=parse_result.get("hyperlinks", []),
            summary=summary,
            key_insights=key_insights,
            chunks=chunks,
            embeddings=embeddings,
            success=True,
        )

    async def process_batch(
        self,
        paths: list[Path | str],
        options: Optional[ProcessingOptions] = None,
        max_concurrency: int = 4,
    ) -> list[ProcessingResult]:
        """Process multiple files with bounded concurrency."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _process_one(p: Path | str) -> ProcessingResult:
            async with semaphore:
                return await self.process(p, options)

        tasks = [_process_one(p) for p in paths]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return list(results)

    async def process_folder(
        self,
        folder: Path | str,
        options: Optional[ProcessingOptions] = None,
        recursive: bool = True,
        max_concurrency: int = 4,
    ) -> list[ProcessingResult]:
        """Recursively scan and process all supported files in a folder."""
        folder = Path(folder)
        pattern = "**/*" if recursive else "*"
        files = [
            p for p in folder.glob(pattern)
            if p.is_file()
            and p.suffix.lstrip(".").lower() in settings.allowed_extensions_set
        ]
        logger.info(f"[Pipeline] Found {len(files)} files in {folder}")
        return await self.process_batch(files, options, max_concurrency)
