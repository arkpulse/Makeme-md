"""
DocFlow — Celery Tasks
Async background tasks for document processing, batch ingestion, and re-indexing.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger

from core.celery_app import celery_app


def _run(coro):
    """Run async coroutine in Celery's sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


@celery_app.task(bind=True, name="docflow.process_file", max_retries=3)
def process_file_task(self, file_path: str, options_dict: dict | None = None):
    """
    Background task: process a single file through the full pipeline.
    Returns serialisable ProcessingResult dict.
    """
    from core.pipeline import DocFlowPipeline, ProcessingOptions

    try:
        pipeline = DocFlowPipeline()
        opts = ProcessingOptions(**(options_dict or {}))
        result = _run(pipeline.process(Path(file_path), opts))
        return result.to_dict()
    except Exception as exc:
        logger.exception(f"[Task] process_file_task failed for {file_path}")
        raise self.retry(exc=exc, countdown=30)


@celery_app.task(bind=True, name="docflow.process_folder", max_retries=2)
def process_folder_task(self, folder_path: str, options_dict: dict | None = None, recursive: bool = True):
    """
    Background task: process all documents in a folder.
    Returns list of result dicts.
    """
    from core.pipeline import DocFlowPipeline, ProcessingOptions

    try:
        pipeline = DocFlowPipeline()
        opts = ProcessingOptions(**(options_dict or {}))
        results = _run(pipeline.process_folder(Path(folder_path), opts, recursive=recursive))
        return [r.to_dict() for r in results]
    except Exception as exc:
        logger.exception(f"[Task] process_folder_task failed for {folder_path}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(name="docflow.reindex")
def reindex_task(vector_store: str = "chroma"):
    """Re-index all stored chunks into the vector store."""
    logger.info(f"[Task] Starting full reindex on {vector_store}")
    # Implementation: query all chunks from DB, regenerate embeddings, upsert
    return {"status": "reindex_started", "vector_store": vector_store}
