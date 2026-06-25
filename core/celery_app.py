"""
DocFlow — Celery Task Queue
Background processing tasks for async batch ingestion.
"""
from __future__ import annotations

from celery import Celery
from loguru import logger

from core.config import settings

celery_app = Celery(
    "docflow",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["core.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=600,   # 10 minutes soft limit
    task_time_limit=900,        # 15 minutes hard limit
    result_expires=3600,        # Results expire after 1 hour
)
