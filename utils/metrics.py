"""
DocFlow — Prometheus Metrics
Custom application metrics for monitoring document processing performance.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest

# ── Counters ──────────────────────────────────────────────────────────────────
documents_processed = Counter(
    "docflow_documents_processed_total",
    "Total number of documents processed",
    ["file_type", "status"],  # labels
)

ocr_calls = Counter(
    "docflow_ocr_calls_total",
    "Total OCR engine invocations",
    ["engine"],
)

embedding_calls = Counter(
    "docflow_embedding_calls_total",
    "Total embedding generation calls",
    ["provider"],
)

api_requests = Counter(
    "docflow_api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status_code"],
)

# ── Histograms ─────────────────────────────────────────────────────────────────
processing_duration = Histogram(
    "docflow_processing_duration_seconds",
    "Document processing time in seconds",
    ["file_type"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

chunk_count_per_doc = Histogram(
    "docflow_chunks_per_document",
    "Number of chunks produced per document",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
)

# ── Gauges ────────────────────────────────────────────────────────────────────
active_processing_jobs = Gauge(
    "docflow_active_processing_jobs",
    "Number of documents currently being processed",
)

vector_store_size = Gauge(
    "docflow_vector_store_chunk_count",
    "Total number of chunks in the vector store",
    ["store"],
)

# ── Summary ───────────────────────────────────────────────────────────────────
embedding_latency = Summary(
    "docflow_embedding_latency_seconds",
    "Embedding generation latency",
    ["provider"],
)


def record_processing(file_type: str, duration_s: float, chunk_count: int, success: bool):
    """Record metrics for a completed processing job."""
    status = "success" if success else "error"
    documents_processed.labels(file_type=file_type, status=status).inc()
    processing_duration.labels(file_type=file_type).observe(duration_s)
    if success:
        chunk_count_per_doc.observe(chunk_count)


def get_metrics() -> bytes:
    """Return Prometheus metrics as bytes for /metrics endpoint."""
    return generate_latest()
