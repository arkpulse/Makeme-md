"""
DocFlow — Locust Load Tests
Simulates concurrent document processing and search API load.

Usage:
    locust -f tests/locustfile.py --host=http://localhost:8000 --users=50 --spawn-rate=5
"""
from __future__ import annotations

import io
import random
import string

from locust import HttpUser, between, task


def _random_text(words: int = 200) -> str:
    """Generate random text content for upload simulation."""
    vocab = ["document", "processing", "machine", "learning", "text",
             "analysis", "embedding", "vector", "semantic", "chunk",
             "extraction", "intelligence", "pipeline", "inference"]
    return " ".join(random.choices(vocab, k=words))


class DocFlowUser(HttpUser):
    """Simulates a typical DocFlow API user."""
    wait_time = between(1, 3)

    def on_start(self):
        """Verify API is healthy before starting."""
        resp = self.client.get("/api/v1/health")
        if resp.status_code != 200:
            self.environment.runner.quit()

    @task(3)
    def health_check(self):
        """Lightweight health probe — most frequent."""
        self.client.get("/api/v1/health", name="/api/v1/health")

    @task(5)
    def chunk_text(self):
        """Chunk a random text document."""
        payload = {
            "text": _random_text(300),
            "strategy": random.choice(["recursive", "heading", "token"]),
            "chunk_size": random.choice([256, 512, 1024]),
            "chunk_overlap": 64,
        }
        self.client.post("/api/v1/chunk", json=payload, name="/api/v1/chunk")

    @task(2)
    def upload_text_file(self):
        """Upload a small text file."""
        content = _random_text(500).encode()
        filename = f"test_{''.join(random.choices(string.ascii_lowercase, k=6))}.txt"
        self.client.post(
            "/api/v1/upload",
            files={"file": (filename, io.BytesIO(content), "text/plain")},
            name="/api/v1/upload",
        )

    @task(2)
    def convert_text_file(self):
        """Full convert pipeline with a text file."""
        content = _random_text(400).encode()
        self.client.post(
            "/api/v1/convert",
            files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
            data={
                "enable_ocr": "false",
                "enable_summarization": "false",
                "enable_embeddings": "false",
                "chunk_strategy": "recursive",
                "chunk_size": "512",
            },
            name="/api/v1/convert",
        )

    @task(1)
    def semantic_search(self):
        """Semantic search query."""
        queries = [
            "machine learning concepts",
            "document processing pipeline",
            "vector embeddings semantic search",
            "text extraction OCR",
        ]
        payload = {
            "query": random.choice(queries),
            "top_k": 5,
            "vector_store": "chroma",
        }
        self.client.post("/api/v1/search", json=payload, name="/api/v1/search")


class HeavyUser(DocFlowUser):
    """Simulates a power user with larger payloads."""
    wait_time = between(3, 8)

    @task(1)
    def chunk_large_text(self):
        payload = {
            "text": _random_text(2000),
            "strategy": "semantic",
            "chunk_size": 512,
        }
        self.client.post("/api/v1/chunk", json=payload, name="/api/v1/chunk [large]")
