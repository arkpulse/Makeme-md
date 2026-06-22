"""
DocFlow — Test Suite
Unit, integration, and API tests covering core pipeline, parsers, OCR, chunker,
embedding engine, and FastAPI endpoints.
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_text() -> str:
    return """# Introduction

This is the first section of our test document.
It contains multiple paragraphs and sentences.

## Background

Machine learning has transformed how we process documents.
Natural language processing enables understanding of complex texts.

### Key Concepts

- Tokenisation splits text into meaningful units.
- Embeddings capture semantic meaning in vector space.
- Transformers enable contextual language understanding.

## Conclusion

Document AI platforms like DocFlow accelerate knowledge extraction
from unstructured data at enterprise scale.
"""


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Create a minimal valid PDF for testing."""
    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Test Document\n\nThis is a test PDF for DocFlow.")
        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue()
    except ImportError:
        pytest.skip("PyMuPDF not installed")


@pytest.fixture
def tmp_text_file(tmp_path, sample_text) -> Path:
    f = tmp_path / "test.txt"
    f.write_text(sample_text)
    return f


@pytest.fixture
def tmp_pdf_file(tmp_path, sample_pdf_bytes) -> Path:
    f = tmp_path / "test.pdf"
    f.write_bytes(sample_pdf_bytes)
    return f


# ── Chunker Tests ─────────────────────────────────────────────────────────────
class TestChunker:
    def setup_method(self):
        from rag.chunker import Chunker
        self.chunker = Chunker()

    def test_recursive_chunk_basic(self, sample_text):
        chunks = self.chunker.chunk(sample_text, strategy="recursive", chunk_size=200)
        assert len(chunks) > 0
        for c in chunks:
            assert "chunk_id" in c
            assert "text" in c
            assert "token_count" in c
            assert len(c["text"]) >= 1

    def test_heading_chunk(self, sample_text):
        chunks = self.chunker.chunk(sample_text, strategy="heading", chunk_size=500)
        assert len(chunks) > 0
        # At least some chunks should have section_title
        titles = [c.get("section_title") for c in chunks if c.get("section_title")]
        assert len(titles) > 0

    def test_empty_text_returns_empty(self):
        assert self.chunker.chunk("") == []
        assert self.chunker.chunk("   ") == []

    def test_chunk_metadata_propagation(self, sample_text):
        meta = {"source": "test.pdf", "file_id": "abc123"}
        chunks = self.chunker.chunk(sample_text, metadata=meta)
        for c in chunks:
            assert c["source"] == "test.pdf"
            assert c["file_id"] == "abc123"

    def test_chunk_ids_unique(self, sample_text):
        chunks = self.chunker.chunk(sample_text, chunk_size=100)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_min_size_filter(self):
        text = "A " * 5  # 10 chars — below min_size of 100
        from core.config import settings
        chunks = self.chunker.chunk(text, chunk_size=512)
        assert len(chunks) == 0

    def test_token_chunk(self, sample_text):
        try:
            import tiktoken
            chunks = self.chunker.chunk(sample_text, strategy="token", chunk_size=50)
            assert len(chunks) > 0
        except ImportError:
            pytest.skip("tiktoken not installed")


# ── Markdown Utilities Tests ──────────────────────────────────────────────────
class TestMarkdownUtils:
    def test_tables_to_markdown_basic(self):
        from utils.markdown_utils import tables_to_markdown
        rows = [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]]
        md = tables_to_markdown(rows)
        assert "Name" in md
        assert "Alice" in md
        assert "|" in md
        assert "---" in md

    def test_tables_to_markdown_empty(self):
        from utils.markdown_utils import tables_to_markdown
        assert tables_to_markdown([]) == ""

    def test_clean_markdown(self):
        from utils.markdown_utils import clean_markdown
        dirty = "Hello\n\n\n\n\nWorld  \n\nEnd"
        cleaned = clean_markdown(dirty)
        assert "\n\n\n" not in cleaned
        assert "Hello" in cleaned

    def test_pipe_escape(self):
        from utils.markdown_utils import tables_to_markdown
        rows = [["Col|1", "Col 2"], ["A|B", "C"]]
        md = tables_to_markdown(rows)
        assert "\\|" in md  # Pipes should be escaped


# ── File Utilities Tests ──────────────────────────────────────────────────────
class TestFileUtils:
    def test_validate_file_not_found(self, tmp_path):
        from utils.file_utils import validate_file
        with pytest.raises(ValueError, match="not found"):
            validate_file(tmp_path / "missing.pdf")

    def test_validate_file_empty(self, tmp_path):
        from utils.file_utils import validate_file
        empty = tmp_path / "empty.txt"
        empty.write_text("")
        with pytest.raises(ValueError, match="empty"):
            validate_file(empty)

    def test_get_extension(self, tmp_path):
        from utils.file_utils import get_extension
        assert get_extension(Path("document.PDF")) == "pdf"
        assert get_extension(Path("data.csv")) == "csv"

    def test_validate_file_unsupported_extension(self, tmp_path):
        from utils.file_utils import validate_file
        bad = tmp_path / "file.xyz"
        bad.write_text("content")
        with pytest.raises(ValueError, match="Unsupported"):
            validate_file(bad)


# ── Parser Tests ──────────────────────────────────────────────────────────────
class TestTextParser:
    @pytest.mark.asyncio
    async def test_parse_text_file(self, tmp_text_file):
        from parsers.text_parser import TextParser
        opts = MagicMock(enable_ocr=False, extract_images=False, extract_tables=False)
        result = await TextParser().parse(tmp_text_file, opts)
        assert "Introduction" in result["markdown"]
        assert result["text"]

    @pytest.mark.asyncio
    async def test_parse_json_file(self, tmp_path):
        from parsers.html_parser import JSONParser
        data = {"key": "value", "items": [1, 2, 3]}
        f = tmp_path / "test.json"
        f.write_text(json.dumps(data))
        opts = MagicMock()
        result = await JSONParser().parse(f, opts)
        assert "```json" in result["markdown"]
        assert result["structured"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_parse_html_file(self, tmp_path):
        try:
            from parsers.html_parser import HTMLParser
        except ImportError:
            pytest.skip("html parser deps not installed")
        html = "<html><body><h1>Title</h1><p>Content here.</p></body></html>"
        f = tmp_path / "test.html"
        f.write_text(html)
        opts = MagicMock(extract_tables=False)
        result = await HTMLParser().parse(f, opts)
        assert "Title" in result["markdown"] or "Title" in result["text"]


class TestXLSXParser:
    @pytest.mark.asyncio
    async def test_parse_csv(self, tmp_path):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        from parsers.xlsx_parser import CSVParser
        csv_content = "Name,Score\nAlice,95\nBob,87\n"
        f = tmp_path / "data.csv"
        f.write_text(csv_content)
        opts = MagicMock(extract_tables=True, extract_images=False)
        result = await CSVParser().parse(f, opts)
        assert "Alice" in result["markdown"]
        assert len(result["tables"]) > 0


# ── Pipeline Integration Tests ────────────────────────────────────────────────
class TestPipeline:
    @pytest.mark.asyncio
    async def test_process_text_file(self, tmp_text_file):
        from core.pipeline import DocFlowPipeline, ProcessingOptions
        pipeline = DocFlowPipeline()
        opts = ProcessingOptions(
            enable_ocr=False,
            enable_summarization=False,
            enable_embeddings=False,
            store_in_vectordb=False,
        )
        result = await pipeline.process(tmp_text_file, opts)
        assert result.success
        assert result.file_name == tmp_text_file.name
        assert "Introduction" in result.markdown
        assert len(result.chunks) > 0

    @pytest.mark.asyncio
    async def test_process_invalid_file(self, tmp_path):
        from core.pipeline import DocFlowPipeline, ProcessingOptions
        pipeline = DocFlowPipeline()
        bad = tmp_path / "missing.pdf"
        result = await pipeline.process(bad)
        assert not result.success
        assert result.errors

    @pytest.mark.asyncio
    async def test_batch_processing(self, tmp_path, sample_text):
        from core.pipeline import DocFlowPipeline, ProcessingOptions
        pipeline = DocFlowPipeline()
        files = []
        for i in range(3):
            f = tmp_path / f"doc{i}.txt"
            f.write_text(f"Document {i}\n\n{sample_text}")
            files.append(f)

        opts = ProcessingOptions(
            enable_ocr=False,
            enable_summarization=False,
            enable_embeddings=False,
            store_in_vectordb=False,
        )
        results = await pipeline.process_batch(files, opts)
        assert len(results) == 3
        assert all(r.success for r in results)


# ── Embedding Engine Tests ─────────────────────────────────────────────────────
class TestEmbeddingEngine:
    @pytest.mark.asyncio
    async def test_sentence_transformers_embed(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            pytest.skip("sentence-transformers not installed")
        from embeddings.engine import EmbeddingEngine
        engine = EmbeddingEngine()
        result = await engine.embed("Hello world", provider="sentence_transformers")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_embedding_cache(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            pytest.skip("sentence-transformers not installed")
        from embeddings.engine import EmbeddingEngine
        engine = EmbeddingEngine()
        e1 = await engine.embed("Cache test sentence", provider="sentence_transformers")
        e2 = await engine.embed("Cache test sentence", provider="sentence_transformers")
        assert e1 == e2  # Should be identical (cached)

    @pytest.mark.asyncio
    async def test_batch_embedding(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            pytest.skip("sentence-transformers not installed")
        from embeddings.engine import EmbeddingEngine
        engine = EmbeddingEngine()
        texts = ["First sentence", "Second sentence", "Third sentence"]
        results = await engine.embed_batch(texts)
        assert len(results) == 3
        assert all(len(e) > 0 for e in results)


# ── API Tests ─────────────────────────────────────────────────────────────────
class TestAPI:
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        from api.main import app
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_root_endpoint(self):
        from api.main import app
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_chunk_endpoint(self, sample_text):
        from api.main import app
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chunk",
                json={"text": sample_text, "strategy": "recursive", "chunk_size": 200},
            )
        assert response.status_code == 200
        data = response.json()
        assert "chunks" in data
        assert data["chunk_count"] > 0

    @pytest.mark.asyncio
    async def test_upload_endpoint_valid(self, tmp_text_file):
        from api.main import app
        async with AsyncClient(app=app, base_url="http://test") as client:
            with open(tmp_text_file, "rb") as f:
                response = await client.post(
                    "/api/v1/upload",
                    files={"file": ("test.txt", f, "text/plain")},
                )
        assert response.status_code == 200
        data = response.json()
        assert "file_id" in data

    @pytest.mark.asyncio
    async def test_upload_endpoint_invalid_type(self):
        from api.main import app
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/upload",
                files={"file": ("malware.exe", b"binary content", "application/octet-stream")},
            )
        assert response.status_code == 400


# ── Dispatcher Tests ──────────────────────────────────────────────────────────
class TestDispatcher:
    def test_resolve_pdf(self, tmp_path):
        from core.dispatcher import FormatDispatcher
        d = FormatDispatcher()
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4")
        assert d.resolve_parser_type(f) == "pdf"

    def test_resolve_docx(self, tmp_path):
        from core.dispatcher import FormatDispatcher
        d = FormatDispatcher()
        f = tmp_path / "doc.docx"
        f.write_bytes(b"dummy")
        assert d.resolve_parser_type(f) == "docx"

    def test_resolve_unknown_fallback(self, tmp_path):
        from core.dispatcher import FormatDispatcher
        d = FormatDispatcher()
        f = tmp_path / "unknown.txt"
        f.write_text("content")
        assert d.resolve_parser_type(f) == "text"
