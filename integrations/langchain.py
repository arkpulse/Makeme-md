"""
DocFlow — LangChain Integration
Custom LangChain-compatible document loaders and retrievers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, List

from loguru import logger


class DocFlowLoader:
    """
    LangChain-compatible document loader.
    Loads one or more files/folders through the DocFlow pipeline.

    Usage:
        loader = DocFlowLoader("./docs/")
        docs = loader.load()

        # Or in a RAG chain:
        retriever = DocFlowRetriever(top_k=5)
    """

    def __init__(
        self,
        path: str | Path,
        recursive: bool = True,
        enable_ocr: bool = True,
        chunk_strategy: str = "recursive",
    ):
        self.path = Path(path)
        self.recursive = recursive
        self.enable_ocr = enable_ocr
        self.chunk_strategy = chunk_strategy

    def load(self) -> List[Any]:
        """Synchronously load documents (for LangChain compatibility)."""
        import asyncio
        return asyncio.run(self.aload())

    async def aload(self) -> List[Any]:
        """Asynchronously load documents."""
        try:
            from langchain_core.documents import Document
        except ImportError:
            from langchain.schema import Document

        from core.pipeline import DocFlowPipeline, ProcessingOptions

        pipeline = DocFlowPipeline()
        opts = ProcessingOptions(
            enable_ocr=self.enable_ocr,
            chunk_strategy=self.chunk_strategy,
            enable_embeddings=False,  # Let LangChain handle embeddings
        )

        if self.path.is_file():
            result = await pipeline.process(self.path, opts)
            results = [result]
        else:
            results = await pipeline.process_folder(self.path, opts, recursive=self.recursive)

        docs = []
        for result in results:
            if not result.success:
                logger.warning(f"[DocFlowLoader] Failed: {result.file_name}")
                continue
            for chunk in result.chunks:
                docs.append(Document(
                    page_content=chunk["text"],
                    metadata={
                        "source": result.file_name,
                        "file_id": result.file_id,
                        "chunk_id": chunk["chunk_id"],
                        "chunk_index": chunk["chunk_index"],
                        "token_count": chunk.get("token_count", 0),
                        "section_title": chunk.get("section_title", ""),
                    },
                ))

        logger.info(f"[DocFlowLoader] Loaded {len(docs)} chunks from {len(results)} files")
        return docs

    def lazy_load(self) -> Iterator[Any]:
        """Iterator version for memory-efficient loading."""
        for doc in self.load():
            yield doc


class DocFlowRetriever:
    """
    LangChain-compatible retriever backed by DocFlow's vector store.

    Usage:
        retriever = DocFlowRetriever(top_k=5)
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    """

    def __init__(self, top_k: int = 5, vector_store: str = "chroma", filters: dict | None = None):
        self.top_k = top_k
        self.vector_store_name = vector_store
        self.filters = filters

    def get_relevant_documents(self, query: str) -> List[Any]:
        import asyncio
        return asyncio.run(self.aget_relevant_documents(query))

    async def aget_relevant_documents(self, query: str) -> List[Any]:
        try:
            from langchain_core.documents import Document
        except ImportError:
            from langchain.schema import Document

        from embeddings.engine import EmbeddingEngine
        from vectorstores.base import get_vector_store

        engine = EmbeddingEngine()
        store = get_vector_store(self.vector_store_name)

        embedding = await engine.embed(query)
        results = await store.search(embedding, top_k=self.top_k, filters=self.filters)

        return [
            Document(
                page_content=r["text"],
                metadata={k: v for k, v in r.items() if k != "text"},
            )
            for r in results
        ]

    # LangChain v0.2+ interface
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Any]:
        return self.get_relevant_documents(query)
