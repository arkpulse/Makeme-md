"""DocFlow — Semantic Search Route"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from embeddings.engine import EmbeddingEngine
from vectorstores.base import get_vector_store

router = APIRouter()
_embedding_engine = EmbeddingEngine()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: dict | None = None
    vector_store: str = "chroma"


@router.post("/search")
async def semantic_search(req: SearchRequest):
    """Perform semantic similarity search over indexed documents."""
    query_embedding = await _embedding_engine.embed(req.query)
    store = get_vector_store(req.vector_store)
    results = await store.search(query_embedding, top_k=req.top_k, filters=req.filters)
    return {"query": req.query, "results": results, "count": len(results)}
