"""DocFlow — Embeddings Route"""
from fastapi import APIRouter
from pydantic import BaseModel
from embeddings.engine import EmbeddingEngine

router = APIRouter()
_engine = EmbeddingEngine()


class EmbedRequest(BaseModel):
    texts: list[str]
    provider: str = "sentence_transformers"


@router.post("/embed")
async def embed_texts(req: EmbedRequest):
    embeddings = await _engine.embed_batch(req.texts, provider=req.provider)
    return {"count": len(embeddings), "dimension": len(embeddings[0]) if embeddings else 0, "embeddings": embeddings}
