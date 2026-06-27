"""DocFlow — Chunk Route"""
from fastapi import APIRouter
from pydantic import BaseModel
from rag.chunker import Chunker

router = APIRouter()
_chunker = Chunker()


class ChunkRequest(BaseModel):
    text: str
    strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64


@router.post("/chunk")
async def chunk_text(req: ChunkRequest):
    chunks = _chunker.chunk(
        req.text,
        strategy=req.strategy,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
    )
    return {"chunk_count": len(chunks), "chunks": chunks}
