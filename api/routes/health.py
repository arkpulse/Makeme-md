"""
DocFlow — Health Check Route
"""
from fastapi import APIRouter
from pydantic import BaseModel
from core.config import settings

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        environment=settings.app_env,
    )
