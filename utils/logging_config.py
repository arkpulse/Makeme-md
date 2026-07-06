"""
DocFlow — Structured Logging Configuration
Sets up Loguru with JSON-structured output, file rotation, and Sentry integration.
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from core.config import settings

LOG_DIR = Path("./logs")


def setup_logging() -> None:
    """Configure Loguru for structured, rotated logging."""
    LOG_DIR.mkdir(exist_ok=True)

    # Remove default handler
    logger.remove()

    # ── Console handler ───────────────────────────────────────────────────────
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        format=log_format,
        level=settings.log_level,
        colorize=True,
        backtrace=True,
        diagnose=settings.debug,
    )

    # ── File handler (JSON, rotated daily, 7-day retention) ───────────────────
    logger.add(
        LOG_DIR / "docflow_{time:YYYY-MM-DD}.log",
        format="{time} | {level} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="00:00",
        retention="7 days",
        compression="zip",
        serialize=True,  # JSON output
        backtrace=True,
        diagnose=False,  # Don't expose variable values in production logs
    )

    # ── Error-only file ───────────────────────────────────────────────────────
    logger.add(
        LOG_DIR / "errors.log",
        format="{time} | {level} | {name}:{function}:{line} | {message}\n{exception}",
        level="ERROR",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True,
    )

    # ── Sentry integration (optional) ─────────────────────────────────────────
    if settings.sentry_dsn:
        try:
            import sentry_sdk
            sentry_sdk.init(
                dsn=settings.sentry_dsn,
                environment=settings.app_env,
                traces_sample_rate=0.1,
                profiles_sample_rate=0.1,
            )
            logger.info("[Logging] Sentry integration enabled")
        except ImportError:
            logger.warning("[Logging] sentry-sdk not installed, Sentry disabled")

    logger.info(f"[Logging] DocFlow {settings.app_version} logger initialised ({settings.log_level})")
