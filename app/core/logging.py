"""Loguru-based logger setup. Call `setup_logging()` once at process start.

Outputs:
  - stdout: colourised, compact format for development
  - logs/papermind_{date}.log: structured JSON-like format, daily rotation
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from app.core.config import settings


def setup_logging() -> None:
    logger.remove()

    # --- stdout sink (development) ---
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        backtrace=False,
        diagnose=False,
        enqueue=True,
    )

    # --- daily file sink (persistent audit trail) ---
    log_dir = Path(settings.log_dir)
    if not log_dir.is_absolute():
        # Resolve relative to repo root (two levels up from app/core/)
        log_dir = Path(__file__).resolve().parents[2] / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_dir / "papermind_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        rotation="00:00",
        retention=f"{settings.log_retention_days} days",
        compression="gz",
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )
    logger.info("File logging enabled: {}", log_dir.resolve())


__all__ = ["logger", "setup_logging"]
