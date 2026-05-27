"""FastAPI entrypoint.

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
import os
from contextlib import asynccontextmanager

# LangChain 1.x may import transformers -> torch while loading API routers on
# Windows. Preload torch before those transitive imports to avoid DLL order bugs.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch  # noqa: F401
except Exception:
    pass

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_v1_router
from app.core.config import settings
from app.core.logging import logger, setup_logging
from app.services.kafka_service import get_kafka_producer
from app.services.minio_service import get_minio_service
from app.services.vectorstore_service import ensure_indices as ensure_es_indices


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info(
        "Starting {name} (env={env}) on {host}:{port}",
        name=settings.app_name,
        env=settings.app_env,
        host=settings.app_host,
        port=settings.app_port,
    )

    # Ensure the MinIO bucket exists once at startup so the first upload is fast.
    try:
        await run_in_threadpool(get_minio_service().ensure_bucket)
    except Exception as exc:
        # Don't crash the API if MinIO is briefly unavailable; bucket-ensure is
        # idempotent and will be retried on the first upload.
        logger.warning("MinIO bucket-ensure at startup failed: {}", exc)

    # Ensure ES parent + child indices exist with the right metadata mapping.
    try:
        await run_in_threadpool(ensure_es_indices)
    except Exception as exc:
        logger.warning("ES index-ensure at startup failed: {}", exc)

    # Start the Kafka producer; upload will 503 until Kafka is available.
    try:
        await get_kafka_producer().start()
    except Exception as exc:
        logger.warning("Kafka producer start failed (upload will be unavailable): {}", exc)

    try:
        yield
    finally:
        await get_kafka_producer().stop()
        logger.info("Shutting down {name}", name=settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="RAG-based scientific-paper Q&A backend.",
    lifespan=lifespan,
)

# Allow browser access from the Vite dev server / preview; adjust origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_v1_router)


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok", "app": settings.app_name, "env": settings.app_env}
