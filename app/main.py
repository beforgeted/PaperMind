"""PaperMind FastAPI entrypoint.

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 2222
"""
import os

# Preload torch before langchain_core → transformers → torch DLL chain
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch  # noqa: F401
except Exception:
    pass

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging
from app.middleware.request_logging import RequestLoggingMiddleware
from app.api.v1.router import api_v1_router
from app.services.storage.kafka import get_kafka_producer
from app.services.minio_service import MinIOService
from app.services.storage.es import ensure_indices as ensure_es_indices

import logging
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Starting PaperMind on %s:%s", settings.app_host, settings.app_port)

    # Ensure MinIO bucket exists
    try:
        minio_svc = MinIOService()
        await run_in_threadpool(minio_svc.ensure_bucket)
    except Exception as exc:
        logger.warning("MinIO bucket-ensure at startup failed: %s", exc)

    # Ensure ES indices exist
    try:
        await run_in_threadpool(ensure_es_indices)
    except Exception as exc:
        logger.warning("ES index-ensure at startup failed: %s", exc)

    # Start Kafka producer
    try:
        await get_kafka_producer().start()
    except Exception as exc:
        logger.warning("Kafka producer start failed: %s", exc)

    # Connect MCP servers (模式 A)
    try:
        from app.agent.tools.registry import clear_tools_cache
        from app.mcp_gateway.server_manager import mcp_server_manager

        await mcp_server_manager.refresh()
        clear_tools_cache()
        logger.info(
            "MCP servers ready, tools=%s",
            list(mcp_server_manager.list_tools().keys()),
        )
    except Exception as exc:
        logger.error("MCP server warmup failed: %s", exc)

    try:
        yield
    finally:
        await get_kafka_producer().stop()
        logger.info("PaperMind shutting down")


app = FastAPI(
    title=settings.project_name,
    description="PaperMind 学术研究多 Agent 平台",
    version="0.1.0",
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

app.include_router(api_v1_router)


@app.get("/")
async def root():
    return {
        "message": "PaperMind Platform 主服务已启动",
        "environment": settings.environment,
    }


@app.get("/health")
async def health_check():
    status = {"status": "ok", "app": settings.app_name, "env": settings.app_env}

    # Check Elasticsearch
    try:
        from app.services.storage.es import get_es_client
        es = get_es_client()
        es_ok = await asyncio.to_thread(es.ping)
        status["elasticsearch"] = "ok" if es_ok else "unreachable"
    except Exception as e:
        status["elasticsearch"] = f"error: {e}"

    # Check Redis
    try:
        from redis import Redis
        r = Redis.from_url(settings.redis_url, decode_responses=True)
        await asyncio.to_thread(r.ping)
        status["redis"] = "ok"
    except Exception as e:
        status["redis"] = f"error: {e}"

    # Check Kafka (minimal — just check producer exists)
    try:
        from app.services.storage.kafka import get_kafka_producer
        kp = get_kafka_producer()
        status["kafka"] = "ok" if kp._producer is not None else "not started"
    except Exception as e:
        status["kafka"] = f"error: {e}"

    # Check MinIO
    try:
        from app.services.minio_service import get_minio_service
        s3 = get_minio_service()
        await asyncio.to_thread(s3._client.list_buckets)
        status["minio"] = "ok"
    except Exception as e:
        status["minio"] = f"error: {e}"

    # Check MySQL (if enabled)
    if settings.mysql_enabled:
        try:
            from app.services.mysql_service import mysql_service
            await mysql_service.require_engine()
            await mysql_service.fetchone("SELECT 1")
            status["mysql"] = "ok"
        except Exception as e:
            status["mysql"] = f"error: {e}"

    return status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.app_port, reload=True)
