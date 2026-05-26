"""Aggregate v1 routers."""
from fastapi import APIRouter

from app.api.v1 import answer, query, tasks, upload

papers_router = APIRouter(prefix="/papers", tags=["papers"])
# Static paths before `/{task_id}` so FastAPI does not capture them as task IDs.
papers_router.include_router(upload.router)
papers_router.include_router(query.router)
papers_router.include_router(answer.router)
papers_router.include_router(tasks.router)

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(papers_router)
