"""Aggregate v1 routers."""
from fastapi import APIRouter

from app.agent.api import router as agent_router
from app.api.v1 import sessions, tasks, upload

papers_router = APIRouter(prefix="/papers", tags=["papers"])
papers_router.include_router(upload.router)
papers_router.include_router(tasks.router)

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(papers_router)
api_v1_router.include_router(agent_router)
api_v1_router.include_router(sessions.router)
