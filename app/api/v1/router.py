"""Aggregate v1 routers."""
from fastapi import APIRouter

from app.api.mcp_v2_api import router as mcp_router
from app.api.v1 import agent_routes, sessions, task_routes as tasks, upload

papers_router = APIRouter(prefix="/papers", tags=["papers"])
papers_router.include_router(upload.router)
papers_router.include_router(tasks.router)

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(papers_router)
api_v1_router.include_router(sessions.router)
api_v1_router.include_router(agent_routes.agent_router)
api_v1_router.include_router(mcp_router)
