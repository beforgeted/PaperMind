"""AgentRun 与外部计算任务接口。"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.runtime_schema import (
    AgentRunListResponse,
    AgentRunResumeRequest,
    AgentRunResumeResponse,
    ComputeTaskCompleteRequest,
    ComputeTaskFailRequest,
    ComputeTaskResponse,
)
from app.services.orchestrator_service import orchestrator
from app.services.chat_workflow_service import chat_workflow_service


router = APIRouter(prefix="/api/v1", tags=["工作流运行"])


@router.get("/agent/runs", response_model=AgentRunListResponse)
async def list_agent_runs(limit: int = Query(default=50, ge=1, le=200)) -> AgentRunListResponse:
    runs = await orchestrator.agent_runs.list_runs(limit=limit)
    return AgentRunListResponse(runs=runs)


@router.get("/agent/runs/{run_id}")
async def get_agent_run(run_id: str) -> dict[str, Any]:
    run = await orchestrator.agent_runs.get_run(run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AgentRun 不存在")
    return {"run": run}


@router.post("/agent/runs/{run_id}/resume", response_model=AgentRunResumeResponse)
async def resume_agent_run(run_id: str, request: AgentRunResumeRequest) -> AgentRunResumeResponse:
    run = await orchestrator.agent_runs.get_run(run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AgentRun 不存在")

    task_id = request.task_id or run.pending_task_id
    if not task_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="当前运行没有等待中的计算任务")

    try:
        resumed = await chat_workflow_service.resume_from_task(task_id=task_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return AgentRunResumeResponse(run=resumed)


@router.get("/compute/tasks")
async def list_compute_tasks(
    run_id: Optional[str] = None,
    task_status: Optional[str] = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    tasks = await orchestrator.compute_tasks.list_tasks(run_id=run_id, status=task_status, limit=limit)
    return {"tasks": tasks}


@router.get("/compute/tasks/{task_id}", response_model=ComputeTaskResponse)
async def get_compute_task(task_id: str) -> ComputeTaskResponse:
    task = await orchestrator.compute_tasks.get_task(task_id)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="计算任务不存在")
    return ComputeTaskResponse(task=task)


@router.post("/compute/tasks/{task_id}/complete", response_model=AgentRunResumeResponse)
async def complete_compute_task(
    task_id: str,
    request: ComputeTaskCompleteRequest,
) -> AgentRunResumeResponse:
    task = await orchestrator.compute_tasks.complete_task(task_id=task_id, result=request.result)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="计算任务不存在")

    try:
        run = await chat_workflow_service.resume_from_task(task_id=task_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return AgentRunResumeResponse(run=run)


@router.post("/compute/tasks/{task_id}/fail", response_model=ComputeTaskResponse)
async def fail_compute_task(task_id: str, request: ComputeTaskFailRequest) -> ComputeTaskResponse:
    task = await orchestrator.compute_tasks.fail_task(task_id=task_id, error=request.error)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="计算任务不存在")
    await orchestrator.agent_runs.fail_run(run_id=task.run_id, error=request.error)
    return ComputeTaskResponse(task=task)
