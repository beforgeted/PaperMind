"""任务 API 路由。

端点：
    POST /api/v1/tasks/submit              提交任务，返回 task_id
    GET  /api/v1/tasks/{task_id}/status    轮询进度
    GET  /api/v1/tasks/{task_id}/result    获取结果
"""

import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.orchestrator_service import orchestrator
from app.services.task_service import compute_task_service
from app.task.storage_adapter import ComputeTaskStorageAdapter
from app.task.task_manager import TaskManager, TaskNotFinishedError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tasks", tags=["异步任务"])

task_manager = TaskManager(
    storage=ComputeTaskStorageAdapter(
        compute_task_service,
        kind="generic_compute",
    ),
    runner_registry={},
)


def _load_runner_registry():
    """延迟加载 Runner，避免可选依赖缺失时影响主应用启动。"""
    try:
        from app.task.runners import RUNNER_REGISTRY
    except ModuleNotFoundError as exc:
        logger.exception("任务依赖缺失: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"任务依赖缺失: {exc.name}",
        ) from exc
    return RUNNER_REGISTRY


async def _create_standalone_task_manager(runner_registry) -> TaskManager:
    """为独立任务创建归属 AgentRun/WorkflowStep，并返回 TaskManager。"""
    run = await orchestrator.create_agent_run(
        query="异步任务",
        context={"source": "task_api", "workflow": "standalone_task"},
    )
    await orchestrator.agent_runs.set_decision(
        run_id=run.run_id,
        decision={
            "action": "dispatch",
            "target_agents": ["task-manager"],
            "reason": "独立异步任务",
        },
        selected_agents=["task-manager"],
        agents=[
            {
                "agent_id": "task-manager",
                "name": "异步任务执行器",
                "description": "执行独立异步任务",
            }
        ],
    )
    refreshed = await orchestrator.agent_runs.get_run(run.run_id)
    if not refreshed or not refreshed.steps:
        raise RuntimeError("无法创建独立任务运行记录")
    step = next((item for item in refreshed.steps if item.agent_id == "task-manager"), refreshed.steps[0])
    return TaskManager(
        storage=ComputeTaskStorageAdapter(
            compute_task_service,
            run_id=run.run_id,
            step_id=step.step_id,
            agent_id="task-manager",
            kind="generic_compute",
        ),
        runner_registry=runner_registry,
    )


# ================================================================
# 请求/响应模型
# ================================================================

class SubmitRequest(BaseModel):
    params: dict = Field(default_factory=dict, description="任务参数")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "params": {"key": "value"},
            }]
        }
    }


class SubmitResponse(BaseModel):
    task_id: str
    status: str
    message: str

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "task_id": "a1b2c3d4e5f6",
                "status": "running",
                "message": "Poll /api/v1/tasks/a1b2c3d4e5f6/status for progress.",
            }]
        }
    }


# ================================================================
# 路由
# ================================================================

@router.post("/submit", response_model=SubmitResponse, summary="提交任务")
async def submit_task(request: SubmitRequest):
    """提交异步任务，立即返回 task_id。"""
    manager = await _create_standalone_task_manager(_load_runner_registry())
    task = await manager.submit(
        tool_name="generic_compute",
        params=request.params,
    )
    return SubmitResponse(
        task_id=task.task_id,
        status=task.status.value,
        message=f"Poll /api/v1/tasks/{task.task_id}/status for progress.",
    )


@router.get("/{task_id}/status", summary="查询任务状态")
async def get_task_status(task_id: str):
    """轮询任务的当前状态、进度和当前阶段。"""
    task = await task_manager.get_status(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return task.status_dict()


@router.get("/{task_id}/result", summary="获取任务结果")
async def get_task_result(task_id: str):
    """获取终态任务的结果。"""
    try:
        task = await task_manager.get_result(task_id)
    except TaskNotFinishedError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if task is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return task.result_dict()


@router.get("/", summary="列出最近任务")
async def list_tasks(skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
    """列出所有任务（最新的在前）。"""
    tasks = await task_manager.list_tasks(skip=skip, limit=limit)
    return {
        "total": len(tasks),
        "items": [t.status_dict() for t in tasks],
    }
