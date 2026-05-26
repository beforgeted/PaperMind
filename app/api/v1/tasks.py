"""`/api/v1/papers` task status and deletion endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app.core.logging import logger
from app.core.schemas import DeleteTaskResult, DeleteTasksRequest, DeleteTasksResponse, TaskRecord
from app.services.minio_service import get_minio_service
from app.services.task_status import delete_task, get_task, list_tasks
from app.services.vectorstore_service import delete_by_task

router = APIRouter()


@router.get(
    "/{task_id}",
    response_model=TaskRecord,
    summary="Get the parsing/indexing status of a task.",
)
async def get_task_status(task_id: str) -> TaskRecord:
    record = get_task(task_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Task {task_id} not found.")
    return record


@router.get(
    "",
    response_model=list[TaskRecord],
    summary="List recent tasks (most-recently-updated first).",
)
async def list_recent_tasks(limit: int = 50) -> list[TaskRecord]:
    return list_tasks(limit=limit)


async def _delete_one_task(task_id: str) -> DeleteTaskResult:
    record = get_task(task_id)
    if record is None:
        return DeleteTaskResult(
            task_id=task_id,
            deleted=False,
            message="Task not found.",
        )

    object_names = [record.object_name]
    if record.parsed_object_name:
        object_names.append(record.parsed_object_name)

    try:
        await run_in_threadpool(delete_by_task, task_id)
        await run_in_threadpool(get_minio_service().remove_objects, object_names)
        deleted_record = await run_in_threadpool(delete_task, task_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Delete task {} failed: {}", task_id, exc)
        return DeleteTaskResult(
            task_id=task_id,
            deleted=False,
            message=str(exc),
        )

    return DeleteTaskResult(
        task_id=task_id,
        deleted=deleted_record,
        message="Deleted.",
    )


@router.delete(
    "/batch",
    response_model=DeleteTasksResponse,
    summary="Delete multiple paper tasks and their stored artifacts.",
)
async def delete_paper_tasks(request: DeleteTasksRequest) -> DeleteTasksResponse:
    seen: set[str] = set()
    task_ids = []
    for task_id in request.task_ids:
        if task_id and task_id not in seen:
            seen.add(task_id)
            task_ids.append(task_id)

    results = [await _delete_one_task(task_id) for task_id in task_ids]
    return DeleteTasksResponse(results=results)


@router.delete(
    "/{task_id}",
    response_model=DeleteTaskResult,
    summary="Delete one paper task and its stored artifacts.",
)
async def delete_paper_task(task_id: str) -> DeleteTaskResult:
    result = await _delete_one_task(task_id)
    if not result.deleted and result.message == "Task not found.":
        raise HTTPException(status.HTTP_404_NOT_FOUND, result.message)
    return result
