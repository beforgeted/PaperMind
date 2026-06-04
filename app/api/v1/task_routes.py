"""/api/v1/papers" task status and deletion endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app.core.schemas import DeleteTaskResult, DeleteTasksRequest, DeleteTasksResponse, TaskRecord, TaskStatus
from app.services.minio_service import get_minio_service
from app.services.storage.redis import get_upload_state
from app.services.task_svc import delete_task, get_task, list_tasks
from app.services.storage.es import delete_by_task

logger = logging.getLogger(__name__)

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
    "/",
    response_model=list[TaskRecord],
    summary="List recent tasks (most-recently-updated first). Falls back to ES paper index if MySQL is empty.",
)
async def list_recent_tasks(limit: int = 50) -> list[TaskRecord]:
    records = list_tasks(limit=limit)
    if records:
        return records

    # MySQL empty — fall back to ES papers index for pre-existing indexed papers
    try:
        from app.core.config import settings
        from app.services.storage.es import get_es_client
        import asyncio

        es = get_es_client()
        body: dict = {"query": {"match_all": {}}, "size": limit}
        resp = await asyncio.to_thread(
            es.search,
            index=settings.es_index_papers,
            body=body,
        )
        hits = resp.get("hits", {}).get("hits", [])
        logger.info("ES papers fallback: found %s papers in %s", len(hits), settings.es_index_papers)
        return [
            TaskRecord(
                task_id=h.get("_id", ""),
                object_name=h.get("_id", ""),
                original_filename=h.get("_source", {}).get("source_file") or h.get("_source", {}).get("title") or h.get("_id", ""),
                status=TaskStatus.SUCCEEDED,
                message="已索引",
                num_parents=1,
            )
            for h in hits
        ]
    except Exception as exc:
        logger.warning("ES papers fallback failed: %s", exc)
        return []


def _cleanup_multipart_upload(upload_id: str) -> None:
    """Best-effort cleanup of Redis state + MinIO orphan chunks."""
    minio = get_minio_service()
    state = get_upload_state()

    try:
        chunk_names = state.chunk_object_names(upload_id)
        if chunk_names:
            minio.remove_objects(chunk_names)
    except Exception:
        pass

    try:
        state.cleanup(upload_id)
    except Exception:
        pass

    try:
        minio.remove_prefix(f"multipart/{upload_id}/")
    except Exception:
        pass


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

        if record.upload_id:
            await run_in_threadpool(_cleanup_multipart_upload, record.upload_id)

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
