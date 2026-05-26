"""Demo-grade task-status store.

Uses one JSON file per task under `./storage/tasks/{task_id}.json`. Atomic
writes (write-temp + os.replace) make concurrent updates safe enough for the
Demo. In production, swap this for Redis / Postgres / Elasticsearch.

Both the API (writes initial PENDING) and the worker (drives the lifecycle)
import this module — it is process-safe but not high-throughput.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from app.core.logging import logger
from app.core.schemas import TaskRecord, TaskStatus

_STORAGE_ROOT = Path("storage/tasks")
_LOCK = threading.Lock()


def _path_for(task_id: str) -> Path:
    return _STORAGE_ROOT / f"{task_id}.json"


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def create_task(record: TaskRecord) -> TaskRecord:
    """Persist the initial PENDING record. Called by the upload endpoint."""
    with _LOCK:
        record.created_at = datetime.utcnow()
        record.updated_at = record.created_at
        _atomic_write(_path_for(record.task_id), record.model_dump(mode="json"))
        logger.info("Task {} created (status={})", record.task_id, record.status)
        return record


def get_task(task_id: str) -> Optional[TaskRecord]:
    path = _path_for(task_id)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return TaskRecord.model_validate(json.load(f))


def list_tasks(limit: int = 50) -> List[TaskRecord]:
    if not _STORAGE_ROOT.exists():
        return []
    files = sorted(
        _STORAGE_ROOT.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]
    out: List[TaskRecord] = []
    for f in files:
        try:
            with f.open("r", encoding="utf-8") as fp:
                out.append(TaskRecord.model_validate(json.load(fp)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipped malformed task file {}: {}", f.name, exc)
    return out


def update_task(
    task_id: str,
    *,
    status: Optional[TaskStatus] = None,
    message: Optional[str] = None,
    error: Optional[str] = None,
    **fields,
) -> Optional[TaskRecord]:
    """Patch named fields on a task record. Missing tasks are logged + skipped."""
    with _LOCK:
        existing = get_task(task_id)
        if existing is None:
            logger.warning("update_task: task {} not found", task_id)
            return None

        if status is not None:
            existing.status = status
        if message is not None:
            existing.message = message
        if error is not None:
            existing.error = error
        for key, value in fields.items():
            if hasattr(existing, key) and value is not None:
                setattr(existing, key, value)
        existing.updated_at = datetime.utcnow()

        _atomic_write(_path_for(task_id), existing.model_dump(mode="json"))
        logger.debug("Task {} updated -> status={}", task_id, existing.status)
        return existing


def delete_task(task_id: str) -> bool:
    """Delete a persisted task record if it exists."""
    with _LOCK:
        path = _path_for(task_id)
        if not path.exists():
            return False
        path.unlink()
        logger.info("Task {} deleted", task_id)
        return True
