"""Compute task service — MySQL or in-memory fallback."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.core.config import settings
from app.schemas.runtime_schema import ComputeTaskRecord, TaskRequest, TaskResult
from app.services.run_service import utc_now_iso


def _json_dump(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False)


def _json_load(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


class TaskService:
    async def run_task(self, request: TaskRequest) -> TaskResult:
        return TaskResult(status="pending", content=request.query)


task_service = TaskService()


class ComputeTaskService:
    """External compute task persistence — MySQL primary, in-memory fallback."""

    def __init__(self):
        self._tasks: Dict[str, ComputeTaskRecord] = {}

    @property
    def _use_mysql(self) -> bool:
        return bool(settings.mysql_enabled)

    async def initialize_storage(self) -> None:
        if self._use_mysql:
            from app.services.mysql_service import mysql_service
            await mysql_service.execute(
                """
                CREATE TABLE IF NOT EXISTS compute_tasks (
                    task_id VARCHAR(80) PRIMARY KEY,
                    run_id VARCHAR(80) NOT NULL,
                    step_id VARCHAR(80) NOT NULL,
                    agent_id VARCHAR(80) NULL,
                    kind VARCHAR(80) NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    input JSON NULL,
                    result JSON NULL,
                    error TEXT NULL,
                    created_at VARCHAR(40) NOT NULL,
                    updated_at VARCHAR(40) NOT NULL,
                    KEY idx_compute_tasks_run (run_id),
                    KEY idx_compute_tasks_status_updated (status, updated_at),
                    KEY idx_compute_tasks_step (step_id),
                    CONSTRAINT fk_compute_tasks_run
                        FOREIGN KEY (run_id) REFERENCES agent_runs (run_id)
                        ON DELETE CASCADE,
                    CONSTRAINT fk_compute_tasks_step
                        FOREIGN KEY (step_id) REFERENCES workflow_steps (step_id)
                        ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )

    async def create_task(
        self,
        *,
        task_id: Optional[str] = None,
        run_id: str,
        step_id: str,
        agent_id: Optional[str],
        kind: str,
        input_data: Dict[str, Any],
    ) -> ComputeTaskRecord:
        now = utc_now_iso()
        task = ComputeTaskRecord(
            task_id=task_id or f"task_{uuid4().hex}",
            run_id=run_id,
            step_id=step_id,
            agent_id=agent_id,
            kind=kind,
            status="pending",
            input=dict(input_data),
            created_at=now,
            updated_at=now,
        )
        if self._use_mysql:
            from app.services.mysql_service import mysql_service
            await mysql_service.execute(
                """
                INSERT INTO compute_tasks (
                    task_id, run_id, step_id, agent_id, kind, status,
                    input, result, error, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    task.task_id, task.run_id, task.step_id, task.agent_id,
                    task.kind, task.status, _json_dump(task.input),
                    _json_dump(task.result), task.error,
                    task.created_at, task.updated_at,
                ),
            )
        else:
            self._tasks[task.task_id] = task
        return task

    async def get_task(self, task_id: str) -> Optional[ComputeTaskRecord]:
        if self._use_mysql:
            from app.services.mysql_service import mysql_service
            row = await mysql_service.fetchone(
                "SELECT * FROM compute_tasks WHERE task_id = %s", (task_id,)
            )
            return self._row_to_task(row) if row else None
        return self._tasks.get(task_id)

    async def list_tasks(
        self,
        *,
        run_id: Optional[str] = None,
        status: Optional[str] = None,
        kind: Optional[str] = None,
        limit: int = 50,
    ) -> List[ComputeTaskRecord]:
        if self._use_mysql:
            from app.services.mysql_service import mysql_service
            where: List[str] = []
            params: List[Any] = []
            if run_id:
                where.append("run_id = %s")
                params.append(run_id)
            if status:
                where.append("status = %s")
                params.append(status)
            if kind:
                where.append("kind = %s")
                params.append(kind)
            where_sql = f"WHERE {' AND '.join(where)}" if where else ""
            params.append(max(1, limit))
            rows = await mysql_service.fetchall(
                f"SELECT * FROM compute_tasks {where_sql} ORDER BY created_at DESC LIMIT %s",
                tuple(params),
            )
            return [self._row_to_task(row) for row in rows]

        # In-memory: filter from dict
        results = list(self._tasks.values())
        if run_id:
            results = [t for t in results if t.run_id == run_id]
        if status:
            results = [t for t in results if t.status == status]
        if kind:
            results = [t for t in results if t.kind == kind]
        results.sort(key=lambda t: t.created_at or "", reverse=True)
        return results[:max(1, limit)]

    async def mark_running(self, task_id: str) -> Optional[ComputeTaskRecord]:
        return await self._mutate_task(task_id, status="running")

    async def update_task(
        self,
        *,
        task_id: str,
        status: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Optional[ComputeTaskRecord]:
        changes: Dict[str, Any] = {}
        if status is not None:
            changes["status"] = status
        if input_data is not None:
            changes["input"] = dict(input_data)
        if result is not None:
            changes["result"] = dict(result)
        if error is not None:
            changes["error"] = error
        if not changes:
            return await self.get_task(task_id)
        return await self._mutate_task(task_id, **changes)

    async def complete_task(self, *, task_id: str, result: Dict[str, Any]) -> Optional[ComputeTaskRecord]:
        return await self._mutate_task(task_id, status="succeeded", result=dict(result), error="")

    async def fail_task(self, *, task_id: str, error: str) -> Optional[ComputeTaskRecord]:
        return await self._mutate_task(task_id, status="failed", error=error)

    async def _mutate_task(self, task_id: str, **changes: Any) -> Optional[ComputeTaskRecord]:
        if self._use_mysql:
            task = await self.get_task(task_id)
            if not task:
                return None
            for key, value in changes.items():
                setattr(task, key, value)
            task.updated_at = utc_now_iso()
            from app.services.mysql_service import mysql_service
            await mysql_service.execute(
                """
                UPDATE compute_tasks
                SET status = %s, input = %s, result = %s, error = %s, updated_at = %s
                WHERE task_id = %s
                """,
                (
                    task.status, _json_dump(task.input), _json_dump(task.result),
                    task.error, task.updated_at, task.task_id,
                ),
            )
            saved = await self.get_task(task_id)
            return saved or task

        # In-memory
        task = self._tasks.get(task_id)
        if not task:
            return None
        for key, value in changes.items():
            setattr(task, key, value)
        task.updated_at = utc_now_iso()
        return task

    def _row_to_task(self, row: Dict[str, Any]) -> ComputeTaskRecord:
        return ComputeTaskRecord(
            task_id=str(row["task_id"]),
            run_id=str(row["run_id"]),
            step_id=str(row["step_id"]),
            agent_id=row.get("agent_id"),
            kind=str(row.get("kind") or "external_compute"),
            status=str(row.get("status") or "pending"),
            input=dict(_json_load(row.get("input"), {})),
            result=dict(_json_load(row.get("result"), {})),
            error=str(row.get("error") or ""),
            created_at=str(row.get("created_at") or ""),
            updated_at=str(row.get("updated_at") or ""),
        )


compute_task_service = ComputeTaskService()
