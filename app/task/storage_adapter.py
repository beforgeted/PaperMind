"""存储适配器：桥接 TaskManager 的 TaskInfo 与现有 ComputeTaskService（MySQL）。

提供两种存储后端：
    - InMemoryTaskStorage: 内存存储，仅用于单元测试和原型验证
    - ComputeTaskStorageAdapter: 对接已有的 ComputeTaskService，将 TaskInfo
      映射到 compute_tasks 表。适用于独立异步任务和工作流步骤任务。

用法：
    from app.task.storage_adapter import ComputeTaskStorageAdapter
    from app.services.task_service import compute_task_service

    storage = ComputeTaskStorageAdapter(
        compute_task_service,
        run_id="run_xxx",
        step_id="step_yyy",
    )
    mgr = TaskManager(storage=storage, runner_registry=...)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.task.task_models import TaskInfo, TaskStatus
from app.task.task_manager import TaskStorage

logger = logging.getLogger(__name__)


class ComputeTaskStorageAdapter(TaskStorage):
    """将 TaskManager 的 TaskInfo 读写适配到已有的 ComputeTaskService。

    TaskInfo（运行时） ←→ ComputeTaskRecord（MySQL 持久化）

    字段映射：
        TaskInfo.task_id        → ComputeTaskRecord.task_id
        TaskInfo.status.value   → ComputeTaskRecord.status
        TaskInfo.params         → ComputeTaskRecord.input (task parameters)
        TaskInfo.result         → ComputeTaskRecord.result (ComplianceReport dict)
        TaskInfo.error          → ComputeTaskRecord.error
        TaskInfo.progress       → ComputeTaskRecord.input["progress"]   (内嵌)
        TaskInfo.current_stage  → ComputeTaskRecord.input["stage"]      (内嵌)
        TaskInfo.stages         → ComputeTaskRecord.input["stages"]     (内嵌)

    TaskInfo 独有的运行时字段（progress, current_stage, stages）
    存储在 ComputeTaskRecord.input JSON 中，因为现有表没有对应列。
    """

    def __init__(
        self,
        task_service,          # ComputeTaskService 实例
        run_id: str = "",
        step_id: str = "",
        agent_id: Optional[str] = None,
        kind: str = "external_compute",
    ):
        self._service = task_service
        self._run_id = run_id
        self._step_id = step_id
        self._agent_id = agent_id
        self._kind = kind
        # 内存缓存：TaskInfo 的完整运行时字段
        self._cache: Dict[str, TaskInfo] = {}

    # ── TaskStorage 接口实现 ───────────────────────────────────────

    async def save(self, task: TaskInfo) -> None:
        """保存或更新任务。首次保存时在 MySQL 创建记录，后续更新。"""
        self._cache[task.task_id] = task

        try:
            existing = await self._service.get_task(task.task_id)
        except Exception:
            existing = None

        input_data = self._build_input(task)
        if existing is None:
            await self._service.create_task(
                task_id=task.task_id,
                run_id=self._run_id or "standalone",
                step_id=self._step_id or task.task_id,
                agent_id=self._agent_id,
                kind=self._kind,
                input_data=input_data,
            )

        # 更新 MySQL 中的状态、进度和输入快照。
        await self._service.update_task(
            task_id=task.task_id,
            status=task.status.value,
            input_data=input_data,
            result=task.result or {},
            error=task.error,
        )
        if task.status == TaskStatus.SUCCEEDED:
            await self._service.complete_task(
                task_id=task.task_id,
                result=task.result or {},
            )
        elif task.status == TaskStatus.FAILED:
            await self._service.fail_task(
                task_id=task.task_id,
                error=task.error,
            )

    async def get(self, task_id: str) -> Optional[TaskInfo]:
        """从缓存或 MySQL 获取任务。"""
        if task_id in self._cache:
            return self._cache[task_id]

        try:
            record = await self._service.get_task(task_id)
        except Exception:
            return None

        if record is None:
            return None

        return self._record_to_taskinfo(record)

    async def get_by_idempotency_key(self, key: str) -> Optional[TaskInfo]:
        """ComputeTaskService 不直接支持幂等键，通过缓存查找。"""
        for task in self._cache.values():
            if task.idempotency_key == key:
                return task
        return None

    async def list_all(self) -> list[TaskInfo]:
        """返回缓存和 MySQL 中的所有任务。"""
        tasks = {task_id: task for task_id, task in self._cache.items()}
        try:
            records = await self._service.list_tasks(kind=self._kind, limit=200)
        except Exception:
            return list(tasks.values())
        for record in records:
            tasks.setdefault(record.task_id, self._record_to_taskinfo(record))
        return list(tasks.values())

    async def delete(self, task_id: str) -> bool:
        """从缓存中删除（MySQL 中的记录保留）。"""
        return self._cache.pop(task_id, None) is not None

    # ── 字段转换 ──────────────────────────────────────────────────

    def _build_input(self, task: TaskInfo) -> Dict[str, Any]:
        """将 TaskInfo 的运行时字段打包进 ComputeTaskRecord.input JSON。"""
        return {
            **task.params,
            "progress": task.progress,
            "current_stage": task.current_stage,
            "stages": task.stages,
            "tool_name": task.tool_name,
        }

    def _record_to_taskinfo(self, record) -> TaskInfo:
        """从 ComputeTaskRecord 还原 TaskInfo。"""
        inp = record.input or {}
        status_map = {
            "pending":   TaskStatus.PENDING,
            "queued":    TaskStatus.QUEUED,
            "running":   TaskStatus.RUNNING,
            "succeeded": TaskStatus.SUCCEEDED,
            "failed":    TaskStatus.FAILED,
            "cancelled": TaskStatus.CANCELLED,
        }
        status = status_map.get(record.status, TaskStatus.PENDING)

        task = TaskInfo(
            task_id=record.task_id,
            tool_name=inp.get("tool_name", ""),
            status=status,
            progress=inp.get("progress", 0.0),
            current_stage=inp.get("current_stage", ""),
            stages=inp.get("stages", []),
            params={k: v for k, v in inp.items()
                    if k not in ("progress", "current_stage", "stages", "tool_name")},
            result=record.result or None,
            error=record.error or "",
        )
        self._cache[record.task_id] = task
        return task
