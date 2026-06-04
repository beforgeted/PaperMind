"""任务状态机、数据模型和状态转换规则。

参考 PaperMind 项目模式（NodeStatus + PipelineStage），以独立、可测试的形式实现。

状态机：

    pending（待处理）
       │
       ▼
    queued（已排队） ──────────────►  cancelled（已取消）
       │
       ▼
    running（运行中） ─────────────►  failed（失败）
       │
       ▼
    succeeded（已成功）

每个转换都通过 assert_valid_transition() 进行强制验证。
终端状态（succeeded / failed / cancelled）不会再转换。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ──────────────────────────────── 状态机 ────────────────────────────────


class TaskStatus(str, Enum):
    PENDING   = "pending"
    QUEUED    = "queued"
    RUNNING   = "running"
    SUCCEEDED = "succeeded"
    FAILED    = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        """不会再转换的任务状态。"""
        return self in (TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED)


# 从每个状态出发的有效转换。
_VALID_TRANSITIONS: Dict[TaskStatus, List[TaskStatus]] = {
    TaskStatus.PENDING:   [TaskStatus.QUEUED, TaskStatus.CANCELLED],
    TaskStatus.QUEUED:    [TaskStatus.RUNNING, TaskStatus.CANCELLED, TaskStatus.FAILED],
    TaskStatus.RUNNING:   [TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED],
    TaskStatus.SUCCEEDED: [],
    TaskStatus.FAILED:    [],
    TaskStatus.CANCELLED: [],
}


def can_transition(current: TaskStatus, target: TaskStatus) -> bool:
    return target in _VALID_TRANSITIONS.get(current, [])


def assert_valid_transition(current: TaskStatus, target: TaskStatus) -> None:
    if not can_transition(current, target):
        raise ValueError(
            f"Invalid status transition: {current.value} → {target.value}"
        )


# ──────────────────────────────── 数据模型 ────────────────────────────────


@dataclass
class TaskInfo:
    """单个异步任务的自包含快照。

    参考 PaperMind Execution ORM 模型的模式，包含每阶段进度字段。
    """

    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    tool_name: str = ""
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    current_stage: str = ""
    stages: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: str = ""
    callback_url: Optional[str] = None
    idempotency_key: Optional[str] = None
    timeout_seconds: Optional[float] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # ── 序列化 ─────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tool_name": self.tool_name,
            "status": self.status.value,
            "progress": self.progress,
            "current_stage": self.current_stage,
            "stages": self.stages,
            "params": self.params,
            "result": self.result,
            "error": self.error,
            "callback_url": self.callback_url,
            "idempotency_key": self.idempotency_key,
            "timeout_seconds": self.timeout_seconds,
            "created_at": _iso(self.created_at),
            "started_at": _iso(self.started_at),
            "completed_at": _iso(self.completed_at),
        }

    def status_dict(self) -> Dict[str, Any]:
        """用于 /status 轮询的轻量级负载。"""
        return {
            "task_id": self.task_id,
            "tool_name": self.tool_name,
            "status": self.status.value,
            "progress": self.progress,
            "current_stage": self.current_stage,
            "stages": self.stages,
            "error": self.error,
            "created_at": _iso(self.created_at),
            "started_at": _iso(self.started_at),
            "completed_at": _iso(self.completed_at),
        }

    def result_dict(self) -> Dict[str, Any]:
        """用于 /result 的负载 — 仅对终端任务有意义。"""
        return {
            "task_id": self.task_id,
            "tool_name": self.tool_name,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "created_at": _iso(self.created_at),
            "started_at": _iso(self.started_at),
            "completed_at": _iso(self.completed_at),
        }

    # ── 状态转换 ─────────────────────────────────────────────

    def set_status(self, target: TaskStatus) -> None:
        assert_valid_transition(self.status, target)
        self.status = target
        if target == TaskStatus.RUNNING and self.started_at is None:
            self.started_at = datetime.now(timezone.utc)
        if target.terminal:
            self.completed_at = datetime.now(timezone.utc)

    def mark_failed(self, message: str) -> None:
        """便捷方法：在一次调用中设置 FAILED 状态和错误信息。

        当已经处于 FAILED 状态时调用是安全的（无操作）— 处理超时在
        执行器已经因其他原因失败后触发的情况。
        """
        if self.status == TaskStatus.FAILED:
            return
        if not self.status.terminal:
            self.set_status(TaskStatus.FAILED)
        self.error = message


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None
