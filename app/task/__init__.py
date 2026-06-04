"""任务调度模块 — 异步执行框架 + Runner 接口。

提供：
    - TaskStatus 状态机
    - TaskInfo 运行时任务模型
    - BaseRunner 抽象接口
    - TaskManager 任务生命周期管理
    - StorageAdapter 对接 ComputeTaskService（MySQL）
"""

from app.task.task_models import TaskStatus, TaskInfo, assert_valid_transition
from app.task.runner_interface import BaseRunner, ProgressCallback
from app.task.task_manager import TaskManager, InMemoryTaskStorage

__all__ = [
    "TaskStatus",
    "TaskInfo",
    "assert_valid_transition",
    "BaseRunner",
    "ProgressCallback",
    "TaskManager",
    "InMemoryTaskStorage",
]
