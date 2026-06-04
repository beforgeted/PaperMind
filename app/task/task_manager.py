"""核心任务管理器：提交、轮询、回调、取消。

架构（参考 PaperMind 生产环境将拥有的架构）：

    POST /tasks/submit
        │
        ▼
    TaskManager.submit()
        ├─ idempotency_key 检查 → 如果已存在则返回现有任务
        ├─ 创建 TaskInfo（status=PENDING），存储它
        ├─ 转换到 QUEUED 状态
        ├─ 将 _execute() 作为 asyncio.Task 启动
        └─ 立即返回 task_id

    _execute()
        ├─ 转换到 RUNNING 状态
        ├─ 调用 runner.run(…, progress_callback)  ← 带 asyncio.wait_for 超时
        │     └─ progress_callback → 更新 TaskInfo.progress / current_stage
        ├─ 成功时：转换到 SUCCEEDED，存储结果
        ├─ TimeoutError 时：mark_failed(TIMEOUT)
        ├─ 其他异常时：mark_failed(RUNNER_ERROR)
        └─ 如果设置了 callback_url → _send_callback_with_retry()

在生产环境中，``asyncio.create_task`` 调用会变成
``celery_app.send_task(name, args=…)``, 但 Runner 接口、
存储、状态机、超时、重试和幂等性逻辑保持不变。
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

from app.task.task_models import TaskInfo, TaskStatus, assert_valid_transition, can_transition
from app.task.runner_interface import BaseRunner, ProgressCallback

logger = logging.getLogger(__name__)


# ──────────────────────────────── 存储抽象 ────────────────────────────────


class TaskStorage(ABC):
    """TaskInfo 记录的持久化契约。"""

    @abstractmethod
    async def save(self, task: TaskInfo) -> None:
        ...

    @abstractmethod
    async def get(self, task_id: str) -> Optional[TaskInfo]:
        ...

    @abstractmethod
    async def get_by_idempotency_key(self, key: str) -> Optional[TaskInfo]:
        """如果存在，返回此幂等键对应的现有任务。"""
        ...

    @abstractmethod
    async def list_all(self) -> List[TaskInfo]:
        ...

    @abstractmethod
    async def delete(self, task_id: str) -> bool:
        ...


class InMemoryTaskStorage(TaskStorage):
    """基于字典的存储，用于测试和原型开发。"""

    def __init__(self) -> None:
        self._tasks: Dict[str, TaskInfo] = {}
        self._idem_index: Dict[str, str] = {}  # idempotency_key → task_id

    async def save(self, task: TaskInfo) -> None:
        self._tasks[task.task_id] = task
        if task.idempotency_key:
            self._idem_index[task.idempotency_key] = task.task_id

    async def get(self, task_id: str) -> Optional[TaskInfo]:
        return self._tasks.get(task_id)

    async def get_by_idempotency_key(self, key: str) -> Optional[TaskInfo]:
        task_id = self._idem_index.get(key)
        if task_id is None:
            return None
        return self._tasks.get(task_id)

    async def list_all(self) -> List[TaskInfo]:
        return list(self._tasks.values())

    async def delete(self, task_id: str) -> bool:
        task = self._tasks.pop(task_id, None)
        if task is not None and task.idempotency_key:
            self._idem_index.pop(task.idempotency_key, None)
        return task is not None


# ──────────────────────────────── TaskManager ────────────────────────────────


class TaskManager:
    """异步任务生命周期的中央协调器。

    参数
    ----------
    storage : TaskStorage
    runner_registry : dict[str, BaseRunner]
        映射 tool_name → Runner 实例。
    callback_client : httpx.AsyncClient, optional
        用于出站 webhook 调用的共享 HTTP 客户端。
    default_timeout_seconds : float
        当任务本身未指定时的每任务超时时间。
    max_callback_retries : int
        失败的回调 POST 请求的重试次数。
    callback_retry_base_delay : float
        初始延迟秒数；每次重试翻倍。
    """

    def __init__(
        self,
        storage: TaskStorage,
        runner_registry: Optional[Dict[str, BaseRunner]] = None,
        callback_client: Optional[httpx.AsyncClient] = None,
        *,
        default_timeout_seconds: float = 3600.0,
        max_callback_retries: int = 3,
        callback_retry_base_delay: float = 1.0,
        on_task_succeeded: Optional[Callable[[TaskInfo], Awaitable[None]]] = None,
        on_task_failed: Optional[Callable[[TaskInfo], Awaitable[None]]] = None,
    ) -> None:
        self.storage = storage
        self._runners = runner_registry or {}
        self._callback_client = callback_client
        self.default_timeout_seconds = default_timeout_seconds
        self.max_callback_retries = max_callback_retries
        self.callback_retry_base_delay = callback_retry_base_delay
        self._on_task_succeeded = on_task_succeeded
        self._on_task_failed = on_task_failed
        self._running_tasks: Dict[str, asyncio.Task] = {}

    # ── 公共 API ──────────────────────────────────────────────────────

    async def submit(
        self,
        tool_name: str,
        params: Optional[Dict[str, Any]] = None,
        callback_url: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> TaskInfo:
        """提交任务并立即返回其 TaskInfo。

        参数
        ----------
        idempotency_key : str, optional
            如果提供且已存在具有此键的任务，则立即返回
            现有任务（不重复执行）。
        timeout_seconds : float, optional
            覆盖默认的每任务超时时间。
        """
        # ── 幂等性检查 ──
        if idempotency_key is not None:
            existing = await self.storage.get_by_idempotency_key(idempotency_key)
            if existing is not None:
                logger.info(
                    "Idempotency key %r → returning existing task %s",
                    idempotency_key,
                    existing.task_id,
                )
                return existing

        runner = self._runners.get(tool_name)
        if runner is None:
            raise ValueError(
                f"Unknown tool '{tool_name}'. "
                f"Registered: {list(self._runners.keys())}"
            )

        task = TaskInfo(
            tool_name=tool_name,
            status=TaskStatus.PENDING,
            params=params or {},
            callback_url=callback_url,
            idempotency_key=idempotency_key,
            timeout_seconds=timeout_seconds,
            stages=list(runner.stages),
        )

        await self.storage.save(task)
        task.set_status(TaskStatus.QUEUED)
        await self.storage.save(task)

        coro = self._execute(task.task_id, runner)
        asyncio_task = asyncio.create_task(coro)
        self._running_tasks[task.task_id] = asyncio_task
        asyncio_task.add_done_callback(
            lambda _: self._running_tasks.pop(task.task_id, None)
        )

        return task

    async def get_status(self, task_id: str) -> Optional[TaskInfo]:
        return await self.storage.get(task_id)

    async def get_result(self, task_id: str) -> Optional[TaskInfo]:
        task = await self.storage.get(task_id)
        if task is None:
            return None
        if not task.status.terminal:
            raise TaskNotFinishedError(
                task_id, f"Task is still {task.status.value}"
            )
        return task

    async def cancel(self, task_id: str) -> bool:
        task = await self.storage.get(task_id)
        if task is None:
            return False
        if not can_transition(task.status, TaskStatus.CANCELLED):
            return False

        asyncio_task = self._running_tasks.get(task_id)
        if asyncio_task is not None and not asyncio_task.done():
            asyncio_task.cancel()

        task.mark_failed("Task cancelled by user")
        # mark_failed 转换到 FAILED — 覆盖为 CANCELLED
        task.status = TaskStatus.CANCELLED
        await self.storage.save(task)
        return True

    async def list_tasks(self, skip: int = 0, limit: int = 50) -> List[TaskInfo]:
        all_tasks = await self.storage.list_all()
        all_tasks.sort(key=lambda t: t.created_at, reverse=True)
        return all_tasks[skip : skip + limit]

    # ── 内部方法 ────────────────────────────────────────────────────────

    async def _progress_callback(
        self, task_id: str, stage_name: str, progress: float
    ) -> None:
        task = await self.storage.get(task_id)
        if task is None:
            return
        task.current_stage = stage_name
        task.progress = min(progress, 1.0)
        await self.storage.save(task)

    async def _execute(self, task_id: str, runner: BaseRunner) -> None:
        """作为 asyncio.Task 运行的核心执行循环。"""
        task = await self.storage.get(task_id)
        if task is None:
            return

        # 转换 QUEUED → RUNNING
        try:
            task.set_status(TaskStatus.RUNNING)
            await self.storage.save(task)
        except ValueError:
            return  # 已被取消/失败

        timeout = task.timeout_seconds or self.default_timeout_seconds

        async def _on_stage(stage_name: str, progress: float) -> None:
            await self._progress_callback(task_id, stage_name, progress)

        try:
            result = await asyncio.wait_for(
                runner.run(task.params, _on_stage),
                timeout=timeout,
            )

            task = await self.storage.get(task_id)
            if task is None:
                return
            task.set_status(TaskStatus.SUCCEEDED)
            task.result = result
            await self.storage.save(task)
            await self._send_callback_with_retry(task)
            if self._on_task_succeeded is not None:
                await self._on_task_succeeded(task)

        except asyncio.TimeoutError:
            task = await self.storage.get(task_id)
            if task is not None:
                task.mark_failed(
                    f"Task timed out after {timeout:.0f}s",
                )
                await self.storage.save(task)
                await self._send_callback_with_retry(task)
                if self._on_task_failed is not None:
                    await self._on_task_failed(task)

        except asyncio.CancelledError:
            task = await self.storage.get(task_id)
            if task is not None and not task.status.terminal:
                task.mark_failed("Task cancelled")
                await self.storage.save(task)
                if self._on_task_failed is not None:
                    await self._on_task_failed(task)

        except Exception as exc:
            task = await self.storage.get(task_id)
            if task is None:
                return

            msg = f"{type(exc).__name__}: {exc}"

            task.mark_failed(msg)
            await self.storage.save(task)
            await self._send_callback_with_retry(task)
            if self._on_task_failed is not None:
                await self._on_task_failed(task)

    # ── 回调重试 ──────────────────────────────────────────────────

    async def _send_callback_with_retry(self, task: TaskInfo) -> None:
        """使用指数退避将任务结果 POST 到 callback_url。

        最多重试 ``max_callback_retries`` 次。所有重试后的失败
        会被记录，但*不会*改变任务状态 — 任务本身已经是终端状态。
        """
        if not task.callback_url:
            return

        if self._callback_client is None:
            self._callback_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))

        payload = task.result_dict()
        delay = self.callback_retry_base_delay

        for attempt in range(self.max_callback_retries + 1):
            try:
                resp = await self._callback_client.post(
                    task.callback_url,
                    json=payload,
                    timeout=httpx.Timeout(10.0),
                )
                resp.raise_for_status()
                logger.info(
                    "Callback to %s succeeded (task=%s, attempt=%d)",
                    task.callback_url, task.task_id, attempt + 1,
                )
                return
            except Exception as exc:
                if attempt < self.max_callback_retries:
                    logger.warning(
                        "Callback to %s failed (task=%s, attempt=%d/%d): %s. "
                        "Retrying in %.1fs …",
                        task.callback_url,
                        task.task_id,
                        attempt + 1,
                        self.max_callback_retries + 1,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    logger.error(
                        "Callback to %s exhausted all %d retries (task=%s): %s",
                        task.callback_url,
                        self.max_callback_retries + 1,
                        task.task_id,
                        exc,
                    )

    async def close(self) -> None:
        for t in list(self._running_tasks.values()):
            if not t.done():
                t.cancel()
        if self._callback_client is not None:
            await self._callback_client.aclose()


# ──────────────────────────────── 错误 ────────────────────────────────


class TaskNotFinishedError(Exception):
    """当任务未达到终端状态时，由 get_result() 抛出。"""

    def __init__(self, task_id: str, message: str = "") -> None:
        self.task_id = task_id
        super().__init__(message or f"Task {task_id} is not finished yet")
