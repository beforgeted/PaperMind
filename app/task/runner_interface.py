"""抽象 Runner 接口。

在生产环境中，具体的 Runner 封装 Docker 容器（GenMol、DiffDock 等）
或 LLM 子代理调用。在测试环境中，它们通过 asyncio.sleep() 进行模拟。

核心契约：Runner 接收参数和进度回调函数，返回结果字典（失败时抛出异常）。
TaskManager 负责线程管理和生命周期控制。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, List

# progress_callback(stage_name: str, progress: float) -> None
ProgressCallback = Callable[[str, float], Awaitable[None]]


class BaseRunner(ABC):
    """单个工具/管道的无状态执行契约。

    子类将 ``tool_name`` 和 ``stages`` 设置为类属性（或实例属性），
    以便 TaskManager 能够在注册时不实例化 runner 的情况下发现阶段数量，
    用于进度计算。
    """

    tool_name: str = ""
    stages: List[str] = []

    @abstractmethod
    async def run(
        self,
        params: Dict[str, Any],
        progress_callback: ProgressCallback,
    ) -> Dict[str, Any]:
        """端到端执行工具。

        * 在每个逻辑阶段完成后调用 ``await progress_callback(stage_name, progress_float)``
          （0.0 ≤ progress ≤ 1.0）。
        * 返回一个 JSON 可序列化的字典作为最终结果。
        * 抛出**任何**异常以表示失败 — TaskManager 会捕获它，
          设置 status=FAILED，并存储错误消息。
        """
        ...
