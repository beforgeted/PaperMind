"""MCP 工具异步 Runner。"""

from typing import Any, Dict

from app.mcp_gateway.gateway import mcp_gateway
from app.task.runner_interface import BaseRunner, ProgressCallback


class McpToolRunner(BaseRunner):
    """通过 TaskManager 异步执行任意 MCP 工具。"""

    tool_name = "mcp_tool"
    stages = ["校验工具请求", "执行 MCP 工具", "保存工具结果"]

    async def run(
        self,
        params: Dict[str, Any],
        progress_callback: ProgressCallback,
    ) -> Dict[str, Any]:
        tool_name = str(params.get("tool_name") or "")
        arguments = params.get("arguments") or {}
        if not tool_name:
            raise ValueError("缺少 MCP 工具名")
        if not isinstance(arguments, dict):
            raise ValueError("MCP 工具参数必须是对象")

        await progress_callback("校验工具请求", 0.2)
        import asyncio

        loop = asyncio.get_running_loop()

        async def _async_progress(stage_name: str, progress: float) -> None:
            await progress_callback(stage_name, progress)

        def _progress(stage_name: str, progress: float) -> None:
            asyncio.run_coroutine_threadsafe(_async_progress(stage_name, progress), loop)

        result = await mcp_gateway.call_tool(tool_name, arguments, progress_callback=_progress)
        await progress_callback("执行 MCP 工具", 0.9)
        await progress_callback("保存工具结果", 1.0)
        return {
            "tool_name": tool_name,
            "arguments": arguments,
            "tool_result": result,
        }
