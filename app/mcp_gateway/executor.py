"""MCP 工具执行器 — 委托 McpServerManager 经 MCP 协议调用。"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Callable, Dict

from app.mcp_gateway.server_manager import mcp_server_manager


class ToolExecutor:
    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> Dict[str, Any]:
        del progress_callback
        if not mcp_server_manager.has(tool_name):
            return {
                "ok": False,
                "content": None,
                "error": f"未知工具: {tool_name}",
                "arguments": dict(arguments),
            }
        try:
            content = await mcp_server_manager.call_tool(tool_name, arguments)
            return {"ok": True, "content": content, "error": ""}
        except Exception as exc:
            message = getattr(exc, "message", str(exc))
            return {
                "ok": False,
                "content": None,
                "error": message,
                "arguments": dict(arguments),
            }

    async def execute_stream(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """以统一事件流执行 MCP 工具（单次结果，无细粒度进度）。"""
        yield {
            "type": "start",
            "tool_name": tool_name,
            "stage": "启动 MCP 工具",
            "progress": 0.0,
        }

        if not mcp_server_manager.has(tool_name):
            error = f"未知工具: {tool_name}"
            yield {
                "type": "error",
                "tool_name": tool_name,
                "ok": False,
                "error": error,
                "arguments": dict(arguments),
            }
            yield {"type": "done", "tool_name": tool_name, "ok": False, "error": error}
            return

        yield {
            "type": "progress",
            "tool_name": tool_name,
            "stage": "执行中",
            "progress": 0.5,
        }

        result = await self.execute(tool_name, arguments)
        ok = bool(result.get("ok"))
        error = str(result.get("error") or "")
        yield {
            "type": "result",
            "tool_name": tool_name,
            "ok": ok,
            "content": result.get("content"),
            "error": error,
        }
        if not ok:
            yield {
                "type": "error",
                "tool_name": tool_name,
                "ok": False,
                "error": error,
                "arguments": dict(arguments),
            }
        yield {"type": "done", "tool_name": tool_name, "ok": ok, "error": error}


tool_executor = ToolExecutor()
