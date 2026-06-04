"""MCP 桥接层 — 委托 McpServerManager（模式 A 兼容旧接口）。"""

from __future__ import annotations

import logging
from typing import Any

from app.mcp_gateway.server_manager import mcp_server_manager

logger = logging.getLogger(__name__)


class _McpBridgeRegistry:
    """兼容旧 LocalToolRegistry 接口，实际路由至 MCP Server。"""

    def has(self, name: str) -> bool:
        return mcp_server_manager.has(name)

    def get(self, name: str) -> dict[str, Any]:
        return mcp_server_manager.get_metadata(name)

    def call(self, name: str, progress_callback: Any = None, **kwargs: Any) -> Any:
        del progress_callback
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(mcp_server_manager.call_tool(name, kwargs))

        if loop.is_running():
            raise RuntimeError(
                "同步 registry.call 不能在运行中的事件循环内调用，请使用 mcp_gateway.call_tool"
            )
        return asyncio.run(mcp_server_manager.call_tool(name, kwargs))

    def list_registered(self) -> dict[str, dict[str, Any]]:
        return mcp_server_manager.list_tools()


_bridge = _McpBridgeRegistry()


def get_tool_registry() -> _McpBridgeRegistry:
    return _bridge


def register_platform_tools() -> None:
    """启动时由 lifespan 调用 mcp_server_manager.refresh()，此处保留兼容。"""
    logger.debug("register_platform_tools: 使用 mcp_server_manager.refresh()")


def list_platform_tools() -> dict[str, dict[str, Any]]:
    return mcp_server_manager.list_tools()
