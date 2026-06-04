"""MCP 工具元数据 — 来自已连接的 MCP Server。"""

from typing import Any, Dict

from app.mcp_gateway.server_manager import mcp_server_manager


def list_tools() -> Dict[str, Dict[str, Any]]:
    return mcp_server_manager.list_tools()
