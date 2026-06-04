"""MCP Server 统一管理 — 连接、发现工具、路由 call_tool。"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from app.mcp_gateway.external_client import McpExternalServer

logger = logging.getLogger(__name__)

PAPERMIND_RETRIEVAL_SERVER = "papermind-retrieval"
PAPER_SEARCH_SERVER = "paper-search"

SERVER_MODULES: dict[str, str] = {
    PAPERMIND_RETRIEVAL_SERVER: "app.mcp_servers.papermind_retrieval.server",
    PAPER_SEARCH_SERVER: "paper_search_mcp.server",
}


class McpServerManager:
    """管理多个 MCP Server 实例，维护工具名 → Server 路由表。"""

    def __init__(self) -> None:
        self._servers: dict[str, McpExternalServer] = {}
        self._tools_meta: dict[str, dict[str, Any]] = {}
        self._tool_to_server: dict[str, str] = {}
        self._ready = False

    async def connect_server(
        self,
        server_name: str,
        *,
        required: bool = True,
    ) -> bool:
        """连接单个 MCP Server。"""
        module = SERVER_MODULES.get(server_name)
        if not module:
            logger.warning("未知 MCP Server: %s", server_name)
            return False

        if server_name in self._servers:
            return True

        server = McpExternalServer(name=server_name, module=module)
        try:
            await server.connect(prefer_inproc=True)
            self._servers[server_name] = server
            for tool in await server.list_tools():
                tool_name = str(tool.get("name") or "")
                if not tool_name:
                    continue
                if tool_name in self._tools_meta:
                    logger.warning(
                        "工具名冲突: %s 已属于 %s，忽略 %s 的重复定义",
                        tool_name,
                        self._tool_to_server.get(tool_name),
                        server_name,
                    )
                    continue
                self._tools_meta[tool_name] = {
                    "name": tool_name,
                    "description": tool.get("description") or "",
                    "parameters": tool.get("parameters") or {},
                    "mcp_server": server_name,
                }
                self._tool_to_server[tool_name] = server_name
            logger.info(
                "MCP Server %s 已连接，工具数=%s",
                server_name,
                sum(1 for s in self._tool_to_server.values() if s == server_name),
            )
            return True
        except Exception as exc:
            if required:
                logger.error("必需 MCP Server %s 连接失败: %s", server_name, exc)
                raise
            logger.warning("可选 MCP Server %s 不可用: %s", server_name, exc)
            return False

    async def refresh(
        self,
        *,
        include_paper_search: bool = True,
    ) -> None:
        """连接并刷新全部工具元数据。"""
        self._tools_meta.clear()
        self._tool_to_server.clear()
        self._servers.clear()
        self._ready = False

        await self.connect_server(PAPERMIND_RETRIEVAL_SERVER, required=True)
        if include_paper_search:
            await self.connect_server(PAPER_SEARCH_SERVER, required=False)
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready and bool(self._tools_meta)

    def list_tools(self) -> dict[str, dict[str, Any]]:
        return dict(self._tools_meta)

    def has(self, tool_name: str) -> bool:
        return tool_name in self._tools_meta

    def get_metadata(self, tool_name: str) -> dict[str, Any]:
        if tool_name not in self._tools_meta:
            raise KeyError(f"未知工具: {tool_name}")
        return self._tools_meta[tool_name]

    def resolve_server(self, tool_name: str) -> Optional[str]:
        return self._tool_to_server.get(tool_name)

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> str:
        """调用 MCP 工具并返回文本结果。"""
        del progress_callback  # MCP 工具暂不支持进度回调
        if not self._ready:
            await self.refresh()

        server_name = self.resolve_server(tool_name)
        if not server_name:
            raise KeyError(f"未知工具: {tool_name}")

        server = self._servers.get(server_name)
        if server is None:
            await self.connect_server(
                server_name,
                required=(server_name == PAPERMIND_RETRIEVAL_SERVER),
            )
            server = self._servers.get(server_name)
        if server is None:
            raise RuntimeError(f"MCP Server 未连接: {server_name}")

        return await server.call_tool(tool_name, arguments)

    async def get_server(self, server_name: str) -> McpExternalServer:
        if server_name not in self._servers:
            await self.connect_server(
                server_name,
                required=(server_name == PAPERMIND_RETRIEVAL_SERVER),
            )
        server = self._servers.get(server_name)
        if server is None:
            raise RuntimeError(f"MCP Server 不可用: {server_name}")
        return server


mcp_server_manager = McpServerManager()


async def get_or_create_paper_search_server() -> McpExternalServer:
    """兼容旧调用：返回 paper-search MCP Server 连接。"""
    return await mcp_server_manager.get_server(PAPER_SEARCH_SERVER)
