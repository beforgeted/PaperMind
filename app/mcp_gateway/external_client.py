"""MCP 外部 Server 客户端 — 管理外部 MCP Server 的工具发现与调用。

优先通过 stdio 子进程协议连接；Windows 下 stdio 不可用时回退到同进程 call_tool API。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class McpExternalServer:
    """管理一个外部 MCP Server 的连接与工具调用。

    支持两种模式：
    - stdio: 启动子进程，通过 MCP 协议通信
    - inproc: 同进程直接调用 FastMCP.call_tool()（Windows 兼容）
    """

    def __init__(self, name: str, module: str):
        self.name = name
        self.module = module  # e.g. "paper_search_mcp.server"
        self._mcp_instance: Any = None
        self._session: Optional[Any] = None
        self._transport: Optional[tuple] = None
        self._connected = False
        self._tools: list[dict] = []

    async def connect(self, *, prefer_inproc: bool = True) -> None:
        """连接到 MCP Server 并发现工具。"""
        if self._connected:
            return

        if prefer_inproc:
            try:
                await self._connect_inproc()
                return
            except Exception as exc:
                logger.info("MCP %s inproc 连接失败，尝试 stdio: %s", self.name, exc)

        try:
            await self._connect_stdio()
        except Exception as exc:
            logger.error("MCP %s stdio 连接也失败: %s", self.name, exc)
            await self.disconnect()
            raise

    async def _connect_inproc(self) -> None:
        """同进程模式：直接导入 FastMCP 实例，通过 call_tool() 调用。"""
        import importlib

        mod = importlib.import_module(self.module)
        self._mcp_instance = mod.mcp  # FastMCP 实例

        # 发现工具
        tools_result = await self._mcp_instance.list_tools()
        self._tools = [
            {
                "name": t.name,
                "description": getattr(t, "description", ""),
                "parameters": getattr(t, "inputSchema", {}),
            }
            for t in (tools_result or [])
        ]
        self._connected = True
        logger.info(
            "MCP %s 已连接(inproc)，发现 %s 个工具: %s",
            self.name,
            len(self._tools),
            [t["name"] for t in self._tools],
        )

    async def _connect_stdio(self) -> None:
        """stdio 子进程模式。"""
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from mcp.client.session import ClientSession

        server_params = StdioServerParameters(command=self.module.replace(".server", ""))
        transport = stdio_client(server_params)
        read, write = await transport.__aenter__()
        self._transport = (transport, read, write)

        session = ClientSession(read, write)
        await session.__aenter__()
        self._session = session
        await session.initialize()

        tools_result = await session.list_tools()
        self._tools = [
            {
                "name": t.name,
                "description": getattr(t, "description", ""),
                "parameters": getattr(t, "inputSchema", {}),
            }
            for t in tools_result.tools
        ]
        self._connected = True
        logger.info(
            "MCP %s 已连接(stdio)，发现 %s 个工具: %s",
            self.name,
            len(self._tools),
            [t["name"] for t in self._tools],
        )

    async def disconnect(self) -> None:
        if self._session:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                pass
            self._session = None
        if self._transport:
            transport, read, write = self._transport
            try:
                await transport.__aexit__(None, None, None)
            except Exception:
                pass
            self._transport = None
        self._mcp_instance = None
        self._connected = False
        self._tools = []

    async def list_tools(self) -> list[dict]:
        if not self._connected:
            await self.connect()
        return list(self._tools)

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        if not self._connected:
            await self.connect()

        try:
            if self._mcp_instance:
                result = await self._mcp_instance.call_tool(tool_name, arguments)
            elif self._session:
                result = await self._session.call_tool(tool_name, arguments)
            else:
                return "MCP Server 未连接"

            if hasattr(result, "content") and result.content:
                texts = []
                for item in result.content:
                    if hasattr(item, "text"):
                        texts.append(item.text)
                    elif isinstance(item, str):
                        texts.append(item)
                return "\n\n".join(texts) if texts else str(result)
            return str(result)
        except Exception as exc:
            logger.error("MCP 工具 %s/%s 失败: %s", self.name, tool_name, exc)
            return f"MCP 工具 {tool_name} 调用失败: {str(exc)}"


async def get_or_create_paper_search_server() -> McpExternalServer:
    """获取或创建 paper-search-mcp 的 MCP Server 实例。"""
    from app.mcp_gateway.server_manager import mcp_server_manager

    return await mcp_server_manager.get_server("paper-search")
