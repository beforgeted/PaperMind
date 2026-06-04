"""MCP Server 注册表。"""

from typing import Dict, List, Optional

from app.mcp_servers.base import BaseMCPServer


class MCPServerRegistry:
    """管理所有已注册的 MCP Server。"""

    def __init__(self):
        self._servers: Dict[str, BaseMCPServer] = {}

    def register(self, server: BaseMCPServer) -> None:
        self._servers[server.name] = server

    def get(self, name: str) -> Optional[BaseMCPServer]:
        return self._servers.get(name)

    def list_all(self) -> List[BaseMCPServer]:
        return list(self._servers.values())

    @property
    def server_names(self) -> List[str]:
        return list(self._servers.keys())


mcp_server_registry = MCPServerRegistry()
