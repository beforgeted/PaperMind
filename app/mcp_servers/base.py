"""MCP 服务描述。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class McpServerSpec:
    name: str
    description: str
    tool_names: tuple[str, ...] = ()
