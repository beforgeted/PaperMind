"""MCP 数据模型。"""

from typing import Any, Dict

from pydantic import BaseModel, Field


class McpToolCallRequest(BaseModel):
    tool_name: str = Field(..., description="工具名称")
    arguments: Dict[str, Any] = Field(default_factory=dict)
    mcp_server: str = Field(default="papermind-tools", description="MCP Server 名称")


class McpToolCallResponse(BaseModel):
    ok: bool
    content: Any = None
    error: str = ""
    mcp_server: str = ""
    tool_name: str = ""


# 兼容旧名
McpToolRequest = McpToolCallRequest
McpToolResult = McpToolCallResponse
