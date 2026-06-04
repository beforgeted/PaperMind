"""MCP 网关门面。"""

from typing import AsyncGenerator

from app.mcp_gateway.audit_logger import audit_logger
from app.mcp_gateway.executor import tool_executor
from app.mcp_gateway.permission import permission_service
from app.mcp_gateway.risk_control import risk_control
from app.mcp_gateway.schema_validator import schema_validator


class McpGateway:
    async def call_tool(self, tool_name: str, arguments: dict, progress_callback=None) -> dict:
        validated = schema_validator.validate(tool_name, arguments)
        if not permission_service.allowed(tool_name, validated):
            return {"ok": False, "content": None, "error": "工具权限不足"}
        if not risk_control.approve(tool_name, validated):
            return {"ok": False, "content": None, "error": "工具风险控制未通过"}
        audit_logger.log_call(tool_name, validated)
        return await tool_executor.execute(tool_name, validated, progress_callback=progress_callback)

    async def stream_tool(self, tool_name: str, arguments: dict) -> AsyncGenerator[dict, None]:
        try:
            validated = schema_validator.validate(tool_name, arguments)
        except Exception as exc:
            message = getattr(exc, "message", str(exc))
            yield {"type": "error", "tool_name": tool_name, "ok": False, "error": message}
            yield {"type": "done", "tool_name": tool_name, "ok": False, "error": message}
            return
        if not permission_service.allowed(tool_name, validated):
            yield {"type": "error", "tool_name": tool_name, "ok": False, "error": "工具权限不足"}
            yield {"type": "done", "tool_name": tool_name, "ok": False, "error": "工具权限不足"}
            return
        if not risk_control.approve(tool_name, validated):
            yield {"type": "error", "tool_name": tool_name, "ok": False, "error": "工具风险控制未通过"}
            yield {"type": "done", "tool_name": tool_name, "ok": False, "error": "工具风险控制未通过"}
            return
        audit_logger.log_call(tool_name, validated)
        async for event in tool_executor.execute_stream(tool_name, validated):
            yield event


mcp_gateway = McpGateway()
