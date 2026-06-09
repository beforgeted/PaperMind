"""MCP 网关门面。"""

import time
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
            audit_logger.log_call(tool_name, validated, ok=False, error="工具权限不足")
            return {"ok": False, "content": None, "error": "工具权限不足"}
        if not risk_control.approve(tool_name, validated):
            audit_logger.log_call(tool_name, validated, ok=False, error="工具风险控制未通过")
            return {"ok": False, "content": None, "error": "工具风险控制未通过"}
        start = time.perf_counter()
        try:
            result = await tool_executor.execute(tool_name, validated, progress_callback=progress_callback)
            latency_ms = round((time.perf_counter() - start) * 1000)
            ok = bool(result.get("ok", True)) if isinstance(result, dict) else True
            error = str(result.get("error") or "") if isinstance(result, dict) else ""
            audit_logger.log_call(tool_name, validated, latency_ms=latency_ms, ok=ok, error=error or None)
            return result
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000)
            audit_logger.log_call(
                tool_name,
                validated,
                latency_ms=latency_ms,
                ok=False,
                error=str(exc),
            )
            raise

    async def stream_tool(self, tool_name: str, arguments: dict) -> AsyncGenerator[dict, None]:
        try:
            validated = schema_validator.validate(tool_name, arguments)
        except Exception as exc:
            message = getattr(exc, "message", str(exc))
            audit_logger.log_call(tool_name, arguments, ok=False, error=message)
            yield {"type": "error", "tool_name": tool_name, "ok": False, "error": message}
            yield {"type": "done", "tool_name": tool_name, "ok": False, "error": message}
            return
        if not permission_service.allowed(tool_name, validated):
            audit_logger.log_call(tool_name, validated, ok=False, error="工具权限不足")
            yield {"type": "error", "tool_name": tool_name, "ok": False, "error": "工具权限不足"}
            yield {"type": "done", "tool_name": tool_name, "ok": False, "error": "工具权限不足"}
            return
        if not risk_control.approve(tool_name, validated):
            audit_logger.log_call(tool_name, validated, ok=False, error="工具风险控制未通过")
            yield {"type": "error", "tool_name": tool_name, "ok": False, "error": "工具风险控制未通过"}
            yield {"type": "done", "tool_name": tool_name, "ok": False, "error": "工具风险控制未通过"}
            return
        start = time.perf_counter()
        ok = True
        error_msg = ""
        try:
            async for event in tool_executor.execute_stream(tool_name, validated):
                if isinstance(event, dict):
                    if event.get("type") == "error" or event.get("ok") is False:
                        ok = False
                        error_msg = str(event.get("error") or error_msg)
                yield event
        except Exception as exc:
            ok = False
            error_msg = str(exc)
            raise
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000)
            audit_logger.log_call(
                tool_name,
                validated,
                latency_ms=latency_ms,
                ok=ok,
                error=error_msg or None,
            )


mcp_gateway = McpGateway()
