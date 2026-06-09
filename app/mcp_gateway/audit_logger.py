"""MCP 审计日志，写入 papermind.audit 独立通道。"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from app.core.log_context import get_context
from app.core.logging import get_audit_logger
from app.observability.events import MCP_TOOL_CALL

_audit = get_audit_logger()


def _preview(arguments: Dict[str, Any], limit: int = 200) -> str:
    try:
        text = json.dumps(arguments, ensure_ascii=False, default=str)
    except TypeError:
        text = str(arguments)
    return text[:limit]


class AuditLogger:
    def log_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        *,
        latency_ms: Optional[int] = None,
        ok: bool = True,
        caller_agent: Optional[str] = None,
        run_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """记录 MCP 工具调用审计事件。"""
        ctx = get_context()
        _audit.info(
            "mcp_tool_call",
            extra={
                "event": MCP_TOOL_CALL,
                "tool_name": tool_name,
                "args_preview": _preview(arguments),
                "latency_ms": latency_ms,
                "ok": ok,
                "caller_agent": caller_agent or ctx.get("agent_id"),
                "run_id": run_id or ctx.get("run_id"),
                "error": error or "",
            },
        )


audit_logger = AuditLogger()
