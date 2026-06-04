"""MCP 审计日志。"""

import logging
from typing import Any, Dict


logger = logging.getLogger(__name__)


class AuditLogger:
    def log_call(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        logger.info("MCP tool call: %s args=%s", tool_name, arguments)


audit_logger = AuditLogger()
