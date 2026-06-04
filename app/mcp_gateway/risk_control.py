"""MCP 风险控制钩子。"""

from typing import Any, Dict


class RiskControl:
    def approve(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        del tool_name, arguments
        return True


risk_control = RiskControl()
