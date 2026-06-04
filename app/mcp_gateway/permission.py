"""MCP 权限检查。"""

from typing import Any, Dict


class PermissionService:
    def allowed(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        del tool_name, arguments
        return True


permission_service = PermissionService()
