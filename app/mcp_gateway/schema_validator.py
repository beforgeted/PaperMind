"""MCP 工具参数校验（基于已发现工具的 inputSchema）。"""

from typing import Any, Dict

from app.mcp_gateway.server_manager import mcp_server_manager


class SchemaValidator:
    def validate(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not mcp_server_manager.has(tool_name):
            return dict(arguments)
        meta = mcp_server_manager.get_metadata(tool_name)
        schema = meta.get("parameters") or {}
        required = set(schema.get("required") or [])
        properties = schema.get("properties") or {}
        validated = dict(arguments)
        for key in required:
            if key not in validated or validated[key] is None:
                raise ValueError(f"缺少必填参数: {key}")
        for key in list(validated.keys()):
            if key not in properties and properties:
                continue
        return validated


schema_validator = SchemaValidator()
