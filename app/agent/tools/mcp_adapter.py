"""从 MCP Gateway 元数据生成 LangChain StructuredTool。"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from app.mcp_gateway.gateway import mcp_gateway
from app.mcp_gateway.server_manager import mcp_server_manager
from app.services.mcp_server_service import get_tool_names_for_agent

logger = logging.getLogger(__name__)

_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _json_schema_to_pydantic(tool_name: str, schema: dict[str, Any]) -> Type[BaseModel]:
    """将 MCP inputSchema 转为 Pydantic 参数模型。"""
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    fields: dict[str, Any] = {}

    for prop_name, prop_def in properties.items():
        if not isinstance(prop_def, dict):
            prop_def = {}
        json_type = prop_def.get("type", "string")
        py_type = _JSON_TYPE_MAP.get(json_type, Any)
        description = prop_def.get("description") or ""
        if prop_name in required:
            fields[prop_name] = (py_type, Field(description=description))
        else:
            fields[prop_name] = (
                Optional[py_type],
                Field(default=None, description=description),
            )

    if not fields:
        fields["payload"] = (Optional[dict], Field(default=None, description="工具参数"))

    model_name = "".join(part.capitalize() for part in tool_name.split("_")) + "Args"
    return create_model(model_name, **fields)  # type: ignore[call-overload]


def create_langchain_tool_from_mcp(meta: dict[str, Any]) -> StructuredTool:
    """基于 MCP 工具元数据创建 LangChain 工具。"""
    tool_name = str(meta["name"])
    description = str(meta.get("description") or tool_name)
    parameters = meta.get("parameters") or {}
    args_schema = _json_schema_to_pydantic(tool_name, parameters)

    async def _invoke(**kwargs: Any) -> str:
        clean_args = {k: v for k, v in kwargs.items() if v is not None and k != "payload"}
        result = await mcp_gateway.call_tool(tool_name, clean_args)
        if not result.get("ok"):
            return str(result.get("error") or "MCP 工具调用失败")
        content = result.get("content")
        if content is None:
            return ""
        return content if isinstance(content, str) else str(content)

    return StructuredTool(
        name=tool_name,
        description=description,
        coroutine=_invoke,
        args_schema=args_schema,
    )


async def build_tools_for_agent(agent_id: str) -> list[StructuredTool]:
    """为指定 Agent 构建经 MCP Gateway 调用的 LangChain 工具列表。"""
    if not mcp_server_manager.is_ready():
        await mcp_server_manager.refresh()

    allowlist = set(get_tool_names_for_agent(agent_id))
    if not allowlist:
        return []

    tools: list[StructuredTool] = []
    for tool_name in allowlist:
        if not mcp_server_manager.has(tool_name):
            logger.debug("Agent %s 跳过未发现的 MCP 工具: %s", agent_id, tool_name)
            continue
        meta = mcp_server_manager.get_metadata(tool_name)
        try:
            tools.append(create_langchain_tool_from_mcp(meta))
        except Exception as exc:
            logger.warning("创建 LangChain 工具失败 %s: %s", tool_name, exc)

    return tools
