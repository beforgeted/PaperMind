"""Agent-Tool 绑定 — 模式 A：从 MCP Server 动态生成 LangChain 工具。"""

from __future__ import annotations

import logging
from typing import Dict, List

from langchain_core.tools import StructuredTool

from app.agent.tools.mcp_adapter import build_tools_for_agent

logger = logging.getLogger(__name__)

_tools_cache: Dict[str, List[StructuredTool]] = {}


async def get_tools_for_agent(agent_id: str) -> list:
    """返回指定 Agent 可用的 LangChain 工具（经 MCP Gateway 调用）。"""
    if agent_id in _tools_cache:
        return _tools_cache[agent_id]

    tools = await build_tools_for_agent(agent_id)
    _tools_cache[agent_id] = tools
    logger.info("Agent %s 已绑定 %s 个 MCP 工具", agent_id, len(tools))
    return tools


def clear_tools_cache() -> None:
    """清除 Agent 工具缓存（MCP 重连后调用）。"""
    _tools_cache.clear()
