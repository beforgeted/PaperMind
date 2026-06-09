"""会话历史消息与 LangChain Message 的转换工具。"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


HistoryMessage = Dict[str, str]


def normalize_history_role(role: str) -> str:
    """将存储角色规范为 user / assistant。"""
    normalized = (role or "").strip().lower()
    if normalized in ("user", "human"):
        return "user"
    if normalized == "assistant":
        return "assistant"
    return "user"


def turns_to_history_messages(turns: List[Any]) -> List[HistoryMessage]:
    """将 TurnRecord 或 dict 转为按时间正序的 role/content 列表。"""
    messages: List[HistoryMessage] = []
    for turn in turns:
        if hasattr(turn, "role") and hasattr(turn, "content"):
            role = normalize_history_role(str(turn.role))
            content = str(turn.content or "")
        elif isinstance(turn, dict):
            role = normalize_history_role(str(turn.get("role") or "user"))
            content = str(turn.get("content") or "")
        else:
            continue
        if not content:
            continue
        messages.append({"role": role, "content": content})
    return messages


def history_to_langchain_messages(history: List[HistoryMessage]) -> List[BaseMessage]:
    """将 role/content 历史转为 LangChain 多轮消息（置于 System 之后、当前问题之前）。"""
    messages: List[BaseMessage] = []
    for item in history:
        role = normalize_history_role(str(item.get("role") or "user"))
        content = str(item.get("content") or "")
        if not content:
            continue
        if role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages
