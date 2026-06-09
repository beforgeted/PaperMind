"""ContextBuilder 类型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from typing_extensions import TypedDict


class ContextStage(str, Enum):
    """LLM 调用阶段。"""

    ROUTER = "router"
    SUB_AGENT = "sub_agent"
    SUMMARY = "summary"


@dataclass
class AgentContextPolicy:
    """单个子 Agent 的上下文裁剪策略。"""

    history_turns: int = 0
    include_upstream: bool = False
    use_standalone_query: bool = True
    include_evidence_packets: bool = False


@dataclass
class SummaryContextPolicy:
    """汇总阶段的上下文裁剪策略。"""

    history_turns: int = 4
    include_evidence_packets: bool = True


class ConversationContext(TypedDict, total=False):
    """路由后写入 GraphState 的会话理解结果。"""

    standalone_query: str
    depends_on_history: bool
    resolved_entities: dict[str, str]
    context_requirements: dict[str, dict[str, Any]]


HistoryMessage = dict[str, str]

# Router 固定使用近 10 轮历史
ROUTER_HISTORY_TURNS = 10
