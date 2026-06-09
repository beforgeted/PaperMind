"""各 Agent 默认上下文策略与合并逻辑。"""

from __future__ import annotations

from typing import Any

from app.agent.context.types import AgentContextPolicy, SummaryContextPolicy


DEFAULT_SUB_AGENT_POLICIES: dict[str, AgentContextPolicy] = {
    "retrieval-agent": AgentContextPolicy(
        history_turns=0,
        include_upstream=False,
        use_standalone_query=True,
        include_evidence_packets=False,
    ),
    "writing-agent": AgentContextPolicy(
        history_turns=2,
        include_upstream=True,
        use_standalone_query=True,
        include_evidence_packets=False,
    ),
    "summary-agent": AgentContextPolicy(
        history_turns=0,
        include_upstream=True,
        use_standalone_query=False,
        include_evidence_packets=True,
    ),
    "profile-agent": AgentContextPolicy(
        history_turns=0,
        include_upstream=False,
        use_standalone_query=True,
        include_evidence_packets=False,
    ),
}

DEFAULT_SUMMARY_POLICY = SummaryContextPolicy(
    history_turns=4,
    include_evidence_packets=True,
)


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
    return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def merge_agent_policy(
    agent_id: str,
    override: dict[str, Any] | None = None,
) -> AgentContextPolicy:
    """合并默认策略与主 Agent 的 context_requirements 覆盖项。"""
    base = DEFAULT_SUB_AGENT_POLICIES.get(
        agent_id,
        AgentContextPolicy(),
    )
    if not override:
        return base

    return AgentContextPolicy(
        history_turns=_coerce_int(
            override.get("history_turns"),
            base.history_turns,
        ),
        include_upstream=_coerce_bool(
            override.get("include_upstream"),
            base.include_upstream,
        ),
        use_standalone_query=_coerce_bool(
            override.get("use_standalone_query"),
            base.use_standalone_query,
        ),
        include_evidence_packets=_coerce_bool(
            override.get("include_evidence_packets"),
            base.include_evidence_packets,
        ),
    )
