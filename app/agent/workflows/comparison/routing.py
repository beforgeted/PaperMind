"""Routes after coverage check."""

from __future__ import annotations

from app.agent.state import AgentState
from app.core.config import settings


def route_after_coverage(state: AgentState) -> str:
    """Retry retrieval up to max_retry, then compare even if coverage is weak."""
    report = state.get("coverage_report") or {}
    retry_count = int(state.get("retry_count") or 0)
    max_retry = settings.comparison_max_retry

    if report.get("ok"):
        return "compare_node"

    if retry_count < max_retry:
        return "query_rewrite_node"

    return "compare_node"
