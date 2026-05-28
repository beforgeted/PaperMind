"""Send fan-out dispatchers for parallel paper search and evidence retrieval."""

from __future__ import annotations

from typing import Any

from langgraph.types import Send

from app.agent.state import AgentState
from app.core.logging import logger


def _plan_dict(state: AgentState) -> dict[str, Any]:
    plan = state.get("plan")
    return plan if isinstance(plan, dict) else {}


def dispatch_paper_search(state: AgentState):
    """Fan-out one search task per paper target."""
    plan = _plan_dict(state)
    targets = plan.get("targets") or []

    if not targets:
        logger.info("dispatch_paper_search: no targets, skip to resolve_papers_node")
        return "resolve_papers_node"

    sends = [
        Send("paper_search_node", {"target": target if isinstance(target, dict) else target})
        for target in targets
    ]
    logger.info("dispatch_paper_search: {} parallel searches", len(sends))
    return sends


def _missing_evidence_pairs(state: AgentState) -> list[tuple[str, str]]:
    """Return (alias, aspect_name) pairs that need more evidence."""
    missing = (state.get("coverage_report") or {}).get("missing") or []
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in missing:
        if not isinstance(item, dict):
            continue
        if item.get("reason") != "insufficient_evidence":
            continue
        alias = str(item.get("alias") or "")
        aspect_name = str(item.get("aspect") or "method")
        key = (alias, aspect_name)
        if alias and key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def dispatch_evidence_retrieval(state: AgentState):
    """Fan-out paper x aspect retrieval; on retry only missing pairs."""
    plan = _plan_dict(state)
    aspects = plan.get("aspects") or []
    aspect_by_name = {str(a.get("name")): a for a in aspects if isinstance(a, dict)}
    resolved = state.get("resolved_papers") or {}
    retry_count = int(state.get("retry_count") or 0)
    only_missing = retry_count > 0

    sends = []

    if only_missing:
        for alias, aspect_name in _missing_evidence_pairs(state):
            paper = resolved.get(alias)
            if not isinstance(paper, dict) or paper.get("status") != "resolved":
                continue
            aspect = aspect_by_name.get(aspect_name)
            if not aspect:
                aspect = {"name": aspect_name, "evidence_query": aspect_name}
            sends.append(Send("retrieve_evidence_node", {"alias": alias, "paper": paper, "aspect": aspect}))
        logger.info("dispatch_evidence_retrieval: retry only_missing | {} parallel retrievals", len(sends))
    else:
        for alias, paper in resolved.items():
            if not isinstance(paper, dict) or paper.get("status") != "resolved":
                continue
            for aspect in aspects:
                if not isinstance(aspect, dict):
                    continue
                sends.append(Send("retrieve_evidence_node", {"alias": alias, "paper": paper, "aspect": aspect}))
        logger.info("dispatch_evidence_retrieval: full | {} parallel retrievals", len(sends))

    if not sends:
        logger.warning("dispatch_evidence_retrieval: no sends, go to coverage_check")
        return "coverage_check_node"

    return sends
