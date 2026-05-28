"""Central routing functions for the main agent graph."""

from __future__ import annotations

from app.agent.routing.intent import route_after_intent
from app.agent.schemas.plan import get_plan_task_type
from app.agent.state import AgentState
from app.core.logging import logger

__all__ = ["route_after_intent", "route_by_plan_type"]


def route_by_plan_type(state: AgentState) -> str:
    """Route to comparison subgraph or intent handlers based on plan.task_type."""
    plan = state.get("plan")
    task_type = get_plan_task_type(plan if isinstance(plan, dict) else None)

    if task_type == "paper_comparison":
        logger.info("route_by_plan_type: paper_comparison -> comparison_subgraph")
        return "comparison_subgraph"

    mapping = {
        "paper_qa": "retrieval_handler",
        "paper_profile": "profile_handler",
        "literature_summary": "summary_handler",
        "writing": "writing_handler",
        "chat": "chat_handler",
    }
    route = mapping.get(task_type)
    if route:
        logger.info("route_by_plan_type: {} -> {}", task_type, route)
        return route

    intent = state.get("intent", "retrieval")
    intent_fallback = {
        "retrieval": "retrieval_handler",
        "profile": "profile_handler",
        "summary": "summary_handler",
        "writing": "writing_handler",
        "chat": "chat_handler",
        "comparison": "comparison_subgraph",
    }
    route = intent_fallback.get(intent, "retrieval_handler")
    logger.info("route_by_plan_type: fallback intent={} -> {}", intent, route)
    return route
