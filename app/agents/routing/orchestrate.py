"""High-level orchestration helpers: route_and_answer, resolve_route."""

from __future__ import annotations

from typing import Optional

from app.agents.routing.decide import decide_route
from app.agents.routing.schema import RoutingState
from app.core.schemas import RoutingResult


async def resolve_route(
    query: str,
    top_k: Optional[int] = None,
    task_id: Optional[str] = None,
) -> RoutingResult:
    """Route-only: return the route decision without executing.

    Used by the streaming endpoint to decide which handler to invoke.
    """
    del top_k, task_id
    return await decide_route(query)


async def route_and_answer(
    query: str,
    top_k: Optional[int] = None,
    task_id: Optional[str] = None,
) -> dict:
    """Full pipeline: route → execute → format. Returns {answer, contexts, ...}."""
    from app.agents.routing.graph import get_routing_graph

    graph = get_routing_graph()
    initial: RoutingState = {
        "query": query,
        "top_k": top_k,
        "task_id": task_id,
        "pre_route": None,
        "pre_confidence": 0.0,
        "pre_reason": "",
        "llm_route": None,
        "llm_confidence": 0.0,
        "llm_reason": "",
        "final_route": "",
        "route_confidence": 0.0,
        "route_reason": "",
        "route_source": "rule",
        "answer": "",
        "contexts": [],
        "sources": [],
        "used_tools": [],
        "error": None,
    }
    result = await graph.ainvoke(initial)  # type: ignore[arg-type]
    return {
        "answer": result.get("answer", ""),
        "contexts": result.get("contexts", []),
        "sources": result.get("sources", []),
        "used_tools": result.get("used_tools", []),
        "route": result.get("final_route"),
        "route_confidence": result.get("route_confidence"),
        "route_reason": result.get("route_reason"),
        "route_source": result.get("route_source"),
        "error": result.get("error"),
    }
