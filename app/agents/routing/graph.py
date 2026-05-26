"""LangGraph StateGraph builder with conditional edges for 3-layer routing."""

from __future__ import annotations

from typing import Literal, Optional

from langgraph.graph import END, StateGraph

from app.agents.routing.nodes import (
    execution_node,
    format_answer_node,
    llm_route_node,
    pre_route_node,
)
from app.agents.routing.schema import RoutingState
from app.core.config import settings


def _should_use_llm(state: RoutingState) -> Literal["llm_route_node", "execution_node"]:
    if (
        state.get("route_source") == "rule"
        and state.get("route_confidence", 0) >= settings.rule_route_threshold
    ):
        return "execution_node"
    return "llm_route_node"


def _after_llm(state: RoutingState) -> Literal["execution_node", "__end__"]:
    if state.get("route_source") == "fallback" and state.get("error"):
        return END
    return "execution_node"


_graph: Optional["CompiledStateGraph"] = None


def build_routing_graph():
    """Build and return the routing StateGraph (uncompiled)."""
    builder = StateGraph(RoutingState)  # type: ignore[arg-type]

    builder.add_node("pre_route_node", pre_route_node)
    builder.add_node("llm_route_node", llm_route_node)
    builder.add_node("execution_node", execution_node)
    builder.add_node("format_answer_node", format_answer_node)

    builder.set_entry_point("pre_route_node")

    builder.add_conditional_edges(
        "pre_route_node",
        _should_use_llm,
        {
            "llm_route_node": "llm_route_node",
            "execution_node": "execution_node",
        },
    )

    builder.add_conditional_edges(
        "llm_route_node",
        _after_llm,
        {
            "execution_node": "execution_node",
            END: END,
        },
    )

    builder.add_edge("execution_node", "format_answer_node")
    builder.add_edge("format_answer_node", END)

    return builder.compile()


def get_routing_graph():
    """Return the singleton compiled routing graph."""
    global _graph
    if _graph is None:
        _graph = build_routing_graph()
    return _graph
