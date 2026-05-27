"""Assemble the LangGraph StateGraph with all nodes and conditional edges."""

from __future__ import annotations

from typing import Any, Optional

from langgraph.graph import END, StateGraph

from app.agent.graph.handlers import HANDLERS
from app.agent.graph.intent_router import (
    intent_router_node,
    route_from_planner,
    route_from_router,
)
from app.agent.graph.planner import planner_node
from app.agent.graph.state import AgentState
from app.agent.graph.synthesizer import synthesizer_node

_graph: Optional[Any] = None


def build_agent_graph():
    """Build and compile the agent StateGraph (uncompiled)."""
    builder = StateGraph(AgentState)  # type: ignore[arg-type]

    # Nodes
    builder.add_node("intent_router", intent_router_node)
    builder.add_node("planner", planner_node)
    for name, handler in HANDLERS.items():
        builder.add_node(name, handler)
    builder.add_node("synthesizer", synthesizer_node)

    # Entry
    builder.set_entry_point("intent_router")

    # Router -> Planner or Handler
    handler_names = list(HANDLERS.keys())
    builder.add_conditional_edges(
        "intent_router",
        route_from_router,
        {"planner": "planner", **{h: h for h in handler_names}},
    )

    # Planner -> Handler
    builder.add_conditional_edges(
        "planner",
        route_from_planner,
        {h: h for h in handler_names},
    )

    # All handlers -> Synthesizer -> END
    for name in handler_names:
        builder.add_edge(name, "synthesizer")
    builder.add_edge("synthesizer", END)

    return builder.compile()


def get_agent_graph():
    """Return the singleton compiled agent graph."""
    global _graph
    if _graph is None:
        _graph = build_agent_graph()
    return _graph
