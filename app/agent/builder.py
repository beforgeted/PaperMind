"""Assemble the LangGraph StateGraph.

Single-agent pipeline — ALL intents converge on chat_handler as the unified answer generator:
  intent_router → memory_recall
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   chat_handler   planner      (chat/writing/
   (for research  (for research  profile intents
    intents)       intents)      go direct)
        ▲             │             │
        │             ▼             │
        │        plan_validate      │
        │             │             │
        │    ┌────────┼────────┐    │
        │    ▼        ▼        ▼    │
        │ retrieval summary comparison
        │    │        │        │    │
        └────┴────────┴────────┘    │
                │                   │
                ▼                   │
           chat_handler ◄───────────┘
                │
                ▼
           synthesizer → END
"""

from __future__ import annotations

from typing import Any, Optional

from langgraph.graph import END, StateGraph

from app.agent.handlers import HANDLERS
from app.agent.routing.intent import intent_router_node
from app.agent.routing.memory_recall import memory_recall_node
from app.agent.planning.validator import plan_validate_node
from app.agent.planning.planner import planner_node
from app.agent.routing.routes import route_after_intent, route_by_plan_type
from app.agent.state import AgentState
from app.agent.workflows.comparison import build_comparison_subgraph
from app.agent.synthesizer import synthesizer_node

_graph: Optional[Any] = None


def build_agent_graph():
    """Build and compile the agent StateGraph."""
    builder = StateGraph(AgentState)  # type: ignore[arg-type]

    comparison_subgraph = build_comparison_subgraph()

    # Nodes
    builder.add_node("intent_router", intent_router_node)
    builder.add_node("memory_recall", memory_recall_node)
    builder.add_node("planner", planner_node)
    builder.add_node("plan_validate", plan_validate_node)
    builder.add_node("comparison_subgraph", comparison_subgraph)
    builder.add_node("chat_handler", HANDLERS["chat_handler"])
    builder.add_node("synthesizer", synthesizer_node)

    for name in ("retrieval_handler", "profile_handler", "summary_handler", "writing_handler"):
        builder.add_node(name, HANDLERS[name])

    # ── Edges ──────────────────────────────────────────────────────────

    builder.set_entry_point("intent_router")
    builder.add_edge("intent_router", "memory_recall")

    # After memory_recall: route by intent
    builder.add_conditional_edges(
        "memory_recall",
        route_after_intent,
        {
            "chat_handler": "chat_handler",          # chat → direct to answer
            "writing_handler": "writing_handler",     # writing → context collect → chat
            "profile_handler": "profile_handler",     # profile → context collect → chat
            "planner": "planner",                     # retrieval/summary/comparison → planner
        },
    )

    # Planner path: plan_validate → route to collector
    builder.add_edge("planner", "plan_validate")
    builder.add_conditional_edges(
        "plan_validate",
        route_by_plan_type,
        {
            "comparison_subgraph": "comparison_subgraph",
            "retrieval_handler": "retrieval_handler",
            "summary_handler": "summary_handler",
            "profile_handler": "profile_handler",
            "writing_handler": "writing_handler",
            "chat_handler": "chat_handler",
        },
    )

    # ALL handlers converge to chat_handler (unified answer generator)
    builder.add_edge("retrieval_handler", "chat_handler")
    builder.add_edge("summary_handler", "chat_handler")
    builder.add_edge("profile_handler", "chat_handler")
    builder.add_edge("writing_handler", "chat_handler")
    builder.add_edge("comparison_subgraph", "chat_handler")

    # chat_handler → synthesizer → END
    builder.add_edge("chat_handler", "synthesizer")
    builder.add_edge("synthesizer", END)

    return builder.compile()


def get_agent_graph():
    """Return the singleton compiled agent graph."""
    global _graph
    if _graph is None:
        _graph = build_agent_graph()
    return _graph
