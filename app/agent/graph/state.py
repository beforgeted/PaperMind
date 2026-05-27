"""AgentState TypedDict for the LangGraph research agent."""

from __future__ import annotations

from typing import Any, Optional

from typing_extensions import TypedDict


class PlanStep(TypedDict):
    step: int
    action: str
    description: str
    params: dict[str, Any]


class AgentState(TypedDict, total=False):
    # Input
    query: str
    enriched_query: str
    top_k: Optional[int]
    task_id: Optional[str]

    # Routing
    intent: str  # retrieval | comparison | summary | profile | writing | chat
    intent_confidence: float
    rule_matched: bool
    routing_reason: str

    # Planning (only for complex intents)
    plan: list[PlanStep]
    plan_summary: str

    # Handler output
    handler_answer: str
    handler_contexts: list[dict[str, Any]]
    handler_sources: list[dict[str, Any]]
    handler_used_tools: list[str]

    # Final output
    final_answer: str
    final_contexts: list[dict[str, Any]]
    final_sources: list[dict[str, Any]]
    used_tools: list[str]

    # Control
    error: Optional[str]
    raw_messages: list[dict[str, Any]]
