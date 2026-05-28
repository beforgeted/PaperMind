"""AgentState TypedDict for the LangGraph research agent."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional

from typing_extensions import TypedDict

from app.agent.schemas.reducers import merge_dict, merge_evidence
from app.agent.schemas.types import PlanStep


class AgentState(TypedDict, total=False):
    # Input
    query: str
    enriched_query: str
    top_k: Optional[int]
    task_id: Optional[str]
    session_id: str  # Redis session key for multi-turn continuity

    # Conversation history (accumulated across turns via Redis session)
    messages: Annotated[list[dict[str, Any]], operator.add]

    # Routing
    intent: str  # retrieval | comparison | summary | profile | writing | chat
    intent_confidence: float
    rule_matched: bool
    routing_reason: str

    # Planning — dict (AgentPlan) or legacy list[PlanStep]
    plan: dict[str, Any] | list[PlanStep]
    plan_summary: str
    plan_validated: bool

    # Comparison workflow (parallel reducers)
    paper_candidates: Annotated[dict[str, list[dict[str, Any]]], merge_dict]
    resolved_papers: Annotated[dict[str, dict[str, Any]], merge_dict]
    evidence_blocks: Annotated[list[dict[str, Any]], merge_evidence]
    coverage_report: dict[str, Any]
    retry_count: int
    trace: Annotated[list[dict[str, Any]], operator.add]

    # Memory system
    recalled_memories: dict[str, list[dict[str, Any]]]  # {working: [...], semantic: [...], episodic: [...]}
    memory_context: str  # pre-formatted context block for prompt injection

    # Per-Send branch payload (ephemeral; not merged globally)
    target: dict[str, Any]
    alias: str
    paper: dict[str, Any]
    aspect: dict[str, Any]

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
