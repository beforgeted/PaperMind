"""LangGraph RoutingState TypedDict shared across graph nodes."""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from typing_extensions import TypedDict


class RoutingState(TypedDict, total=False):
    # ---- Input (immutable) ----
    query: str
    top_k: Optional[int]
    task_id: Optional[str]

    # ---- Pre-router output ----
    pre_route: Optional[str]
    pre_confidence: float
    pre_reason: str

    # ---- LLM router output ----
    llm_route: Optional[str]
    llm_confidence: float
    llm_reason: str

    # ---- Resolved route ----
    final_route: str
    route_confidence: float
    route_reason: str
    route_source: Literal["rule", "llm", "fallback"]

    # ---- Execution output ----
    answer: str
    contexts: List[Any]
    sources: List[dict]
    used_tools: List[str]
    error: Optional[str]
