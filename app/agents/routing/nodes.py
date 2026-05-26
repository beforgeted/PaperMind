"""LangGraph nodes: pre_route, llm_route, execution, format_answer."""

from __future__ import annotations

from typing import Any

from app.agents.routing.handlers import execute_route_with_fallback
from app.agents.routing.llm_router import StructuredLLMRouter
from app.agents.routing.pre_router import EnhancedRuleRouter
from app.agents.routing.schema import RoutingState
from app.core.config import settings
from app.core.logging import logger


async def pre_route_node(state: RoutingState) -> dict:
    """Run rule-based pre-routing with confidence scoring."""
    router = EnhancedRuleRouter()
    result = router.route(state["query"])

    updates: dict[str, Any] = {
        "pre_route": result.route,
        "pre_confidence": result.confidence,
        "pre_reason": result.reason,
    }

    if result.confidence >= settings.rule_route_threshold:
        updates["final_route"] = result.route
        updates["route_confidence"] = result.confidence
        updates["route_reason"] = f"[rule] {result.reason}"
        updates["route_source"] = "rule"

    return updates


async def llm_route_node(state: RoutingState) -> dict:
    """Run structured LLM routing for ambiguous queries."""
    try:
        router = StructuredLLMRouter()
        pre_context = None
        pre_route = state.get("pre_route")
        pre_conf = state.get("pre_confidence", 0)
        if pre_route and pre_conf < settings.rule_route_threshold:
            pre_context = (
                f"Most likely route: {pre_route} "
                f"(confidence: {pre_conf:.2f}, reason: {state.get('pre_reason', '')})"
            )

        decision = await router.route(state["query"], pre_context)
        return {
            "llm_route": decision.route,
            "llm_confidence": decision.confidence,
            "llm_reason": decision.reason,
            "final_route": decision.route,
            "route_confidence": decision.confidence,
            "route_reason": f"[llm] {decision.reason}",
            "route_source": "llm",
        }
    except Exception as exc:
        fallback = state.get("pre_route") or settings.llm_route_fallback_route
        logger.exception("LLM routing failed for query={!r}: {}", state["query"], exc)
        return {
            "llm_route": None,
            "llm_confidence": 0.0,
            "llm_reason": "",
            "final_route": fallback,
            "route_confidence": 0.3,
            "route_reason": f"[fallback] LLM error, using {fallback}",
            "route_source": "fallback",
            "error": f"LLM routing failed: {exc}",
        }


async def execution_node(state: RoutingState) -> dict:
    """Dispatch to the shared route handler based on final_route."""
    return await execute_route_with_fallback(
        state.get("final_route"),
        state["query"],
        state.get("top_k"),
        state.get("task_id"),
    )


async def format_answer_node(state: RoutingState) -> dict:
    """Ensure answer is present; no-op for well-formed outputs."""
    if not state.get("answer"):
        return {"answer": "未能生成回答，请尝试更具体的查询。"}
    return {}
