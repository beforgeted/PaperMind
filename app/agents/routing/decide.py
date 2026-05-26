"""Route decision without execution (shared by streaming API and orchestrate)."""

from __future__ import annotations

from app.agents.routing.llm_router import StructuredLLMRouter
from app.agents.routing.pre_router import EnhancedRuleRouter
from app.core.config import settings
from app.core.schemas import RoutingResult


async def decide_route(query: str) -> RoutingResult:
    """Return the final route for *query* using rule pre-router then LLM escalation."""
    pre_result = EnhancedRuleRouter().route(query)
    if pre_result.confidence >= settings.rule_route_threshold:
        return RoutingResult(
            route=pre_result.route,  # type: ignore[arg-type]
            confidence=pre_result.confidence,
            reason=f"[rule] {pre_result.reason}",
            source="rule",
        )

    try:
        pre_context = (
            f"Most likely route: {pre_result.route} "
            f"(confidence: {pre_result.confidence:.2f}, reason: {pre_result.reason})"
        )
        decision = await StructuredLLMRouter().route(query, pre_context)
        return RoutingResult(
            route=decision.route,
            confidence=decision.confidence,
            reason=f"[llm] {decision.reason}",
            source="llm",
        )
    except Exception:
        return RoutingResult(
            route=settings.llm_route_fallback_route,  # type: ignore[arg-type]
            confidence=0.3,
            reason="[fallback] LLM routing error",
            source="fallback",
        )
