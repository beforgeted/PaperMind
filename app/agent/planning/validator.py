"""Validate and normalize structured plans before execution."""

from __future__ import annotations

from typing import Any

from app.agent.schemas.plan import (
    DEFAULT_METHOD_ASPECT,
    AgentPlan,
    CompareAspect,
    PaperTarget,
    aspect_from_name,
    get_plan_task_type,
    minimal_agent_plan,
    parse_plan_from_llm,
)
from app.agent.state import AgentState
from app.core.config import settings
from app.core.logging import logger


def _plan_from_state(state: AgentState) -> AgentPlan:
    raw = state.get("plan")
    intent = state.get("intent", "retrieval")
    query = state.get("query", "")

    if isinstance(raw, dict):
        try:
            return AgentPlan.model_validate(raw)
        except Exception:
            return parse_plan_from_llm(intent, raw, query)
    if isinstance(raw, list):
        return AgentPlan(
            task_type=get_plan_task_type({}),
            summary=state.get("plan_summary", ""),
            steps=[dict(s) if isinstance(s, dict) else s for s in raw],
        )
    return minimal_agent_plan(intent, query)


def _degrade_comparison_if_unexecutable(plan: AgentPlan, query: str) -> tuple[AgentPlan, str | None]:
    """Downgrade comparison to paper_qa when no targets and no discovery topic."""
    if plan.task_type != "paper_comparison":
        return plan, None
    has_targets = bool(plan.targets)
    has_discovery = bool((plan.discovery_query or "").strip())
    if has_targets or has_discovery:
        return plan, None
    fallback = minimal_agent_plan("retrieval", query)
    fallback.task_type = "paper_qa"
    fallback.summary = "Comparison plan had no targets; degraded to paper QA"
    logger.info("plan_validate: degrading paper_comparison -> paper_qa (no targets/discovery)")
    return fallback, "comparison_degraded_to_paper_qa"


async def plan_validate_node(state: AgentState) -> dict[str, Any]:
    """Normalize plan fields and set discovery / default aspects for comparison."""
    intent = state.get("intent", "retrieval")
    query = state.get("query", "")
    degrade_reason: str | None = None

    try:
        plan = _plan_from_state(state)
    except Exception as exc:
        logger.warning("plan_validate failed, using minimal plan: {}", exc)
        if intent == "chat":
            plan = minimal_agent_plan("chat", query)
            degrade_reason = "parse_failed_chat_fallback"
        else:
            plan = minimal_agent_plan(intent, query)
            degrade_reason = "parse_failed_minimal_fallback"

    if intent == "comparison" or plan.task_type == "paper_comparison":
        plan = _normalize_comparison_plan(plan, query)
        plan, comp_degrade = _degrade_comparison_if_unexecutable(plan, query)
        if comp_degrade:
            degrade_reason = comp_degrade

    plan_dict = plan.model_dump(mode="json")
    trace_entry: dict[str, Any] = {
        "node": "plan_validate",
        "task_type": plan.task_type,
        "target_count": len(plan.targets),
        "aspect_count": len(plan.aspects),
    }
    if degrade_reason:
        trace_entry["degrade_reason"] = degrade_reason

    logger.info(
        "plan_validate: task_type={} | targets={} | aspects={}",
        plan.task_type,
        len(plan.targets),
        len(plan.aspects),
    )

    return {
        "plan": plan_dict,
        "plan_validated": True,
        "trace": [trace_entry],
    }


def _normalize_comparison_plan(plan: AgentPlan, query: str) -> AgentPlan:
    """Ensure comparison plans are executable."""
    plan.task_type = "paper_comparison"

    if not plan.targets:
        from app.agent.schemas.plan import extract_comparison_targets_from_query

        plan.targets = extract_comparison_targets_from_query(query)
        if not plan.targets:
            plan.discovery_query = plan.discovery_query or query

    if len(plan.targets) > settings.comparison_max_targets:
        plan.targets = plan.targets[: settings.comparison_max_targets]

    aliases = ["A", "B", "C", "D", "E"]
    normalized_targets: list[PaperTarget] = []
    for i, target in enumerate(plan.targets):
        alias = target.alias or aliases[min(i, 4)]
        if not target.query.strip():
            continue
        normalized_targets.append(
            PaperTarget(alias=alias, query=target.query.strip(), required=target.required)
        )
    plan.targets = normalized_targets

    if not plan.aspects:
        plan.aspects = [DEFAULT_METHOD_ASPECT.model_copy()]

    fixed_aspects: list[CompareAspect] = []
    for asp in plan.aspects:
        preset = aspect_from_name(asp.name)
        fixed_aspects.append(
            CompareAspect(
                name=preset.name,
                evidence_query=asp.evidence_query or preset.evidence_query,
                section_types=asp.section_types or list(preset.section_types),
            )
        )
    plan.aspects = fixed_aspects

    return plan
