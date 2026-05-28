"""Task planner node — structured AgentPlan output per intent.

Only retrieval, summary, and comparison intents reach this node.
Chat, writing, and profile are routed directly to their handlers.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.schemas.plan import (
    intent_to_task_type,
    minimal_agent_plan,
    parse_plan_from_llm,
)
from app.agent.state import AgentState
from app.core.logging import logger
from app.services.llm import get_llm


_COMPARISON_PLANNER_PROMPT = """You are a research paper comparison planner.

Parse the user question into a structured comparison plan. Output ONLY valid JSON.

Required JSON shape:
{
  "task_type": "paper_comparison",
  "summary": "brief plan summary",
  "targets": [
    {"alias": "A", "query": "<paper A name or keyword>", "required": true},
    {"alias": "B", "query": "<paper B name or keyword>", "required": true}
  ],
  "aspects": [
    {
      "name": "method",
      "evidence_query": "method architecture framework 方法 网络结构",
      "section_types": ["method", "approach", "model"]
    }
  ],
  "output_format": "table",
  "discovery_query": ""
}

Rules:
1. Extract each paper to compare as a separate target with its own query string.
2. Do NOT invent paper titles not mentioned or implied by the user.
3. If the user compares papers on a topic without naming them, leave targets [] and set discovery_query to the topic.
4. aspects: include method, experiment, innovation, etc. based on what the user asks to compare.
5. aspect name must be one of: method, architecture, module, loss, experiment, dataset, metric, innovation, limitation."""

_GENERIC_PLANNER_PROMPTS: dict[str, str] = {
    "retrieval": """You are a research QA planner. Output JSON:
{"task_type": "paper_qa", "summary": "...", "steps": [
  {"step": 1, "action": "search_papers", "description": "...", "params": {"query": "<paper or topic>"}},
  {"step": 2, "action": "retrieve_evidence", "description": "...", "params": {"query": "..."}}
]}
Note: answer generation is automatic after evidence retrieval — do NOT include an "answer" step.""",
    "summary": """Literature review planner. Output JSON:
{"task_type": "literature_summary", "summary": "...", "steps": [
  {"step": 1, "action": "search_papers", "description": "...", "params": {}},
  {"step": 2, "action": "retrieve_evidence", "description": "...", "params": {}},
  {"step": 3, "action": "generate_outline", "description": "...", "params": {}}
]}""",
}


async def planner_node(state: AgentState) -> dict[str, Any]:
    """Produce a structured AgentPlan dict in state['plan']."""
    intent = state.get("intent", "retrieval")
    query = state.get("enriched_query") or state.get("query", "")
    original_query = state.get("query", "")

    if intent == "comparison":
        prompt = _COMPARISON_PLANNER_PROMPT
    else:
        prompt = _GENERIC_PLANNER_PROMPTS.get(intent)
        if prompt is None:
            plan = minimal_agent_plan(intent, query)
            logger.info(
                "任务拆解: intent={} | source=minimal | query={:.100}",
                intent,
                original_query,
            )
            return {
                "plan": plan.model_dump(mode="json"),
                "plan_summary": plan.summary,
                "trace": [{"node": "planner", "source": "minimal", "task_type": plan.task_type}],
            }

    llm = get_llm()
    try:
        json_prompt = (
            f"{prompt}\n\n"
            "IMPORTANT: Output ONLY valid JSON, no markdown fences, no extra text."
        )
        messages: list = [
            SystemMessage(content=json_prompt),
            HumanMessage(content=f"Query: {query}\nIntent: {intent}"),
        ]
        raw = await llm.ainvoke(messages)
        text = raw.content if hasattr(raw, "content") else str(raw)
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3]
        parsed = json.loads(text.strip())
        plan = parse_plan_from_llm(intent, parsed, original_query)
        if not plan.task_type:
            plan.task_type = intent_to_task_type(intent)

        logger.info(
            "任务拆解: intent={} | task_type={} | source=llm | query={:.100}",
            intent,
            plan.task_type,
            original_query,
        )
        return {
            "plan": plan.model_dump(mode="json"),
            "plan_summary": plan.summary or f"{plan.task_type} plan",
            "trace": [{"node": "planner", "source": "llm", "task_type": plan.task_type}],
        }
    except Exception as exc:
        logger.info(
            "任务拆解: intent={} | source=fallback | reason={} | query={:.100}",
            intent,
            exc,
            original_query,
        )
        plan = minimal_agent_plan(intent, original_query)
        return {
            "plan": plan.model_dump(mode="json"),
            "plan_summary": plan.summary,
            "trace": [{"node": "planner", "source": "fallback", "task_type": plan.task_type}],
        }
