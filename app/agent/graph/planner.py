"""Task planner node — LLM-driven decomposition for complex intents."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agent.graph.state import AgentState, PlanStep
from app.core.logging import logger
from app.services.llm_service import get_llm


class PlanOutput(BaseModel):
    plan: list[PlanStep] = Field(default_factory=list)
    summary: str = ""


_PLANNER_PROMPTS: dict[str, str] = {
    "retrieval": """You are a research question-answering planner. Given a question about a specific paper or research topic, create steps:

1. search: If the query mentions a specific paper by name, search for it using search_paper_profiles. Use the paper name as the search query.
2. retrieve: Retrieve detailed evidence from paper chunks using retrieve_evidence, targeting relevant sections (method, experiment, etc.).
3. answer: Synthesize the answer from retrieved evidence.

Return JSON: {"plan": [{"step":1,"action":"search_papers","description":"...","params":{"query":"<paper name or topic>"}}, ...], "summary":"..."}""",

    "comparison": """You are a research comparison planner. Given a query asking to compare papers, create steps:

1. search: Search for candidate papers matching the comparison topic
2. retrieve: Retrieve detailed evidence for each candidate
3. compare: Organize findings into a structured comparison table
4. recommend: Summarize differences and practical recommendations

Return JSON: {"plan": [{"step":1,"action":"search_papers","description":"...","params":{}}, ...], "summary":"..."}""",

    "summary": """You are a literature review planner. Given a review topic, create steps:

1. search: Search for papers on the topic
2. retrieve: Retrieve evidence from paper chunks in the knowledge base
3. outline: Generate a structured review outline with thematic organization
4. draft: Draft each review section thematically
5. review: Self-review the draft
6. polish: Polish the final text

Return JSON: {"plan": [{"step":1,"action":"search_papers","description":"...","params":{}}, ...], "summary":"..."}""",

    "writing": """You are an academic writing planner. Given a query about polishing or reviewing text, create steps:

1. analyze: Identify the section type and text to polish
2. load_skill: Load relevant writing skill (nature-polishing)
3. polish: Apply polishing rules following Nature editorial standards
4. review: Optionally perform peer review
5. output: Return polished text with revision notes

Return JSON: {"plan": [{"step":1,"action":"analyze","description":"...","params":{}}, ...], "summary":"..."}""",
}


def _minimal_plan(intent: str, query: str) -> list[PlanStep]:
    """Fallback plan when LLM fails."""
    if intent == "retrieval":
        return [
            {"step": 1, "action": "search_papers", "description": "Search for papers matching the query", "params": {"query": query}},
            {"step": 2, "action": "retrieve_evidence", "description": "Retrieve detailed evidence from matched papers", "params": {"query": query}},
            {"step": 3, "action": "answer", "description": "Generate answer from retrieved evidence", "params": {}},
        ]
    if intent == "comparison":
        return [
            {"step": 1, "action": "search_papers", "description": "Search for papers on comparison topic", "params": {"query": query}},
            {"step": 2, "action": "retrieve_evidence", "description": "Retrieve evidence for each paper", "params": {}},
            {"step": 3, "action": "compare", "description": "Generate comparison table", "params": {}},
        ]
    if intent == "summary":
        return [
            {"step": 1, "action": "search_papers", "description": "Search for papers", "params": {"query": query}},
            {"step": 2, "action": "retrieve_evidence", "description": "Retrieve detailed evidence", "params": {}},
            {"step": 3, "action": "generate_outline", "description": "Generate review outline", "params": {}},
            {"step": 4, "action": "draft_sections", "description": "Draft review sections", "params": {}},
        ]
    return [
        {"step": 1, "action": "polish", "description": "Polish the provided text", "params": {}},
    ]


async def planner_node(state: AgentState) -> dict[str, Any]:
    """Decompose a complex query into a step-by-step plan."""
    intent = state.get("intent", "retrieval")
    query = state.get("enriched_query") or state.get("query", "")
    original_query = state.get("query", "")

    prompt = _PLANNER_PROMPTS.get(intent)
    if prompt is None:
        plan = _minimal_plan(intent, query)
        logger.info(
            "任务拆解: intent={} | steps={} | source=minimal(no prompt) | query={:.100}",
            intent, len(plan), original_query,
        )
        _log_plan_steps(plan)
        return {
            "plan": plan,
            "plan_summary": f"Auto-generated {intent} plan (no planner prompt for this intent)",
        }
    llm = get_llm()
    try:
        # Qwen does not support native function_calling through LangChain.
        # Use plain LLM call with JSON output instruction instead.
        json_prompt = (
            f"{prompt}\n\n"
            "IMPORTANT: Output ONLY valid JSON, no markdown fences, no extra text.\n"
            "Format: {{\"plan\": [{{\"step\": 1, \"action\": \"...\", \"description\": \"...\", \"params\": {{}}}}, ...], \"summary\": \"...\"}}"
        )
        messages: list = [
            SystemMessage(content=json_prompt),
            HumanMessage(content=f"Query: {query}\nIntent: {intent}"),
        ]
        raw = await llm.ainvoke(messages)
        text = raw.content if hasattr(raw, "content") else str(raw)
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3]
        parsed = json.loads(text.strip())
        plan = parsed.get("plan", _minimal_plan(intent, query))
        summary = parsed.get("summary", "")
        logger.info(
            "任务拆解: intent={} | steps={} | source=llm | summary={:.80} | query={:.100}",
            intent, len(plan), summary, original_query,
        )
        _log_plan_steps(plan)
        return {"plan": plan, "plan_summary": summary}
    except Exception as exc:
        logger.info(
            "任务拆解: intent={} | source=fallback | reason={} | query={:.100}",
            intent, exc, original_query,
        )
        plan = _minimal_plan(intent, query)
        _log_plan_steps(plan)
        return {
            "plan": plan,
            "plan_summary": f"Auto-generated {intent} plan (LLM fallback)",
        }


def _log_plan_steps(plan: list) -> None:
    for step in plan:
        logger.info(
            "  Step {}: {} — {}",
            step.get("step", "?"),
            step.get("action", "?"),
            step.get("description", "?"),
        )
