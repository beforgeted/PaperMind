"""PaperMind research agent — LangGraph-based entrypoint.

Replaces the old black-box ReAct agent with explicit intent routing,
planning, and execution via LangGraph StateGraph.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch  # noqa: F401
except Exception:
    pass

from app.agent.builder import get_agent_graph
from app.agent.state import AgentState
from app.agent.session import build_user_query, is_task_status_query
from app.core.logging import logger


async def answer_with_agent(
    query: str,
    top_k: int | None = None,
    task_id: str | None = None,
    session_id: str = "",
) -> dict:
    """Answer a PaperMind question with the LangGraph research agent.

    Args:
        session_id: Redis session key for multi-turn conversation continuity.
                    Same session_id across requests restores full history.
    """
    # Pre-check: task status queries without task_id are rejected early
    if is_task_status_query(query) and not task_id:
        return {
            "answer": "请提供 task_id 后再查询论文解析状态。",
            "contexts": [],
            "sources": [],
            "used_tools": [],
            "raw_messages": [],
        }

    enriched_query = build_user_query(query, top_k, task_id)

    initial_state: AgentState = {
        "query": query,
        "enriched_query": enriched_query,
        "top_k": top_k,
        "task_id": task_id,
        "session_id": session_id,
        "messages": [],
    }

    try:
        graph = get_agent_graph()
        result = await graph.ainvoke(initial_state)
    except Exception as exc:
        error = str(exc)
        return {
            "answer": f"Agent 调用失败：{error}",
            "contexts": [],
            "sources": [],
            "used_tools": [],
            "raw_messages": [],
            "error": error,
        }

    # Auto-generate title if still default
    session_title = ""
    if session_id:
        try:
            from app.services.memory.session_store import get_session_store
            store = get_session_store()
            current_title = store._redis.hget(store.session_key(session_id), "title") or ""
            if not current_title or current_title == "新会话":
                session_title = await _auto_title(session_id, query)
        except Exception:
            pass

    return {
        "answer": result.get("final_answer", ""),
        "contexts": result.get("final_contexts", []),
        "sources": result.get("final_sources", []),
        "used_tools": result.get("used_tools", []),
        "raw_messages": result.get("raw_messages", []),
        "session_title": session_title,
    }


async def _auto_title(session_id: str, first_query: str) -> str:
    """Generate a session title from the first user query."""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from app.services.llm import get_llm
        llm = get_llm()
        title_raw = llm.invoke([
            SystemMessage(content="根据用户的第一个问题，生成一个简短的会话标题（不超过20字）。只输出标题，不要引号、解释或标点。"),
            HumanMessage(content=first_query),
        ])
        title = (title_raw.content if hasattr(title_raw, "content") else str(title_raw)).strip()[:50]
        if title:
            from app.services.memory.session_store import get_session_store
            store = get_session_store()
            store._redis.hset(store.session_key(session_id), "title", title)
            from app.services.memory.episodic import EpisodicMemory
            episodic = EpisodicMemory()
            doc = await episodic.get_session_doc(session_id)
            if doc:
                doc["title"] = title
                await episodic._es.index(index=episodic.config.episodic_es_index, id=session_id, body=doc, refresh=True)
            logger.info("自动标题: session={} title={}", session_id[:12], title)
            return title
    except Exception as exc:
        logger.warning("自动标题生成失败: {}", exc)
    return ""


async def stream_answer_with_agent(
    query: str,
    top_k: int | None = None,
    task_id: str | None = None,
    session_id: str = "",
) -> AsyncIterator[dict]:
    """Stream an agent answer via SSE-compatible events.

    Phase 1 (non-streamed): graph execution — intent routing, planning, retrieval
    Phase 2 (streamed): LLM answer generation, token by token
    Phase 3: metadata — contexts, sources, tools
    """
    if is_task_status_query(query) and not task_id:
        yield {"type": "done", "answer": "请提供 task_id 后再查询论文解析状态。", "contexts": [], "sources": [], "used_tools": []}
        return

    enriched_query = build_user_query(query, top_k, task_id)
    initial_state: AgentState = {
        "query": query,
        "enriched_query": enriched_query,
        "top_k": top_k,
        "task_id": task_id,
        "session_id": session_id,
        "messages": [],
    }

    logger.info("Agent stream start: session_id={} query={:.60}", session_id[:12] if session_id else "(none)", query)

    try:
        graph = get_agent_graph()
        result = await graph.ainvoke(initial_state)
    except Exception as exc:
        error = str(exc)
        yield {"type": "done", "answer": f"Agent 调用失败：{error}", "contexts": [], "sources": [], "used_tools": [], "error": error}
        return

    # Auto-generate title if still default (first turn of new session)
    if session_id:
        try:
            from app.services.memory.session_store import get_session_store
            store = get_session_store()
            current_title = store._redis.hget(store.session_key(session_id), "title") or ""
            if not current_title or current_title == "新会话":
                title = await _auto_title(session_id, query)
                if title:
                    yield {"type": "status", "phase": "title", "title": title, "session_id": session_id}
        except Exception:
            pass

    # Emit routing + planning metadata
    intent = result.get("intent", "retrieval")
    yield {
        "type": "status",
        "phase": "routing",
        "intent": intent,
        "intent_confidence": result.get("intent_confidence", 0),
        "routing_reason": result.get("routing_reason", ""),
    }
    plan = result.get("plan")
    if plan:
        yield {
            "type": "status",
            "phase": "planning",
            "plan_summary": result.get("plan_summary", ""),
            "plan": _format_plan_for_sse(plan),
            "plan_steps": _legacy_plan_steps_for_sse(plan),
        }

    trace = result.get("trace") or []
    if trace:
        yield {
            "type": "status",
            "phase": "trace",
            "trace": trace[-20:],
        }

    final_answer = result.get("final_answer", "")

    # chat_handler is now the unified answer generator for ALL intents.
    # All handlers feed their collected context into chat_handler, which produces
    # the single final answer. No second LLM call needed — use final_answer directly.
    accumulated = final_answer
    if accumulated:
        yield {"type": "delta", "text": accumulated}

    yield {
        "type": "done",
        "answer": accumulated,
        "contexts": result.get("final_contexts", []),
        "sources": result.get("final_sources", []),
        "used_tools": result.get("used_tools", []),
    }


def _format_plan_for_sse(plan: Any) -> dict[str, Any]:
    """Expose structured plan for comparison or legacy steps."""
    if isinstance(plan, dict):
        if plan.get("task_type") == "paper_comparison":
            return {
                "task_type": plan.get("task_type"),
                "targets": plan.get("targets", []),
                "aspects": [
                    {"name": a.get("name"), "evidence_query": a.get("evidence_query", "")[:80]}
                    for a in (plan.get("aspects") or [])
                    if isinstance(a, dict)
                ],
                "output_format": plan.get("output_format", "table"),
            }
        return {
            "task_type": plan.get("task_type"),
            "steps": plan.get("steps", []),
        }
    return {"steps": plan}


def _legacy_plan_steps_for_sse(plan: Any) -> list[dict[str, Any]]:
    """Backward-compatible plan_steps list for clients expecting step objects."""
    if isinstance(plan, dict):
        steps = plan.get("steps")
        if isinstance(steps, list) and steps:
            return [
                {
                    "step": s.get("step"),
                    "action": s.get("action"),
                    "description": s.get("description"),
                }
                for s in steps
                if isinstance(s, dict)
            ]
        targets = plan.get("targets") or []
        aspects = plan.get("aspects") or []
        out: list[dict[str, Any]] = []
        for i, t in enumerate(targets):
            if isinstance(t, dict):
                out.append(
                    {
                        "step": i + 1,
                        "action": "search_papers",
                        "description": f"Search {t.get('alias')}: {t.get('query')}",
                    }
                )
        base = len(out)
        for j, a in enumerate(aspects):
            if isinstance(a, dict):
                out.append(
                    {
                        "step": base + j + 1,
                        "action": "retrieve_evidence",
                        "description": f"Retrieve {a.get('name')} evidence",
                    }
                )
        if out:
            out.append(
                {"step": len(out) + 1, "action": "compare", "description": "Generate comparison"}
            )
        return out
    if isinstance(plan, list):
        return [
            {
                "step": s.get("step"),
                "action": s.get("action"),
                "description": s.get("description"),
            }
            for s in plan
            if isinstance(s, dict)
        ]
    return []


def _format_contexts_for_stream(contexts: list) -> str:
    """Format RetrievedChunk/dict contexts into a prompt-friendly string."""
    if not contexts:
        return "(无相关上下文)"
    blocks = []
    for i, ctx in enumerate(contexts):
        if not isinstance(ctx, dict):
            continue
        md = ctx.get("metadata") or {}
        label = (md.get("paper_id") or md.get("parent_id") or f"ctx-{i}")[:8]
        section = md.get("section_type") or md.get("section_title") or "unknown"
        text = ctx.get("parent_text") or ctx.get("content") or ""
        if text:
            blocks.append(f"[{label} | {section}] {text}")
    return "\n\n".join(blocks) if blocks else "(无相关上下文)"
