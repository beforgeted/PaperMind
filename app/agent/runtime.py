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

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.agent.graph.graph_builder import get_agent_graph
from app.agent.graph.state import AgentState
from app.agent.prompts import build_system_prompt
from app.agent.session import build_user_query, is_task_status_query
from app.core.logging import logger
from app.services.llm_service import get_streaming_llm

SYSTEM_PROMPT = build_system_prompt()

_STREAM_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是科研论文问答助手。基于下方检索到的论文上下文回答用户问题，"
            "在引用论据时使用 [parent_id] 标注来源；若上下文不足以回答，请明确说明。"
            "回答应严谨、忠实于上下文，不要编造未在上下文中出现的事实。"
            "请使用 Markdown 格式组织回答，合理使用标题、列表、表格等结构，提高可读性。",
        ),
        ("human", "上下文：\n{context}\n\n问题：{question}"),
    ]
)


async def answer_with_agent(
    query: str,
    top_k: int | None = None,
    task_id: str | None = None,
) -> dict:
    """Answer a PaperMind question with the LangGraph research agent.

    Returns the same dict shape as the old ReAct agent for API compatibility.
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

    return {
        "answer": result.get("final_answer", ""),
        "contexts": result.get("final_contexts", []),
        "sources": result.get("final_sources", []),
        "used_tools": result.get("used_tools", []),
        "raw_messages": result.get("raw_messages", []),
    }


async def stream_answer_with_agent(
    query: str,
    top_k: int | None = None,
    task_id: str | None = None,
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
    }

    try:
        graph = get_agent_graph()
        result = await graph.ainvoke(initial_state)
    except Exception as exc:
        error = str(exc)
        yield {"type": "done", "answer": f"Agent 调用失败：{error}", "contexts": [], "sources": [], "used_tools": [], "error": error}
        return

    # Emit routing + planning metadata
    intent = result.get("intent", "retrieval")
    yield {
        "type": "status",
        "phase": "routing",
        "intent": intent,
        "intent_confidence": result.get("intent_confidence", 0),
        "routing_reason": result.get("routing_reason", ""),
    }
    plan = result.get("plan", [])
    if plan:
        yield {
            "type": "status",
            "phase": "planning",
            "plan_summary": result.get("plan_summary", ""),
            "plan_steps": [{"step": s.get("step"), "action": s.get("action"), "description": s.get("description")} for s in plan],
        }

    # Build context string from retrieved chunks
    contexts = result.get("final_contexts", [])
    context_text = _format_contexts_for_stream(contexts)

    # Stream token-by-token
    chain = _STREAM_QA_PROMPT | get_streaming_llm() | StrOutputParser()
    accumulated = ""
    try:
        async for token in chain.astream({"context": context_text, "question": query}):
            if token:
                accumulated += token
                yield {"type": "delta", "text": token}
    except Exception as exc:
        logger.warning("Streaming generation interrupted: {}", exc)
        if not accumulated:
            accumulated = result.get("final_answer", "")

    yield {
        "type": "done",
        "answer": accumulated,
        "contexts": contexts,
        "sources": result.get("final_sources", []),
        "used_tools": result.get("used_tools", []),
    }


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
