"""Synthesizer node — deduplicate contexts/sources, format final answer."""

from __future__ import annotations

from typing import Any

from app.agent.graph.state import AgentState


async def synthesizer_node(state: AgentState) -> dict[str, Any]:
    """Merge handler outputs into the final answer format."""
    if state.get("error"):
        return {
            "final_answer": f"处理出错：{state['error']}",
            "final_contexts": [],
            "final_sources": [],
            "used_tools": state.get("handler_used_tools", []),
            "raw_messages": [],
            "error": state["error"],
            "intent": state.get("intent", ""),
            "intent_confidence": state.get("intent_confidence"),
            "routing_reason": state.get("routing_reason", ""),
            "plan_summary": state.get("plan_summary", ""),
            "plan": state.get("plan", []),
        }

    contexts = _dedup_contexts(state.get("handler_contexts", []))
    sources = _dedup_sources(state.get("handler_sources", []))
    answer = state.get("handler_answer", "")
    used_tools = state.get("handler_used_tools", [])
    original_query = state.get("query", "")

    # Ensure sources include context-derived sources
    for ctx in contexts:
        src = _source_from_context(ctx)
        sources = _add_source(sources, src)

    # Build raw_messages for API backward compatibility
    raw_messages: list[dict[str, Any]] = [
        {"type": "human", "name": None, "content": original_query},
        {"type": "ai", "name": "agent", "content": answer},
    ]
    for tool_name in used_tools:
        raw_messages.append({"type": "tool", "name": tool_name, "content": "{}"})

    return {
        "final_answer": answer,
        "final_contexts": contexts,
        "final_sources": sources,
        "used_tools": used_tools,
        "raw_messages": raw_messages,
        # Pass through routing/planning metadata for debug visibility
        "intent": state.get("intent", ""),
        "intent_confidence": state.get("intent_confidence"),
        "routing_reason": state.get("routing_reason", ""),
        "plan_summary": state.get("plan_summary", ""),
        "plan": state.get("plan", []),
    }


def _ctx_key(item: dict) -> tuple:
    return (
        item.get("paper_id"),
        item.get("parent_id"),
        item.get("title"),
    )


def _dedup_contexts(contexts: list[dict] | None) -> list[dict]:
    if not contexts or not isinstance(contexts, list):
        return []
    seen: set[tuple] = set()
    result: list[dict] = []
    for ctx in contexts:
        if not isinstance(ctx, dict):
            continue
        key = _ctx_key(ctx)
        if key not in seen:
            seen.add(key)
            result.append(ctx)
    return result


def _dedup_sources(sources: list[dict] | None) -> list[dict]:
    if not sources or not isinstance(sources, list):
        return []
    seen: set[tuple] = set()
    result: list[dict] = []
    for src in sources:
        if not isinstance(src, dict):
            continue
        key = (src.get("paper_id"), src.get("parent_id"), src.get("title"))
        if key not in seen:
            seen.add(key)
            result.append(src)
    return result


def _source_from_context(ctx: dict) -> dict:
    return {
        "paper_id": ctx.get("paper_id"),
        "title": ctx.get("title"),
        "section": ctx.get("section"),
        "section_type": ctx.get("section_type"),
        "chunk_ids": ctx.get("chunk_ids") or [],
        "parent_id": ctx.get("parent_id"),
        "score": ctx.get("score"),
        "metadata": ctx.get("metadata") or {},
    }


def _add_source(sources: list[dict], new_src: dict) -> list[dict]:
    key = (new_src.get("paper_id"), new_src.get("title"))
    for s in sources:
        if (s.get("paper_id"), s.get("title")) == key:
            return sources
    if any(v for v in new_src.values() if v):
        return sources + [new_src]
    return sources
