"""Synthesizer node — deduplicate contexts/sources, format final answer, persist session turns."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

from app.agent.state import AgentState
from app.core.logging import logger
from app.shared.dedup import dedup_by_key, source_from_context, context_key, source_key


async def synthesizer_node(state: AgentState) -> dict[str, Any]:
    """Merge handler outputs into the final answer format. Persist turns to Redis + ES."""
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
            "plan": state.get("plan", {}),
            "trace": state.get("trace", []),
        }

    contexts = dedup_by_key(state.get("handler_contexts", []), key_fn=context_key)
    sources = dedup_by_key(state.get("handler_sources", []), key_fn=source_key)
    answer = state.get("handler_answer", "")
    used_tools = state.get("handler_used_tools", [])
    original_query = state.get("query", "")
    session_id = state.get("session_id", "")

    # Ensure sources include context-derived sources
    for ctx in contexts:
        src = source_from_context(ctx)
        sources = _add_source(sources, src)

    # Build raw_messages for API backward compatibility
    raw_messages: list[dict[str, Any]] = [
        {"type": "human", "name": None, "content": original_query},
        {"type": "ai", "name": "agent", "content": answer},
    ]
    for tool_name in used_tools:
        raw_messages.append({"type": "tool", "name": tool_name, "content": "{}"})

    # ── Persist turns to Redis (hot cache) + ES Episodic (permanent) ──
    if session_id:
        now = datetime.now(timezone.utc).isoformat()
        intent = state.get("intent", "")
        user_turn_id = uuid.uuid4().hex[:8]
        assistant_turn_id = uuid.uuid4().hex[:8]

        # 1. Redis hot cache (same-session last 10 turns)
        try:
            from app.services.memory.session_store import TurnRecord, get_session_store
            store = get_session_store()

            store.append_turn(session_id, TurnRecord(
                turn_id=user_turn_id,
                role="user",
                content=original_query[:2000],
                intent=intent,
                timestamp=now,
            ))
            store.append_turn(session_id, TurnRecord(
                turn_id=assistant_turn_id,
                role="assistant",
                content=answer[:4000] if answer else "",
                intent=intent,
                used_tools=list(used_tools),
                timestamp=now,
            ))
        except Exception as exc:
            logger.warning("Redis turn append failed: {}", exc)

        # 2. ES Episodic permanent storage (full original text)
        try:
            from app.services.memory.manager import get_memory_manager
            manager = get_memory_manager()

            user_turn = {
                "turn_id": user_turn_id,
                "role": "user",
                "content": original_query[:2000],
                "intent": intent,
                "used_tools": [],
                "timestamp": now,
            }
            await asyncio.to_thread(manager.episodic.append_turn, session_id, user_turn)

            assistant_turn = {
                "turn_id": assistant_turn_id,
                "role": "assistant",
                "content": answer[:4000] if answer else "",
                "intent": intent,
                "used_tools": list(used_tools),
                "timestamp": now,
            }
            await asyncio.to_thread(manager.episodic.append_turn, session_id, assistant_turn)
        except Exception:
            logger.debug("ES episodic turn append skipped (non-fatal)")

    result: dict[str, Any] = {
        "final_answer": answer,
        "final_contexts": contexts,
        "final_sources": sources,
        "used_tools": used_tools,
        "raw_messages": raw_messages,
        "intent": state.get("intent", ""),
        "intent_confidence": state.get("intent_confidence"),
        "routing_reason": state.get("routing_reason", ""),
        "plan_summary": state.get("plan_summary", ""),
        "plan": state.get("plan", {}),
        "trace": state.get("trace", []),
    }
    return result



def _add_source(sources: list[dict], new_src: dict) -> list[dict]:
    key = (new_src.get("paper_id"), new_src.get("title"))
    for s in sources:
        if (s.get("paper_id"), s.get("title")) == key:
            return sources
    if any(v for v in new_src.values() if v):
        return sources + [new_src]
    return sources
