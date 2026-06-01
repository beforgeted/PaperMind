"""Memory recall node — restores conversation context from Redis session history.

Only pulls turns from the current session_id (no cross-session mixing).
Stores context in memory_context (for LLM prompts) without polluting enriched_query
(which is used for retrieval queries).
"""

from __future__ import annotations

from typing import Any

from app.agent.state import AgentState
from app.core.logging import logger
from app.services.memory.manager import get_memory_manager
from app.services.memory.session_store import get_session_store


async def memory_recall_node(state: AgentState) -> dict[str, Any]:
    """Recall current session's conversation history from Redis."""
    query = state.get("query", "")
    if not query:
        return {"recalled_memories": {}, "memory_context": "", "messages": []}

    session_id = state.get("session_id", "")
    recalled_memories: dict[str, list[dict[str, Any]]] = {}
    context_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []

    if session_id:
        try:
            store = get_session_store()
            store.ensure_session(session_id)
            history_msgs = store.get_history(session_id, max_turns=10, as_messages=True)
            if history_msgs:
                session_context = store.build_context_for_prompt(session_id, query)
                context_parts.append(session_context)
                messages = list(history_msgs)
                sample = history_msgs[0].get("content", "")[:80] if history_msgs else ""

                logger.info(
                    "会话恢复: session={} | turns={} | sample={!r} | context_len={}",
                    session_id[:12], len(history_msgs), sample, len(session_context),
                )
                trace.append({"node": "memory_recall", "session_id": session_id[:12], "turns_restored": len(history_msgs)})
        except Exception as exc:
            logger.warning("Redis session history fetch failed: {}", exc)

    try:
        manager = get_memory_manager()
        semantic_recall = await manager.recall(
            query=query,
            scope="user",
            include_working=False,
            include_semantic=True,
            include_episodic=False,
            session_id=session_id or None,
        )
        semantic_items = semantic_recall.get("semantic", [])
        if semantic_items:
            recalled_memories["semantic"] = [_memory_to_dict(item) for item in semantic_items]
            semantic_context = manager.build_context_prompt({"semantic": semantic_items})
            if semantic_context:
                context_parts.append(semantic_context)
            trace.append({"node": "memory_recall", "semantic_recalled": len(semantic_items)})
    except Exception as exc:
        logger.warning("Semantic memory recall skipped: {}", exc)

    return {
        "recalled_memories": recalled_memories,
        "memory_context": "\n\n".join(part for part in context_parts if part),
        "messages": messages,
        "trace": trace or [{"node": "memory_recall", "turns_restored": 0, "semantic_recalled": 0}],
    }


def _memory_to_dict(item: Any) -> dict[str, Any]:
    """Serialize MemoryItem-like objects for AgentState."""
    if isinstance(item, dict):
        return dict(item)
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json")
    return {
        "key": getattr(item, "key", ""),
        "content": getattr(item, "content", ""),
    }
