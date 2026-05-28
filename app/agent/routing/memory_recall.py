"""Memory recall node — restores conversation context from Redis session history.

Only pulls turns from the current session_id (no cross-session mixing).
Stores context in memory_context (for LLM prompts) without polluting enriched_query
(which is used for retrieval queries).
"""

from __future__ import annotations

from typing import Any

from app.agent.state import AgentState
from app.core.logging import logger
from app.services.memory.session_store import get_session_store


async def memory_recall_node(state: AgentState) -> dict[str, Any]:
    """Recall current session's conversation history from Redis."""
    query = state.get("query", "")
    if not query:
        return {"recalled_memories": {}, "memory_context": "", "messages": []}

    session_id = state.get("session_id", "")

    if session_id:
        try:
            store = get_session_store()
            store.ensure_session(session_id)
            history_msgs = store.get_history(session_id, max_turns=10, as_messages=True)
            if history_msgs:
                session_context = store.build_context_for_prompt(session_id, query)
                sample = history_msgs[0].get("content", "")[:80] if history_msgs else ""

                logger.info(
                    "会话恢复: session={} | turns={} | sample={!r} | context_len={}",
                    session_id[:12], len(history_msgs), sample, len(session_context),
                )
                return {
                    "recalled_memories": {},
                    "memory_context": session_context,
                    "messages": list(history_msgs),
                    "trace": [{"node": "memory_recall", "session_id": session_id[:12], "turns_restored": len(history_msgs)}],
                }
        except Exception as exc:
            logger.warning("Redis session history fetch failed: {}", exc)

    return {
        "recalled_memories": {},
        "memory_context": "",
        "messages": [],
        "trace": [{"node": "memory_recall", "turns_restored": 0}],
    }
