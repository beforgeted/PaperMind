"""Memory tools for PaperMind agents — three-layer memory (working / semantic / episodic).

Inspired by Hello Agents Ch.8 memory architecture.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from app.agent.tools.contracts import error_response, json_response
from app.agent.tools.decorators import tool
from app.services.memory import get_memory_store
from app.services.memory.manager import MemoryManager

_memory_manager: MemoryManager | None = None


def _get_manager() -> MemoryManager:
    """Lazy singleton for the three-layer memory manager."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def _parse_value(value: str) -> Any:
    """Parse JSON values when possible; otherwise keep the original string."""
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


# ── Legacy tools (JsonMemoryStore, backward compat) ──────────────────────

@tool
def remember_fact(scope: str, key: str, value: str) -> str:
    """Store a fact in session, user, or project memory (JSON-backed)."""
    tool_name = "remember_fact"
    query = f"{scope}:{key}"
    try:
        entry = get_memory_store().remember(scope, key, _parse_value(value))
        return json_response(tool_name=tool_name, query=query, results=[entry], confidence=1.0)
    except Exception as exc:
        return error_response(tool_name, query, exc)


@tool
def recall_memory(scope: str, query: str = "") -> str:
    """Recall facts from session, user, or project memory (JSON-backed)."""
    tool_name = "recall_memory"
    try:
        results = get_memory_store().recall(scope, query)
        return json_response(tool_name=tool_name, query=f"{scope}:{query}", results=results, confidence=1.0 if results else 0.0)
    except Exception as exc:
        return error_response(tool_name, f"{scope}:{query}", exc)


@tool
def update_workspace_state(key: str, value: str) -> str:
    """Update current research workspace state for long-running tasks."""
    tool_name = "update_workspace_state"
    try:
        state = get_memory_store().update_workspace_state(key, _parse_value(value))
        return json_response(tool_name=tool_name, query=key, results=[{"workspace": state}], confidence=1.0)
    except Exception as exc:
        return error_response(tool_name, key, exc)


@tool
def get_workspace_state() -> str:
    """Return the current research workspace state."""
    tool_name = "get_workspace_state"
    try:
        state = get_memory_store().get_workspace_state()
        return json_response(tool_name=tool_name, query="workspace", results=[{"workspace": state}], confidence=1.0 if state else 0.0)
    except Exception as exc:
        return error_response(tool_name, "workspace", exc)


# ── New three-layer memory tools ────────────────────────────────────────

@tool
def search_memories(query: str, scope: str = "user", include_semantic: bool = True, include_episodic: bool = True) -> str:
    """Semantic search across all memory layers. Use to find past preferences, research context, or interaction history related to a topic."""
    tool_name = "search_memories"
    try:
        result = asyncio.run(_get_manager().recall(
            query=query,
            scope=scope,  # type: ignore[arg-type]
            include_working=True,
            include_semantic=include_semantic,
            include_episodic=include_episodic,
        ))
        total = sum(len(v) for v in result.values())
        context = _get_manager().build_context_prompt(result)
        return json_response(
            tool_name=tool_name,
            query=query,
            results=[{"layers": {k: len(v) for k, v in result.items()}, "context": context}],
            confidence=min(1.0, total * 0.2),
        )
    except Exception as exc:
        return error_response(tool_name, query, exc)


@tool
def consolidate_memories() -> str:
    """Move high-importance working memories to long-term semantic storage. Call periodically or after important interactions."""
    tool_name = "consolidate_memories"
    try:
        count = asyncio.run(_get_manager().consolidate())
        return json_response(
            tool_name=tool_name,
            query="consolidate",
            results=[{"consolidated_count": count}],
            confidence=1.0 if count > 0 else 0.0,
        )
    except Exception as exc:
        return error_response(tool_name, "consolidate", exc)


@tool
def forget_memories() -> str:
    """Clean up expired or stale memories across all layers."""
    tool_name = "forget_memories"
    try:
        result = asyncio.run(_get_manager().forget_expired())
        return json_response(
            tool_name=tool_name,
            query="forget",
            results=[result],
            confidence=1.0 if sum(result.values()) > 0 else 0.0,
        )
    except Exception as exc:
        return error_response(tool_name, "forget", exc)


@tool
def get_memory_stats() -> str:
    """Return statistics about the current memory system state."""
    tool_name = "get_memory_stats"
    try:
        stats = _get_manager().stats()
        return json_response(tool_name=tool_name, query="stats", results=[stats], confidence=1.0)
    except Exception as exc:
        return error_response(tool_name, "stats", exc)


MEMORY_TOOLS = [
    remember_fact,
    recall_memory,
    update_workspace_state,
    get_workspace_state,
    search_memories,
    consolidate_memories,
    forget_memories,
    get_memory_stats,
]

