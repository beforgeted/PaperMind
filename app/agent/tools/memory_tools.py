"""Memory tools for PaperMind agents."""

from __future__ import annotations

import json
from typing import Any

from app.services.memory_store import get_memory_store
from app.agent.tools.contracts import error_response, json_response
from app.agent.tools.decorators import tool


def _parse_value(value: str) -> Any:
    """Parse JSON values when possible; otherwise keep the original string."""
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


@tool
def remember_fact(scope: str, key: str, value: str) -> str:
    """Store a fact in session, user, or project memory."""
    tool_name = "remember_fact"
    query = f"{scope}:{key}"
    try:
        entry = get_memory_store().remember(scope, key, _parse_value(value))
        return json_response(
            tool_name=tool_name,
            query=query,
            results=[entry],
            confidence=1.0,
        )
    except Exception as exc:  # noqa: BLE001
        return error_response(tool_name, query, exc)


@tool
def recall_memory(scope: str, query: str = "") -> str:
    """Recall facts from session, user, or project memory."""
    tool_name = "recall_memory"
    try:
        results = get_memory_store().recall(scope, query)
        return json_response(
            tool_name=tool_name,
            query=f"{scope}:{query}",
            results=results,
            confidence=1.0 if results else 0.0,
        )
    except Exception as exc:  # noqa: BLE001
        return error_response(tool_name, f"{scope}:{query}", exc)


@tool
def update_workspace_state(key: str, value: str) -> str:
    """Update current research workspace state for long-running tasks."""
    tool_name = "update_workspace_state"
    try:
        state = get_memory_store().update_workspace_state(key, _parse_value(value))
        return json_response(
            tool_name=tool_name,
            query=key,
            results=[{"workspace": state}],
            confidence=1.0,
        )
    except Exception as exc:  # noqa: BLE001
        return error_response(tool_name, key, exc)


@tool
def get_workspace_state() -> str:
    """Return the current research workspace state."""
    tool_name = "get_workspace_state"
    try:
        state = get_memory_store().get_workspace_state()
        return json_response(
            tool_name=tool_name,
            query="workspace",
            results=[{"workspace": state}],
            confidence=1.0 if state else 0.0,
        )
    except Exception as exc:  # noqa: BLE001
        return error_response(tool_name, "workspace", exc)


MEMORY_TOOLS = [
    remember_fact,
    recall_memory,
    update_workspace_state,
    get_workspace_state,
]
