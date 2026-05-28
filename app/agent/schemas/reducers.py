"""State reducers for parallel LangGraph branches."""

from __future__ import annotations

import operator
from typing import Annotated


def merge_dict(left: dict | None, right: dict | None) -> dict:
    """Merge two dicts (parallel paper_search / resolve branches)."""
    result = dict(left or {})
    result.update(right or {})
    return result


def merge_evidence(left: list | None, right: list | None) -> list:
    """Merge evidence blocks with dedup by paper_id + aspect + parent_id."""
    result: list = []
    seen: set[tuple] = set()

    for item in (left or []) + (right or []):
        if not isinstance(item, dict):
            continue
        key = (
            item.get("paper_id"),
            item.get("aspect"),
            item.get("chunk_id") or item.get("parent_id"),
        )
        if key not in seen:
            seen.add(key)
            result.append(item)

    return result


# Re-export for Annotated fields in AgentState
trace_reducer = operator.add
