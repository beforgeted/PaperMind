"""Shared utilities for agent graph handlers and nodes."""

from __future__ import annotations

from typing import Any, Callable, Sequence


def dedup_by_key(
    items: Sequence[dict[str, Any]] | None,
    key_fn: Callable[[dict[str, Any]], tuple],
) -> list[dict[str, Any]]:
    """Deduplicate a list of dicts by a computed key tuple. First occurrence wins."""
    if not items:
        return []
    seen: set[tuple] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        key = key_fn(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def merge_deduped(
    *lists: list[dict[str, Any]] | None,
    key_fn: Callable[[dict[str, Any]], tuple] | None = None,
) -> list[dict[str, Any]]:
    """Merge multiple context/source lists with dedup. First occurrence wins."""
    if key_fn is None:
        key_fn = _default_context_key
    combined: list[dict[str, Any]] = []
    for lst in lists:
        if lst:
            combined.extend(lst)
    return dedup_by_key(combined, key_fn=key_fn)


def _default_context_key(item: dict[str, Any]) -> tuple:
    return (
        item.get("paper_id"),
        item.get("parent_id"),
        item.get("title"),
    )


def context_key(item: dict[str, Any]) -> tuple:
    """Standard dedup key for handler contexts."""
    return _default_context_key(item)


def source_key(item: dict[str, Any]) -> tuple:
    """Dedup key for citation sources."""
    return (
        item.get("paper_id"),
        item.get("parent_id"),
        item.get("title"),
    )


def source_from_context(ctx: dict[str, Any]) -> dict[str, Any]:
    """Extract a source dict from a context dict."""
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
