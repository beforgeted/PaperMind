"""Retrieval context collector — gathers evidence, feeds into chat_handler."""

from __future__ import annotations

from typing import Any

from app.agent.state import AgentState
from app.agent.tools.paper_serializers import chunk_to_result, result_to_source
from app.core.logging import logger
from app.services.papers.search import search_papers_by_query
from app.services.retrieval import _mqe_retrieve, _should_trigger_mqe, hybrid_retrieve


async def retrieval_handler(state: AgentState) -> dict[str, Any]:
    """Collect evidence from the knowledge base. chat_handler will generate the answer."""
    query = state.get("enriched_query") or state.get("query", "")
    original_query = state.get("query", "")
    top_k = state.get("top_k")
    task_id = state.get("task_id")

    contexts, sources, used_tools = await _collect_evidence(query, top_k, task_id)

    logger.info(
        "检索完成: tools={} | contexts={} | query={:.100}",
        used_tools, len(contexts), original_query,
    )

    return {
        "handler_contexts": contexts,
        "handler_sources": sources,
        "handler_used_tools": used_tools,
    }


async def _collect_evidence(
    query: str,
    top_k: int | None,
    task_id: str | None,
) -> tuple[list[dict], list[dict], list[str]]:
    """Retrieve evidence from knowledge base. Returns (contexts, sources, tools)."""
    used_tools: list[str] = []
    all_contexts: list[dict] = []

    # Attempt paper discovery
    try:
        rows = await search_papers_by_query(query=query, task_id=task_id)
        used_tools.append("search_paper_profiles")
        if rows:
            for row in rows:
                pid = getattr(row, "paper_id", "")
                title = getattr(row, "title", "")
                all_contexts.append({
                    "paper_id": pid, "title": title or "",
                    "content": getattr(row, "abstract", "") or "",
                })
            if rows:
                task_id = getattr(rows[0], "paper_id", None) or task_id
    except Exception:
        pass

    # First retrieval
    first_chunks = await hybrid_retrieve(query=query, top_k=top_k, task_id=task_id)
    used_tools.append("retrieve_evidence")

    # MQE fallback for weak retrieval
    if _should_trigger_mqe(query, first_chunks):
        chunks = await _mqe_retrieve(query, top_k=top_k, task_id=task_id)
        used_tools.append("multi_query_expansion")
        retrieve_contexts = [chunk_to_result(c) for c in chunks]
    else:
        retrieve_contexts = [chunk_to_result(c) for c in first_chunks]

    all_contexts = _merge_unique(all_contexts, retrieve_contexts)
    sources = [result_to_source(c) for c in all_contexts]

    return all_contexts, sources, used_tools


def _merge_unique(*lists: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    result: list[dict] = []
    for lst in lists:
        for item in lst:
            if not isinstance(item, dict):
                continue
            key = (item.get("paper_id"), item.get("parent_id"), item.get("title"))
            if key not in seen:
                seen.add(key)
                result.append(item)
    return result
