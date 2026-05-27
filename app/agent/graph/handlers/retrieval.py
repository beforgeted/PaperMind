"""RAG QA handler — supports plan-driven, direct, and MQE-augmented retrieval paths."""

from __future__ import annotations

from typing import Any

from app.agent.graph.state import AgentState
from app.agent.tools.paper_serializers import chunk_to_result, result_to_source
from app.core.logging import logger
from app.services.paper_search_service import search_papers_by_query
from app.services.qa_service import answer as qa_answer
from app.services.retrieval_service import (
    _mqe_retrieve,
    _should_trigger_mqe,
    hybrid_retrieve,
)


async def retrieval_handler(state: AgentState) -> dict[str, Any]:
    """Handle retrieval queries, optionally following a planner-decomposed plan."""
    query = state.get("enriched_query") or state.get("query", "")
    original_query = state.get("query", "")
    top_k = state.get("top_k")
    task_id = state.get("task_id")
    plan: list[dict] = state.get("plan", [])

    if plan:
        logger.debug("retrieval_handler: 使用 plan 路径 (steps={}), query={:.100}", len(plan), original_query)
        result = await _execute_plan(plan, query, top_k, task_id)
    else:
        logger.debug("retrieval_handler: 使用 direct 路径, query={:.100}", original_query)
        result = await _direct_retrieve(query, top_k, task_id)

    logger.info(
        "检索完成: tools={} | contexts={} | query={:.100}",
        result.get("handler_used_tools", []),
        len(result.get("handler_contexts", [])),
        original_query,
    )
    return result


async def _execute_plan(
    plan: list[dict],
    query: str,
    top_k: int | None,
    task_id: str | None,
) -> dict[str, Any]:
    """Execute planner-decomposed steps: search_papers → retrieve_evidence → answer.

    After paper discovery, subsequent retrievals are scoped to the matched
    paper(s) via task_id filtering, preventing cross-paper contamination.
    """
    used_tools: list[str] = []
    paper_contexts: list[dict] = []
    discovered_paper_ids: list[str] = []
    discovered_paper_names: list[str] = []
    evidence_query = query

    for step in plan:
        action = step.get("action", "")
        params: dict = step.get("params", {}) if isinstance(step.get("params"), dict) else {}

        if action == "search_papers":
            search_query = params.get("query") or query
            try:
                rows = await search_papers_by_query(query=search_query, task_id=task_id)
                used_tools.append("search_paper_profiles")
                if rows:
                    for row in rows:
                        pid = getattr(row, "paper_id", "")
                        title = getattr(row, "title", "")
                        paper_contexts.append({
                            "paper_id": pid,
                            "title": title or "",
                            "content": getattr(row, "abstract", "") or "",
                        })
                        if pid:
                            discovered_paper_ids.append(str(pid))
                        if title:
                            discovered_paper_names.append(str(title))
                    logger.debug(
                        "Paper discovery found {} papers (ids={}) for query: {}",
                        len(rows), discovered_paper_ids, search_query,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Paper search failed (non-fatal): {}", exc)

        elif action == "retrieve_evidence":
            evidence_query = params.get("query") or query

    # Scope retrieval to discovered paper(s) to avoid cross-paper contamination
    filter_task_id: str | None = None
    if discovered_paper_ids:
        filter_task_id = discovered_paper_ids[0]  # top-match paper
        logger.info(
            "检索范围限定: paper_id={} ({} papers found)",
            filter_task_id, len(discovered_paper_ids),
        )
    elif task_id:
        filter_task_id = task_id

    # First retrieval — scoped to the matched paper
    qa_result = await qa_answer(query=evidence_query, top_k=top_k, task_id=filter_task_id)
    first_chunks = await hybrid_retrieve(query=evidence_query, top_k=top_k, task_id=filter_task_id)

    # Conditionally trigger MQE when first retrieval is weak
    if _should_trigger_mqe(evidence_query, first_chunks):
        logger.debug("MQE triggered for plan query: {}", evidence_query[:80])
        chunks = await _mqe_retrieve(evidence_query, top_k=top_k, task_id=filter_task_id)
        used_tools.append("multi_query_expansion")
        retrieve_contexts_mqe = [chunk_to_result(c) for c in chunks]
        combined_mqe = _merge_contexts(paper_contexts, list(qa_result.get("contexts", [])))
        combined_mqe = _merge_contexts(combined_mqe, retrieve_contexts_mqe)

        enriched_answer = await qa_answer(query=evidence_query, top_k=top_k, task_id=filter_task_id)
        return {
            "handler_answer": enriched_answer.get("answer", qa_result.get("answer", "")),
            "handler_contexts": combined_mqe,
            "handler_sources": [result_to_source(c) for c in combined_mqe],
            "handler_used_tools": used_tools + ["answer_with_rag", "retrieve_evidence"],
        }

    used_tools.extend(["answer_with_rag", "retrieve_evidence"])
    qa_contexts = list(qa_result.get("contexts", []))
    retrieve_contexts = [chunk_to_result(c) for c in first_chunks]
    combined = _merge_contexts(paper_contexts, qa_contexts)
    combined = _merge_contexts(combined, retrieve_contexts)

    return {
        "handler_answer": qa_result.get("answer", ""),
        "handler_contexts": combined,
        "handler_sources": [result_to_source(c) for c in combined],
        "handler_used_tools": used_tools,
    }


async def _direct_retrieve(
    query: str,
    top_k: int | None,
    task_id: str | None,
) -> dict[str, Any]:
    """Direct retrieval with conditional MQE fallback."""
    # First retrieval
    qa_result = await qa_answer(query=query, top_k=top_k, task_id=task_id)
    first_chunks = await hybrid_retrieve(query=query, top_k=top_k, task_id=task_id)

    # Conditionally trigger MQE when first retrieval is weak
    if _should_trigger_mqe(query, first_chunks):
        logger.debug("MQE triggered for query: {}", query[:80])
        chunks = await _mqe_retrieve(query, top_k=top_k, task_id=task_id)
        retrieve_contexts = [chunk_to_result(c) for c in chunks]
        combined = _merge_contexts(list(qa_result.get("contexts", [])), retrieve_contexts)

        # Re-generate answer with expanded contexts for better coverage
        enriched_answer = await qa_answer(query=query, top_k=top_k, task_id=task_id)
        return {
            "handler_answer": enriched_answer.get("answer", qa_result.get("answer", "")),
            "handler_contexts": combined,
            "handler_sources": [result_to_source(c) for c in combined],
            "handler_used_tools": ["answer_with_rag", "retrieve_evidence", "multi_query_expansion"],
        }

    qa_contexts = qa_result.get("contexts", [])
    retrieve_contexts = [chunk_to_result(c) for c in first_chunks]
    combined = _merge_contexts(qa_contexts, retrieve_contexts)

    return {
        "handler_answer": qa_result.get("answer", ""),
        "handler_contexts": combined,
        "handler_sources": [result_to_source(c) for c in combined],
        "handler_used_tools": ["answer_with_rag", "retrieve_evidence"],
    }


def _merge_contexts(qa_contexts: list[Any], retrieve_contexts: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    result: list[dict] = []
    for item in list(qa_contexts) + retrieve_contexts:
        if not isinstance(item, dict):
            continue
        key = (item.get("paper_id"), item.get("parent_id"), item.get("title"))
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result
