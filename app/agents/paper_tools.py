"""LangChain/LangGraph tools backed by PaperMind retrieval services."""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.tools import tool

from app.core.schemas import PaperProfile, PaperSearchResult, RetrievedChunk, TaskRecord
from app.services.paper_index_service import get_paper_profile as fetch_paper_profile
from app.services.paper_index_service import search_papers
from app.services.paper_search_service import (
    deep_search_papers_by_query,
    search_papers_by_query,
)
from app.services.retrieval_service import hybrid_retrieve
from app.services.task_status import get_task

_MAX_CONTENT_CHARS = 1500


def _truncate(value: Any, max_chars: int = _MAX_CONTENT_CHARS) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...(truncated)"


def _json_response(
    *,
    tool_name: str,
    query: str,
    results: list[dict],
    error: Optional[str] = None,
) -> str:
    payload = {
        "tool_name": tool_name,
        "query": query,
        "result_count": len(results),
        "results": results,
        "error": error,
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def _error_response(tool_name: str, query: str, exc: Exception) -> str:
    return _json_response(
        tool_name=tool_name,
        query=query,
        results=[],
        error=str(exc),
    )


def _chunk_to_result(chunk: RetrievedChunk) -> dict:
    md = chunk.metadata or {}
    return {
        "paper_id": md.get("paper_id"),
        "title": md.get("title"),
        "section": md.get("section_title"),
        "section_type": md.get("section_type"),
        "chunk_ids": chunk.child_ids or [],
        "parent_id": chunk.parent_id,
        "score": float(chunk.score or 0.0),
        "content": _truncate(chunk.parent_text),
        "metadata": md,
    }


def _profile_to_result(profile: PaperProfile) -> dict:
    metadata = profile.model_dump(mode="json")
    content_parts = [
        profile.summary,
        profile.abstract_summary,
        profile.method_summary,
        profile.contribution_summary,
        profile.experiment_summary,
    ]
    content = "\n".join(part for part in content_parts if part).strip()
    return {
        "paper_id": profile.paper_id,
        "title": profile.title,
        "section": "paper_profile",
        "section_type": "paper_profile",
        "chunk_ids": [],
        "parent_id": None,
        "score": 1.0,
        "content": _truncate(content or profile.abstract or profile.paper_search_text),
        "metadata": metadata,
    }


def _paper_search_result_to_result(item: PaperSearchResult) -> dict:
    evidence = "\n".join(item.evidence_chunks or [])
    content_parts = [
        item.summary,
        item.abstract_summary,
        item.method_summary,
        item.contribution_summary,
        evidence,
    ]
    content = "\n".join(part for part in content_parts if part).strip()
    return {
        "paper_id": item.paper_id,
        "title": item.title or item.source_file,
        "section": "paper_profile",
        "section_type": "paper_profile",
        "chunk_ids": [],
        "parent_id": None,
        "score": float(item.score or 0.0),
        "content": _truncate(content),
        "metadata": item.model_dump(mode="json"),
    }


def _task_to_result(task: TaskRecord) -> dict:
    metadata = task.model_dump(mode="json")
    return {
        "paper_id": None,
        "title": task.original_filename,
        "section": "task_status",
        "section_type": "task_status",
        "chunk_ids": [],
        "parent_id": None,
        "score": 1.0,
        "content": _truncate(task.message or task.error or ""),
        "metadata": metadata,
    }


def _find_profile_by_title(title: str) -> Optional[PaperProfile]:
    if not title.strip():
        return None
    rows = search_papers(
        {
            "query": {
                "multi_match": {
                    "query": title,
                    "fields": ["title^4", "source_file^2", "paper_search_text"],
                }
            }
        },
        size=1,
    )
    if not rows:
        return None
    return fetch_paper_profile(rows[0].paper_id)


@tool
async def search_paper_chunks(
    query: str,
    top_k: Optional[int] = None,
    task_id: Optional[str] = None,
) -> str:
    """Search fine-grained paper content for methods, experiments, metrics, datasets, ablations, and module design evidence."""
    tool_name = "search_paper_chunks"
    try:
        chunks = await hybrid_retrieve(query=query, top_k=top_k, task_id=task_id)
        return _json_response(
            tool_name=tool_name,
            query=query,
            results=[_chunk_to_result(chunk) for chunk in chunks],
        )
    except Exception as exc:  # noqa: BLE001
        return _error_response(tool_name, query, exc)


@tool
async def search_paper_profiles(
    query: str,
    task_id: Optional[str] = None,
) -> str:
    """Search paper-level profiles for paper discovery, filtering, and recommendation questions."""
    tool_name = "search_paper_profiles"
    try:
        results = await search_papers_by_query(query=query, task_id=task_id)
        return _json_response(
            tool_name=tool_name,
            query=query,
            results=[_paper_search_result_to_result(item) for item in results],
        )
    except Exception as exc:  # noqa: BLE001
        return _error_response(tool_name, query, exc)


@tool
async def deep_search_papers(
    query: str,
    task_id: Optional[str] = None,
) -> str:
    """Search paper-level profiles and supporting evidence for questions asking which papers match and why."""
    tool_name = "deep_search_papers"
    try:
        results = await deep_search_papers_by_query(query=query, task_id=task_id)
        return _json_response(
            tool_name=tool_name,
            query=query,
            results=[_paper_search_result_to_result(item) for item in results],
        )
    except Exception as exc:  # noqa: BLE001
        return _error_response(tool_name, query, exc)


@tool
def get_paper_profile(
    paper_id: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """Get one paper profile by paper_id first, or by title as a lightweight fallback."""
    tool_name = "get_paper_profile"
    query = paper_id or title or ""
    try:
        profile = fetch_paper_profile(paper_id) if paper_id else None
        if profile is None and title:
            profile = _find_profile_by_title(title)
        results = [_profile_to_result(profile)] if profile else []
        return _json_response(tool_name=tool_name, query=query, results=results)
    except Exception as exc:  # noqa: BLE001
        return _error_response(tool_name, query, exc)


@tool
def get_task_status(task_id: str) -> str:
    """Get upload, parsing, chunking, embedding, and indexing status by task_id."""
    tool_name = "get_task_status"
    try:
        task = get_task(task_id)
        results = [_task_to_result(task)] if task else []
        error = None if task else f"Task {task_id} not found."
        return _json_response(
            tool_name=tool_name,
            query=task_id,
            results=results,
            error=error,
        )
    except Exception as exc:  # noqa: BLE001
        return _error_response(tool_name, task_id, exc)


PAPER_TOOLS = [
    search_paper_chunks,
    search_paper_profiles,
    deep_search_papers,
    get_paper_profile,
    get_task_status,
]


def get_paper_tools() -> list:
    """Return PaperMind tools for LangGraph/LangChain agents."""
    return PAPER_TOOLS
