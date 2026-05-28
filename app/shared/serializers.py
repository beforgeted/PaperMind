"""Serializers shared by paper-related tools."""

from __future__ import annotations

from typing import Optional

from app.core.schemas import PaperProfile, PaperSearchResult, RetrievedChunk, TaskRecord
from app.services.papers.index import get_paper_profile as fetch_paper_profile
from app.services.papers.index import search_papers
from app.agent.tools.contracts import ToolSource, truncate


def chunk_to_result(chunk: RetrievedChunk) -> dict:
    """Convert a retrieved parent chunk into a compact tool result."""
    metadata = chunk.metadata or {}
    return {
        "paper_id": metadata.get("paper_id"),
        "title": metadata.get("title"),
        "section": metadata.get("section_title"),
        "section_type": metadata.get("section_type"),
        "chunk_ids": chunk.child_ids or [],
        "parent_id": chunk.parent_id,
        "score": float(chunk.score or 0.0),
        "content": truncate(chunk.parent_text),
        "metadata": metadata,
    }


def result_to_source(result: dict) -> ToolSource:
    """Extract citation metadata from a normalized paper result."""
    return ToolSource(
        paper_id=result.get("paper_id"),
        title=result.get("title"),
        section=result.get("section"),
        section_type=result.get("section_type"),
        parent_id=result.get("parent_id"),
        chunk_ids=[str(item) for item in result.get("chunk_ids") or []],
        score=result.get("score"),
        metadata=result.get("metadata") or {},
    )


def profile_to_result(profile: PaperProfile) -> dict:
    """Convert a paper profile into the common tool result shape."""
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
        "content": truncate(content or profile.abstract or profile.paper_search_text),
        "metadata": metadata,
    }


def paper_search_result_to_result(item: PaperSearchResult) -> dict:
    """Convert a paper-level search row into the common tool result shape."""
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
        "content": truncate(content),
        "metadata": item.model_dump(mode="json"),
    }


def task_to_result(task: TaskRecord) -> dict:
    """Convert task status into the common tool result shape."""
    metadata = task.model_dump(mode="json")
    return {
        "paper_id": None,
        "title": task.original_filename,
        "section": "task_status",
        "section_type": "task_status",
        "chunk_ids": [],
        "parent_id": None,
        "score": 1.0,
        "content": truncate(task.message or task.error or ""),
        "metadata": metadata,
    }


def find_profile_by_title(title: str) -> Optional[PaperProfile]:
    """Find the closest paper profile by title or source filename."""
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
