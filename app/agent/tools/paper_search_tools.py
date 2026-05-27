"""Paper discovery and profile tools."""

from __future__ import annotations

from typing import Optional

from app.services.paper_index_service import get_paper_profile as fetch_paper_profile
from app.services.paper_search_service import (
    deep_search_papers_by_query,
    search_papers_by_query,
)
from app.agent.tools.contracts import error_response, json_response
from app.agent.tools.decorators import tool
from app.agent.tools.paper_serializers import (
    find_profile_by_title,
    paper_search_result_to_result,
    profile_to_result,
    result_to_source,
)


@tool
async def search_paper_profiles(
    query: str,
    task_id: Optional[str] = None,
) -> str:
    """Search paper-level profiles for discovery, filtering, and recommendation."""
    tool_name = "search_paper_profiles"
    try:
        rows = await search_papers_by_query(query=query, task_id=task_id)
        results = [paper_search_result_to_result(item) for item in rows]
        return json_response(
            tool_name=tool_name,
            query=query,
            results=results,
            sources=[result_to_source(item) for item in results],
            confidence=1.0 if results else 0.0,
        )
    except Exception as exc:  # noqa: BLE001
        return error_response(tool_name, query, exc)


@tool
async def deep_search_papers(
    query: str,
    task_id: Optional[str] = None,
) -> str:
    """Search paper-level profiles and supporting evidence for match explanations."""
    tool_name = "deep_search_papers"
    try:
        rows = await deep_search_papers_by_query(query=query, task_id=task_id)
        results = [paper_search_result_to_result(item) for item in rows]
        return json_response(
            tool_name=tool_name,
            query=query,
            results=results,
            sources=[result_to_source(item) for item in results],
            confidence=1.0 if results else 0.0,
        )
    except Exception as exc:  # noqa: BLE001
        return error_response(tool_name, query, exc)


@tool
def get_paper_profile(
    paper_id: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """Get one paper profile by paper_id first, or by title as a fallback."""
    tool_name = "get_paper_profile"
    query = paper_id or title or ""
    try:
        profile = fetch_paper_profile(paper_id) if paper_id else None
        if profile is None and title:
            profile = find_profile_by_title(title)
        results = [profile_to_result(profile)] if profile else []
        return json_response(
            tool_name=tool_name,
            query=query,
            results=results,
            sources=[result_to_source(item) for item in results],
            confidence=1.0 if results else 0.0,
        )
    except Exception as exc:  # noqa: BLE001
        return error_response(tool_name, query, exc)


PAPER_SEARCH_TOOLS = [search_paper_profiles, deep_search_papers, get_paper_profile]
