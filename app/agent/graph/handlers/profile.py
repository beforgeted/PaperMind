"""Paper profile handler — search and lookup paper metadata."""

from __future__ import annotations

from typing import Any

from app.agent.graph.state import AgentState
from app.agent.tools.paper_serializers import paper_search_result_to_result, result_to_source
from app.services.paper_search_service import (
    render_paper_search_answer,
    search_papers_by_query,
)


async def profile_handler(state: AgentState) -> dict[str, Any]:
    query = state.get("enriched_query") or state.get("query", "")
    task_id = state.get("task_id")

    results = await search_papers_by_query(query=query, task_id=task_id)

    if not results:
        return {
            "handler_answer": "当前知识库中未找到匹配的论文，请尝试更宽泛的关键词或先上传相关论文。",
            "handler_contexts": [],
            "handler_sources": [],
            "handler_used_tools": ["search_paper_profiles"],
        }

    answer = render_paper_search_answer(query, results)
    contexts = [paper_search_result_to_result(r) for r in results]

    return {
        "handler_answer": answer,
        "handler_contexts": contexts,
        "handler_sources": [result_to_source(c) for c in contexts],
        "handler_used_tools": ["search_paper_profiles", "get_paper_profile"],
    }
