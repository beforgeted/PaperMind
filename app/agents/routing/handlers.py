"""Route execution handlers shared by LangGraph nodes and streaming API."""

from __future__ import annotations

import re
from typing import Any, Callable

from app.core.config import settings
from app.core.logging import logger

_TASK_ID_RE = re.compile(r"task_id[=:]\s*(\S+)")


def extract_task_id(query: str, explicit_task_id: str | None) -> str | None:
    """Resolve task_id from request body or embedded in the query string."""
    if explicit_task_id:
        return explicit_task_id
    match = _TASK_ID_RE.search(query)
    if match:
        return match.group(1).strip("'\"")
    return None


async def execute_paper_search(query: str, _top_k: int | None, task_id: str | None) -> dict:
    from app.services.paper_search_service import (
        render_paper_search_answer,
        search_papers_by_query,
    )

    results = await search_papers_by_query(query=query, task_id=task_id)
    answer = render_paper_search_answer(query, results)
    return {"answer": answer, "contexts": [], "sources": [], "used_tools": ["search_paper_profiles"]}


async def execute_paper_deep_search(query: str, _top_k: int | None, task_id: str | None) -> dict:
    from app.services.paper_search_service import (
        deep_search_papers_by_query,
        render_paper_deep_search_answer,
    )

    results = await deep_search_papers_by_query(query=query, task_id=task_id)
    answer = render_paper_deep_search_answer(results)
    return {"answer": answer, "contexts": [], "sources": [], "used_tools": ["deep_search_papers"]}


async def execute_chunk_qa(query: str, top_k: int | None, task_id: str | None) -> dict:
    from app.services.qa_service import answer as qa_answer

    result = await qa_answer(query=query, top_k=top_k, task_id=task_id)
    return {
        "answer": result["answer"],
        "contexts": result["contexts"],
        "sources": [],
        "used_tools": ["search_paper_chunks"],
    }


async def execute_task_status(query: str, _top_k: int | None, task_id: str | None) -> dict:
    from app.services.task_status import get_task

    tid = extract_task_id(query, task_id)
    if not tid:
        return {
            "answer": "请提供 task_id 后再查询论文解析状态。您可以通过 /api/v1/papers 接口获取任务列表。",
            "contexts": [],
            "sources": [],
            "used_tools": [],
        }
    task = get_task(tid)
    if task is None:
        return {
            "answer": f"未找到 task_id={tid} 对应的任务，请检查 task_id 是否正确。",
            "contexts": [],
            "sources": [],
            "used_tools": [],
        }
    lines = [
        f"任务 {tid}",
        f"- 文件名：{task.original_filename}",
        f"- 状态：{task.status.value}",
        f"- 消息：{task.message}",
    ]
    if task.num_pages is not None:
        lines.append(f"- 页数：{task.num_pages}")
    if task.num_parents is not None:
        lines.append(f"- 段落数：{task.num_parents}")
    if task.error:
        lines.append(f"- 错误：{task.error}")
    return {
        "answer": "\n".join(lines),
        "contexts": [],
        "sources": [],
        "used_tools": ["get_task_status"],
    }


async def execute_paper_profile(query: str, _top_k: int | None, task_id: str | None) -> dict:
    from app.services.paper_search_service import search_papers_by_query

    results = await search_papers_by_query(query=query, task_id=task_id)
    if not results:
        return {
            "answer": "未找到相关论文，请尝试更具体的查询或检查知识库中是否已上传相关论文。",
            "contexts": [],
            "sources": [],
            "used_tools": ["get_paper_profile"],
        }
    if len(results) == 1:
        paper = results[0]
        lines = [f"论文《{paper.title or paper.source_file}》"]
        if paper.main_task:
            lines.append(f"- 主要任务：{paper.main_task}")
        tags_info = []
        if paper.method_tags:
            tags_info.append(f"核心方法：{'、'.join(paper.method_tags[:5])}")
        if paper.task_tags:
            tags_info.append(f"任务标签：{'、'.join(paper.task_tags[:5])}")
        if paper.dataset_tags:
            tags_info.append(f"数据集：{'、'.join(paper.dataset_tags[:5])}")
        if tags_info:
            lines.append("- " + "；".join(tags_info))
        if paper.abstract_summary:
            lines.append(f"- 摘要：{paper.abstract_summary}")
        if paper.method_summary:
            lines.append(f"- 方法：{paper.method_summary}")
        if paper.contribution_summary:
            lines.append(f"- 贡献：{paper.contribution_summary}")
        return {
            "answer": "\n".join(lines),
            "contexts": [],
            "sources": [],
            "used_tools": ["get_paper_profile"],
        }

    answer = f"找到 {len(results)} 篇相关论文，请指定具体论文：\n"
    for index, paper in enumerate(results[:5], 1):
        answer += f"{index}. {paper.title or paper.source_file}"
        if paper.main_task:
            answer += f"（{paper.main_task}）"
        answer += "\n"
    return {
        "answer": answer,
        "contexts": [],
        "sources": [],
        "used_tools": ["get_paper_profile"],
    }


_ROUTE_EXECUTORS: dict[str, Callable[..., Any]] = {
    "paper_search": execute_paper_search,
    "paper_deep_search": execute_paper_deep_search,
    "chunk_qa": execute_chunk_qa,
    "task_status": execute_task_status,
    "paper_profile": execute_paper_profile,
}


async def execute_route(
    route: str,
    query: str,
    top_k: int | None = None,
    task_id: str | None = None,
) -> dict:
    """Run the handler for *route* and return {answer, contexts, sources, used_tools}."""
    executor = _ROUTE_EXECUTORS.get(route)
    if executor is None:
        return {
            "answer": f"未知路由：{route}",
            "contexts": [],
            "sources": [],
            "used_tools": [],
            "error": f"Unknown route: {route}",
        }

    try:
        return await executor(query, top_k, task_id)
    except Exception as exc:
        logger.exception("Execution failed for route={} query={!r}: {}", route, query, exc)
        return {
            "answer": f"执行失败（{route}）：{exc}",
            "contexts": [],
            "sources": [],
            "used_tools": [],
            "error": str(exc),
        }


async def execute_route_with_fallback(
    route: str | None,
    query: str,
    top_k: int | None = None,
    task_id: str | None = None,
) -> dict:
    """Like execute_route but falls back to configured default when route is empty."""
    resolved = route or settings.llm_route_fallback_route
    return await execute_route(resolved, query, top_k, task_id)
