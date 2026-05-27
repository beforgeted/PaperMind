"""Multi-paper comparison handler — search candidates, retrieve evidence, generate comparison table."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.graph.state import AgentState
from app.agent.tools.paper_serializers import paper_search_result_to_result, result_to_source
from app.core.logging import logger
from app.services.llm_service import get_llm
from app.services.paper_search_service import (
    deep_search_papers_by_query,
    search_papers_by_query,
)
from app.services.retrieval_service import hybrid_retrieve
from app.services.skill_loader import get_skill_loader

_COMPARISON_SYSTEM = """You are a research paper comparison assistant.

Given search results and evidence chunks from multiple papers, generate a structured comparison.
Use Markdown tables when comparing across dimensions (method, dataset, metric, contribution).
Cite specific evidence from each paper. Be fair and accurate.

Output format:
1. Brief overview of each paper being compared
2. Comparison table (method | dataset | key metric | contribution)
3. Summary of key differences and practical recommendations

Do not invent data. If evidence is insufficient for a dimension, mark it as 'N/A'."""


async def comparison_handler(state: AgentState) -> dict[str, Any]:
    query = state.get("enriched_query") or state.get("query", "")
    original_query = state.get("query", "")
    task_id = state.get("task_id")
    used_tools: list[str] = []
    all_contexts: list[dict] = []

    logger.info(
        "对比任务开始: query={:.100}",
        original_query,
    )

    # Step 1: Search for candidate papers
    search_results = await search_papers_by_query(query=query, task_id=task_id)
    used_tools.append("search_paper_profiles")
    logger.info(
        "对比-论文搜索: found={} papers, query={:.80}",
        len(search_results) if search_results else 0, query,
    )

    if not search_results:
        logger.info("对比-论文搜索: 无结果，尝试 deep_search")
        search_results = await deep_search_papers_by_query(query=query, task_id=task_id)
        used_tools.append("deep_search_papers")
        logger.info(
            "对比-deep_search: found={} papers",
            len(search_results) if search_results else 0,
        )

    if not search_results:
        logger.info("对比任务: 无可用论文，返回空结果")
        return _no_results(used_tools)

    for i, p in enumerate(search_results):
        logger.info(
            "对比-候选论文[{}]: id={} | title={} | score={:.4f}",
            i, getattr(p, "paper_id", "?"), getattr(p, "title", "?")[:80], getattr(p, "score", 0.0),
        )

    # Step 2: Retrieve detailed evidence for top papers (max 5)
    top_papers = search_results[:5]
    evidence_blocks: list[str] = []
    for paper in top_papers:
        paper_id = getattr(paper, "paper_id", "")
        paper_title = getattr(paper, "title", "") or getattr(paper, "source_file", "")
        chunks = await hybrid_retrieve(
            query=f"{query} {paper_title}",
            task_id=paper_id,
            top_k=3,
        )
        used_tools.append("retrieve_evidence")
        logger.info(
            "对比-证据检索: paper={} | chunks={}",
            paper_title[:60], len(chunks),
        )
        for chunk in chunks:
            ctx = paper_search_result_to_result(paper)
            ctx["evidence_content"] = chunk.parent_text
            all_contexts.append(ctx)

        paper_evidence = "\n".join(c.parent_text[:400] for c in chunks if c.parent_text)
        if not paper_evidence:
            paper_evidence = f"(no evidence chunks retrieved for {paper_title})"
            logger.warning("对比-证据检索: paper={} 无有效证据", paper_title[:60])
        evidence_blocks.append(
            f"## {paper_title}\n"
            f"Paper ID: {paper_id}\n"
            f"Main task: {getattr(paper, 'main_task', 'N/A')}\n"
            f"Evidence:\n{paper_evidence}"
        )

    # Step 3: Optionally load writing skill for comparison standards
    try:
        loader = get_skill_loader()
        loader.load_skill("nature-polishing")
        used_tools.append("load_skill")
    except Exception:
        pass

    # Step 4: LLM generates structured comparison
    evidence_text = "\n\n".join(evidence_blocks)
    logger.info(
        "对比-LLM生成: papers={} | evidence_chars={}",
        len(top_papers), len(evidence_text),
    )
    llm = get_llm()
    response = await llm.ainvoke([
        SystemMessage(content=_COMPARISON_SYSTEM),
        HumanMessage(content=f"Comparison query: {original_query}\n\nEvidence:\n{evidence_text}"),
    ])
    answer = response.content if hasattr(response, "content") else str(response)

    logger.info(
        "对比任务完成: papers={} | answer_chars={} | tools={}",
        len(top_papers), len(answer), list(set(used_tools)),
    )

    return {
        "handler_answer": answer,
        "handler_contexts": [c for c in all_contexts if isinstance(c, dict)],
        "handler_sources": _dedup([result_to_source(c) for c in all_contexts if isinstance(c, dict)]),
        "handler_used_tools": list(set(used_tools)),
    }


def _no_results(used_tools: list[str]) -> dict[str, Any]:
    return {
        "handler_answer": "未找到可对比的论文。请先上传相关论文或调整查询条件。",
        "handler_contexts": [],
        "handler_sources": [],
        "handler_used_tools": used_tools,
    }


def _dedup(sources: list) -> list:
    seen: set[tuple] = set()
    result: list = []
    for src in sources:
        if isinstance(src, dict):
            key = (src.get("paper_id"), src.get("title"))
        else:
            key = (getattr(src, "paper_id", None), getattr(src, "title", None))
        if key not in seen:
            seen.add(key)
            result.append(src)
    return result
