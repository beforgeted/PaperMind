"""All LangGraph nodes for the comparison workflow subgraph.

Consolidated from the 7 original node files plus the entry marker.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.state import AgentState
from app.core.config import settings
from app.core.logging import logger
from app.services.llm import get_llm
from app.services.papers.search import search_papers_by_query
from app.services.retrieval import hybrid_retrieve
from app.shared.serializers import result_to_source


# ── helpers ──────────────────────────────────────────────────────────────

def _serialize_search_candidate(row: Any) -> dict[str, Any]:
    if hasattr(row, "model_dump"):
        return row.model_dump()
    return {
        "paper_id": getattr(row, "paper_id", ""),
        "title": getattr(row, "title", "") or getattr(row, "source_file", ""),
        "score": getattr(row, "score", 0.0),
        "main_task": getattr(row, "main_task", ""),
        "source_file": getattr(row, "source_file", ""),
    }


def _serialize_resolve_candidate(row: Any) -> dict[str, Any]:
    if hasattr(row, "model_dump"):
        return row.model_dump()
    return {
        "paper_id": getattr(row, "paper_id", ""),
        "title": getattr(row, "title", "") or getattr(row, "source_file", ""),
        "score": getattr(row, "score", 0.0),
    }


_REWRITE_SUFFIX = (
    " detailed description implementation architecture module network "
    "详细 实现 结构 模块 方法"
)


# ── entry node ───────────────────────────────────────────────────────────

async def comparison_entry_node(state: AgentState) -> dict[str, Any]:
    """Subgraph entry — trace marker for SSE."""
    plan = state.get("plan")
    target_count = (
        len((plan or {}).get("targets") or [])
        if isinstance(plan, dict)
        else 0
    )
    return {
        "trace": [
            {"node": "comparison_subgraph", "phase": "start", "target_count": target_count}
        ],
    }


# ── dispatch marker nodes ────────────────────────────────────────────────

async def dispatch_paper_search_node(state: AgentState) -> dict[str, Any]:
    """Record planned paper-search fan-out before conditional Send."""
    plan = state.get("plan")
    targets = (plan.get("targets") or []) if isinstance(plan, dict) else []
    count = len(targets) if targets else 0
    logger.info("dispatch_paper_search_node: targets={}", count)
    return {
        "trace": [{"node": "dispatch_paper_search", "parallel_searches": count}],
    }


async def dispatch_evidence_node(state: AgentState) -> dict[str, Any]:
    """Record planned evidence-retrieval fan-out before conditional Send."""
    plan = state.get("plan")
    aspects = (plan.get("aspects") or []) if isinstance(plan, dict) else []
    resolved = state.get("resolved_papers") or {}
    resolved_count = sum(
        1 for p in resolved.values()
        if isinstance(p, dict) and p.get("status") == "resolved"
    )
    retry_count = int(state.get("retry_count") or 0)
    only_missing = retry_count > 0
    missing = (state.get("coverage_report") or {}).get("missing") or []
    if only_missing:
        pairs = [
            m for m in missing
            if isinstance(m, dict) and m.get("reason") == "insufficient_evidence"
        ]
        parallel = len(pairs)
    else:
        parallel = resolved_count * len(aspects)

    logger.info("dispatch_evidence_node: only_missing={} | parallel={}", only_missing, parallel)
    return {
        "trace": [
            {"node": "dispatch_evidence_retrieval", "only_missing": only_missing, "parallel_retrievals": parallel}
        ],
    }


# ── paper search ─────────────────────────────────────────────────────────

async def paper_search_node(state: AgentState) -> dict[str, Any]:
    """Run search_papers_by_query for one target alias."""
    target = state.get("target") or {}
    if not isinstance(target, dict):
        target = {}

    alias = str(target.get("alias") or "?")
    paper_query = str(target.get("query") or "").strip()
    task_id = state.get("task_id")

    if not paper_query:
        return {
            "paper_candidates": {alias: []},
            "trace": [{"node": "paper_search_node", "alias": alias, "candidate_count": 0}],
        }

    rows = await search_papers_by_query(query=paper_query, task_id=task_id)
    candidates = [_serialize_search_candidate(row) for row in rows]

    logger.info("paper_search_node: alias={} | query={:.60} | found={}", alias, paper_query, len(candidates))

    return {
        "paper_candidates": {alias: candidates},
        "trace": [{"node": "paper_search_node", "alias": alias, "query": paper_query, "candidate_count": len(candidates)}],
    }


# ── paper resolver ───────────────────────────────────────────────────────

async def resolve_papers_node(state: AgentState) -> dict[str, Any]:
    """Pick top candidate per alias; run discovery when targets were empty."""
    plan = state.get("plan")
    plan_dict = plan if isinstance(plan, dict) else {}
    paper_candidates = dict(state.get("paper_candidates") or {})
    resolved: dict[str, dict[str, Any]] = {}
    task_id = state.get("task_id")

    targets = plan_dict.get("targets") or []
    if not targets and not paper_candidates:
        discovery_query = str(plan_dict.get("discovery_query") or state.get("query") or "")
        if discovery_query:
            rows = await search_papers_by_query(query=discovery_query, task_id=task_id)
            aliases = ["A", "B", "C", "D", "E"]
            for i, row in enumerate(rows[: settings.comparison_discovery_top_k]):
                alias = aliases[min(i, 4)]
                paper_candidates[alias] = [_serialize_resolve_candidate(row)]

    for alias, candidates in paper_candidates.items():
        if not candidates:
            resolved[alias] = {"status": "not_found", "paper_id": None, "title": None, "score": 0.0}
            continue

        best = candidates[0]
        score = float(best.get("score") or 0.0)
        min_score = settings.comparison_paper_resolve_min_score

        if score < min_score and len(candidates) == 1:
            resolved[alias] = {"status": "low_confidence", "paper_id": best.get("paper_id"), "title": best.get("title"), "score": score}
        else:
            resolved[alias] = {"status": "resolved", "paper_id": best.get("paper_id"), "title": best.get("title"), "score": score}

    logger.info("resolve_papers_node: resolved={}", resolved)

    return {
        "resolved_papers": resolved,
        "trace": [{"node": "resolve_papers_node", "resolved": resolved}],
    }


# ── evidence retriever ───────────────────────────────────────────────────

async def retrieve_evidence_node(state: AgentState) -> dict[str, Any]:
    """Scoped hybrid_retrieve within a single paper."""
    alias = str(state.get("alias") or "?")
    paper = state.get("paper") or {}
    aspect = state.get("aspect") or {}

    if not isinstance(paper, dict) or not isinstance(aspect, dict):
        return {"evidence_blocks": [], "trace": []}

    paper_id = paper.get("paper_id")
    title = paper.get("title") or ""
    aspect_name = str(aspect.get("name") or "method")
    evidence_query = str(aspect.get("evidence_query") or aspect_name)
    section_types = list(aspect.get("section_types") or [])

    if not paper_id or paper.get("status") not in ("resolved", "low_confidence"):
        return {
            "evidence_blocks": [],
            "trace": [{"node": "retrieve_evidence_node", "alias": alias, "aspect": aspect_name, "evidence_count": 0, "skipped": True}],
        }

    chunks = await hybrid_retrieve(
        query=evidence_query,
        task_id=str(paper_id),
        top_k=settings.comparison_evidence_top_k,
        section_types=section_types or None,
    )

    evidence_blocks: list[dict[str, Any]] = []
    for chunk in chunks:
        md = chunk.metadata or {}
        evidence_blocks.append({
            "alias": alias, "paper_id": paper_id, "title": title, "aspect": aspect_name,
            "parent_id": chunk.parent_id,
            "chunk_id": chunk.child_ids[0] if chunk.child_ids else chunk.parent_id,
            "section_type": md.get("section_type"),
            "content": chunk.parent_text, "score": chunk.score,
        })

    logger.info("retrieve_evidence_node: alias={} | aspect={} | chunks={}", alias, aspect_name, len(evidence_blocks))

    return {
        "evidence_blocks": evidence_blocks,
        "trace": [{"node": "retrieve_evidence_node", "alias": alias, "paper_id": paper_id, "aspect": aspect_name, "evidence_count": len(evidence_blocks)}],
    }


# ── coverage check ───────────────────────────────────────────────────────

async def coverage_check_node(state: AgentState) -> dict[str, Any]:
    """Build coverage_report for routing to compare or rewrite."""
    plan = state.get("plan")
    plan_dict = plan if isinstance(plan, dict) else {}
    resolved = state.get("resolved_papers") or {}
    evidence_blocks = state.get("evidence_blocks") or []
    aspects = plan_dict.get("aspects") or []
    min_count = settings.comparison_min_evidence_per_aspect

    report: dict[str, Any] = {"ok": True, "missing": []}

    for alias, paper in resolved.items():
        if not isinstance(paper, dict):
            continue
        if paper.get("status") == "not_found":
            report["ok"] = False
            report["missing"].append({"alias": alias, "reason": "paper_not_resolved"})
            continue
        if paper.get("status") == "low_confidence":
            report["ok"] = False
            report["missing"].append({"alias": alias, "reason": "low_confidence", "paper_id": paper.get("paper_id")})

        for aspect in aspects:
            aspect_name = str(aspect.get("name") if isinstance(aspect, dict) else aspect)
            matched = [
                e for e in evidence_blocks
                if isinstance(e, dict) and e.get("alias") == alias and e.get("aspect") == aspect_name
            ]
            if len(matched) < min_count:
                report["ok"] = False
                report["missing"].append({
                    "alias": alias, "paper_id": paper.get("paper_id"),
                    "aspect": aspect_name, "reason": "insufficient_evidence", "count": len(matched),
                })

    return {
        "coverage_report": report,
        "trace": [{"node": "coverage_check_node", "coverage_report": report}],
    }


# ── query rewriter ───────────────────────────────────────────────────────

async def query_rewrite_node(state: AgentState) -> dict[str, Any]:
    """Expand evidence_query for missing (alias, aspect) pairs."""
    from app.agent.schemas.plan import aspect_from_name

    plan = state.get("plan")
    plan_dict = dict(plan) if isinstance(plan, dict) else {}
    aspects = list(plan_dict.get("aspects") or [])
    missing = (state.get("coverage_report") or {}).get("missing") or []
    retry_count = int(state.get("retry_count") or 0)

    aspect_by_name: dict[str, dict[str, Any]] = {}
    for asp in aspects:
        if isinstance(asp, dict):
            aspect_by_name[str(asp.get("name"))] = dict(asp)

    for item in missing:
        if not isinstance(item, dict):
            continue
        if item.get("reason") != "insufficient_evidence":
            continue
        aspect_name = str(item.get("aspect") or "method")
        asp = aspect_by_name.get(aspect_name)
        if not asp:
            continue
        old_q = str(asp.get("evidence_query") or aspect_name)
        asp["evidence_query"] = old_q + _REWRITE_SUFFIX
        preset = aspect_from_name(aspect_name)
        if not asp.get("section_types"):
            asp["section_types"] = list(preset.section_types)

    plan_dict["aspects"] = list(aspect_by_name.values()) if aspect_by_name else aspects

    logger.info("query_rewrite_node: retry_count={} -> {}", retry_count, retry_count + 1)

    return {
        "plan": plan_dict,
        "retry_count": retry_count + 1,
        "trace": [{"node": "query_rewrite_node", "retry_count": retry_count + 1}],
    }


# ── comparer ─────────────────────────────────────────────────────────────

_COMPARISON_SYSTEM = """You are a research paper comparison assistant.

Given resolved papers, evidence blocks, and coverage report, generate a structured comparison.
Use Markdown tables when comparing across dimensions (method, dataset, metric, contribution).
Cite specific evidence from each paper. Be fair and accurate.

Output format:
1. Brief overview of each paper being compared
2. Comparison table (method | dataset | key metric | contribution)
3. Summary of key differences and practical recommendations

Do not invent data. If evidence is insufficient for a dimension, mark it as 'N/A' and explain."""


async def compare_node(state: AgentState) -> dict[str, Any]:
    """LLM synthesis only — no retrieval."""
    import json

    plan = state.get("plan")
    plan_dict = plan if isinstance(plan, dict) else {}
    resolved = state.get("resolved_papers") or {}
    evidence_blocks = state.get("evidence_blocks") or []
    coverage_report = state.get("coverage_report") or {}
    original_query = state.get("query", "")

    evidence_text = json.dumps(evidence_blocks, ensure_ascii=False, indent=2)[:12000]

    llm = get_llm()
    response = await llm.ainvoke([
        SystemMessage(content=_COMPARISON_SYSTEM),
        HumanMessage(content=(
            f"Comparison query: {original_query}\n\n"
            f"Resolved papers:\n{json.dumps(resolved, ensure_ascii=False, indent=2)}\n\n"
            f"Aspects:\n{json.dumps(plan_dict.get('aspects', []), ensure_ascii=False)}\n\n"
            f"Coverage:\n{json.dumps(coverage_report, ensure_ascii=False)}\n\n"
            f"Evidence blocks:\n{evidence_text}"
        )),
    ])
    answer = response.content if hasattr(response, "content") else str(response)

    contexts = [_evidence_to_context(e) for e in evidence_blocks if isinstance(e, dict)]
    sources = [result_to_source(c) for c in contexts]

    logger.info("compare_node: papers={} | evidence={} | answer_len={}", len(resolved), len(evidence_blocks), len(answer))

    return {
        "handler_answer": answer,
        "handler_contexts": contexts,
        "handler_sources": sources,
        "handler_used_tools": ["search_paper_profiles", "retrieve_evidence", "compare_node"],
        "trace": [{"node": "compare_node", "evidence_count": len(evidence_blocks)}],
    }


def _evidence_to_context(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": block.get("paper_id"),
        "title": block.get("title"),
        "parent_id": block.get("parent_id"),
        "content": block.get("content") or "",
        "section_type": block.get("section_type"),
        "metadata": {
            "alias": block.get("alias"), "aspect": block.get("aspect"),
            "paper_id": block.get("paper_id"), "title": block.get("title"),
        },
        "chunk_ids": [block.get("chunk_id")] if block.get("chunk_id") else [],
        "score": block.get("score"),
    }
