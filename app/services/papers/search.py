"""Paper-level search and answer formatting."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Optional

from app.core.config import settings
import logging; logger = logging.getLogger(__name__)
from app.core.schemas import PaperSearchResult
from app.services.storage.embedding import get_embeddings
from app.services.papers.index import fetch_paper_evidence_chunks, search_papers
from app.services.storage.es import get_es_client

_DEEP_TOP_PAPERS = 5
_CANDIDATE_SIZE = 30
_RRF_K = 10




def _dedupe_terms(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        normalized = term.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            out.append(normalized)
    return out


def _expand_query_terms(query: str) -> list[str]:
    """Extract key terms from query for search expansion."""
    # Extract English words and Chinese phrases
    import re
    terms = [query]
    # English words (2+ chars)
    en_words = re.findall(r'[A-Za-z]{2,}', query)
    terms.extend(en_words)
    # Chinese bigrams
    cjk_chars = re.findall(r'[一-鿿]', query)
    for i in range(len(cjk_chars) - 1):
        terms.append(cjk_chars[i] + cjk_chars[i+1])
    return _dedupe_terms(terms)[:20]  # Limit to 20 terms


def _build_query(query: str, task_id: Optional[str]) -> tuple[dict, list[str]]:
    """根据问题构建 paper 索引查询。"""
    should: list[dict] = []
    evidence_terms: list[str] = []

    # Generic multi_match across all relevant fields
    should.append({
        "multi_match": {
            "query": query,
            "fields": ["title^3", "main_task^2", "summary^2", "abstract",
                       "method_summary", "contribution_summary",
                       "task_tags^2", "method_tags^2", "domain_tags"],
            "type": "best_fields",
            "tie_breaker": 0.3,
        }
    })

    must: list[dict] = []
    if task_id:
        must.append({"term": {"task_id": task_id}})

    return {"query": {"bool": {"should": should, "must": must, "minimum_should_match": 1}}}, evidence_terms


def _paper_should_clauses(query: str, expanded_terms: list[str]) -> list[dict]:
    should: list[dict] = [
        {
            "multi_match": {
                "query": query,
                "fields": [
                    "title^4",
                    "paper_search_text^3",
                    "main_task^2",
                    "abstract^2",
                    "clean_abstract^2",
                    "abstract_summary^2",
                    "method_summary^2",
                    "contribution_summary^2",
                    "research_problem^1.5",
                    "method_name^1.5",
                    "summary",
                ],
            }
        }
    ]
    for term in expanded_terms[1:]:
        should.append({"match_phrase": {"paper_search_text": {"query": term, "boost": 1.5}}})
    return should


def _paper_filter(task_id: Optional[str]) -> list[dict]:
    return [{"term": {"paper_id": task_id}}] if task_id else []


def _search_paper_bm25(query: str, expanded_terms: list[str], task_id: Optional[str]) -> list[dict]:
    client = get_es_client()
    body: dict = {
        "size": _CANDIDATE_SIZE,
        "query": {
            "bool": {
                "should": _paper_should_clauses(query, expanded_terms),
                "minimum_should_match": 1,
                "filter": _paper_filter(task_id),
            }
        },
    }
    response = client.search(index=settings.es_index_papers, body=body)
    return response.get("hits", {}).get("hits", [])


def _search_paper_vector(query: str, task_id: Optional[str]) -> list[dict]:
    client = get_es_client()
    vector = get_embeddings().embed_query(query)
    knn: dict = {
        "field": "paper_search_vector",
        "query_vector": vector,
        "k": _CANDIDATE_SIZE,
        "num_candidates": max(_CANDIDATE_SIZE * 4, 80),
    }
    filters = _paper_filter(task_id)
    if filters:
        knn["filter"] = filters[0]
    response = client.search(
        index=settings.es_index_papers,
        body={"size": _CANDIDATE_SIZE, "knn": knn},
    )
    return response.get("hits", {}).get("hits", [])


def _search_paper_tags(query: str, task_id: Optional[str]) -> list[dict]:
    expanded_terms = _expand_query_terms(query)
    should: list[dict] = []
    for term in expanded_terms:
        should.extend(
            [
                {"term": {"task_tags": {"value": term, "boost": 2.0}}},
                {"term": {"method_tags": {"value": term, "boost": 2.0}}},
                {"term": {"domain_tags": {"value": term, "boost": 1.5}}},
                {"term": {"dataset_tags": {"value": term, "boost": 1.2}}},
                {"term": {"metric_tags": {"value": term, "boost": 1.2}}},
            ]
        )
    if not should:
        return []
    client = get_es_client()
    response = client.search(
        index=settings.es_index_papers,
        body={
            "size": _CANDIDATE_SIZE,
            "query": {
                "bool": {
                    "should": should,
                    "minimum_should_match": 1,
                    "filter": _paper_filter(task_id),
                }
            },
        },
    )
    return response.get("hits", {}).get("hits", [])


def _rank_set_from_hits(hits: list[dict], *, nested_paper_id: bool = False) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for rank, hit in enumerate(hits, start=1):
        source = hit.get("_source") or {}
        metadata = source.get("metadata") or {}
        if nested_paper_id:
            paper_id = metadata.get("paper_id") or metadata.get("task_id")
        else:
            paper_id = source.get("paper_id")
        if paper_id and paper_id not in ranks:
            ranks[paper_id] = rank
    return ranks


def _rrf_scores(*rank_sets: dict[str, int], k: int = _RRF_K) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for rank_set in rank_sets:
        for paper_id, rank in rank_set.items():
            scores[paper_id] += 1.0 / (k + rank)
    return dict(scores)


def _source_to_result(source: dict, paper_id: str, score: float) -> PaperSearchResult:
    return PaperSearchResult(
        paper_id=paper_id,
        title=source.get("title", ""),
        main_task=source.get("main_task", ""),
        modality_tags=source.get("modality_tags") or [],
        task_tags=source.get("task_tags") or [],
        method_tags=source.get("method_tags") or [],
        domain_tags=source.get("domain_tags") or [],
        dataset_tags=source.get("dataset_tags") or [],
        metric_tags=source.get("metric_tags") or [],
        image_confidence=float(source.get("image_confidence") or 0.0),
        frequency_confidence=float(source.get("frequency_confidence") or 0.0),
        summary=source.get("summary", ""),
        abstract_summary=source.get("abstract_summary", ""),
        method_summary=source.get("method_summary", ""),
        contribution_summary=source.get("contribution_summary", ""),
        source_file=source.get("source_file", ""),
        score=score,
    )


def _collect_paper_sources(*hit_lists: list[dict]) -> dict[str, dict]:
    sources: dict[str, dict] = {}
    for hits in hit_lists:
        for hit in hits:
            source = hit.get("_source") or {}
            paper_id = source.get("paper_id") or hit.get("_id")
            if paper_id and paper_id not in sources:
                sources[paper_id] = source
    return sources


def _chunk_query(query: str, expanded_terms: list[str], candidate_paper_ids: list[str]) -> dict:
    should: list[dict] = [{"match": {"text": {"query": query, "boost": 2.0}}}]
    should.extend({"match_phrase": {"text": {"query": term, "boost": 1.3}}} for term in expanded_terms[1:])
    return {
        "bool": {
            "filter": [{"terms": {"metadata.task_id": candidate_paper_ids}}],
            "should": should,
            "minimum_should_match": 1,
        }
    }


def _search_chunk_bm25(query: str, expanded_terms: list[str], candidate_paper_ids: list[str]) -> list[dict]:
    if not candidate_paper_ids:
        return []
    client = get_es_client()
    response = client.search(
        index=settings.es_index_child,
        body={
            "size": 80,
            "_source": ["text", "metadata.task_id", "metadata.paper_id", "metadata.section_type", "metadata.section_title"],
            "query": _chunk_query(query, expanded_terms, candidate_paper_ids),
        },
    )
    return response.get("hits", {}).get("hits", [])


def _search_chunk_vector(query: str, candidate_paper_ids: list[str]) -> list[dict]:
    if not candidate_paper_ids:
        return []
    client = get_es_client()
    vector = get_embeddings().embed_query(query)
    response = client.search(
        index=settings.es_index_child,
        body={
            "size": 80,
            "_source": ["text", "metadata.task_id", "metadata.paper_id", "metadata.section_type", "metadata.section_title"],
            "knn": {
                "field": "vector",
                "query_vector": vector,
                "k": 80,
                "num_candidates": 200,
                "filter": {"terms": {"metadata.task_id": candidate_paper_ids}},
            },
        },
    )
    return response.get("hits", {}).get("hits", [])


def _aggregate_chunk_evidence(*hit_lists: list[dict]) -> dict[str, list[str]]:
    evidence: dict[str, list[str]] = defaultdict(list)
    for hits in hit_lists:
        for hit in hits:
            source = hit.get("_source") or {}
            metadata = source.get("metadata") or {}
            paper_id = metadata.get("paper_id") or metadata.get("task_id")
            text = (source.get("text") or "").strip().replace("\n", " ")
            if not paper_id:
                continue
            if text and len(evidence[paper_id]) < 2:
                section = metadata.get("section_type") or "chunk"
                evidence[paper_id].append(f"[{section}] {text[:260]}")
    return dict(evidence)


def _tag_boost(result: PaperSearchResult, query: str) -> float:
    lowered = query.lower()
    boost = 0.0
    for tag in result.task_tags + result.method_tags + result.domain_tags:
        if tag.lower() in lowered:
            boost += 0.01
    return boost


def _reason_for_deep_result(result: PaperSearchResult, query: str) -> str:
    reasons: list[str] = []
    if result.image_confidence >= 0.6:
        reasons.append(f"image confidence {result.image_confidence:.2f}")
    if result.frequency_confidence >= 0.6:
        reasons.append(f"frequency confidence {result.frequency_confidence:.2f}")
    tags = result.task_tags + result.method_tags + result.domain_tags
    if tags:
        reasons.append(f"tags matched: {', '.join(tags[:4])}")
    if result.evidence_chunks:
        reasons.append("chunk evidence hit in candidate papers")
    return "; ".join(reasons) or "paper profile and body text jointly match query"


def _deep_search_sync(query: str, task_id: Optional[str]) -> list[PaperSearchResult]:
    expanded_terms = _expand_query_terms(query)
    try:
        paper_bm25_hits = _search_paper_bm25(query, expanded_terms, task_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Paper BM25 search failed for query={!r}: {}", query, exc)
        paper_bm25_hits = []
    try:
        paper_vector_hits = _search_paper_vector(query, task_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Paper vector search skipped for query={!r}: {}", query, exc)
        paper_vector_hits = []
    try:
        tag_hits = _search_paper_tags(query, task_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Paper tag search failed for query={!r}: {}", query, exc)
        tag_hits = []

    paper_sources = _collect_paper_sources(paper_bm25_hits, paper_vector_hits, tag_hits)
    paper_scores = _rrf_scores(
        _rank_set_from_hits(paper_bm25_hits),
        _rank_set_from_hits(paper_vector_hits),
        _rank_set_from_hits(tag_hits),
    )
    candidate_paper_ids = [
        paper_id
        for paper_id, _score in sorted(paper_scores.items(), key=lambda item: item[1], reverse=True)[:_CANDIDATE_SIZE]
    ]
    if not candidate_paper_ids:
        return []

    try:
        chunk_bm25_hits = _search_chunk_bm25(query, expanded_terms, candidate_paper_ids)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Candidate chunk BM25 search failed for query={!r}: {}", query, exc)
        chunk_bm25_hits = []
    try:
        chunk_vector_hits = _search_chunk_vector(query, candidate_paper_ids)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Candidate chunk vector search skipped for query={!r}: {}", query, exc)
        chunk_vector_hits = []

    chunk_evidence = _aggregate_chunk_evidence(chunk_bm25_hits, chunk_vector_hits)
    fused_scores = _rrf_scores(
        _rank_set_from_hits(paper_bm25_hits),
        _rank_set_from_hits(paper_vector_hits),
        _rank_set_from_hits(tag_hits),
        _rank_set_from_hits(chunk_bm25_hits, nested_paper_id=True),
        _rank_set_from_hits(chunk_vector_hits, nested_paper_id=True),
    )

    results: list[PaperSearchResult] = []
    for paper_id, score in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True):
        source = paper_sources.get(paper_id)
        if not source:
            continue
        item = _source_to_result(source, paper_id, score)
        item.evidence_chunks = chunk_evidence.get(paper_id, [])
        item.score = score + _tag_boost(item, query)
        item.reason = _reason_for_deep_result(item, query)
        results.append(item)
    return sorted(results, key=lambda item: item.score, reverse=True)[:_DEEP_TOP_PAPERS]


async def search_papers_by_query(query: str, task_id: Optional[str] = None) -> list[PaperSearchResult]:
    """查询论文级索引并补充少量 chunk 证据。"""
    body, evidence_terms = _build_query(query, task_id)
    try:
        rows = search_papers(body, settings.paper_search_top_k)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Paper search failed for query={!r}: {}", query, exc)
        return []

    for row in rows:
        evidence = fetch_paper_evidence_chunks(
            row.paper_id,
            evidence_terms,
            limit=2,
        )
        row.evidence_chunks = evidence
        if evidence:
            row.reason = f"证据片段命中关键词：{', '.join(evidence_terms[:4])}"
        elif row.method_tags:
            row.reason = f"方法标签匹配：{', '.join(row.method_tags[:3])}"
        elif row.task_tags:
            row.reason = f"任务标签匹配：{', '.join(row.task_tags[:3])}"
        else:
            row.reason = "标题/摘要/总结与查询语义匹配"
    return rows


async def deep_search_papers_by_query(query: str, task_id: Optional[str] = None) -> list[PaperSearchResult]:
    """论文级初筛 + 候选论文内 chunk 精检 + RRF 聚合。"""
    return await asyncio.to_thread(_deep_search_sync, query, task_id)


def render_paper_search_answer(query: str, results: list[PaperSearchResult]) -> str:
    """将论文级结果渲染为可读答案文本。"""
    if not results:
        return (
            '当前知识库中暂未检索到明确相关论文。你可以尝试使用更宽泛的关键词或英文术语。'
        )

    lines = [f"知识库中检索到 {len(results)} 篇相关论文：", ""]
    for idx, item in enumerate(results, start=1):
        methods = ", ".join(item.method_tags[:4]) or "未标注"
        modality = ", ".join(item.modality_tags[:3]) or "未标注"
        summary = item.summary or "暂无摘要总结。"
        lines.append(f"{idx}. 《{item.title or item.source_file or item.paper_id}》")
        lines.append(f"   - 主要任务：{item.main_task or '未标注'}")
        lines.append(f"   - 数据类型：{modality}")
        lines.append(f"   - 核心方法：{methods}")
        lines.append(f"   - 相关原因：{item.reason}")
        if item.evidence_chunks:
            lines.append(f"   - 证据片段：{item.evidence_chunks[0]}")
        lines.append(f"   - 简要总结：{summary}")
        lines.append("")
    if "多少" in query or "几篇" in query:
        lines.append(f"统计结果：共 {len(results)} 篇。")
    return "\n".join(lines).strip()


def render_paper_deep_search_answer(results: list[PaperSearchResult]) -> str:
    """将 paper_deep_search 结果渲染为带证据的答案文本。"""
    if not results:
        return (
            "当前知识库中没有检索到高置信度相关论文。建议尝试更宽泛关键词或英文术语。"
        )

    lines = [f"知识库中检索到 {len(results)} 篇较相关论文，Top {len(results)} 如下：", ""]
    for idx, item in enumerate(results, start=1):
        methods = ", ".join(item.method_tags[:4]) or item.method_summary or "未标注"
        tags = ", ".join((item.task_tags + item.method_tags + item.domain_tags + item.dataset_tags + item.metric_tags)[:6]) or "未标注"
        evidence = item.evidence_chunks[0] if item.evidence_chunks else "暂无高亮 chunk，可查看论文摘要与方法总结。"
        summary = item.method_summary or item.contribution_summary or item.summary or item.abstract_summary or "暂无摘要总结。"
        lines.append(f"{idx}. 《{item.title or item.source_file or item.paper_id}》")
        lines.append(f"   - 主要任务：{item.main_task or '未标注'}")
        lines.append(f"   - 核心方法：{methods}")
        lines.append(f"   - 相关标签：{tags}")
        lines.append(f"   - 匹配原因：{item.reason}")
        lines.append(f"   - 证据片段：{evidence}")
        lines.append(f"   - 简要总结：{summary}")
        lines.append("")
    return "\n".join(lines).strip()
