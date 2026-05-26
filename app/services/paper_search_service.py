"""Paper-level search and answer formatting."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Optional

from app.core.config import settings
from app.core.logging import logger
from app.core.schemas import PaperSearchResult
from app.services.embedding_service import get_embeddings
from app.services.paper_index_service import fetch_paper_evidence_chunks, search_papers
from app.services.vectorstore_service import get_es_client

_IMAGE_HINTS = ("图像", "视觉", "image", "vision", "enhancement", "检测", "分割")
_FREQ_HINTS = ("傅里叶", "频域", "频谱", "fft", "dft", "fourier", "spectrum", "phase", "amplitude")
_TRANSFORMER_HINTS = ("transformer",)
_ATTENTION_HINTS = ("attention", "注意力")
_ENHANCEMENT_HINTS = ("图像增强", "image enhancement", "underwater image enhancement")
_UNDERWATER_HINTS = ("水下", "underwater")
_DEEP_TOP_PAPERS = 5
_CANDIDATE_SIZE = 30
_RRF_K = 60

_QUERY_EXPANSIONS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("傅里叶", "频域", "频谱"), ("Fourier", "FFT", "DFT", "frequency domain", "spectrum")),
    (("幅度谱",), ("magnitude spectrum", "amplitude spectrum")),
    (("相位谱",), ("phase spectrum",)),
    (("图像增强",), ("image enhancement", "image restoration")),
    (("图像恢复",), ("image restoration", "image enhancement")),
    (("水下",), ("underwater", "underwater image enhancement")),
    (("注意力",), ("attention", "attention mechanism")),
)


def _contains_any(query: str, terms: tuple[str, ...]) -> bool:
    lowered = query.lower()
    return any(term.lower() in lowered for term in terms)


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
    lowered = query.lower()
    terms = [query]
    for triggers, additions in _QUERY_EXPANSIONS:
        if any(trigger.lower() in lowered for trigger in triggers):
            terms.extend(additions)
    if _contains_any(lowered, _IMAGE_HINTS):
        terms.extend(["image", "vision", "computer vision"])
    if _contains_any(lowered, _FREQ_HINTS):
        terms.extend(["Fourier", "FFT", "frequency domain", "spectrum"])
    return _dedupe_terms(terms)


def _build_query(query: str, task_id: Optional[str]) -> tuple[dict, list[str]]:
    """根据问题构建 paper 索引查询。"""
    should: list[dict] = []
    evidence_terms: list[str] = []
    lowered = query.lower()

    if _contains_any(lowered, _IMAGE_HINTS):
        should.extend(
            [
                {"term": {"is_image_related": True}},
                {"term": {"modality_tags": "image"}},
                {"term": {"domain_tags": "computer vision"}},
                {"match": {"task_tags": {"query": "image enhancement", "boost": 2.0}}},
                {"match": {"main_task": {"query": "image", "boost": 1.5}}},
            ]
        )
        evidence_terms.extend(["image", "vision", "enhancement", "segmentation", "detection"])

    if _contains_any(lowered, _FREQ_HINTS):
        should.extend(
            [
                {"term": {"is_frequency_related": True}},
                {"match": {"method_tags": {"query": "Fourier Transform", "boost": 2.0}}},
                {"match": {"method_tags": {"query": "FFT", "boost": 2.0}}},
                {"match": {"summary": {"query": "frequency domain", "boost": 1.5}}},
                {"match": {"abstract": {"query": "spectrum", "boost": 1.2}}},
            ]
        )
        evidence_terms.extend(["Fourier", "FFT", "DFT", "frequency domain", "spectrum"])

    if _contains_any(lowered, _TRANSFORMER_HINTS):
        should.extend(
            [
                {"match": {"method_tags": {"query": "Transformer", "boost": 2.0}}},
                {"match": {"summary": {"query": "Transformer", "boost": 1.2}}},
                {"match": {"abstract": {"query": "Transformer", "boost": 1.2}}},
            ]
        )
        evidence_terms.append("Transformer")

    if _contains_any(lowered, _ATTENTION_HINTS):
        should.extend(
            [
                {"match": {"method_tags": {"query": "Attention", "boost": 2.0}}},
                {"match": {"summary": {"query": "attention", "boost": 1.2}}},
                {"match": {"abstract": {"query": "attention", "boost": 1.2}}},
            ]
        )
        evidence_terms.append("Attention")

    if _contains_any(lowered, _ENHANCEMENT_HINTS):
        should.extend(
            [
                {"match": {"task_tags": {"query": "image enhancement", "boost": 2.0}}},
                {"match": {"task_tags": {"query": "underwater image enhancement", "boost": 2.0}}},
                {"match": {"main_task": {"query": "image enhancement", "boost": 1.5}}},
                {"match": {"summary": {"query": "图像增强", "boost": 1.2}}},
            ]
        )
        evidence_terms.extend(["image enhancement", "underwater image"])

    must: list[dict] = []
    if _contains_any(lowered, _UNDERWATER_HINTS):
        # 对“水下”问题启用强约束，避免仅靠宽泛 image 标签误命中。
        must.append(
            {
                "bool": {
                    "should": [
                        {"match": {"title": {"query": "underwater", "boost": 2.0}}},
                        {"match": {"main_task": {"query": "underwater", "boost": 2.0}}},
                        {"match": {"summary": {"query": "水下", "boost": 2.0}}},
                        {"match": {"abstract": {"query": "underwater", "boost": 1.6}}},
                        {"match": {"task_tags": {"query": "underwater image enhancement", "boost": 2.0}}},
                        {"match": {"domain_tags": {"query": "underwater vision", "boost": 1.8}}},
                        {"wildcard": {"source_file": {"value": "*underwater*"}}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        )
        evidence_terms.extend(["underwater", "水下"])

    should.append(
        {
            "multi_match": {
                "query": query,
                "fields": [
                    "title^3",
                    "main_task^2",
                    "paper_search_text^2",
                    "abstract_summary^1.5",
                    "method_summary^1.5",
                    "contribution_summary^1.5",
                    "summary",
                    "abstract",
                ],
            }
        }
    )
    bool_query: dict = {"should": should, "minimum_should_match": 1}
    if must:
        bool_query["must"] = must
    if task_id:
        bool_query["filter"] = [{"term": {"paper_id": task_id}}]
    return {"query": {"bool": bool_query}}, evidence_terms


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
    lowered = query.lower()
    if _contains_any(lowered, _IMAGE_HINTS):
        should.extend(
            [
                {"term": {"is_image_related": {"value": True, "boost": 2.0}}},
                {"term": {"modality_tags": {"value": "image", "boost": 1.8}}},
                {"term": {"domain_tags": {"value": "computer vision", "boost": 1.5}}},
            ]
        )
    if _contains_any(lowered, _FREQ_HINTS):
        should.extend(
            [
                {"term": {"is_frequency_related": {"value": True, "boost": 2.0}}},
                {"term": {"domain_tags": {"value": "frequency domain", "boost": 1.8}}},
                {"terms": {"method_tags": ["Fourier Transform", "FFT", "Wavelet Transform"], "boost": 2.0}},
            ]
        )
    if _contains_any(lowered, _UNDERWATER_HINTS):
        should.extend(
            [
                {"term": {"domain_tags": {"value": "underwater vision", "boost": 1.8}}},
                {"match_phrase": {"paper_search_text": {"query": "underwater", "boost": 1.8}}},
            ]
        )
    if _contains_any(lowered, _ATTENTION_HINTS):
        should.extend(
            [
                {"term": {"method_tags": {"value": "Attention", "boost": 1.8}}},
                {"match_phrase": {"paper_search_text": {"query": "attention", "boost": 1.5}}},
            ]
        )
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
    lowered = query.lower()
    if _contains_any(lowered, _IMAGE_HINTS):
        should.append({"term": {"is_image_related": {"value": True, "boost": 2.0}}})
    if _contains_any(lowered, _FREQ_HINTS):
        should.append({"term": {"is_frequency_related": {"value": True, "boost": 2.0}}})
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
    if _contains_any(lowered, _IMAGE_HINTS) and result.image_confidence >= 0.6:
        boost += 0.01
    if _contains_any(lowered, _FREQ_HINTS) and result.frequency_confidence >= 0.6:
        boost += 0.01
    if _contains_any(lowered, _UNDERWATER_HINTS) and any("underwater" in tag.lower() for tag in result.domain_tags + result.task_tags):
        boost += 0.01
    return boost


def _reason_for_deep_result(result: PaperSearchResult, query: str) -> str:
    reasons: list[str] = []
    if _contains_any(query, _IMAGE_HINTS) and result.image_confidence >= 0.6:
        reasons.append(f"图像相关置信度 {result.image_confidence:.2f}")
    if _contains_any(query, _FREQ_HINTS) and result.frequency_confidence >= 0.6:
        reasons.append(f"频域相关置信度 {result.frequency_confidence:.2f}")
    tags = result.task_tags + result.method_tags + result.domain_tags
    if tags:
        reasons.append(f"标签匹配：{', '.join(tags[:4])}")
    if result.evidence_chunks:
        reasons.append("候选论文内 chunk 命中查询证据")
    return "；".join(reasons) or "论文画像与正文片段共同匹配查询"


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
            "当前知识库中暂未检索到明确相关论文。你可以尝试使用更宽泛的关键词，"
            "例如“频域”“FFT”“spectrum”“image enhancement”等。"
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
            "当前知识库中没有检索到高置信度相关论文。建议尝试更宽泛关键词，"
            "如“频域”“FFT”“spectrum”“image enhancement”等。"
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

