"""Hybrid retrieval with application-side RRF.

Elasticsearch Basic supports BM25 and dense-vector kNN, but native RRF can be
license-gated. We run BM25 and kNN separately on child chunks, fuse child hits
by their parent id in Python, then load parent chunks from the ES docstore.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.documents import Document

from app.core.config import settings
from app.core.logging import logger
from app.core.schemas import RetrievedChunk
from app.services.docstore_service import get_docstore
from app.services.embedding_service import get_embeddings
from app.services.vectorstore_service import get_es_client
from app.utils.paper_structure import extract_entities


@dataclass
class _FusedParentHit:
    parent_id: str
    score: float = 0.0
    child_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _RetrievalIntent:
    section_types: list[str]
    entities: list[str]


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CJK_ASCII_BOUNDARY_RE = re.compile(
    r"(?<=[\u4e00-\u9fff])(?=[A-Za-z0-9])|(?<=[A-Za-z0-9])(?=[\u4e00-\u9fff])"
)
_QUERY_ENTITY_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]{1,}")

_ZH_QUERY_EXPANSIONS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("怎么实现", "如何实现", "实现", "怎么做", "如何做"), ("implementation", "implemented", "how it works")),
    (("怎么设计", "如何设计", "设计"), ("design", "architecture", "structure")),
    (("方法", "模块", "网络", "架构", "框架"), ("method", "module", "network", "architecture", "framework")),
    (("提出", "贡献", "创新"), ("propose", "contribution", "novel")),
    (("问题", "挑战", "动机"), ("problem", "challenge", "motivation")),
    (("实验", "结果", "指标", "对比", "评估"), ("experiment", "result", "metric", "comparison", "evaluation")),
    (("消融",), ("ablation",)),
    (("数据集", "数据"), ("dataset", "data")),
    (("结论", "总结"), ("conclusion", "summary")),
    (("简单回答", "简要", "简洁"), ("briefly",)),
)


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


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


def _expand_query_for_retrieval(query: str, intent: _RetrievalIntent) -> str:
    """Expand Chinese questions with English terms used by paper text."""
    if not _contains_cjk(query):
        return query

    additions: list[str] = []
    for triggers, translations in _ZH_QUERY_EXPANSIONS:
        if any(trigger in query for trigger in triggers):
            additions.extend(translations)
    for section_type in intent.section_types:
        additions.append(section_type.replace("_", " "))

    expanded_terms = _dedupe_terms(additions)
    if not expanded_terms:
        return query
    return f"{query} {' '.join(expanded_terms)}"


def _extract_query_entities(query: str) -> list[str]:
    spaced_query = _CJK_ASCII_BOUNDARY_RE.sub(" ", query)
    entities = set(extract_entities(spaced_query))
    for token in _QUERY_ENTITY_RE.findall(spaced_query):
        if token.isupper() or token.endswith("Net") or any(char.isdigit() for char in token):
            entities.add(token)
    return sorted(entities, key=lambda item: (item.lower(), item))


def _infer_intent(query: str) -> _RetrievalIntent:
    normalized = query.lower()
    section_types: list[str] = []

    def add_section(section_type: str) -> None:
        if section_type not in section_types:
            section_types.append(section_type)

    if any(word in normalized for word in ("消融", "ablation")):
        add_section("ablation")
    if any(word in normalized for word in ("实验", "结果", "对比", "指标", "quantitative", "qualitative", "psnr", "ssim", "uiebd", "euvp", "lsui", "experiment", "evaluation", "comparison")):
        add_section("experiment")
    if any(word in normalized for word in ("怎么设计", "如何设计", "模块", "架构", "网络", "方法", "设计", "method", "approach", "architecture", "module", "network")):
        add_section("method")
    if any(word in normalized for word in ("相关工作", "related work", "background")):
        add_section("related_work")
    if any(word in normalized for word in ("摘要", "abstract")):
        add_section("abstract")
    if any(word in normalized for word in ("讨论", "局限", "discussion", "limitation")):
        add_section("discussion")
    if any(word in normalized for word in ("结论", "conclusion", "future work")):
        add_section("conclusion")
    if any(word in normalized for word in ("图", "figure", "fig.")):
        add_section("figure_caption")
    if any(word in normalized for word in ("表", "table")):
        add_section("table_caption")
    if any(word in normalized for word in ("参考文献", "reference", "bibliography")):
        add_section("reference")

    return _RetrievalIntent(section_types=section_types, entities=_extract_query_entities(query))


def _task_filter(task_id: Optional[str]) -> Optional[dict]:
    if not task_id:
        return None
    return {"term": {"metadata.task_id": task_id}}


def _metadata_filters(task_id: Optional[str], intent: _RetrievalIntent, *, section_filter: bool) -> list[dict]:
    filters: list[dict] = []
    filter_clause = _task_filter(task_id)
    if filter_clause:
        filters.append(filter_clause)
    if section_filter and intent.section_types:
        filters.append({"terms": {"metadata.section_type": intent.section_types}})
    return filters


def _metadata_should(intent: _RetrievalIntent) -> list[dict]:
    should: list[dict] = []
    if intent.entities:
        should.append({"terms": {"metadata.entities": intent.entities, "boost": 3.0}})
        should.append({"terms": {"metadata.keywords": intent.entities, "boost": 1.5}})
    if intent.section_types:
        should.append({"terms": {"metadata.section_type": intent.section_types, "boost": 2.0}})
    return should


def _bm25_query(query: str, task_id: Optional[str], intent: _RetrievalIntent, *, section_filter: bool) -> dict:
    text_query = {"match": {"text": query}}
    filter_clauses = _metadata_filters(task_id, intent, section_filter=section_filter)
    should_clauses = _metadata_should(intent)
    if not filter_clauses and not should_clauses:
        return text_query
    bool_query: dict = {"must": [text_query]}
    if filter_clauses:
        bool_query["filter"] = filter_clauses
    if should_clauses:
        bool_query["should"] = should_clauses
    return {"bool": bool_query}


def _collect_parent_ranks(hits: list[dict]) -> dict[str, tuple[int, list[str]]]:
    ranks: dict[str, tuple[int, list[str]]] = {}
    for rank, hit in enumerate(hits, start=1):
        metadata = (hit.get("_source") or {}).get("metadata") or {}
        parent_id = metadata.get("doc_id")
        if not parent_id:
            continue
        child_id = hit.get("_id")
        if parent_id not in ranks:
            ranks[parent_id] = (rank, [child_id] if child_id else [])
        elif child_id:
            ranks[parent_id][1].append(child_id)
    return ranks


def _rrf_fuse(*rank_sets: dict[str, tuple[int, list[str]]]) -> list[_FusedParentHit]:
    fused: dict[str, _FusedParentHit] = {}
    for rank_set in rank_sets:
        for parent_id, (rank, child_ids) in rank_set.items():
            item = fused.setdefault(parent_id, _FusedParentHit(parent_id=parent_id))
            item.score += 1.0 / (settings.rrf_k + rank)
            for child_id in child_ids:
                if child_id not in item.child_ids:
                    item.child_ids.append(child_id)
    return sorted(fused.values(), key=lambda item: item.score, reverse=True)


def _search_bm25(
    query: str,
    task_id: Optional[str],
    intent: _RetrievalIntent,
    size: int,
    *,
    section_filter: bool,
) -> dict[str, tuple[int, list[str]]]:
    client = get_es_client()
    response = client.search(
        index=settings.es_index_child,
        body={
            "size": size,
            "_source": ["metadata.doc_id"],
            "query": _bm25_query(query, task_id, intent, section_filter=section_filter),
        },
    )
    return _collect_parent_ranks(response["hits"]["hits"])


def _search_knn(
    query: str,
    task_id: Optional[str],
    intent: _RetrievalIntent,
    size: int,
    *,
    section_filter: bool,
) -> dict[str, tuple[int, list[str]]]:
    client = get_es_client()
    vector = get_embeddings().embed_query(query)
    knn: dict = {
        "field": "vector",
        "query_vector": vector,
        "k": size,
        "num_candidates": max(size * 5, 50),
    }
    filter_clauses = _metadata_filters(task_id, intent, section_filter=section_filter)
    if len(filter_clauses) == 1:
        knn["filter"] = filter_clauses[0]
    elif filter_clauses:
        knn["filter"] = {"bool": {"filter": filter_clauses}}

    response = client.search(
        index=settings.es_index_child,
        body={
            "size": size,
            "_source": ["metadata.doc_id"],
            "knn": knn,
        },
    )
    return _collect_parent_ranks(response["hits"]["hits"])


def _search_with_structural_fallback(
    query: str,
    task_id: Optional[str],
    intent: _RetrievalIntent,
) -> tuple[dict[str, tuple[int, list[str]]], dict[str, tuple[int, list[str]]], bool]:
    section_filter = bool(intent.section_types)
    bm25_ranks = _search_bm25(
        query,
        task_id,
        intent,
        settings.retrieve_top_k_bm25,
        section_filter=section_filter,
    )
    knn_ranks = _search_knn(
        query,
        task_id,
        intent,
        settings.retrieve_top_k_knn,
        section_filter=section_filter,
    )
    if section_filter and not bm25_ranks and not knn_ranks:
        logger.debug(
            "Structured recall empty for sections {}; falling back to broad search.",
            intent.section_types,
        )
        bm25_ranks = _search_bm25(
            query,
            task_id,
            intent,
            settings.retrieve_top_k_bm25,
            section_filter=False,
        )
        knn_ranks = _search_knn(
            query,
            task_id,
            intent,
            settings.retrieve_top_k_knn,
            section_filter=False,
        )
        return bm25_ranks, knn_ranks, False
    return bm25_ranks, knn_ranks, section_filter


def _retrieve_parent_documents_sync(
    query: str,
    *,
    top_k: Optional[int] = None,
    task_id: Optional[str] = None,
) -> List[Document]:
    final_top_k = top_k or settings.final_top_k
    intent = _infer_intent(query)
    retrieval_query = _expand_query_for_retrieval(query, intent)
    bm25_ranks, knn_ranks, used_section_filter = _search_with_structural_fallback(
        retrieval_query,
        task_id,
        intent,
    )
    fused_hits = _rrf_fuse(bm25_ranks, knn_ranks)[:final_top_k]

    parent_ids = [hit.parent_id for hit in fused_hits]
    parents = get_docstore().mget(parent_ids)
    out: list[Document] = []
    for hit, doc in zip(fused_hits, parents):
        if doc is None:
            continue
        metadata = dict(doc.metadata or {})
        metadata["doc_id"] = hit.parent_id
        metadata["_hybrid_score"] = hit.score
        metadata["_child_ids"] = hit.child_ids
        metadata["_retrieval_intent"] = {
            "section_types": intent.section_types,
            "entities": intent.entities,
            "used_section_filter": used_section_filter,
            "retrieval_query": retrieval_query,
        }
        out.append(Document(page_content=doc.page_content, metadata=metadata))
    logger.debug(
        "Hybrid recall: {} parents (bm25={}, knn={}, task_id={}, sections={}, entities={}, expanded_query={})",
        len(out),
        len(bm25_ranks),
        len(knn_ranks),
        task_id,
        intent.section_types,
        intent.entities,
        retrieval_query != query,
    )
    return out


def _to_retrieved_chunk(doc, rank: int) -> RetrievedChunk:
    md = doc.metadata or {}
    # ParentDocumentRetriever uses MultiVectorRetriever's id_key default
    # ("doc_id"), which holds the parent's docstore key (a UUID).
    return RetrievedChunk(
        parent_id=md.get("doc_id") or f"rank-{rank}",
        parent_text=doc.page_content,
        child_ids=md.get("_child_ids") or [],
        score=md.get("_hybrid_score") or 1.0 / (rank + 1),
        metadata={
            "paper_id": md.get("paper_id") or md.get("task_id"),
            "task_id": md.get("task_id"),
            "original_filename": md.get("original_filename"),
            "title": md.get("title"),
            "section_title": md.get("section_title"),
            "section_type": md.get("section_type"),
            "subsection": md.get("subsection"),
            "page": md.get("page"),
            "keywords": md.get("keywords") or [],
            "entities": md.get("entities") or [],
            "content_type": md.get("content_type"),
            "retrieval_intent": md.get("_retrieval_intent"),
        },
    )


async def retrieve_parent_documents(
    query: str,
    top_k: Optional[int] = None,
    task_id: Optional[str] = None,
) -> List[Document]:
    return await asyncio.to_thread(
        _retrieve_parent_documents_sync,
        query,
        top_k=top_k,
        task_id=task_id,
    )


async def hybrid_retrieve(
    query: str,
    top_k: Optional[int] = None,
    task_id: Optional[str] = None,
) -> List[RetrievedChunk]:
    parent_docs = await retrieve_parent_documents(query, top_k=top_k, task_id=task_id)
    return [_to_retrieved_chunk(d, i) for i, d in enumerate(parent_docs)]
