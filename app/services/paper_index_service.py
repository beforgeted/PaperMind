"""Paper-level index operations for `settings.es_index_papers`."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from elasticsearch import NotFoundError

from app.core.config import settings
from app.core.logging import logger
from app.core.schemas import PaperProfile, PaperSearchResult
from app.services.embedding_service import get_embeddings
from app.services.vectorstore_service import get_es_client


def upsert_paper_profile(profile: PaperProfile) -> None:
    """写入或更新论文画像，不让失败影响主链路。"""
    now = datetime.utcnow()
    payload = profile.model_dump(mode="json")
    payload["updated_at"] = now.isoformat()
    if not payload.get("created_at"):
        payload["created_at"] = now.isoformat()
    if profile.paper_search_text:
        try:
            payload["paper_search_vector"] = get_embeddings().embed_query(profile.paper_search_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Paper profile embedding skipped (paper_id={}): {}", profile.paper_id, exc)
    client = get_es_client()
    client.index(
        index=settings.es_index_papers,
        id=profile.paper_id,
        document=payload,
        refresh=True,
    )


def get_paper_profile(paper_id: str) -> Optional[PaperProfile]:
    """按 paper_id 获取论文画像。"""
    client = get_es_client()
    try:
        res = client.get(index=settings.es_index_papers, id=paper_id)
    except NotFoundError:
        return None
    return PaperProfile.model_validate(res.get("_source") or {})


def search_papers(query_body: dict, size: int) -> list[PaperSearchResult]:
    """执行论文级 ES 查询，并转换为标准结果。"""
    client = get_es_client()
    response = client.search(
        index=settings.es_index_papers,
        body={**query_body, "size": size},
    )
    out: list[PaperSearchResult] = []
    for hit in response.get("hits", {}).get("hits", []):
        source = hit.get("_source") or {}
        out.append(
            PaperSearchResult(
                paper_id=source.get("paper_id") or hit.get("_id", ""),
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
                score=float(hit.get("_score") or 0.0),
            )
        )
    return out


def fetch_paper_evidence_chunks(
    paper_id: str,
    keywords: list[str],
    *,
    limit: int = 2,
) -> list[str]:
    """从 chunk 索引提取少量证据片段，用于解释“为何相关”。"""
    client = get_es_client()
    should = [{"match_phrase": {"text": kw}} for kw in keywords if kw]
    body: dict = {
        "size": limit,
        "_source": ["text"],
        "query": {
            "bool": {
                "filter": [{"term": {"metadata.task_id": paper_id}}],
            }
        },
    }
    if should:
        body["query"]["bool"]["should"] = should
        body["query"]["bool"]["minimum_should_match"] = 1
    try:
        response = client.search(index=settings.es_index_child, body=body)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fetch paper evidence failed (paper_id={}): {}", paper_id, exc)
        return []

    snippets: list[str] = []
    for hit in response.get("hits", {}).get("hits", []):
        text = ((hit.get("_source") or {}).get("text") or "").strip()
        if text:
            snippets.append(text[:240].replace("\n", " "))
    return snippets

