"""LangChain `ElasticsearchStore` wrapper for child chunks.

Uses dense-vector kNN search for child chunks. We avoid Elasticsearch's native
RRF hybrid mode because it is license-gated in some self-hosted distributions.

The store is keyed by `settings.es_index_child` and stores the embedding plus
the LangChain `Document.metadata`. `ensure_indices()` is idempotent and must be
called once at startup so metadata filters use stable field types.
"""

from __future__ import annotations

import logging
from typing import Optional

from elasticsearch import ApiError, Elasticsearch, NotFoundError
from langchain_elasticsearch import DenseVectorStrategy, ElasticsearchStore

from app.core.config import settings
from app.services.storage.embedding import get_embeddings

logger = logging.getLogger(__name__)


def _build_es_client() -> Elasticsearch:
    auth = None
    if settings.es_username:
        auth = (settings.es_username, settings.es_password or "")
    return Elasticsearch(
        hosts=settings.es_hosts_list,
        basic_auth=auth,
        request_timeout=30,
    )


def _chunk_order_metadata_fields() -> dict:
    """Ordered parent/child linkage fields (parent + child indices share one mapping)."""
    return {
        "doc_id": {"type": "keyword"},
        "parent_index": {"type": "integer"},
        "parent_count": {"type": "integer"},
        "prev_parent_id": {"type": "keyword"},
        "next_parent_id": {"type": "keyword"},
        "child_index": {"type": "integer"},
        "child_count": {"type": "integer"},
    }


def _shared_chunk_metadata_fields() -> dict:
    base = {
        "paper_id": {"type": "keyword"},
        "task_id": {"type": "keyword"},
        "original_filename": {"type": "keyword"},
        "title": {
            "type": "text",
            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
        },
        "section_title": {"type": "keyword"},
        "section_type": {"type": "keyword"},
        "subsection": {"type": "keyword"},
        "content_type": {"type": "keyword"},
        "entities": {"type": "keyword"},
        "keywords": {"type": "keyword"},
        "page": {"type": "integer"},
        "section_index": {"type": "integer"},
    }
    base.update(_chunk_order_metadata_fields())
    return base


def _child_mapping(dims: int) -> dict:
    return {
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "vector": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                },
                "metadata": {"properties": _shared_chunk_metadata_fields()},
            }
        }
    }


def _parent_mapping() -> dict:
    return {
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "metadata": {"properties": _shared_chunk_metadata_fields()},
            }
        }
    }


def _papers_mapping() -> dict:
    """论文级索引 mapping（每篇论文 1 条）。"""
    dims = get_embeddings().dim or settings.es_vector_dims
    return {
        "mappings": {
            "properties": {
                "paper_id": {"type": "keyword"},
                "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "abstract": {"type": "text"},
                "clean_abstract": {"type": "text"},
                "abstract_summary": {"type": "text"},
                "introduction_summary": {"type": "text"},
                "method_summary": {"type": "text"},
                "contribution_summary": {"type": "text"},
                "experiment_summary": {"type": "text"},
                "research_problem": {"type": "text"},
                "method_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "authors": {"type": "keyword"},
                "year": {"type": "keyword"},
                "source_file": {"type": "keyword"},
                "main_task": {"type": "text"},
                "modality_tags": {"type": "keyword"},
                "task_tags": {"type": "keyword"},
                "method_tags": {"type": "keyword"},
                "domain_tags": {"type": "keyword"},
                "dataset_tags": {"type": "keyword"},
                "metric_tags": {"type": "keyword"},
                "is_image_related": {"type": "boolean"},
                "is_frequency_related": {"type": "boolean"},
                "paper_search_text": {"type": "text"},
                "paper_search_vector": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                },
                "matched_keywords": {"type": "keyword"},
                "image_confidence": {"type": "float"},
                "frequency_confidence": {"type": "float"},
                "image_evidence": {"type": "object", "enabled": True},
                "frequency_evidence": {"type": "object", "enabled": True},
                "summary": {"type": "text"},
                "created_at": {"type": "date"},
                "updated_at": {"type": "date"},
            }
        }
    }


_vectorstore: Optional[ElasticsearchStore] = None
_indices_ready = False


def _metadata_mapping_update() -> dict:
    """Additive metadata fields only (safe on indices created before chunk ordering)."""
    return {"properties": {"metadata": {"properties": _chunk_order_metadata_fields()}}}


def _apply_metadata_mapping_update(client: Elasticsearch, index: str) -> None:
    try:
        client.indices.put_mapping(index=index, body=_metadata_mapping_update())
    except ApiError as exc:
        logger.warning("Index {} metadata mapping update skipped: {}", index, exc)


def get_es_client() -> Elasticsearch:
    """Bare ES client for low-level ops (indices admin, mget, delete_by_query)."""
    return _build_es_client()


def ensure_indices() -> None:
    """Create both indices with strict metadata typing if they do not exist.

    Safe to call repeatedly. Honours `embedding.dim` so the dense_vector
    mapping matches whatever backend is active (DashScope 1024, local model
    might differ).
    """
    global _indices_ready
    if _indices_ready:
        return

    client = _build_es_client()
    embeddings = get_embeddings()
    dims = embeddings.dim or settings.es_vector_dims

    if not client.indices.exists(index=settings.es_index_parent):
        client.indices.create(index=settings.es_index_parent, body=_parent_mapping())
        logger.info("Created index {}", settings.es_index_parent)
    else:
        _apply_metadata_mapping_update(client, settings.es_index_parent)
    if not client.indices.exists(index=settings.es_index_child):
        client.indices.create(
            index=settings.es_index_child, body=_child_mapping(dims)
        )
        logger.info(
            "Created index {} (dense_vector dims={})",
            settings.es_index_child,
            dims,
        )
    else:
        _apply_metadata_mapping_update(client, settings.es_index_child)
    if not client.indices.exists(index=settings.es_index_papers):
        client.indices.create(index=settings.es_index_papers, body=_papers_mapping())
        logger.info("Created index {}", settings.es_index_papers)
    else:
        try:
            client.indices.put_mapping(
                index=settings.es_index_papers,
                body=_papers_mapping()["mappings"],
            )
        except ApiError as exc:
            logger.warning(
                "Index {} mapping update skipped: {}", settings.es_index_papers, exc
            )

    _indices_ready = True


def get_vectorstore() -> ElasticsearchStore:
    """Singleton ElasticsearchStore configured for dense-vector retrieval."""
    global _vectorstore
    if _vectorstore is None:
        es_url = settings.es_hosts_list[0] if settings.es_hosts_list else settings.es_hosts
        _vectorstore = ElasticsearchStore(
            es_url=es_url,
            index_name=settings.es_index_child,
            embedding=get_embeddings(),
            strategy=DenseVectorStrategy(),
        )
        logger.info(
            "ElasticsearchStore ready (url={}, index={}, strategy=dense_vector)",
            es_url,
            settings.es_index_child,
        )
    return _vectorstore


def delete_by_task(task_id: str) -> None:
    """Best-effort cleanup of a task's child + parent docs (used for retries / dev)."""
    client = _build_es_client()
    for idx in (settings.es_index_parent, settings.es_index_child):
        try:
            client.delete_by_query(
                index=idx,
                body={"query": {"term": {"metadata.task_id": task_id}}},
                refresh=True,
            )
        except NotFoundError:
            pass
    try:
        client.delete(index=settings.es_index_papers, id=task_id, refresh=True)
    except NotFoundError:
        pass
