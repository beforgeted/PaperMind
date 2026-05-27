"""Markdown → token-sized parent/child chunks with ordered metadata.

Uses `chunk_indexing.index_documents` (parent docstore + child vectors) instead
of LangChain's default `ParentDocumentRetriever.add_documents`, so each parent
has `parent_index` / `prev_parent_id` / `next_parent_id` and each child has
`child_index` + `doc_id`.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings
from app.core.logging import logger
from app.core.schemas import ParsedDocument
from app.services.chunk_indexing import index_documents
from app.services.vectorstore_service import ensure_indices, get_es_client
from app.utils.chunking import parsed_to_documents


@dataclass
class IndexingStats:
    num_parents: int
    num_children: int


def run_indexing(parsed: ParsedDocument) -> IndexingStats:
    """Chunk → embed → bulk-index. Synchronous; call from a worker thread."""
    documents = parsed_to_documents(parsed)
    if not documents:
        logger.warning(
            "Task {}: empty markdown — skipping indexing.", parsed.task_id
        )
        return IndexingStats(num_parents=0, num_children=0)

    ensure_indices()
    index_documents(documents)

    # Refresh + count what landed for this task.
    client = get_es_client()
    client.indices.refresh(
        index=[settings.es_index_parent, settings.es_index_child]
    )
    num_parents = client.count(
        index=settings.es_index_parent,
        body={"query": {"term": {"metadata.task_id": parsed.task_id}}},
    )["count"]
    num_children = client.count(
        index=settings.es_index_child,
        body={"query": {"term": {"metadata.task_id": parsed.task_id}}},
    )["count"]

    logger.info(
        "Task {}: indexed {} parents / {} children",
        parsed.task_id,
        num_parents,
        num_children,
    )
    return IndexingStats(num_parents=num_parents, num_children=num_children)
