"""Markdown → ParentDocumentRetriever.add_documents.

ParentDocumentRetriever does:
  1. parent_splitter.split_documents(docs)        # parent chunks
  2. child_splitter.split_documents(parent_chunks) # child chunks (vectorstore)
  3. vectorstore.add_documents(child_chunks)
  4. docstore.mset([(parent_id, parent_chunk)])

We just hand it a single Document per paper and read back the resulting
counts via the docstore + vectorstore for status reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings
from app.core.logging import logger
from app.core.schemas import ParsedDocument
from app.services.retriever_service import get_parent_retriever
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
    retriever = get_parent_retriever()
    retriever.add_documents(documents)

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
