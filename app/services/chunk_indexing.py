"""Parent/child indexing with ordered metadata and explicit parent links.

Replaces `ParentDocumentRetriever.add_documents` so each parent stores
`parent_index`, `prev_parent_id`, `next_parent_id`, and each child stores
`child_index` plus `doc_id` (parent key). Vectors are written only for children.
"""

from __future__ import annotations

import uuid
from typing import List, Sequence, Tuple

from langchain_core.documents import Document

from app.core.logging import logger
from app.services.docstore_service import get_docstore
from app.services.vectorstore_service import get_vectorstore
from app.utils.token_splitting import child_text_splitter, parent_text_splitter

ParentPair = Tuple[str, Document]


def _annotate_parent_chain(parents: List[Document]) -> List[ParentPair]:
    """Split section docs into parents and stamp sequential + neighbor metadata."""
    if not parents:
        return []

    parent_ids = [str(uuid.uuid4()) for _ in parents]
    count = len(parents)
    pairs: List[ParentPair] = []

    for index, parent_doc in enumerate(parents):
        metadata = dict(parent_doc.metadata or {})
        metadata["parent_index"] = index
        metadata["parent_count"] = count
        if index > 0:
            metadata["prev_parent_id"] = parent_ids[index - 1]
        if index < count - 1:
            metadata["next_parent_id"] = parent_ids[index + 1]
        pairs.append(
            (parent_ids[index], Document(page_content=parent_doc.page_content, metadata=metadata))
        )
    return pairs


def _split_section_into_parents(section_doc: Document) -> List[ParentPair]:
    splitter = parent_text_splitter()
    parents = splitter.split_documents([section_doc])
    return _annotate_parent_chain(parents)


def _split_parent_into_children(
    parent_id: str,
    parent_doc: Document,
) -> List[Document]:
    """Child chunks inherit parent metadata and add child_index + doc_id."""
    splitter = child_text_splitter()
    raw_children = splitter.split_documents([parent_doc])
    child_count = len(raw_children)
    children: List[Document] = []

    for child_index, child in enumerate(raw_children):
        metadata = dict(parent_doc.metadata or {})
        metadata["doc_id"] = parent_id
        metadata["child_index"] = child_index
        metadata["child_count"] = child_count
        children.append(
            Document(page_content=child.page_content, metadata=metadata)
        )
    return children


def build_parent_child_batches(
    section_documents: Sequence[Document],
) -> Tuple[List[ParentPair], List[Document]]:
    """Build all parent store pairs and child vectorstore documents."""
    parent_pairs: List[ParentPair] = []
    child_documents: List[Document] = []

    for section_doc in section_documents:
        section_parents = _split_section_into_parents(section_doc)
        parent_pairs.extend(section_parents)
        for parent_id, parent_doc in section_parents:
            child_documents.extend(_split_parent_into_children(parent_id, parent_doc))

    return parent_pairs, child_documents


def index_documents(section_documents: List[Document]) -> Tuple[int, int]:
    """Persist parents to docstore and embed/index children. Returns (parents, children)."""
    if not section_documents:
        return 0, 0

    parent_pairs, child_documents = build_parent_child_batches(section_documents)
    if not parent_pairs:
        return 0, 0

    get_docstore().mset(parent_pairs)
    if child_documents:
        get_vectorstore().add_documents(child_documents)

    logger.debug(
        "Chunk index: {} parents, {} children from {} sections",
        len(parent_pairs),
        len(child_documents),
        len(section_documents),
    )
    return len(parent_pairs), len(child_documents)


__all__ = [
    "build_parent_child_batches",
    "index_documents",
]
