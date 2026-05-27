"""`ParentDocumentRetriever` factories.

We keep two splitter instances (parent + child) shared across API and worker
processes, plus a default retriever singleton. The `for_task(task_id)` helper
returns a *new* retriever bound to a filtered child vectorstore (LangChain's
`as_retriever(search_kwargs={"filter": ...})` plumbing isn't directly exposed
by ParentDocumentRetriever, so we override `child_metadata_fields` /
`search_kwargs` instead).

Note: the indexing path always calls `add_documents` on the *unfiltered*
retriever. Filtering only applies on retrieve.
"""

from __future__ import annotations

from typing import Optional

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.services.docstore_service import get_docstore
from app.services.vectorstore_service import get_vectorstore
from app.utils.token_splitting import child_text_splitter, parent_text_splitter


def _parent_splitter() -> RecursiveCharacterTextSplitter:
    return parent_text_splitter()


def _child_splitter() -> RecursiveCharacterTextSplitter:
    return child_text_splitter()


_default: Optional[ParentDocumentRetriever] = None


def get_parent_retriever() -> ParentDocumentRetriever:
    """Singleton retriever used for indexing and unfiltered retrieval."""
    global _default
    if _default is None:
        _default = ParentDocumentRetriever(
            vectorstore=get_vectorstore(),
            docstore=get_docstore(),
            child_splitter=_child_splitter(),
            parent_splitter=_parent_splitter(),
        )
    return _default


def get_parent_retriever_for(
    task_id: Optional[str] = None,
    top_k: Optional[int] = None,
) -> ParentDocumentRetriever:
    """Build a retriever scoped to a single task (or unscoped if task_id is None).

    Re-uses the same vectorstore + docstore + splitters; only the child
    `search_kwargs` differ. We construct a fresh `ParentDocumentRetriever`
    rather than mutating the singleton so concurrent queries with different
    `task_id` filters don't race.
    """
    search_kwargs: dict = {"k": top_k or settings.retrieve_top_k_children}
    if task_id:
        search_kwargs["filter"] = [{"term": {"metadata.task_id": task_id}}]

    return ParentDocumentRetriever(
        vectorstore=get_vectorstore(),
        docstore=get_docstore(),
        child_splitter=_child_splitter(),
        parent_splitter=_parent_splitter(),
        search_kwargs=search_kwargs,
    )
