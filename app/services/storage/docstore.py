"""ES-backed `BaseStore[str, Document]` for parent documents.

`ParentDocumentRetriever` calls:
- `mset` on add (parent_id -> Document)
- `mget` on retrieve (parent_ids returned by the child vectorstore search)

Storing parents in ES (rather than in-memory) keeps the architecture
single-source-of-truth and survives API/worker restarts. Each parent doc body
is `{text, metadata}`, indexed by `parent_id` as the ES `_id`.
"""

from __future__ import annotations

import logging
from typing import Iterator, List, Optional, Sequence, Tuple

from elasticsearch import NotFoundError, helpers
from langchain_core.documents import Document
from langchain_core.stores import BaseStore

from app.core.config import settings
from app.services.storage.es import get_es_client

logger = logging.getLogger(__name__)


class ESDocumentStore(BaseStore[str, Document]):
    """Parent-document store backed by `papermind_parents`."""

    def __init__(self) -> None:
        self._client = get_es_client()
        self._index = settings.es_index_parent

    # ---------------- mget ----------------

    def mget(self, keys: Sequence[str]) -> List[Optional[Document]]:
        if not keys:
            return []
        try:
            resp = self._client.mget(index=self._index, body={"ids": list(keys)})
        except NotFoundError:
            return [None] * len(keys)
        out: List[Optional[Document]] = []
        for doc in resp.get("docs", []):
            if not doc.get("found"):
                out.append(None)
                continue
            src = doc.get("_source", {})
            out.append(
                Document(
                    page_content=src.get("text", ""),
                    metadata=src.get("metadata", {}),
                )
            )
        return out

    # ---------------- mset ----------------

    def mset(self, key_value_pairs: Sequence[Tuple[str, Document]]) -> None:
        if not key_value_pairs:
            return
        actions = (
            {
                "_op_type": "index",
                "_index": self._index,
                "_id": key,
                "_source": {
                    "text": doc.page_content,
                    "metadata": doc.metadata or {},
                },
            }
            for key, doc in key_value_pairs
        )
        success, errors = helpers.bulk(self._client, actions, refresh=False)
        if errors:
            logger.error("Parent bulk index reported {} errors", len(errors))

    # ---------------- mdelete ----------------

    def mdelete(self, keys: Sequence[str]) -> None:
        if not keys:
            return
        actions = (
            {"_op_type": "delete", "_index": self._index, "_id": key} for key in keys
        )
        helpers.bulk(self._client, actions, refresh=False, raise_on_error=False)

    # ---------------- yield_keys ----------------

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        query: dict = {"match_all": {}} if not prefix else {
            "prefix": {"metadata.parent_id": prefix}
        }
        # Scroll through all matching ids, source-disabled.
        resp = self._client.search(
            index=self._index,
            body={"query": query, "_source": False},
            scroll="2m",
            size=500,
        )
        scroll_id = resp.get("_scroll_id")
        try:
            while True:
                hits = resp["hits"]["hits"]
                if not hits:
                    break
                for h in hits:
                    yield h["_id"]
                resp = self._client.scroll(scroll_id=scroll_id, scroll="2m")
                scroll_id = resp.get("_scroll_id")
        finally:
            if scroll_id:
                try:
                    self._client.clear_scroll(scroll_id=scroll_id)
                except Exception:  # noqa: BLE001
                    pass


_docstore: Optional[ESDocumentStore] = None


def get_docstore() -> ESDocumentStore:
    global _docstore
    if _docstore is None:
        _docstore = ESDocumentStore()
    return _docstore
