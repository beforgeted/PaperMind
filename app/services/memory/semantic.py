"""Semantic memory — ES-backed vector search for long-term knowledge.

Maps to the "encyclopedia" layer from Hello Agents Ch.8.
Stores user preferences, research topics, paper notes with dense vectors.
"""

from __future__ import annotations

import asyncio
import math
import uuid
from datetime import datetime, timezone
from typing import Any

import logging; logger = logging.getLogger(__name__)
from app.services.memory.config import MemoryConfig
from app.services.memory.models import MemoryItem, MemoryType
from app.services.storage.es import get_es_client
from app.services.storage.embedding import get_embeddings


class SemanticMemory:
    """Elasticsearch-backed semantic memory with vector + keyword hybrid retrieval."""

    def __init__(
        self,
        config: MemoryConfig | None = None,
        es_client: Any | None = None,
        embedding_model: Any | None = None,
    ):
        self.config = config or MemoryConfig()
        self._embedding_model = embedding_model or get_embeddings()
        self._es = es_client or get_es_client()
        self._ensure_index()

    # ── index management ─────────────────────────────────────────────

    def _ensure_index(self) -> None:
        """Create the semantic memory index if it doesn't exist."""
        index = self.config.semantic_es_index
        if not self._es.indices.exists(index=index):
            self._es.indices.create(
                index=index,
                body={
                    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
                    "mappings": {
                        "properties": {
                            "memory_id": {"type": "keyword"},
                            "memory_type": {"type": "keyword"},
                            "scope": {"type": "keyword"},
                            "user_id": {"type": "keyword"},
                            "project_id": {"type": "keyword"},
                            "memory_kind": {"type": "keyword"},
                            "key": {"type": "keyword"},
                            "content": {"type": "text"},
                            "importance": {"type": "float"},
                            "confidence": {"type": "float"},
                            "access_count": {"type": "integer"},
                            "created_at": {"type": "date"},
                            "updated_at": {"type": "date"},
                            "last_accessed_at": {"type": "date"},
                            "last_evidence_at": {"type": "date"},
                            "source_session_ids": {"type": "keyword"},
                            "embedding": {
                                "type": "dense_vector",
                                "dims": self.config.semantic_embedding_dims,
                            },
                        }
                    },
                },
            )
            logger.info("SemanticMemory: created index {}", index)
        else:
            try:
                self._es.indices.put_mapping(
                    index=index,
                    body={
                        "properties": {
                            "user_id": {"type": "keyword"},
                            "project_id": {"type": "keyword"},
                            "memory_kind": {"type": "keyword"},
                            "confidence": {"type": "float"},
                            "last_evidence_at": {"type": "date"},
                            "source_session_ids": {"type": "keyword"},
                        }
                    },
                )
            except Exception as exc:
                logger.debug("SemanticMemory: mapping update skipped: {}", exc)

    # ── encode / store ───────────────────────────────────────────────

    async def store(self, item: MemoryItem) -> MemoryItem:
        """Embed the content and index into ES. Enforces capacity limit."""
        item.memory_type = MemoryType.SEMANTIC
        item.updated_at = datetime.now(timezone.utc).isoformat()

        if not item.embedding:
            text_to_embed = f"{item.key}: {item.content}"[:2000]
            try:
                vectors = await self._embedding_model.aembed_documents([text_to_embed])
                item.embedding = vectors[0] if vectors else []
            except Exception as exc:
                logger.warning("SemanticMemory: embedding failed for key={!r}: {}", item.key, exc)
                item.embedding = []

        doc = item.model_dump(mode="json")
        for date_field in ("last_accessed_at", "last_evidence_at", "expires_at"):
            if doc.get(date_field) == "":
                doc.pop(date_field, None)
        try:
            await asyncio.to_thread(
                self._es.index,
                index=self.config.semantic_es_index,
                id=item.memory_id,
                body=doc,
                refresh=True,
            )
        except Exception as exc:
            logger.warning("SemanticMemory: ES index failed for key={!r}: {}", item.key, exc)

        await self._enforce_capacity()
        return item

    async def upsert_long_term_memory(
        self,
        *,
        key: str,
        content: str,
        memory_kind: str = "fact",
        scope: str = "user",
        user_id: str = "",
        project_id: str = "",
        source_session_ids: list[str] | None = None,
        confidence: float = 0.7,
        importance: float = 0.7,
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> MemoryItem:
        """Insert or merge a distilled long-term semantic memory."""
        now = datetime.now(timezone.utc).isoformat()
        stable_id = memory_id or self._stable_memory_id(user_id, project_id, memory_kind, key)
        source_ids = list(dict.fromkeys(source_session_ids or []))
        existing = await self._get_existing(stable_id)

        if existing:
            old_sources = list(existing.source_session_ids or [])
            source_ids = list(dict.fromkeys([*old_sources, *source_ids]))
            content = self._merge_content(existing.content, content)
            importance = max(existing.importance, importance)
            confidence = max(existing.confidence, confidence)
            created_at = existing.created_at
            access_count = existing.access_count
        else:
            created_at = now
            access_count = 0

        item = MemoryItem(
            memory_id=stable_id,
            memory_type=MemoryType.SEMANTIC,
            scope=scope,  # type: ignore[arg-type]
            key=key,
            content=content,
            user_id=user_id,
            project_id=project_id,
            memory_kind=memory_kind,  # type: ignore[arg-type]
            source_session_ids=source_ids,
            confidence=confidence,
            importance=importance,
            metadata=metadata or {},
            created_at=created_at,
            updated_at=now,
            last_evidence_at=now,
            access_count=access_count,
        )
        return await self.store(item)

    # ── retrieve ─────────────────────────────────────────────────────

    async def recall(
        self,
        query: str,
        scope: str | None = None,
        user_id: str = "",
        project_id: str = "",
        memory_kind: str = "",
        limit: int | None = None,
    ) -> list[MemoryItem]:
        """Semantic + keyword hybrid retrieval with scoring formula."""
        if limit is None:
            limit = self.config.semantic_retrieval_top_k

        # Get query embedding for vector search
        query_vector: list[float] = []
        try:
            vectors = await self._embedding_model.aembed_documents([query[:2000]])
            query_vector = vectors[0] if vectors else []
        except Exception as exc:
            logger.warning("SemanticMemory: query embedding failed: {}", exc)

        # Build hybrid search body
        must_clauses: list[dict] = []
        if scope:
            must_clauses.append({"term": {"scope": scope}})
        if user_id:
            must_clauses.append({"term": {"user_id": user_id}})
        if project_id:
            must_clauses.append({"term": {"project_id": project_id}})
        if memory_kind:
            must_clauses.append({"term": {"memory_kind": memory_kind}})

        body: dict[str, Any] = {
            "size": limit,
            "query": {
                "bool": {
                    "must": must_clauses,
                    "should": [
                        {"match": {"content": {"query": query, "boost": 1.0}}},
                        {"match": {"key": {"query": query, "boost": 2.0}}},
                    ],
                }
            },
        }

        if query_vector:
            body["query"]["bool"]["should"].append({
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_vector},
                    },
                    "boost": 2.0,
                }
            })

        try:
            result = await asyncio.to_thread(
                self._es.search, index=self.config.semantic_es_index, body=body
            )
        except Exception as exc:
            logger.warning("SemanticMemory: ES search failed: {}", exc)
            return []

        items: list[MemoryItem] = []
        for hit in result["hits"]["hits"]:
            src = hit["_source"]
            item = MemoryItem(**{k: v for k, v in src.items() if k != "embedding"})
            item.embedding = src.get("embedding")

            # Apply scoring formula
            score = self._compute_score(item, hit["_score"] or 0.0, query)
            if score >= self.config.semantic_min_score:
                item.metadata["_retrieval_score"] = score
                item.touch()
                items.append(item)

        items.sort(key=lambda i: i.metadata.get("_retrieval_score", 0), reverse=True)
        logger.debug("SemanticMemory: recall query={!r} → {} results", query[:60], len(items))
        return items[:limit]

    def _compute_score(self, item: MemoryItem, es_score: float, query: str) -> float:
        """Retrieval scoring formula: (vec × kw) × time_decay × importance.

        score = (sim_w + kw_w) × e^(-λ × days) × (0.5 + imp_w × importance)
        """
        lamb = self.config.score_time_decay_lambda
        iw = self.config.score_importance_weight

        # Time decay
        try:
            updated = datetime.fromisoformat(item.updated_at.replace("Z", "+00:00"))
            days_ago = (datetime.now(timezone.utc) - updated).total_seconds() / 86400
        except Exception:
            days_ago = 0
        time_weight = math.exp(-lamb * days_ago)

        # Base score from ES (_score is normalized BM25 + vector boost)
        base_score = min(es_score / 5.0, 1.0)  # normalize rough ES score to [0, 1]

        # Combined
        raw = base_score * time_weight * (0.5 + iw * item.importance)
        return round(raw, 4)

    # ── forget ───────────────────────────────────────────────────────

    def forget(self, memory_id: str) -> bool:
        """Remove a semantic memory by ID."""
        try:
            self._es.delete(index=self.config.semantic_es_index, id=memory_id, refresh=True)
            return True
        except Exception:
            return False

    def forget_stale(self, max_age_days: int | None = None) -> int:
        """Delete items older than max_age_days. Returns count removed."""
        max_age = max_age_days or self.config.forget_max_age_days
        try:
            result = self._es.delete_by_query(
                index=self.config.semantic_es_index,
                body={
                    "query": {
                        "range": {
                            "updated_at": {
                                "lt": f"now-{max_age}d",
                            }
                        }
                    }
                },
                refresh=True,
            )
            removed = result.get("deleted", 0)
            if removed:
                logger.info("SemanticMemory: forgot {} stale items (>{} days)", removed, max_age)
            return removed
        except Exception as exc:
            logger.warning("SemanticMemory: forget_stale failed: {}", exc)
            return 0

    # ── internal ─────────────────────────────────────────────────────

    async def _enforce_capacity(self) -> None:
        """Delete lowest-importance items if over capacity."""
        try:
            count_resp = await asyncio.to_thread(
                self._es.count, index=self.config.semantic_es_index
            )
            total = count_resp.get("count", 0)
            overflow = total - self.config.semantic_max_items
            if overflow <= 0:
                return

            result = await asyncio.to_thread(
                self._es.search,
                index=self.config.semantic_es_index,
                body={
                    "size": overflow,
                    "sort": [{"importance": {"order": "asc"}}, {"access_count": {"order": "asc"}}],
                    "_source": ["memory_id"],
                },
            )
            for hit in result["hits"]["hits"]:
                await asyncio.to_thread(
                    self._es.delete, index=self.config.semantic_es_index, id=hit["_id"]
                )
            if overflow > 0:
                logger.info("SemanticMemory: evicted {} low-importance items (capacity)", overflow)
        except Exception as exc:
            logger.warning("SemanticMemory: capacity enforcement failed: {}", exc)

    @property
    def size(self) -> int:
        try:
            return self._es.count(index=self.config.semantic_es_index).get("count", 0)
        except Exception:
            return 0

    async def _get_existing(self, memory_id: str) -> MemoryItem | None:
        try:
            result = await asyncio.to_thread(
                self._es.get, index=self.config.semantic_es_index, id=memory_id
            )
            source = result.get("_source", {})
            return MemoryItem(**{k: v for k, v in source.items() if k != "embedding"})
        except Exception:
            return None

    @staticmethod
    def _stable_memory_id(user_id: str, project_id: str, memory_kind: str, key: str) -> str:
        raw = f"{user_id}|{project_id}|{memory_kind}|{key}".encode("utf-8")
        return uuid.uuid5(uuid.NAMESPACE_URL, raw.decode("utf-8")).hex[:16]

    @staticmethod
    def _merge_content(existing: str, incoming: str) -> str:
        existing_clean = existing.strip()
        incoming_clean = incoming.strip()
        if not existing_clean:
            return incoming_clean
        if not incoming_clean or incoming_clean in existing_clean:
            return existing_clean
        if existing_clean in incoming_clean:
            return incoming_clean
        return f"{existing_clean}\n{incoming_clean}"
