"""Semantic memory — ES-backed vector search for long-term knowledge.

Maps to the "encyclopedia" layer from Hello Agents Ch.8.
Stores user preferences, research topics, paper notes with dense vectors.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from app.core.logging import logger
from app.services.memory.config import MemoryConfig
from app.services.memory.models import MemoryItem, MemoryType
from app.services.storage.es import get_es_client
from app.services.storage.embedding import get_embeddings


class SemanticMemory:
    """Elasticsearch-backed semantic memory with vector + keyword hybrid retrieval."""

    def __init__(self, config: MemoryConfig | None = None):
        self.config = config or MemoryConfig()
        self._embedding_model = get_embeddings()
        self._es = get_es_client()
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
                            "key": {"type": "keyword"},
                            "content": {"type": "text"},
                            "importance": {"type": "float"},
                            "access_count": {"type": "integer"},
                            "created_at": {"type": "date"},
                            "updated_at": {"type": "date"},
                            "last_accessed_at": {"type": "date"},
                            "embedding": {
                                "type": "dense_vector",
                                "dims": self.config.semantic_embedding_dims,
                            },
                        }
                    },
                },
            )
            logger.info("SemanticMemory: created index {}", index)

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
        try:
            self._es.index(
                index=self.config.semantic_es_index,
                id=item.memory_id,
                body=doc,
                refresh=True,
            )
        except Exception as exc:
            logger.warning("SemanticMemory: ES index failed for key={!r}: {}", item.key, exc)

        self._enforce_capacity()
        return item

    # ── retrieve ─────────────────────────────────────────────────────

    async def recall(
        self,
        query: str,
        scope: str | None = None,
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
            "sort": [{"importance": {"order": "desc"}}, {"updated_at": {"order": "desc"}}],
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
            result = self._es.search(index=self.config.semantic_es_index, body=body)
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
        vw = self.config.score_vector_weight
        kw = self.config.score_keyword_weight
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
        cutoff = datetime.now(timezone.utc).isoformat()
        try:
            result = self._es.delete_by_query(
                index=self.config.semantic_es_index,
                body={
                    "query": {
                        "bool": {
                            "must": [
                                {"range": {"updated_at": {"lte": cutoff, "boost": 1.0}}},
                            ],
                            "filter": [
                                {"script": {
                                    "script": {
                                        "source": f"(new Date().getTime() - doc['updated_at'].value.getMillis()) > {max_age_days * 86400000}L"
                                    }
                                }},
                            ],
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

    def _enforce_capacity(self) -> None:
        """Delete lowest-importance items if over capacity."""
        try:
            count_resp = self._es.count(index=self.config.semantic_es_index)
            total = count_resp.get("count", 0)
            overflow = total - self.config.semantic_max_items
            if overflow <= 0:
                return

            result = self._es.search(
                index=self.config.semantic_es_index,
                body={
                    "size": overflow,
                    "sort": [{"importance": {"order": "asc"}}, {"access_count": {"order": "asc"}}],
                    "_source": ["memory_id"],
                },
            )
            for hit in result["hits"]["hits"]:
                self._es.delete(index=self.config.semantic_es_index, id=hit["_id"])
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
