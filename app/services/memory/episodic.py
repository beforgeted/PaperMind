"""Episodic memory — per-session permanent storage backed by ES.

One document per session, each turn appended with full original text.
Redis is the hot cache (10 turns), ES is the permanent archive.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.core.logging import logger
from app.services.memory.config import MemoryConfig
from app.services.storage.es import get_es_client


class EpisodicMemory:
    """ES-backed per-session document store with turns array."""

    def __init__(self, config: MemoryConfig | None = None):
        self.config = config or MemoryConfig()
        self._es = get_es_client()
        self._ensure_index()

    # ── index ────────────────────────────────────────────────────────

    def _ensure_index(self) -> None:
        index = self.config.episodic_es_index
        if not self._es.indices.exists(index=index):
            self._es.indices.create(
                index=index,
                body={
                    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
                    "mappings": {
                        "properties": {
                            "session_id": {"type": "keyword"},
                            "title": {"type": "text"},
                            "created_at": {"type": "date"},
                            "updated_at": {"type": "date"},
                            "turn_count": {"type": "integer"},
                            "status": {"type": "keyword"},
                            "turns": {
                                "type": "nested",
                                "properties": {
                                    "turn_id": {"type": "keyword"},
                                    "role": {"type": "keyword"},
                                    "content": {"type": "text"},
                                    "intent": {"type": "keyword"},
                                    "used_tools": {"type": "keyword"},
                                    "timestamp": {"type": "date"},
                                },
                            },
                        }
                    },
                },
            )
            logger.info("EpisodicMemory: created index {}", index)

    # ── session doc lifecycle ───────────────────────────────────────

    async def create_session_doc(self, session_id: str, title: str, created_at: str) -> dict[str, Any]:
        """Create a new empty session document in ES."""
        doc = {
            "session_id": session_id,
            "title": title,
            "created_at": created_at,
            "updated_at": created_at,
            "turn_count": 0,
            "status": "active",
            "turns": [],
        }
        try:
            self._es.index(index=self.config.episodic_es_index, id=session_id, body=doc, refresh=True)
        except Exception as exc:
            logger.warning("EpisodicMemory: create_session_doc failed: {}", exc)
        return doc

    async def append_turn(self, session_id: str, turn: dict[str, Any]) -> int:
        """Append a turn (full original content) to a session document. Returns new turn_count."""
        try:
            result = self._es.get(index=self.config.episodic_es_index, id=session_id)
            doc = result["_source"]
        except Exception:
            # Session doc doesn't exist yet — create it
            now = datetime.now(timezone.utc).isoformat()
            doc = await self.create_session_doc(session_id, "新会话", now)

        turns: list = doc.get("turns", [])
        if not isinstance(turns, list):
            turns = []

        turn["turn_id"] = turn.get("turn_id", "")
        turn["timestamp"] = turn.get("timestamp", datetime.now(timezone.utc).isoformat())
        turns.append(turn)

        doc["turns"] = turns
        doc["turn_count"] = len(turns)
        doc["updated_at"] = datetime.now(timezone.utc).isoformat()

        try:
            self._es.index(index=self.config.episodic_es_index, id=session_id, body=doc, refresh=True)
        except Exception as exc:
            logger.warning("EpisodicMemory: append_turn failed: {}", exc)

        return len(turns)

    async def get_session_doc(self, session_id: str) -> dict[str, Any] | None:
        """Get full session document with all turns."""
        try:
            result = self._es.get(index=self.config.episodic_es_index, id=session_id)
            return result.get("_source", {})  # type: ignore[no-any-return]
        except Exception:
            return None

    async def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """List all session documents, newest first."""
        try:
            result = self._es.search(
                index=self.config.episodic_es_index,
                body={
                    "size": limit,
                    "query": {"match_all": {}},
                    "sort": [{"updated_at": {"order": "desc"}}],
                },
            )
            return [hit["_source"] for hit in result["hits"]["hits"]]
        except Exception as exc:
            logger.warning("EpisodicMemory: list_sessions failed: {}", exc)
            return []

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session document from ES."""
        try:
            self._es.delete(index=self.config.episodic_es_index, id=session_id, refresh=True)
            return True
        except Exception:
            return False

    # ── retrieval ───────────────────────────────────────────────────

    async def recall(
        self,
        query: str = "",
        session_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search across session documents. If session_id given, return turns from that session only."""
        try:
            if session_id:
                doc = await self.get_session_doc(session_id)
                if doc:
                    turns = doc.get("turns", [])
                    # Flatten: each turn becomes a dict with session context
                    return [
                        {
                            "session_id": session_id,
                            "session_title": doc.get("title", ""),
                            **turn,
                        }
                        for turn in turns[-limit:]
                    ]
                return []

            # Cross-session search by query
            result = self._es.search(
                index=self.config.episodic_es_index,
                body={
                    "size": limit,
                    "query": {
                        "nested": {
                            "path": "turns",
                            "query": {
                                "bool": {
                                    "should": [
                                        {"match": {"turns.content": query}},
                                    ],
                                }
                            },
                            "inner_hits": {"size": 3},
                        }
                    },
                    "sort": [{"updated_at": {"order": "desc"}}],
                },
            )
            hits = []
            for hit in result["hits"]["hits"]:
                src = hit["_source"]
                inner = hit.get("inner_hits", {}).get("turns", {}).get("hits", {}).get("hits", [])
                for ih in inner:
                    turn = ih.get("_source", {})
                    turn["session_id"] = src.get("session_id", "")
                    turn["session_title"] = src.get("title", "")
                    hits.append(turn)
            return hits[:limit]
        except Exception as exc:
            logger.warning("EpisodicMemory: recall failed: {}", exc)
            return []

    @property
    def size(self) -> int:
        try:
            return self._es.count(index=self.config.episodic_es_index).get("count", 0)
        except Exception:
            return 0
