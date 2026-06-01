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
                            "user_id": {"type": "keyword"},
                            "project_id": {"type": "keyword"},
                            "title": {"type": "text"},
                            "created_at": {"type": "date"},
                            "updated_at": {"type": "date"},
                            "turn_count": {"type": "integer"},
                            "consolidated_at": {"type": "date"},
                            "consolidated_turn_count": {"type": "integer"},
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
        else:
            try:
                self._es.indices.put_mapping(
                    index=index,
                    body={
                        "properties": {
                            "user_id": {"type": "keyword"},
                            "project_id": {"type": "keyword"},
                            "consolidated_at": {"type": "date"},
                            "consolidated_turn_count": {"type": "integer"},
                        }
                    },
                )
            except Exception as exc:
                logger.debug("EpisodicMemory: mapping update skipped: {}", exc)

    # ── session doc lifecycle ───────────────────────────────────────

    def create_session_doc(
        self,
        session_id: str,
        title: str,
        created_at: str,
        user_id: str = "",
        project_id: str = "",
    ) -> dict[str, Any]:
        """Create a new empty session document in ES."""
        doc = {
            "session_id": session_id,
            "user_id": user_id,
            "project_id": project_id,
            "title": title,
            "created_at": created_at,
            "updated_at": created_at,
            "turn_count": 0,
            "consolidated_at": None,
            "consolidated_turn_count": 0,
            "status": "active",
            "turns": [],
        }
        try:
            self._es.index(index=self.config.episodic_es_index, id=session_id, body=doc, refresh=True)
        except Exception as exc:
            logger.warning("EpisodicMemory: create_session_doc failed: {}", exc)
        return doc

    def append_turn(self, session_id: str, turn: dict[str, Any]) -> int:
        """Append a turn atomically using Painless script. Returns new turn_count.

        Uses ES update + upsert to avoid read-then-write race conditions.
        """
        now = datetime.now(timezone.utc).isoformat()
        turn["turn_id"] = turn.get("turn_id", "")
        turn["timestamp"] = turn.get("timestamp", now)

        script_source = """
            if (ctx._source.turns == null) { ctx._source.turns = []; }
            ctx._source.turns.add(params.turn);
            ctx._source.turn_count = ctx._source.turns.length;
            ctx._source.updated_at = params.now;
        """

        upsert_doc = {
            "session_id": session_id,
            "user_id": turn.get("user_id", ""),
            "project_id": turn.get("project_id", ""),
            "title": "新会话",
            "created_at": now,
            "updated_at": now,
            "turn_count": 1,
            "consolidated_at": None,
            "consolidated_turn_count": 0,
            "status": "active",
            "turns": [turn],
        }

        try:
            self._es.update(
                index=self.config.episodic_es_index,
                id=session_id,
                script={"source": script_source, "lang": "painless", "params": {"turn": turn, "now": now}},
                upsert=upsert_doc,
                refresh=True,
            )
        except Exception as exc:
            logger.warning("EpisodicMemory: append_turn failed: {}", exc)

        # Read back turn count for the return value
        doc = self.get_session_doc(session_id)
        return len(doc.get("turns", [])) if doc else 1

    def get_session_doc(self, session_id: str) -> dict[str, Any] | None:
        """Get full session document with all turns."""
        try:
            result = self._es.get(index=self.config.episodic_es_index, id=session_id)
            return result.get("_source", {})  # type: ignore[no-any-return]
        except Exception:
            return None

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
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

    def list_sessions_needing_consolidation(self, limit: int = 50) -> list[dict[str, Any]]:
        """List sessions whose archive has turns not yet consolidated."""
        try:
            result = self._es.search(
                index=self.config.episodic_es_index,
                body={
                    "size": limit,
                    "query": {
                        "script": {
                            "script": {
                                "source": (
                                    "doc['turn_count'].size()!=0 && "
                                    "(doc['consolidated_turn_count'].size()==0 || "
                                    "doc['turn_count'].value > doc['consolidated_turn_count'].value)"
                                )
                            }
                        }
                    },
                    "sort": [{"updated_at": {"order": "asc"}}],
                },
            )
            return [hit["_source"] for hit in result["hits"]["hits"]]
        except Exception as exc:
            logger.warning("EpisodicMemory: list_sessions_needing_consolidation failed: {}", exc)
            return []

    def mark_consolidated(self, session_id: str, turn_count: int) -> None:
        """Record how many turns have been consolidated into semantic memory.

        Uses ES partial doc update to avoid read-then-write race.
        """
        now = datetime.now(timezone.utc).isoformat()
        try:
            self._es.update(
                index=self.config.episodic_es_index,
                id=session_id,
                body={"doc": {"consolidated_at": now, "consolidated_turn_count": int(turn_count)}},
                refresh=True,
            )
        except Exception as exc:
            logger.warning("EpisodicMemory: mark_consolidated failed: {}", exc)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session document from ES."""
        try:
            self._es.delete(index=self.config.episodic_es_index, id=session_id, refresh=True)
            return True
        except Exception:
            return False

    def forget_stale(self, max_age_days: int | None = None) -> int:
        """Delete archived sessions older than max_age_days."""
        max_age = max_age_days or self.config.forget_max_age_days
        try:
            result = self._es.delete_by_query(
                index=self.config.episodic_es_index,
                body={"query": {"range": {"updated_at": {"lt": f"now-{max_age}d"}}}},
                refresh=True,
            )
            removed = result.get("deleted", 0)
            if removed:
                logger.info("EpisodicMemory: forgot {} stale sessions (>{} days)", removed, max_age)
            return removed
        except Exception as exc:
            logger.warning("EpisodicMemory: forget_stale failed: {}", exc)
            return 0

    # ── retrieval ───────────────────────────────────────────────────

    def recall(
        self,
        query: str = "",
        session_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search across session documents. If session_id given, return turns from that session only."""
        try:
            if session_id:
                doc = self.get_session_doc(session_id)
                if doc:
                    turns = doc.get("turns", [])
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
