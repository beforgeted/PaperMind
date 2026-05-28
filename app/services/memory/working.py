"""Working memory — in-memory dict with TTL expiration.

Maps to the "scratchpad" layer from Hello Agents Ch.8.
Fastest access, auto-expires, feeds into consolidation pipeline.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

from app.core.logging import logger
from app.services.memory.config import MemoryConfig
from app.services.memory.models import MemoryItem, MemoryType


class WorkingMemory:
    """In-process short-term memory with TTL-based expiration."""

    def __init__(self, config: MemoryConfig | None = None):
        self.config = config or MemoryConfig()
        self._items: dict[str, MemoryItem] = {}
        self._lock = threading.Lock()
        self._last_cleanup: float = time.monotonic()

    # ── encode / store ────────────────────────────────────────────────

    def store(self, item: MemoryItem) -> MemoryItem:
        """Store or update a working memory item. Enforces capacity limit."""
        item.memory_type = MemoryType.WORKING
        if not item.expires_at:
            from datetime import timedelta
            expires = datetime.now(timezone.utc) + timedelta(seconds=item.ttl_seconds or self.config.working_default_ttl_seconds)
            item.expires_at = expires.isoformat()
        item.updated_at = datetime.now(timezone.utc).isoformat()

        with self._lock:
            self._maybe_cleanup()
            if len(self._items) >= self.config.working_max_items and item.memory_id not in self._items:
                self._evict_lowest_importance()
            self._items[item.memory_id] = item

        logger.debug("WorkingMemory: stored item key={!r} total={}", item.key, len(self._items))
        return item

    # ── retrieve ──────────────────────────────────────────────────────

    def recall(self, query: str = "", scope: str | None = None, limit: int = 20) -> list[MemoryItem]:
        """Return working memory items filtered by scope and optional query substring."""
        results: list[MemoryItem] = []
        needle = query.strip().lower()

        with self._lock:
            self._maybe_cleanup()
            for item in list(self._items.values()):
                if item.is_expired():
                    continue
                if scope and item.scope != scope:
                    continue
                if needle:
                    searchable = f"{item.key} {item.content}".lower()
                    if needle not in searchable:
                        continue
                item.touch()
                results.append(item)

        # Sort by importance × recency (approximate via access_count)
        results.sort(key=lambda i: i.importance * (1.0 + 0.1 * i.access_count), reverse=True)
        return results[:limit]

    def get(self, memory_id: str) -> MemoryItem | None:
        """Get a specific item by ID."""
        with self._lock:
            item = self._items.get(memory_id)
            if item and not item.is_expired():
                item.touch()
                return item
            return None

    # ── consolidate ────────────────────────────────────────────────────

    def get_consolidation_candidates(self) -> list[MemoryItem]:
        """Return items ready for consolidation to semantic memory."""
        candidates: list[MemoryItem] = []
        with self._lock:
            self._maybe_cleanup()
            for item in list(self._items.values()):
                if item.is_expired():
                    continue
                if item.importance >= self.config.consolidation_min_importance:
                    if item.access_count >= self.config.consolidation_min_access_count:
                        candidates.append(item)
        return candidates

    # ── forget ────────────────────────────────────────────────────────

    def forget(self, memory_id: str) -> bool:
        """Explicitly remove an item."""
        with self._lock:
            if memory_id in self._items:
                del self._items[memory_id]
                return True
            return False

    def forget_expired(self) -> int:
        """Remove all expired items. Returns count removed."""
        removed = 0
        with self._lock:
            expired_ids = [mid for mid, item in self._items.items() if item.is_expired()]
            for mid in expired_ids:
                del self._items[mid]
                removed += 1
        if removed:
            logger.debug("WorkingMemory: forgot {} expired items", removed)
        return removed

    # ── internal ──────────────────────────────────────────────────────

    def _maybe_cleanup(self) -> None:
        """Periodic TTL-based cleanup."""
        now = time.monotonic()
        if now - self._last_cleanup > self.config.working_cleanup_interval:
            self.forget_expired()
            self._last_cleanup = now

    def _evict_lowest_importance(self) -> None:
        """Remove the lowest-importance non-expired item to make room."""
        candidates = [(mid, item) for mid, item in self._items.items() if not item.is_expired()]
        if not candidates:
            return
        candidates.sort(key=lambda x: x[1].importance)
        victim_id, _ = candidates[0]
        del self._items[victim_id]
        logger.debug("WorkingMemory: evicted low-importance item {}", victim_id)

    @property
    def size(self) -> int:
        return len(self._items)

    def stats(self) -> dict[str, Any]:
        return {
            "total_items": len(self._items),
            "expired": sum(1 for item in self._items.values() if item.is_expired()),
            "avg_importance": sum(i.importance for i in self._items.values()) / max(len(self._items), 1),
        }
