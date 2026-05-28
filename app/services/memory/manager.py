"""MemoryManager — unified orchestrator for the three-layer memory system.

Lifecycle: encode → store → retrieve → consolidate → forget

Integrates with LangGraph agent pipeline:
  - recall before planner (context injection)
  - encode after synthesizer (episodic recording)
  - consolidate periodically (working → semantic)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from app.core.logging import logger
from app.services.memory.config import MemoryConfig
from app.services.memory.models import MemoryItem, MemoryScope, MemoryType
from app.services.memory.working import WorkingMemory
from app.services.memory.semantic import SemanticMemory
from app.services.memory.episodic import EpisodicMemory


class MemoryManager:
    """Orchestrates Working → Semantic → Episodic memory layers."""

    def __init__(self, config: MemoryConfig | None = None):
        self.config = config or MemoryConfig()
        self.working = WorkingMemory(self.config)
        self.semantic = SemanticMemory(self.config)
        self.episodic = EpisodicMemory(self.config)

    # ── encoding ──────────────────────────────────────────────────────

    async def encode_working(
        self,
        key: str,
        content: str,
        scope: MemoryScope = "session",
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        """Create a working memory item. Fast, in-memory, TTL'd."""
        item = MemoryItem(
            memory_type=MemoryType.WORKING,
            scope=scope,
            key=key,
            content=content,
            importance=importance,
            metadata=metadata or {},
            ttl_seconds=self.config.working_default_ttl_seconds,
        )
        return self.working.store(item)

    async def encode_semantic(
        self,
        key: str,
        content: str,
        scope: MemoryScope = "user",
        importance: float = 0.7,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        """Create a semantic memory item with vector embedding."""
        item = MemoryItem(
            memory_type=MemoryType.SEMANTIC,
            scope=scope,
            key=key,
            content=content,
            importance=importance,
            metadata=metadata or {},
        )
        return await self.semantic.store(item)

    async def encode_episodic(
        self,
        key: str,
        content: str,
        scope: MemoryScope = "user",
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        """Record an agent interaction as an episodic memory."""
        item = MemoryItem(
            memory_type=MemoryType.EPISODIC,
            scope=scope,
            key=key,
            content=content,
            importance=importance,
            metadata=metadata or {},
        )
        return self.episodic.store(item)

    # ── retrieval ────────────────────────────────────────────────────

    async def recall(
        self,
        query: str,
        scope: MemoryScope = "user",
        include_working: bool = True,
        include_semantic: bool = True,
        include_episodic: bool = True,
        session_id: str | None = None,
    ) -> dict[str, list[MemoryItem]]:
        """Unified recall across all memory layers."""
        result: dict[str, list[MemoryItem]] = {}

        if include_working and self.config.auto_recall_working:
            result["working"] = self.working.recall(query=query, scope=scope)

        if include_semantic and self.config.auto_recall_semantic:
            result["semantic"] = await self.semantic.recall(query, scope=scope)

        if include_episodic:
            result["episodic"] = self.episodic.recall(query=query, scope=scope, session_id=session_id)

        total = sum(len(v) for v in result.values())
        logger.debug("MemoryManager.recall: query={!r} → {} items across {} layers", query[:60], total, len(result))
        return result

    def recall_sync(
        self,
        query: str = "",
        scope: MemoryScope = "session",
    ) -> list[MemoryItem]:
        """Synchronous recall from working memory only (for graph nodes)."""
        return self.working.recall(query=query, scope=scope)

    # ── consolidation ────────────────────────────────────────────────

    async def consolidate(self) -> int:
        """Move high-importance working memories to semantic memory.

        Returns count of items consolidated.
        """
        candidates = self.working.get_consolidation_candidates()
        if not candidates:
            return 0

        consolidated = 0
        for item in candidates:
            try:
                # Create a semantic version with elevated importance
                await self.encode_semantic(
                    key=item.key,
                    content=item.content,
                    scope=item.scope,
                    importance=min(item.importance * 1.1, 1.0),
                    metadata={**item.metadata, "consolidated_from": "working", "original_id": item.memory_id},
                )
                self.working.forget(item.memory_id)
                consolidated += 1
            except Exception as exc:
                logger.warning("MemoryManager.consolidate: failed for {}: {}", item.key, exc)

        if consolidated:
            logger.info("MemoryManager.consolidate: moved {} items working → semantic", consolidated)
        return consolidated

    # ── forgetting ───────────────────────────────────────────────────

    async def forget_expired(self) -> dict[str, int]:
        """Run forgetting across all layers. Returns counts by layer."""
        result = {
            "working": self.working.forget_expired(),
            "semantic": self.semantic.forget_stale(self.config.forget_max_age_days),
            "episodic": self.episodic.forget_stale(self.config.forget_max_age_days),
        }
        total = sum(result.values())
        if total:
            logger.info("MemoryManager.forget: cleaned {} items (w={} s={} e={})", total, *result.values())
        return result

    # ── agent integration helpers ───────────────────────────────────

    def build_context_prompt(self, recalled: dict[str, list[MemoryItem]], max_items: int = 5) -> str:
        """Build a prompt-safe context block from recalled memories."""
        parts: list[str] = []
        for layer, items in recalled.items():
            if not items:
                continue
            parts.append(f"## {layer} memory:")
            for item in items[:max_items]:
                parts.append(f"- [{item.key}] {item.content[:300]}")
        return "\n".join(parts) if parts else ""

    async def record_interaction(
        self,
        query: str,
        intent: str,
        answer_summary: str,
        used_tools: list[str],
        task_type: str = "",
        paper_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> MemoryItem:
        """Record a full agent interaction as an episodic memory after each invocation."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        key = f"interaction:{timestamp}"
        content = f"Q: {query[:500]}\nIntent: {intent}\nAnswer: {answer_summary[:300]}"

        return await self.encode_episodic(
            key=key,
            content=content,
            metadata={
                "query": query[:500],
                "intent": intent,
                "answer_summary": answer_summary[:500],
                "used_tools": used_tools,
                "task_type": task_type,
                "paper_ids": paper_ids or [],
                "session_id": session_id or "",
            },
        )

    def stats(self) -> dict[str, Any]:
        return {
            "working": self.working.stats(),
            "semantic_count": self.semantic.size,
            "episodic_count": self.episodic.size,
        }
