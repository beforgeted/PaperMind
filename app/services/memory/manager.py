"""MemoryManager — orchestrator for simplified PaperMind memory.

Lifecycle: restore session → recall long-term semantic memory → archive turns →
consolidate episodic archive into semantic memory.

Integrates with LangGraph agent pipeline:
  - Redis SessionStore restores current-session context
  - ES Episodic stores raw sessions
  - ES Semantic stores distilled long-term facts and preferences
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.core.logging import logger
from app.services.memory.config import MemoryConfig
from app.services.memory.models import MemoryItem, MemoryScope, MemoryType
from app.services.memory.semantic import SemanticMemory
from app.services.memory.episodic import EpisodicMemory
from app.services.memory.session_store import SessionStore, get_session_store


class MemoryManager:
    """Coordinates Redis session memory, ES archives, and semantic memory."""

    def __init__(
        self,
        config: MemoryConfig | None = None,
        session_store: SessionStore | None = None,
        semantic: SemanticMemory | None = None,
        episodic: EpisodicMemory | None = None,
    ):
        self.config = config or MemoryConfig()
        self.session_store = session_store or get_session_store()
        self.semantic = semantic or SemanticMemory(self.config)
        self.episodic = episodic or EpisodicMemory(self.config)

    # ── encoding ──────────────────────────────────────────────────────

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

    async def encode_long_term(
        self,
        *,
        key: str,
        content: str,
        memory_kind: str = "fact",
        scope: MemoryScope = "user",
        user_id: str = "",
        project_id: str = "",
        source_session_ids: list[str] | None = None,
        confidence: float = 0.7,
        importance: float = 0.7,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        """Store a distilled long-term semantic memory."""
        return await self.semantic.upsert_long_term_memory(
            key=key,
            content=content,
            memory_kind=memory_kind,
            scope=scope,
            user_id=user_id,
            project_id=project_id,
            source_session_ids=source_session_ids or [],
            confidence=confidence,
            importance=importance,
            metadata=metadata or {},
        )

    def encode_episodic(
        self,
        key: str,
        content: str,
        scope: MemoryScope = "user",
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Archive an interaction turn in the ES episodic session document."""
        session_id = (metadata or {}).get("session_id", "")
        if not session_id:
            raise ValueError("session_id is required for episodic archive writes")
        turn = {
            "role": (metadata or {}).get("role", "assistant"),
            "content": content,
            "intent": (metadata or {}).get("intent", ""),
            "used_tools": (metadata or {}).get("used_tools", []),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        turn_count = self.episodic.append_turn(session_id, turn)
        return {"session_id": session_id, "turn_count": turn_count, "key": key}

    # ── retrieval ────────────────────────────────────────────────────

    async def recall(
        self,
        query: str,
        scope: MemoryScope = "user",
        include_working: bool = False,
        include_semantic: bool = True,
        include_episodic: bool = True,
        session_id: str | None = None,
        user_id: str = "",
        project_id: str = "",
    ) -> dict[str, list[Any]]:
        """Recall long-term semantic memories and optional session archive turns."""
        result: dict[str, list[Any]] = {}

        if include_semantic and self.config.auto_recall_semantic:
            semantic_items = await self.semantic.recall(
                query,
                scope=scope,
                user_id=user_id,
                project_id=project_id,
            )
            result["semantic"] = [self._memory_to_dict(item) for item in semantic_items]

        if include_episodic:
            result["episodic"] = self.episodic.recall(
                query=query,
                session_id=session_id,
                limit=self.config.episodic_retrieval_top_k,
            )

        total = sum(len(v) for v in result.values())
        logger.debug("MemoryManager.recall: query={!r} -> {} items across {} stores", query[:60], total, len(result))
        return result

    # ── consolidation ────────────────────────────────────────────────

    async def consolidate(self) -> int:
        """Consolidate pending ES episodic sessions into semantic memory."""
        from app.services.memory.consolidation import MemoryConsolidator

        consolidator = MemoryConsolidator(episodic=self.episodic, semantic=self.semantic)
        return await consolidator.consolidate_pending()

    # ── forgetting ───────────────────────────────────────────────────

    def forget_expired(self) -> dict[str, int]:
        """Run forgetting across all layers. Returns counts by layer."""
        result = {
            "semantic": self.semantic.forget_stale(self.config.forget_max_age_days),
            "episodic": self.episodic.forget_stale(self.config.forget_max_age_days),
        }
        total = sum(result.values())
        if total:
            logger.info("MemoryManager.forget: cleaned {} items (semantic={}, episodic={})", total, result["semantic"], result["episodic"])
        return result

    # ── agent integration helpers ───────────────────────────────────

    def build_context_prompt(self, recalled: dict[str, list[Any]], max_items: int = 5) -> str:
        """Build a prompt-safe context block from recalled memories."""
        parts: list[str] = []
        for layer, items in recalled.items():
            if not items:
                continue
            title = "长期记忆" if layer == "semantic" else "历史归档"
            parts.append(f"## {title}")
            for item in items[:max_items]:
                if isinstance(item, MemoryItem):
                    key = item.key
                    content = item.content
                else:
                    key = str(item.get("key") or item.get("memory_kind") or item.get("role") or "memory")
                    content = str(item.get("content") or "")
                if content:
                    parts.append(f"- [{key}] {content[:300]}")
        return "\n".join(parts) if parts else ""

    def record_interaction(
        self,
        query: str,
        intent: str,
        answer_summary: str,
        used_tools: list[str],
        task_type: str = "",
        paper_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Record a full agent interaction summary into the episodic archive."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        key = f"interaction:{timestamp}"
        content = f"Q: {query[:500]}\nIntent: {intent}\nAnswer: {answer_summary[:300]}"

        return self.encode_episodic(
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
                "role": "assistant",
            },
        )

    def stats(self) -> dict[str, Any]:
        return {
            "session_store": "redis",
            "semantic_count": self.semantic.size,
            "episodic_count": self.episodic.size,
        }

    @staticmethod
    def _memory_to_dict(item: Any) -> dict[str, Any]:
        if isinstance(item, dict):
            return dict(item)
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json")
        return {
            "key": getattr(item, "key", ""),
            "content": getattr(item, "content", ""),
        }


# ── singleton ────────────────────────────────────────────────────────────

_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    """Return the process-level MemoryManager singleton."""
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager
