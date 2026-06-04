"""Memory data models for PaperMind agent memory system.

The current architecture keeps short-term conversation state in Redis,
archives raw sessions in ES Episodic, and stores distilled long-term facts in
Semantic memory.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


MemoryScope = Literal["session", "user", "project"]
MemoryKind = Literal["preference", "research_interest", "project_state", "writing_style", "fact"]


class MemoryType(str, Enum):
    WORKING = "working"       # 工作记忆 — 当前会话上下文, TTL 过期
    SEMANTIC = "semantic"     # 语义记忆 — 论文知识/用户偏好, ES 索引
    EPISODIC = "episodic"     # 情景记忆 — 历史交互记录, 时间序列


class MemoryItem(BaseModel):
    """Single unit of memory, serializable to all backends."""

    memory_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    memory_type: MemoryType = MemoryType.WORKING
    scope: MemoryScope = "session"

    # Content
    key: str = ""               # Short label (e.g. "preferred_language", "query:2026-05-28")
    content: str = ""           # Free-text body
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Long-term semantic memory dimensions
    user_id: str = ""
    project_id: str = ""
    memory_kind: MemoryKind = "fact"
    source_session_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    last_evidence_at: str = ""

    # Scoring
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    access_count: int = 0
    last_accessed_at: str = ""

    # Lifecycle
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ttl_seconds: int = 3600     # TTL for working memory (default 1 hour)
    expires_at: str = ""

    # Vector embedding (populated for semantic memories stored in ES)
    embedding: list[float] | None = None

    def touch(self) -> None:
        """Mark as accessed, bump counter, update timestamp."""
        self.access_count += 1
        self.last_accessed_at = datetime.now(timezone.utc).isoformat()

    def is_expired(self) -> bool:
        """Check if this working memory item has exceeded its TTL."""
        if self.memory_type != MemoryType.WORKING:
            return False
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc).isoformat() > self.expires_at

    def summary(self) -> str:
        """One-line summary for logging / trace."""
        return f"[{self.memory_type.value}:{self.key}] imp={self.importance:.2f} acc={self.access_count}"
