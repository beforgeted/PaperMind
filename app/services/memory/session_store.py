"""Redis-based stateless sliding-window session memory.

Dual-key architecture:
  session:{sid}  → Hash  (metadata, TTL, connection state)
  history:{sid}  → List  (sliding window of conversation turns)

Decouples HTTP/WebSocket connection lifecycle from chat business data.
On reconnect, full conversation context restored from Redis via session_id.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from redis import Redis

from app.core.config import settings
from app.core.logging import logger

# ── constants ────────────────────────────────────────────────────────────

DEFAULT_WINDOW_SIZE = 10        # max turns kept in sliding window (single session only)
DEFAULT_SESSION_TTL = 86_400    # 24 hours idle expiry
DEFAULT_MESSAGE_MAX_CHARS = 4_000

SessionStatus = Literal["active", "expired"]


# ── models ──────────────────────────────────────────────────────────────

class TurnRecord:
    """A single conversation turn stored in the history list."""

    __slots__ = ("turn_id", "role", "content", "intent", "timestamp", "used_tools", "metadata")

    def __init__(
        self,
        *,
        role: str,
        content: str,
        turn_id: str | None = None,
        intent: str = "",
        timestamp: str = "",
        used_tools: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.turn_id = turn_id or uuid.uuid4().hex[:8]
        self.role = role
        self.content = content[:DEFAULT_MESSAGE_MAX_CHARS]
        self.intent = intent
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.used_tools = used_tools or []
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "role": self.role,
            "content": self.content,
            "intent": self.intent,
            "timestamp": self.timestamp,
            "used_tools": self.used_tools,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TurnRecord":
        return cls(
            turn_id=d.get("turn_id", ""),
            role=d.get("role", "user"),
            content=d.get("content", ""),
            intent=d.get("intent", ""),
            timestamp=d.get("timestamp", ""),
            used_tools=list(d.get("used_tools") or []),
            metadata=dict(d.get("metadata") or {}),
        )


# ── store ───────────────────────────────────────────────────────────────

class SessionStore:
    """Redis-backed session memory with dual-key sliding window."""

    def __init__(
        self,
        redis_client: Redis | None = None,
        window_size: int = DEFAULT_WINDOW_SIZE,
        session_ttl: int = DEFAULT_SESSION_TTL,
    ):
        self._redis = redis_client or Redis.from_url(settings.redis_url, decode_responses=True)
        self.window_size = window_size
        self.session_ttl = session_ttl

    # ── key helpers ──────────────────────────────────────────────────

    @staticmethod
    def session_key(session_id: str) -> str:
        return f"session:{session_id}"

    @staticmethod
    def history_key(session_id: str) -> str:
        return f"history:{session_id}"

    # ── session lifecycle ────────────────────────────────────────────

    def ensure_session(
        self,
        session_id: str,
        user_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create or refresh a session. Idempotent — safe to call on every request."""
        skey = self.session_key(session_id)
        now = datetime.now(timezone.utc).isoformat()

        exists = self._redis.exists(skey)
        if not exists:
            fields = {
                "session_id": session_id,
                "status": "active",
                "created_at": now,
                "last_active": now,
                "turn_count": "0",
                "window_size": str(self.window_size),
                "user_id": user_id or "",
                "metadata": json.dumps(metadata or {}, ensure_ascii=False),
            }
            self._hset_fields(self._redis, skey, fields)
            logger.debug("SessionStore: created session {}", session_id)
        else:
            self._redis.hset(skey, "last_active", now)

        self._refresh_ttl(session_id)
        return self._redis.hgetall(skey)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get session metadata. Returns None if not found or expired."""
        skey = self.session_key(session_id)
        if not self._redis.exists(skey):
            return None
        data = self._redis.hgetall(skey)
        if not data:
            return None
        self._redis.hset(skey, "last_active", datetime.now(timezone.utc).isoformat())
        self._refresh_ttl(session_id)
        return data

    def end_session(self, session_id: str) -> None:
        """Mark session as expired, keep history for a grace period."""
        skey = self.session_key(session_id)
        self._redis.hset(skey, "status", "expired")
        self._redis.expire(skey, 300)         # 5 min grace
        self._redis.expire(self.history_key(session_id), 300)

    def delete_session(self, session_id: str) -> None:
        """Immediately delete session + history."""
        self._redis.delete(self.session_key(session_id), self.history_key(session_id))
        logger.debug("SessionStore: deleted session {}", session_id)

    # ── turn history ────────────────────────────────────────────────

    def append_turn(self, session_id: str, turn: TurnRecord) -> int:
        """Push a turn to the front of the history list, trim to window size.

        Returns the new turn count.
        """
        hkey = self.history_key(session_id)
        payload = json.dumps(turn.to_dict(), ensure_ascii=False)

        # LPUSH = newest first, LTRIM = cap at window_size
        pipe = self._redis.pipeline()
        pipe.lpush(hkey, payload)
        pipe.ltrim(hkey, 0, self.window_size - 1)
        pipe.execute()

        # Increment turn count
        skey = self.session_key(session_id)
        count = self._redis.hincrby(skey, "turn_count", 1)
        self._redis.hset(skey, "last_active", turn.timestamp)
        self._refresh_ttl(session_id)

        logger.debug("SessionStore: turn {} appended to {} (total={})", turn.turn_id, session_id, count)
        return int(count)

    def get_history(
        self,
        session_id: str,
        max_turns: int | None = None,
        as_messages: bool = True,
    ) -> list[TurnRecord] | list[dict[str, str]]:
        """Retrieve conversation history, newest first (then reversed if as_messages).

        If as_messages=True, returns [{"role": "user", "content": "..."}, ...] in chronological order.
        Otherwise returns list of TurnRecord in newest-first order.
        """
        hkey = self.history_key(session_id)
        limit = max_turns or self.window_size
        raw = self._redis.lrange(hkey, 0, limit - 1)

        turns: list[TurnRecord] = []
        for item in raw:
            try:
                turns.append(TurnRecord.from_dict(json.loads(item)))
            except (json.JSONDecodeError, TypeError):
                continue

        if as_messages:
            # Chronological order (oldest first) for LLM prompt injection
            messages: list[dict[str, str]] = []
            for t in reversed(turns):
                role = t.role
                if role not in ("user", "assistant"):
                    role = "user" if t.role in ("human", "user") else "assistant"
                messages.append({"role": role, "content": t.content})
            return messages  # type: ignore[return-value]

        return turns  # type: ignore[return-value]

    def get_turn_count(self, session_id: str) -> int:
        """Get the current number of turns in the session."""
        raw = self._redis.hget(self.session_key(session_id), "turn_count")
        return int(raw) if raw else 0

    def last_n_turns(self, session_id: str, n: int = 5) -> list[TurnRecord]:
        """Get the last N turns (convenience)."""
        result = self.get_history(session_id, max_turns=n, as_messages=False)
        return list(result) if isinstance(list(result)[0] if result else None, TurnRecord) else []

    # ── session listing ─────────────────────────────────────────────

    def list_active_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Scan for active sessions (for admin/debug)."""
        return self.list_sessions(limit=limit)

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Scan Redis for all session metadata. Returns all statuses (active + expired)."""
        sessions: list[dict[str, Any]] = []
        cursor = 0
        pattern = self.session_key("*")
        while True:
            cursor, keys = self._redis.scan(cursor=cursor, match=pattern, count=min(limit, 50))
            for key in keys:
                data = self._redis.hgetall(key)
                if data:
                    sessions.append(data)
            if cursor == 0 or len(sessions) >= limit:
                break
        return sessions[:limit]

    # ── context builder ─────────────────────────────────────────────

    def build_context_for_prompt(
        self,
        session_id: str,
        current_query: str,
        max_history_turns: int = 10,
    ) -> str:
        """Build a compact context block from session history for prompt injection.

        Used by memory_recall_node to inject conversation continuity.
        """
        turns = self.get_history(session_id, max_turns=max_history_turns, as_messages=True)
        if not turns:
            return ""

        parts: list[str] = ["## 对话历史 (最近 {} 轮)".format(len(turns))]
        for i, msg in enumerate(turns):
            role_label = "用户" if msg["role"] == "user" else "助手"
            content_short = msg["content"][:200]
            parts.append(f"[{i + 1}] {role_label}: {content_short}")

        parts.append(f"\n## 当前问题\n{current_query}")
        return "\n".join(parts)

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _hset_fields(client: Redis, key: str, fields: dict[str, str]) -> None:
        for field, value in fields.items():
            client.hset(key, field, value)

    def _refresh_ttl(self, session_id: str) -> None:
        self._redis.expire(self.session_key(session_id), self.session_ttl)
        self._redis.expire(self.history_key(session_id), self.session_ttl)


# ── singleton ────────────────────────────────────────────────────────────

_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    """Return the process-level SessionStore singleton."""
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
