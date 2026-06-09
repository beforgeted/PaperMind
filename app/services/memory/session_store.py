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
import logging; logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────

DEFAULT_WINDOW_SIZE = 10        # max turns kept in sliding window (single session only)
DEFAULT_SESSION_TTL = 86_400    # 24 hours idle expiry
DEFAULT_MESSAGE_MAX_CHARS = 4_000
ASSISTANT_MESSAGE_MAX_CHARS = 32_000

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
        allow_long: bool = False,
    ):
        self.turn_id = turn_id or uuid.uuid4().hex[:8]
        self.role = role
        max_chars = (
            ASSISTANT_MESSAGE_MAX_CHARS
            if allow_long or role == "assistant"
            else DEFAULT_MESSAGE_MAX_CHARS
        )
        self.content = content[:max_chars]
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
    def from_dict(cls, d: dict[str, Any], *, allow_long: bool = True) -> "TurnRecord":
        """从 Redis / ES 反序列化；默认保留完整 content，避免二次截断。"""
        return cls(
            turn_id=d.get("turn_id", ""),
            role=d.get("role", "user"),
            content=d.get("content", ""),
            intent=d.get("intent", ""),
            timestamp=d.get("timestamp", ""),
            used_tools=list(d.get("used_tools") or []),
            metadata=dict(d.get("metadata") or {}),
            allow_long=allow_long,
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

    def get_title(self, session_id: str) -> str:
        """Return session title from Redis, or placeholder if unset."""
        skey = self.session_key(session_id)
        title = self._redis.hget(skey, "title")
        return str(title) if title else "新会话"

    def set_title(self, session_id: str, title: str) -> None:
        """Update session title and refresh TTL."""
        skey = self.session_key(session_id)
        self.ensure_session(session_id)
        self._redis.hset(skey, "title", (title or "新会话").strip()[:120])
        self._redis.hset(skey, "last_active", datetime.now(timezone.utc).isoformat())
        self._refresh_ttl(session_id)

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
        turns = self._read_redis_turns(session_id, max_turns=max_turns)
        if as_messages:
            from app.services.memory.message_utils import turns_to_history_messages

            return turns_to_history_messages(list(reversed(turns)))  # type: ignore[return-value]
        return turns  # type: ignore[return-value]

    def _read_redis_turns(
        self,
        session_id: str,
        *,
        max_turns: int | None = None,
    ) -> list[TurnRecord]:
        """从 Redis 读取 turn 列表（newest-first）。"""
        hkey = self.history_key(session_id)
        limit = max_turns or self.window_size
        raw = self._redis.lrange(hkey, 0, limit - 1)

        turns: list[TurnRecord] = []
        for item in raw:
            try:
                turns.append(TurnRecord.from_dict(json.loads(item)))
            except (json.JSONDecodeError, TypeError):
                continue
        return turns

    def has_redis_history(self, session_id: str) -> bool:
        """Redis 中是否存在会话历史。"""
        return bool(self._redis.exists(self.history_key(session_id)))

    def restore_history(self, session_id: str, turns: list[TurnRecord]) -> None:
        """用 ES 等来源的 turn 回填 Redis 滑动窗口（turns 为时间正序）。"""
        self.ensure_session(session_id)
        hkey = self.history_key(session_id)
        capped = turns[-self.window_size :]
        pipe = self._redis.pipeline()
        pipe.delete(hkey)
        # 按时间正序 LPUSH，使最新 turn 留在列表头部（与 append_turn 一致）
        for turn in capped:
            pipe.lpush(hkey, json.dumps(turn.to_dict(), ensure_ascii=False))
        pipe.ltrim(hkey, 0, self.window_size - 1)
        pipe.execute()
        self._refresh_ttl(session_id)
        logger.info(
            "SessionStore: restored {} turns to Redis for {}",
            len(capped),
            session_id,
        )

    def _turn_from_es_dict(self, payload: dict[str, Any]) -> TurnRecord:
        """将 ES turn 文档转为 TurnRecord（保留完整字段）。"""
        return TurnRecord(
            turn_id=str(payload.get("turn_id") or ""),
            role=str(payload.get("role") or "user"),
            content=str(payload.get("content") or ""),
            intent=str(payload.get("intent") or ""),
            timestamp=str(payload.get("timestamp") or ""),
            used_tools=list(payload.get("used_tools") or []),
            metadata=dict(payload.get("metadata") or {}),
            allow_long=True,
        )

    def load_history_messages(
        self,
        session_id: str,
        max_turns: int | None = None,
    ) -> list[dict[str, str]]:
        """加载近 N 轮对话为 role/content 列表（时间正序、完整 content）。

        优先 Redis 热缓存；若无则回源 ES 最近 N 轮并回填 Redis。
        两侧均无数据时返回空列表。
        """
        limit = max_turns or self.window_size
        redis_turns = self._read_redis_turns(session_id, max_turns=limit)
        if redis_turns:
            from app.services.memory.message_utils import turns_to_history_messages

            return turns_to_history_messages(list(reversed(redis_turns)))

        from app.services.memory.episodic import EpisodicMemory
        from app.services.memory.message_utils import turns_to_history_messages

        episodic = EpisodicMemory()
        es_turns = episodic.recall(session_id=session_id, limit=limit)
        if not es_turns:
            return []

        chronological = [self._turn_from_es_dict(item) for item in es_turns]
        self.restore_history(session_id, chronological)
        redis_turns = self._read_redis_turns(session_id, max_turns=limit)
        return turns_to_history_messages(list(reversed(redis_turns)))

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
        """兼容旧接口：返回 role/content 序列化文本（完整 content，不截断）。"""
        history = self.load_history_messages(session_id, max_turns=max_history_turns)
        if not history:
            return ""

        parts: list[str] = [f"## 对话历史 (最近 {len(history)} 轮)"]
        for index, msg in enumerate(history, 1):
            parts.append(
                f"[{index}] role={msg['role']}\ncontent:\n{msg['content']}"
            )
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
