"""Session CRUD API — ChatGPT-style conversation session management."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status

from app.core.logging import logger
from app.core.schemas import (
    SessionHistoryMessage,
    SessionHistoryResponse,
    SessionListResponse,
    SessionRecord,
)
from app.services.memory.episodic import EpisodicMemory
from app.services.memory.session_store import TurnRecord, get_session_store

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def _get_episodic() -> EpisodicMemory:
    return EpisodicMemory()


def _turns_from_redis(session_id: str, *, max_turns: int) -> list[dict]:
    """Chronological turn dicts from Redis sliding window."""
    store = get_session_store()
    raw = store.get_history(session_id, max_turns=max_turns, as_messages=False)
    if not raw:
        return []
    turns: list[dict] = []
    for item in raw:
        if isinstance(item, TurnRecord):
            turns.append(item.to_dict())
        elif isinstance(item, dict):
            turns.append(item)
    turns.reverse()
    return turns


def _load_session_turns(session_id: str, *, max_turns: int) -> tuple[list[dict], str]:
    """Load turns: prefer ES episodic (full archive), else Redis (recent window)."""
    try:
        episodic = _get_episodic()
        doc = episodic.get_session_doc(session_id)
        if doc:
            es_turns = doc.get("turns") or []
            if isinstance(es_turns, list) and es_turns:
                return list(es_turns)[-max_turns:], "episodic"
    except Exception as exc:
        logger.warning("ES session history load failed: {}", exc)

    redis_turns = _turns_from_redis(session_id, max_turns=max_turns)
    if redis_turns:
        return redis_turns, "redis"
    return [], "none"


def _pair_turns_to_messages(turns: list[dict]) -> list[SessionHistoryMessage]:
    """Merge alternating user/assistant turns into chat UI messages."""
    messages: list[SessionHistoryMessage] = []
    i = 0
    while i < len(turns):
        turn = turns[i]
        role = str(turn.get("role") or "").lower()
        if role != "user":
            i += 1
            continue

        question = str(turn.get("content") or "")
        msg_id = str(turn.get("turn_id") or f"turn-{len(messages)}")
        created_at = str(turn.get("timestamp") or "")
        answer = ""
        used_tools: list[str] = []
        i += 1

        if i < len(turns) and str(turns[i].get("role") or "").lower() == "assistant":
            assistant = turns[i]
            answer = str(assistant.get("content") or "")
            raw_tools = assistant.get("used_tools")
            if isinstance(raw_tools, list):
                used_tools = [str(t) for t in raw_tools]
            if not created_at:
                created_at = str(assistant.get("timestamp") or "")
            i += 1

        messages.append(
            SessionHistoryMessage(
                id=msg_id,
                question=question,
                answer=answer,
                created_at=created_at,
                used_tools=used_tools,
            )
        )
    return messages


# ── CRUD ────────────────────────────────────────────────────────────────


@router.post("", response_model=SessionRecord, status_code=status.HTTP_201_CREATED)
async def create_session() -> SessionRecord:
    """Create a new empty conversation session."""
    sid = _new_session_id()
    now = _utc_now()
    store = get_session_store()
    store.ensure_session(sid)
    store._redis.hset(store.session_key(sid), "title", "新会话")
    store._redis.hset(store.session_key(sid), "created_at", now)

    # Also init ES episodic doc
    try:
        episodic = _get_episodic()
        episodic.create_session_doc(sid, "新会话", now)
    except Exception as exc:
        logger.warning("Episodic session doc init failed (non-fatal): {}", exc)

    return SessionRecord(session_id=sid, title="新会话", created_at=now, updated_at=now, turn_count=0, status="active")


@router.get("", response_model=SessionListResponse)
async def list_sessions() -> SessionListResponse:
    """List all sessions, newest first. Sources from ES episodic (persistent) + Redis (active)."""
    sessions: list[SessionRecord] = []

    # 1. Try ES episodic (permanent archive)
    try:
        episodic = _get_episodic()
        es_sessions = episodic.list_sessions(limit=50)
        for s in es_sessions:
            sessions.append(SessionRecord(
                session_id=s.get("session_id", ""),
                title=s.get("title", "新会话"),
                created_at=s.get("created_at", ""),
                updated_at=s.get("updated_at", ""),
                turn_count=int(s.get("turn_count") or 0),
                status=s.get("status", "active"),
            ))
    except Exception as exc:
        logger.warning("ES session list failed: {}", exc)

    # 2. Supplement with Redis active sessions not yet in ES
    try:
        store = get_session_store()
        redis_sessions = store.list_sessions(limit=50)
        es_ids = {s.session_id for s in sessions}
        for rs in redis_sessions:
            sid = rs.get("session_id", "")
            if sid and sid not in es_ids:
                sessions.append(SessionRecord(
                    session_id=sid,
                    title=rs.get("title", "新会话"),
                    created_at=rs.get("created_at", ""),
                    updated_at=rs.get("last_active", ""),
                    turn_count=int(rs.get("turn_count") or 0),
                    status=rs.get("status", "active"),
                ))
    except Exception as exc:
        logger.warning("Redis session list failed: {}", exc)

    sessions.sort(key=lambda s: s.updated_at or s.created_at, reverse=True)
    return SessionListResponse(sessions=sessions)


@router.get("/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str, limit: int = 100) -> SessionHistoryResponse:
    """Return conversation turns for UI restore (ES archive, Redis fallback)."""
    limit = max(1, min(limit, 200))
    title = "新会话"
    found = False

    store = get_session_store()
    redis_meta = store.get_session(session_id)
    if redis_meta:
        found = True
        title = redis_meta.get("title") or title

    es_doc: dict | None = None
    try:
        es_doc = _get_episodic().get_session_doc(session_id)
        if es_doc:
            found = True
            title = es_doc.get("title") or title
    except Exception as exc:
        logger.warning("ES session lookup failed: {}", exc)

    if not found:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Session {session_id} not found")

    turns, source = _load_session_turns(session_id, max_turns=limit)
    return SessionHistoryResponse(
        session_id=session_id,
        title=title,
        messages=_pair_turns_to_messages(turns),
        source=source,
    )


@router.get("/{session_id}", response_model=SessionRecord)
async def get_session(session_id: str) -> SessionRecord:
    """Get a single session's metadata."""
    store = get_session_store()
    data = store.get_session(session_id)
    if not data:
        try:
            episodic = _get_episodic()
            doc = episodic.get_session_doc(session_id)
            if doc:
                return SessionRecord(
                    session_id=doc.get("session_id", session_id),
                    title=doc.get("title", "新会话"),
                    created_at=doc.get("created_at", ""),
                    updated_at=doc.get("updated_at", ""),
                    turn_count=int(doc.get("turn_count") or 0),
                    status=doc.get("status", "active"),
                )
        except Exception:
            pass
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Session {session_id} not found")

    return SessionRecord(
        session_id=session_id,
        title=data.get("title", "新会话"),
        created_at=data.get("created_at", ""),
        updated_at=data.get("last_active", ""),
        turn_count=int(data.get("turn_count") or 0),
        status=data.get("status", "active"),
    )


@router.patch("/{session_id}", response_model=SessionRecord)
async def update_session(session_id: str, title: str | None = None) -> SessionRecord:
    """Update session title or status."""
    store = get_session_store()
    data = store.get_session(session_id)
    if not data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Session {session_id} not found")

    if title is not None:
        store._redis.hset(store.session_key(session_id), "title", title)
        # Also update ES
        try:
            episodic = _get_episodic()
            es = episodic.get_session_doc(session_id)
            if es:
                es["title"] = title
                episodic._es.index(
                    index=episodic.config.episodic_es_index,
                    id=session_id,
                    body=es,
                    refresh=True,
                )
        except Exception:
            pass

    data = store.get_session(session_id) or data
    return SessionRecord(
        session_id=session_id,
        title=data.get("title", ""),
        created_at=data.get("created_at", ""),
        updated_at=data.get("last_active", ""),
        turn_count=int(data.get("turn_count") or 0),
        status=data.get("status", "active"),
    )


@router.delete("/{session_id}", status_code=status.HTTP_200_OK)
async def delete_session(session_id: str) -> dict[str, str]:
    """Delete a session and all its data from Redis and ES."""
    store = get_session_store()
    store.delete_session(session_id)

    try:
        episodic = _get_episodic()
        episodic.delete_session(session_id)
    except Exception as exc:
        logger.warning("ES session delete failed (non-fatal): {}", exc)

    return {"status": "deleted", "session_id": session_id}
