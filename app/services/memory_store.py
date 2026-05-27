"""Simple JSON-backed memory store for PaperMind agents."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

MemoryScope = Literal["session", "user", "project"]

_DEFAULT_MEMORY_PATH = Path("storage/memory/agent_memory.json")
_VALID_SCOPES = {"session", "user", "project"}


def _utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class JsonMemoryStore:
    """Small persistent store for scoped agent memory and workspace state."""

    def __init__(self, path: Path = _DEFAULT_MEMORY_PATH):
        self.path = path
        self._lock = threading.Lock()

    def _empty_payload(self) -> dict[str, Any]:
        return {
            "session": {},
            "user": {},
            "project": {},
            "workspace": {},
            "updated_at": None,
        }

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty_payload()
        with self.path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        merged = self._empty_payload()
        merged.update(payload if isinstance(payload, dict) else {})
        for scope in _VALID_SCOPES:
            if not isinstance(merged.get(scope), dict):
                merged[scope] = {}
        if not isinstance(merged.get("workspace"), dict):
            merged["workspace"] = {}
        return merged

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload["updated_at"] = _utc_now()
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, self.path)

    def _validate_scope(self, scope: str) -> MemoryScope:
        if scope not in _VALID_SCOPES:
            raise ValueError(f"scope must be one of {sorted(_VALID_SCOPES)}")
        return scope  # type: ignore[return-value]

    def remember(self, scope: str, key: str, value: Any) -> dict[str, Any]:
        """Store one memory value under a scope/key pair."""
        if not key.strip():
            raise ValueError("key must not be empty")
        normalized_scope = self._validate_scope(scope)
        with self._lock:
            payload = self._read()
            entry = {
                "scope": normalized_scope,
                "key": key,
                "value": value,
                "updated_at": _utc_now(),
            }
            payload[normalized_scope][key] = entry
            self._write(payload)
            return entry

    def recall(self, scope: str, query: str = "") -> list[dict[str, Any]]:
        """Return memories in scope whose key or value matches query."""
        normalized_scope = self._validate_scope(scope)
        payload = self._read()
        entries = list(payload[normalized_scope].values())
        needle = query.strip().lower()
        if not needle:
            return entries
        hits: list[dict[str, Any]] = []
        for entry in entries:
            searchable = json.dumps(
                {"key": entry.get("key"), "value": entry.get("value")},
                ensure_ascii=False,
                default=str,
            ).lower()
            if needle in searchable:
                hits.append(entry)
        return hits

    def update_workspace_state(self, key: str, value: Any) -> dict[str, Any]:
        """Set one workspace state value without clearing other keys."""
        if not key.strip():
            raise ValueError("key must not be empty")
        with self._lock:
            payload = self._read()
            payload["workspace"][key] = value
            self._write(payload)
            return dict(payload["workspace"])

    def get_workspace_state(self) -> dict[str, Any]:
        """Return the current workspace state map."""
        return dict(self._read()["workspace"])


_store: JsonMemoryStore | None = None


def get_memory_store() -> JsonMemoryStore:
    """Return the process-level memory store singleton."""
    global _store
    if _store is None:
        _store = JsonMemoryStore()
    return _store
