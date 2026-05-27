"""Shared JSON contract for PaperMind tools."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

DEFAULT_MAX_CONTENT_CHARS = 1500


def truncate(value: Any, max_chars: int = DEFAULT_MAX_CONTENT_CHARS) -> str:
    """Return a compact string safe for tool payloads."""
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...(truncated)"


@dataclass(frozen=True)
class ToolSource:
    """Citation/source metadata attached to a tool result."""

    paper_id: Optional[str] = None
    title: Optional[str] = None
    section: Optional[str] = None
    section_type: Optional[str] = None
    parent_id: Optional[str] = None
    chunk_ids: list[str] = field(default_factory=list)
    score: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the source with JSON-friendly defaults."""
        return asdict(self)


@dataclass(frozen=True)
class ToolResult:
    """Stable envelope returned by every PaperMind tool."""

    tool_name: str
    query: str
    results: list[dict[str, Any]] = field(default_factory=list)
    sources: list[ToolSource | dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    confidence: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result envelope used by agents and workflows."""
        sources = [
            source.to_dict() if isinstance(source, ToolSource) else source
            for source in self.sources
        ]
        return {
            "tool_name": self.tool_name,
            "query": self.query,
            "result_count": len(self.results),
            "results": self.results,
            "sources": sources,
            "error": self.error,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Return the envelope as UTF-8 friendly JSON text."""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


def json_response(
    *,
    tool_name: str,
    query: str,
    results: list[dict[str, Any]],
    sources: Optional[list[ToolSource | dict[str, Any]]] = None,
    error: Optional[str] = None,
    confidence: Optional[float] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    """Build a standardized JSON response for a tool."""
    return ToolResult(
        tool_name=tool_name,
        query=query,
        results=results,
        sources=sources or [],
        error=error,
        confidence=confidence,
        metadata=metadata or {},
    ).to_json()


def error_response(tool_name: str, query: str, exc: Exception) -> str:
    """Build a standardized failed tool response."""
    return json_response(
        tool_name=tool_name,
        query=query,
        results=[],
        sources=[],
        error=str(exc),
    )
