"""NDJSON streaming helpers for routed answers."""

from __future__ import annotations

import json
from typing import AsyncIterator

from app.agents.routing.handlers import execute_route
from app.core.logging import logger
from app.services.qa_service import stream_answer as qa_stream_answer


async def _metadata_event(query: str) -> str:
    return json.dumps(
        {"type": "metadata", "query": query, "contexts": []},
        ensure_ascii=False,
    ) + "\n"


async def _delta_and_done(text: str) -> AsyncIterator[str]:
    yield json.dumps({"type": "delta", "text": text}, ensure_ascii=False) + "\n"
    yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"


async def iter_routed_stream_events(
    *,
    route: str,
    query: str,
    top_k: int | None,
    task_id: str | None,
) -> AsyncIterator[str]:
    """Yield NDJSON lines for a resolved route (chunk_qa uses token streaming)."""
    if route == "chunk_qa":
        try:
            async for event in qa_stream_answer(query=query, top_k=top_k, task_id=task_id):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Streaming answer failed for query={!r}: {}", query, exc)
            yield json.dumps(
                {
                    "type": "error",
                    "message": "Answer generation failed; check ES / embedding / LLM services.",
                },
                ensure_ascii=False,
            ) + "\n"
        return

    result = await execute_route(route, query, top_k, task_id)
    yield await _metadata_event(query)
    async for chunk in _delta_and_done(result.get("answer", "")):
        yield chunk
