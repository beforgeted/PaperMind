"""/api/v1/agent — Agent-first research assistant endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.agent.runtime import answer_with_agent, stream_answer_with_agent
from app.core.logging import logger
from app.core.schemas import AgentChatRequest, AgentChatResponse, RetrievedChunk
from app.services.tasks import get_task

router = APIRouter(prefix="/agent", tags=["agent"])


def agent_result_to_contexts(result: dict) -> list[RetrievedChunk]:
    """Convert Agent tool results into RetrievedChunk contexts."""
    raw_items = result.get("contexts") or result.get("sources") or []
    contexts: list[RetrievedChunk] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        metadata = dict(item.get("metadata") or {})
        paper_id = item.get("paper_id") or metadata.get("paper_id")
        title = item.get("title") or metadata.get("title")
        section = item.get("section") or metadata.get("section_title")
        section_type = item.get("section_type") or metadata.get("section_type")
        parent_id = item.get("parent_id") or paper_id or title or f"agent-source-{index}"
        child_ids = item.get("chunk_ids") or item.get("child_ids") or []
        if not isinstance(child_ids, list):
            child_ids = [str(child_ids)]
        # preserve original source info; only tag as agent when none exists
        existing_source = metadata.get("source") or metadata.get("original_filename") or metadata.get("source_file")
        metadata.update(
            {
                "paper_id": paper_id or metadata.get("paper_id"),
                "title": title or metadata.get("title"),
                "section_title": section or metadata.get("section_title"),
                "section_type": section_type or metadata.get("section_type"),
                "source": existing_source or "papermind_agent",
                "original_filename": metadata.get("original_filename") or metadata.get("source_file") or metadata.get("original_filename", ""),
            }
        )
        contexts.append(
            RetrievedChunk(
                parent_id=str(parent_id),
                parent_text=item.get("content") or "",
                child_ids=[str(child_id) for child_id in child_ids],
                score=float(item.get("score") or 0.0),
                metadata=metadata,
            )
        )
    return contexts


@router.post(
    "/chat",
    response_model=AgentChatResponse,
    summary="Agent-first research assistant backed by PaperMind tools.",
)
async def chat_with_agent(request: AgentChatRequest) -> AgentChatResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty query.")

    if request.task_id and get_task(request.task_id) is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Task {request.task_id} not found.",
        )

    try:
        result = await answer_with_agent(
            query=query,
            top_k=request.top_k,
            task_id=request.task_id,
            session_id=request.session_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent chat failed for query={!r}: {}", query, exc)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"Agent chat failed: {exc}",
        ) from exc

    return AgentChatResponse(
        query=query,
        answer=result.get("answer", ""),
        contexts=agent_result_to_contexts(result),
        sources=result.get("sources") or [],
        used_tools=result.get("used_tools") or [],
    )


@router.post(
    "/chat/stream",
    summary="Agent-first research assistant with SSE streaming.",
)
async def chat_with_agent_stream(request: AgentChatRequest):
    """Stream the agent answer via Server-Sent Events.

    Events:
      - status: routing/planning metadata
      - delta: answer text token
      - done: final metadata (contexts, sources, tools)
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty query.")

    if request.task_id and get_task(request.task_id) is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Task {request.task_id} not found.",
        )

    async def event_stream():
        try:
            async for event in stream_answer_with_agent(
                query=query,
                top_k=request.top_k,
                task_id=request.task_id,
                session_id=request.session_id,
            ):
                yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent streaming failed for query={!r}: {}", query, exc)
            error_event = json.dumps(
                {"type": "done", "answer": f"Agent 调用失败：{exc}", "contexts": [], "sources": [], "used_tools": [], "error": str(exc)},
                ensure_ascii=False,
            )
            yield f"event: done\ndata: {error_event}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
