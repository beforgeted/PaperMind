"""`/api/v1/papers/answer*` — routed Q&A and agent mode."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.agents.papermind_agent import answer_with_agent
from app.agents.routing import resolve_route, route_and_answer
from app.agents.routing.stream import iter_routed_stream_events
from app.core.logging import logger
from app.core.schemas import AnswerRequest, AnswerResponse, RetrievedChunk
from app.services.task_status import get_task

router = APIRouter()


def agent_result_to_contexts(result: dict) -> list[RetrievedChunk]:
    """Convert Agent tool results into the existing /answer context schema."""
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
        metadata.update(
            {
                "paper_id": paper_id,
                "title": title,
                "section_title": section,
                "section_type": section_type,
                "source": "papermind_agent",
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
    "/answer",
    response_model=AnswerResponse,
    summary="Hybrid retrieval + Qwen LLM answer with [parent_id] citations.",
)
async def answer_question(request: AnswerRequest) -> AnswerResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty query.")

    if request.task_id and get_task(request.task_id) is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Task {request.task_id} not found.",
        )

    if request.use_agent:
        try:
            result = await answer_with_agent(
                query=query,
                top_k=request.top_k,
                task_id=request.task_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent answer failed for query={!r}: {}", query, exc)
            result = {
                "answer": f"Agent 调用失败：{exc}",
                "contexts": [],
                "sources": [],
            }
        return AnswerResponse(
            query=query,
            answer=result.get("answer", ""),
            contexts=agent_result_to_contexts(result),
        )

    try:
        result = await route_and_answer(
            query=query,
            top_k=request.top_k,
            task_id=request.task_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Routing/answer failed for query={!r}: {}", query, exc)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"Answer generation failed: {exc}",
        ) from exc

    return AnswerResponse(
        query=query,
        answer=result["answer"],
        contexts=result["contexts"],
    )


@router.post(
    "/answer/stream",
    summary="Hybrid retrieval + Qwen LLM streaming answer with citations.",
)
async def stream_answer_question(request: AnswerRequest) -> StreamingResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty query.")

    if request.task_id and get_task(request.task_id) is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Task {request.task_id} not found.",
        )

    route = await resolve_route(query=query, top_k=request.top_k, task_id=request.task_id)

    async def generate():
        async for line in iter_routed_stream_events(
            route=route.route,
            query=query,
            top_k=request.top_k,
            task_id=request.task_id,
        ):
            yield line

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson; charset=utf-8",
    )
