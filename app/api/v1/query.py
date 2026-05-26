"""`/api/v1/papers/query` — hybrid retrieval."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.core.logging import logger
from app.core.schemas import QueryRequest, QueryResponse
from app.services.retrieval_service import hybrid_retrieve
from app.services.task_status import get_task

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Hybrid retrieval: BM25 + KNN with app-side RRF.",
)
async def query_papers(request: QueryRequest) -> QueryResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty query.")

    if request.task_id and get_task(request.task_id) is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Task {request.task_id} not found.",
        )

    try:
        contexts = await hybrid_retrieve(
            query=query,
            top_k=request.top_k,
            task_id=request.task_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Retrieval failed for query={!r}: {}", query, exc)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "Retrieval failed; check ES / embedding service.",
        ) from exc

    return QueryResponse(query=query, contexts=contexts)
