"""RAG tools backed by PaperMind retrieval and QA services."""

from __future__ import annotations

from typing import Optional

from app.services.qa import answer as qa_answer
from app.services.retrieval import hybrid_retrieve
from app.agent.tools.contracts import error_response, json_response
from app.agent.tools.decorators import tool
from app.shared.serializers import chunk_to_result, result_to_source


@tool
async def retrieve_evidence(
    query: str,
    top_k: Optional[int] = None,
    task_id: Optional[str] = None,
) -> str:
    """Search fine-grained paper evidence without generating a final answer."""
    tool_name = "retrieve_evidence"
    try:
        chunks = await hybrid_retrieve(query=query, top_k=top_k, task_id=task_id)
        results = [chunk_to_result(chunk) for chunk in chunks]
        return json_response(
            tool_name=tool_name,
            query=query,
            results=results,
            sources=[result_to_source(item) for item in results],
            confidence=1.0 if results else 0.0,
        )
    except Exception as exc:  # noqa: BLE001
        return error_response(tool_name, query, exc)


@tool
async def answer_with_rag(
    query: str,
    top_k: Optional[int] = None,
    task_id: Optional[str] = None,
) -> str:
    """Answer a question with RAG and return answer plus citation contexts."""
    tool_name = "answer_with_rag"
    try:
        result = await qa_answer(query=query, top_k=top_k, task_id=task_id)
        contexts = [chunk_to_result(chunk) for chunk in result.get("contexts", [])]
        payload = {
            "answer": result.get("answer", ""),
            "contexts": contexts,
        }
        return json_response(
            tool_name=tool_name,
            query=query,
            results=[payload],
            sources=[result_to_source(item) for item in contexts],
            confidence=1.0 if contexts else 0.3,
        )
    except Exception as exc:  # noqa: BLE001
        return error_response(tool_name, query, exc)


@tool
async def search_paper_chunks(
    query: str,
    top_k: Optional[int] = None,
    task_id: Optional[str] = None,
) -> str:
    """Backward-compatible alias for retrieve_evidence."""
    return await retrieve_evidence.ainvoke(  # type: ignore[attr-defined]
        {"query": query, "top_k": top_k, "task_id": task_id}
    ) if hasattr(retrieve_evidence, "ainvoke") else await retrieve_evidence(query, top_k, task_id)


RAG_TOOLS = [retrieve_evidence, answer_with_rag, search_paper_chunks]
