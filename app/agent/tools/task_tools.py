"""Task status tools for uploaded papers."""

from __future__ import annotations

from app.services.tasks import get_task
from app.agent.tools.contracts import error_response, json_response
from app.agent.tools.decorators import tool
from app.shared.serializers import task_to_result


@tool
def get_task_status(task_id: str) -> str:
    """Get upload, parsing, chunking, embedding, and indexing status by task_id."""
    tool_name = "get_task_status"
    try:
        task = get_task(task_id)
        results = [task_to_result(task)] if task else []
        error = None if task else f"Task {task_id} not found."
        return json_response(
            tool_name=tool_name,
            query=task_id,
            results=results,
            sources=[],
            error=error,
            confidence=1.0 if task else 0.0,
        )
    except Exception as exc:  # noqa: BLE001
        return error_response(tool_name, task_id, exc)


TASK_TOOLS = [get_task_status]
