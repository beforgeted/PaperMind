"""Session-level helpers for PaperMind research agent calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AgentSessionConfig:
    """Request-scoped constraints passed to the research agent."""

    top_k: Optional[int] = None
    task_id: Optional[str] = None


def build_user_query(
    query: str,
    top_k: Optional[int] = None,
    task_id: Optional[str] = None,
) -> str:
    """Append explicit retrieval constraints to the user query."""
    constraints: list[str] = []
    if top_k is not None:
        constraints.append(
            f"检索工具如支持 top_k，请优先使用 top_k={top_k}；top_k 只是检索上限，不代表必须回答 {top_k} 条。"
        )
        constraints.append("最终结果数量必须以工具返回的 result_count 或 results 实际长度为准。")
    if task_id:
        constraints.append(
            f"当前问题限定在 task_id={task_id} 对应的任务或论文上传结果范围内。"
        )
    if not constraints:
        return query
    return f"{query}\n\n约束：\n" + "\n".join(f"- {item}" for item in constraints)


def is_task_status_query(query: str) -> bool:
    """Return whether a query is asking about upload/indexing task status."""
    normalized = query.strip().lower()
    task_terms = (
        "任务状态",
        "解析状态",
        "解析任务",
        "处理进度",
        "task status",
        "task_id",
        "任务",
        "上传",
        "入库",
        "向量化",
    )
    status_terms = (
        "状态",
        "进度",
        "完成",
        "完了吗",
        "失败",
        "成功",
        "status",
        "入库",
        "向量化",
    )
    return any(term in normalized for term in task_terms) and any(
        term in normalized for term in status_terms
    )
