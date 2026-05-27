"""Chat handler — casual conversation and capability questions."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.graph.state import AgentState
from app.services.llm_service import get_llm

_CHAT_PROMPT = """你是 PaperMind 论文研究助手。你可以帮助用户：
- 检索论文知识库中的证据和内容（retrieve_evidence）
- 查找、筛选、推荐论文（search_paper_profiles、deep_search_papers）
- 对比多篇论文（comparison）
- 生成文献综述（summary）
- 学术写作润色和修改（polish_academic_text、peer_review_draft）

对于闲聊和问候，请友好简短回复。
对于具体的研究问题，请引导用户提出明确需求。
不要承诺未经授权的操作。"""


async def chat_handler(state: AgentState) -> dict[str, Any]:
    llm = get_llm()
    response = await llm.ainvoke([
        SystemMessage(content=_CHAT_PROMPT),
        HumanMessage(content=state.get("query", "")),
    ])
    content = response.content if hasattr(response, "content") else str(response)
    return {
        "handler_answer": content,
        "handler_contexts": [],
        "handler_sources": [],
        "handler_used_tools": [],
    }
