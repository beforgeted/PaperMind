"""Unified Chat handler — single LLM answer generator for ALL intents.

All other handlers (retrieval, profile, summary, writing, comparison) feed their
collected evidence into state.collected_contexts. Chat is the only node that
calls LLM and produces handler_answer.

Architecture:
  intent_router → memory_recall → [context_collectors] → chat_handler → synthesizer
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.state import AgentState
from app.core.logging import logger
from app.services.llm import get_llm
from app.services.memory.session_store import get_session_store

_CHAT_SYSTEM = """你是 PaperMind 论文研究助手，在以下几个场景下工作：

1. **普通对话**：基于对话历史，友好简短回复。使用对话历史中用户告诉你的信息（姓名、偏好等）。
2. **论文检索**：下方「检索上下文」中有论文片段时，基于这些片段回答用户问题。引用时标注论文来源。若证据不足，明确说明。
3. **论文对比**：下方有对比结果时，基于对比表格和分析回答用户问题。
4. **学术写作**：下方有润色/评审结果时，基于结果回答用户问题。

要求：
- 使用 Markdown 格式组织回答，合理使用标题、列表、表格
- 回答严谨、忠实于上下文，不要编造
- 若上下文不足以回答，明确说明，但不要编造解释"""


async def chat_handler(state: AgentState) -> dict[str, Any]:
    llm = get_llm()
    query = state.get("query", "")
    session_id = state.get("session_id", "")

    messages: list = [SystemMessage(content=_CHAT_SYSTEM)]

    # 1. Conversation history restored by memory_recall_node; fallback to Redis.
    restored_history = state.get("messages", [])
    if restored_history:
        for h in restored_history:
            role = h.get("role", "")
            content = h.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        logger.info("chat_handler: restored history_turns={}", len(restored_history))
    elif session_id:
        try:
            store = get_session_store()
            history = store.get_history(session_id, max_turns=10, as_messages=True)
            for h in history:
                role = h.get("role", "")
                content = h.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
            if history:
                logger.info("chat_handler: session={} history_turns={}", session_id[:12], len(history))
        except Exception as exc:
            logger.warning("chat_handler: Redis history read failed: {}", exc)

    # 2. If upstream handler already generated an answer (summary/writing), pass through
    existing_answer = state.get("handler_answer", "")
    if existing_answer:
        logger.info("chat_handler: passing through existing answer (len={})", len(existing_answer))
        return {
            "handler_answer": existing_answer,
            "handler_contexts": state.get("handler_contexts", []),
            "handler_sources": state.get("handler_sources", []),
            "handler_used_tools": state.get("handler_used_tools", []),
        }

    # 3. Collected evidence from context collectors (retrieval, profile, comparison)
    collected = state.get("handler_contexts", [])
    context_text = _format_collected_contexts(collected)
    memory_context = state.get("memory_context", "")

    # 4. Current query with evidence context
    prompt_parts: list[str] = []
    if memory_context:
        prompt_parts.append(f"记忆上下文：\n{memory_context}")
    if context_text:
        prompt_parts.append(f"检索上下文：\n{context_text}")
    prompt_parts.append(f"问题：{query}" if prompt_parts else query)
    prompt = "\n\n".join(prompt_parts)

    messages.append(HumanMessage(content=prompt))

    response = await llm.ainvoke(messages)
    content = response.content if hasattr(response, "content") else str(response)

    return {
        "handler_answer": content,
        "handler_contexts": collected,
        "handler_sources": state.get("handler_sources", []),
        "handler_used_tools": state.get("handler_used_tools", []),
    }


def _format_collected_contexts(contexts: list[dict[str, Any]]) -> str:
    """Format collected evidence chunks into a prompt-friendly string."""
    if not contexts:
        return ""

    blocks: list[str] = []
    for i, ctx in enumerate(contexts):
        if not isinstance(ctx, dict):
            continue
        title = ctx.get("title") or ctx.get("paper_id") or f"来源-{i + 1}"
        content = ctx.get("content") or ctx.get("parent_text") or ""
        section = ctx.get("section") or ctx.get("section_type") or ""
        score = ctx.get("score")

        header = f"### [{title}]"
        if section:
            header += f" ({section})"
        if score is not None:
            header += f" [相关度: {float(score):.2f}]"

        if content:
            blocks.append(f"{header}\n{content[:1000]}")
        else:
            blocks.append(header)

    return "\n\n".join(blocks) if blocks else ""
