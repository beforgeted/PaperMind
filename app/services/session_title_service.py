"""Generate short session titles from the user's first message."""

from __future__ import annotations

import logging
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.services.llm_service import llm_service, message_content_to_text


logger = logging.getLogger(__name__)

_DEFAULT_TITLES = frozenset({"", "新会话", "新对话", "新学术研讨会话", "New Session"})

_TITLE_SYSTEM_PROMPT = """你是会话标题生成器。根据用户的第一条问题，生成一个简短的中文会话标题。

要求：
- 8～20 个汉字（或等价长度），概括用户意图
- 不要引号、不要句号、不要「会话」「标题」等元描述
- 只输出标题本身一行，不要其他内容"""


def is_default_session_title(title: Optional[str]) -> bool:
    """Return True when the session still uses the placeholder title."""
    normalized = (title or "").strip()
    return normalized in _DEFAULT_TITLES


def fallback_title_from_query(query: str, *, max_len: int = 24) -> str:
    """Heuristic title when LLM is unavailable."""
    text = re.sub(r"\s+", " ", (query or "").strip())
    if not text:
        return "新会话"
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


async def generate_session_title(query: str) -> str:
    """Summarize the first user message into a short session title."""
    raw = (query or "").strip()
    if not raw:
        return "新会话"

    try:
        llm = llm_service.create_chat_model(
            llm_service.main_agent_model_config(temperature=0.2),
            streaming=False,
            timeout=30,
            max_tokens=64,
        )
        response = await llm.ainvoke(
            [
                SystemMessage(content=_TITLE_SYSTEM_PROMPT),
                HumanMessage(content=raw[:2000]),
            ]
        )
        title = message_content_to_text(getattr(response, "content", "")).strip()
        title = title.strip("\"'「」『』")
        title = re.sub(r"\s+", " ", title)
        if title and not is_default_session_title(title) and len(title) <= 40:
            logger.info("会话标题 LLM 生成成功: title=%s", title[:40])
            return title
    except Exception as exc:
        logger.warning("会话标题 LLM 生成失败，使用截断标题: %s", exc)

    return fallback_title_from_query(raw)
