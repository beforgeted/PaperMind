"""DashScope-backed Chat LLM for the LCEL QA chain.

Reuses `DASHSCOPE_API_KEY` already required by the embedding service.
`ChatTongyi` lives in `langchain_community.chat_models`; for plain text answers
(no tool/structured output) it works out of the box.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from app.core.config import settings


class LLMConfigError(RuntimeError):
    pass


_llm: Optional["BaseChatModel"] = None
_streaming_llm: Optional["BaseChatModel"] = None


def _load_chat_tongyi() -> Any:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import torch  # noqa: F401
    except Exception:
        # ChatTongyi uses remote DashScope calls; defer any torch failure to the
        # actual community integration import below instead of breaking API import.
        pass
    from langchain_community.chat_models import ChatTongyi

    return ChatTongyi


def get_llm() -> "BaseChatModel":
    global _llm
    if _llm is None:
        if not settings.dashscope_api_key:
            raise LLMConfigError(
                "DASHSCOPE_API_KEY is not set; cannot build the LLM."
            )
        ChatTongyi = _load_chat_tongyi()
        _llm = ChatTongyi(
            model=settings.llm_model,
            dashscope_api_key=settings.dashscope_api_key,
            temperature=settings.llm_temperature,
        )
    return _llm


def get_streaming_llm() -> "BaseChatModel":
    global _streaming_llm
    if _streaming_llm is None:
        if not settings.dashscope_api_key:
            raise LLMConfigError(
                "DASHSCOPE_API_KEY is not set; cannot build the streaming LLM."
            )
        ChatTongyi = _load_chat_tongyi()
        _streaming_llm = ChatTongyi(
            model=settings.llm_model,
            dashscope_api_key=settings.dashscope_api_key,
            temperature=settings.llm_temperature,
            streaming=True,
        )
    return _streaming_llm
