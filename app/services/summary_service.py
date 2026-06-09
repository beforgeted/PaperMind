"""最终回答汇总服务。"""

from __future__ import annotations

from typing import AsyncGenerator, Awaitable, Callable, Optional

from app.agent.context import context_builder
from app.agent.graphs.graph_state import PaperMindState
from app.services.llm_service import llm_service, message_content_to_text


class SummaryService:
    """把子智能体输出汇总成面向用户的最终回答。"""

    async def stream_summary(
        self,
        *,
        state: PaperMindState,
    ) -> AsyncGenerator[str, None]:
        llm = llm_service.create_chat_model(
            llm_service.main_agent_model_config(),
            streaming=True,
            timeout=120,
            max_tokens=4096,
        )
        messages = context_builder.build_for_summary(state)
        async for chunk in llm.astream(messages):
            text = message_content_to_text(getattr(chunk, "content", ""))
            if text:
                yield text

    async def summarize_results(
        self,
        *,
        state: PaperMindState,
        chunk_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> str:
        chunks = []
        async for text in self.stream_summary(state=state):
            chunks.append(text)
            if chunk_callback is not None:
                await chunk_callback(text)
        return "".join(chunks).strip()


summary_service = SummaryService()
