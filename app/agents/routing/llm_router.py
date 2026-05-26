"""Structured LLM router for ambiguous queries.

Primary path: llm.with_structured_output(RouterDecision)
Fallback path: JsonOutputParser when structured output is unsupported.
"""

from __future__ import annotations

from typing import Optional

from app.core.config import settings
from app.core.logging import logger
from app.core.schemas import RouterDecision
from app.services.llm_service import get_llm

_ROUTER_SYSTEM_PROMPT = """You are a query router for an academic paper knowledge base.
Classify the user's question into exactly ONE of these 5 routes:

1. paper_search — Paper discovery / inventory / filtering.
   Examples: "有哪些相关论文", "知识库中有多少篇频域论文", "列出图像增强相关论文"

2. paper_deep_search — Paper discovery REQUIRING supporting evidence / reasoning.
   Examples: "哪些论文用了频域方法，具体依据是什么", "找出图像增强论文并说明理由"

3. chunk_qa — Fine-grained paper content QA (methods, experiments, metrics, datasets).
   Examples: "这个方法具体怎么实现的", "实验PSNR是多少", "消融实验的结果是什么"

4. task_status — Task / parsing / indexing status queries.
   Examples: "论文解析完了吗", "任务状态是什么", "上传进度如何"
   Only route to task_status if the query asks about processing status, NOT paper content.

5. paper_profile — Single paper profile / overview.
   Examples: "介绍一下这篇论文", "这篇论文的摘要是什么"

Rules:
- If the query asks about paper discovery AND requires evidence/reasoning -> paper_deep_search
- If the query asks about processing status -> task_status
- If the query asks for a single paper overview/summary -> paper_profile
- If the query asks about paper discovery/filtering/counting -> paper_search
- Otherwise -> chunk_qa
- Return your confidence in the route (0.0-1.0) and a brief reason in Chinese.
"""


class StructuredLLMRouter:
    """LLM-based router with structured output + JSON fallback."""

    def __init__(self) -> None:
        base_llm = get_llm()
        self._base_llm = base_llm
        self._structured_llm = None
        self._use_structured = None

    def _ensure_structured(self) -> None:
        if self._use_structured is not None:
            return
        try:
            self._structured_llm = self._base_llm.with_structured_output(RouterDecision)
            self._use_structured = True
        except (NotImplementedError, AttributeError, TypeError) as exc:
            logger.debug("Structured output not supported by ChatTongyi: {}", exc)
            self._use_structured = False

    async def route(
        self,
        query: str,
        pre_route_context: Optional[str] = None,
    ) -> RouterDecision:
        self._ensure_structured()
        if self._use_structured:
            return await self._route_structured(query, pre_route_context)
        return await self._route_prompt_fallback(query, pre_route_context)

    async def _route_structured(
        self,
        query: str,
        pre_route_context: Optional[str],
    ) -> RouterDecision:
        messages: list = [("system", _ROUTER_SYSTEM_PROMPT)]
        if pre_route_context:
            messages.append(
                (
                    "human",
                    f"Pre-router analysis: {pre_route_context}\n\nUser query: {query}",
                )
            )
        else:
            messages.append(("human", query))

        result = await self._structured_llm.ainvoke(messages)  # type: ignore[union-attr]
        return RouterDecision(
            route=result.route if hasattr(result, "route") else str(result.get("route", "chunk_qa")),
            confidence=float(
                result.confidence
                if hasattr(result, "confidence")
                else result.get("confidence", 0.5)
            ),
            reason=str(
                result.reason
                if hasattr(result, "reason")
                else result.get("reason", "")
            ),
        )

    async def _route_prompt_fallback(
        self,
        query: str,
        pre_route_context: Optional[str],
    ) -> RouterDecision:
        from langchain_core.output_parsers import JsonOutputParser

        prompt = _ROUTER_SYSTEM_PROMPT
        if pre_route_context:
            prompt += f"\n\nPre-router analysis: {pre_route_context}"

        messages: list = [("system", prompt), ("human", query)]
        chain = self._base_llm | JsonOutputParser()
        try:
            result = await chain.ainvoke(messages)
        except Exception:
            return RouterDecision(
                route=settings.llm_route_fallback_route,  # type: ignore[arg-type]
                confidence=0.3,
                reason="LLM JSON parse failed, using fallback",
            )
        if not isinstance(result, dict):
            return RouterDecision(
                route=settings.llm_route_fallback_route,  # type: ignore[arg-type]
                confidence=0.3,
                reason="LLM non-dict output, using fallback",
            )
        return RouterDecision(
            route=str(result.get("route", settings.llm_route_fallback_route)),
            confidence=float(result.get("confidence", 0.5)),
            reason=str(result.get("reason", "")),
        )
