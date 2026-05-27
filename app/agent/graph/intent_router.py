"""Two-stage intent routing: rule pre-check + LLM structured output fallback."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agent.graph.state import AgentState
from app.core.config import settings
from app.core.logging import logger
from app.services.llm_service import get_llm
from app.services.retrieval_service import _extract_query_entities

_INTENT_LITERAL = "retrieval", "comparison", "summary", "profile", "writing", "chat"


class IntentRouteResult(BaseModel):
    intent: str = Field(description="One of: retrieval, comparison, summary, profile, writing, chat")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    reason: str = Field(default="", description="Routing rationale")


@dataclass(frozen=True)
class _IntentPattern:
    intent: str
    strong_hints: tuple[str, ...] = ()   # high-confidence trigger words
    support_hints: tuple[str, ...] = ()  # supporting keywords that boost confidence


_PATTERNS: list[_IntentPattern] = [
    _IntentPattern(
        "comparison",
        strong_hints=(
            "对比", "比较", "区别", "不同", "vs", "versus", "哪个更好", "优劣",
            "compare", "difference", "comparison", "which is better", "better than",
            "横向对比", "哪个方法更好", "性能对比", "对比下", "对比一下",
            "方法对比", "比较一下", "区别是什么", "有什么不同",
        ),
        support_hints=(
            "之间", "benchmark", "论文A", "论文B", "哪个方法",
            "的优缺点",
        ),
    ),
    _IntentPattern(
        "summary",
        strong_hints=(
            "综述", "文献综述", "总结", "概述", "概览", "literature review", "survey",
            "summarize", "总结所有", "概括全文", "系统回顾", "全面梳理", "写综述",
            "生成综述", "帮我写综述", "研究现状",
        ),
        support_hints=(
            "领域发展", "起草", "整理", "帮我写一篇", "概括", "梳理",
        ),
    ),
    _IntentPattern(
        "writing",
        strong_hints=(
            "润色", "改写", "polish", "修改文字", "改进语法", "措辞",
            "academic writing", "peer review", "review draft", "帮我改",
            "优化文字", "学术表达", "润色这段", "改写这段", "润色这",
            "改写这", "修改这段", "改这段", "polish this",
        ),
        support_hints=(
            "段落", "文字", "句子", "语法", "表达", "流畅", "帮我看看",
            "使其更", "更学术", "更专业", "更流畅", "更通顺",
        ),
    ),
    _IntentPattern(
        "profile",
        strong_hints=(
            "哪篇论文", "有哪些论文", "有哪些相关", "论文信息", "paper profile",
            "paper details", "论文基本信息", "查找论文", "知识库中有", "列出",
            "有多少篇", "找一下", "有哪些", "搜索论文", "find papers",
        ),
        support_hints=(
            "find paper", "paper info", "论文画像", "这篇论文", "作者", "发表信息",
            "论文详情", "知识库", "几篇", "筛选", "哪些论文", "哪些相关",
        ),
    ),
    _IntentPattern(
        "retrieval",
        strong_hints=(
            "什么是", "如何", "怎么", "为什么", "解释", "证据", "chunk", "细节",
            "详细说明", "实验", "方法", "结果", "数据集", "指标", "实现", "架构",
            "模块", "网络结构", "what is", "how does", "explain", "tell me about",
            "describe", "检索", "查找段落", "具体内容",
        ),
        support_hints=(
            "回答", "answer", "问答", "论文内容", "论文中", "知识库",
        ),
    ),
    _IntentPattern(
        "chat",
        strong_hints=(
            "你好", "谢谢", "hello", "hi", "help", "你能做什么", "你是谁",
            "再见", "goodbye", "what can you do", "who are you", "嗨",
        ),
        support_hints=(
            "帮助", "功能", "介绍自己", "能力", "谢谢", "感谢",
        ),
    ),
]

_INTENT_ROUTER_SYSTEM = """You are a query intent classifier for an academic paper research assistant.

Classify the user's query into exactly ONE of these 6 intents:

1. retrieval — Fine-grained paper content QA (methods, experiments, metrics, datasets).
   Examples: "这个方法怎么实现的", "实验结果PSNR是多少", "频域增强的原理是什么"

2. comparison — Multi-paper comparison or benchmarking.
   Examples: "对比论文A和论文B的方法", "哪种方法更好", "比较这些论文的性能"

3. summary — Literature review or topic summary generation.
   Examples: "生成水下图像增强综述", "总结频域方法的研究现状", "帮我写文献综述"

4. profile — Paper discovery, filtering, or single paper profile lookup.
   Examples: "知识库有哪些频域论文", "这篇论文的基本信息", "列出图像增强相关论文"

5. writing — Academic text polishing, rewriting, or peer review.
   Examples: "润色这段文字", "帮我改这篇摘要", "review这个段落"

6. chat — Casual conversation, greetings, capability questions.
   Examples: "你好", "你能做什么", "谢谢"

Rules:
- If the query asks about specific paper content -> retrieval
- If the query asks to compare multiple papers -> comparison
- If the query asks for a comprehensive review/summary on a topic -> summary
- If the query asks to find/pick papers -> profile
- If the query asks to improve/edit text -> writing
- If the query is casual conversation -> chat
- Return confidence 0.0-1.0 and a brief reason."""


def _rule_classify(query: str) -> tuple[str, float, str]:
    """Rule-based intent classification with weighted keyword matching."""
    normalized = query.strip().lower()
    candidates: list[tuple[str, float, str]] = []

    for pattern in _PATTERNS:
        strong_matches = sum(1 for h in pattern.strong_hints if h in normalized)
        support_matches = sum(1 for h in pattern.support_hints if h in normalized)

        if strong_matches == 0 and support_matches == 0:
            continue

        confidence = min(strong_matches * 0.45 + support_matches * 0.15, 0.95)
        reason = f"strong={strong_matches}, support={support_matches}"
        candidates.append((pattern.intent, confidence, reason))

    if not candidates:
        return "retrieval", 0.40, "no rule match, default"

    candidates.sort(key=lambda x: x[1], reverse=True)
    best = candidates[0]

    # Ambiguity penalty when top-2 are close
    if len(candidates) > 1 and (best[1] - candidates[1][1]) < 0.10:
        return (best[0], round(best[1] * 0.85, 4), f"ambiguous: {best[0]} vs {candidates[1][0]} ({best[2]})")

    return (best[0], round(best[1], 4), best[2])


async def _llm_classify(query: str) -> IntentRouteResult:
    """LLM intent classification via plain JSON output (compatible with Qwen)."""
    llm = get_llm()
    try:
        json_prompt = (
            f"{_INTENT_ROUTER_SYSTEM}\n\n"
            "IMPORTANT: Output ONLY valid JSON, no markdown fences, no extra text.\n"
            'Format: {{"intent": "<one of six>", "confidence": 0.0-1.0, "reason": "..."}}'
        )
        messages: list = [
            SystemMessage(content=json_prompt),
            HumanMessage(content=f"Query: {query}"),
        ]
        raw = await llm.ainvoke(messages)
        text = raw.content if hasattr(raw, "content") else str(raw)
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3]
        parsed = json.loads(text.strip())
        return IntentRouteResult(
            intent=parsed.get("intent", "retrieval"),
            confidence=float(parsed.get("confidence", 0.5)),
            reason=str(parsed.get("reason", "")),
        )
    except Exception as exc:
        logger.debug("LLM intent routing failed, using fallback: {}", exc)
        return IntentRouteResult(intent="retrieval", confidence=0.4, reason="[fallback] LLM routing failed")


async def intent_router_node(state: AgentState) -> dict[str, Any]:
    """Route the query to an intent via rule first, then LLM if needed.

    Classification runs on the ORIGINAL query, not the enriched one.
    Enrichment adds retrieval constraints (top_k, task_id) that would
    artificially inflate retrieval confidence and cause misrouting.
    """
    original_query = state.get("query", "")

    rule_intent, rule_confidence, rule_reason = _rule_classify(original_query)

    if rule_confidence >= settings.rule_route_threshold:
        result = {
            "intent": rule_intent,
            "intent_confidence": rule_confidence,
            "rule_matched": True,
            "routing_reason": f"[rule] {rule_reason}",
        }
        logger.info(
            "意图识别: intent={} | confidence={:.2f} | method=rule({}) | query={:.100}",
            rule_intent, rule_confidence, rule_reason, original_query,
        )
        return result

    llm_result = await _llm_classify(original_query)
    result = {
        "intent": llm_result.intent,
        "intent_confidence": llm_result.confidence,
        "rule_matched": False,
        "routing_reason": f"[llm] {llm_result.reason}",
    }
    logger.info(
        "意图识别: intent={} | confidence={:.2f} | method=llm({}) | query={:.100}",
        llm_result.intent, llm_result.confidence, llm_result.reason, original_query,
    )
    return result


def _query_mentions_paper(query: str) -> bool:
    """Check if a retrieval query references a specific paper by name."""
    entities = _extract_query_entities(query)
    if entities:
        return True
    # Also check for paper citation patterns: "论文<Name>" or "the <Name> paper"
    paper_patterns = (
        "论文", "这篇文章", "这篇论文", "paper", "the paper",
    )
    normalized = query.lower()
    return any(p in normalized for p in paper_patterns) and len(query) > 30


def route_from_router(state: AgentState) -> str:
    """Conditional edge: complex intents go to planner, simple intents to handler.

    Retrieval queries that mention a specific paper are treated as complex
    and routed through the planner for paper-discovery + evidence retrieval.
    """
    intent = state.get("intent", "retrieval")
    query = state.get("enriched_query") or state.get("query", "")

    if intent in ("comparison", "summary", "writing"):
        logger.info("路由决策: {} -> planner (complex intent)", intent)
        return "planner"

    if intent == "retrieval" and _query_mentions_paper(query):
        logger.info("路由决策: retrieval -> planner (检测到论文名/实体)", intent)
        return "planner"

    logger.info("路由决策: {} -> {}_handler (直接执行)", intent, intent)
    return f"{intent}_handler"


def route_from_planner(state: AgentState) -> str:
    """Conditional edge: planner output goes to the intent's handler."""
    intent = state.get("intent", "retrieval")
    return f"{intent}_handler"
