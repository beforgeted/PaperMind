"""Academic writing handler — polish text, peer review drafts using nature-skills."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.state import AgentState
from app.services.llm import get_llm
from app.services.skills import get_skill_loader

_POLISH_SYSTEM = """You are an academic writing editor trained on Nature-journal editorial standards.

Rules:
- Language serves argument. Do not polish sentences while leaving reasoning broken.
- Write with empathy for the reader: relevance first, then novelty, then trust, then reuse, then meaning.
- Do not invent data, references, mechanisms, or novelty claims.
- Sentences should be 10-30 words. Split overloaded sentences.
- Each paragraph: one controlling idea + support.
- Results vs Discussion: Results = what we observed (past tense). Discussion = how we understand it.
- Output polished text followed by 3-5 concise revision notes."""

_PEER_REVIEW_SYSTEM = """You are a constructive peer reviewer. Review the draft:
1. Overall assessment (score 0-100)
2. What works well (2-3 points)
3. What needs improvement (2-3 prioritized points)
4. Specific revision suggestions
5. Overclaim / unsupported claim check
Be specific and actionable."""

_SECTION_HINTS = {
    "abstract": ("abstract", "摘要"),
    "introduction": ("introduction", "引言", "介绍", "背景"),
    "methods": ("method", "方法", "实验设置", "materials"),
    "results": ("result", "结果", "experiment", "实验"),
    "discussion": ("discussion", "讨论", "discuss"),
    "conclusion": ("conclusion", "结论", "总结"),
}


def _detect_section_type(text: str) -> str:
    lowered = text.lower()[:200]
    for section, hints in _SECTION_HINTS.items():
        if any(h in lowered for h in hints):
            return section
    return "body"


async def writing_handler(state: AgentState) -> dict[str, Any]:
    query = state.get("query", "")
    used_tools: list[str] = []

    # Step 1: Load nature-polishing skill if available
    try:
        loader = get_skill_loader()
        loader.load_skill("nature-polishing")
        used_tools.append("load_skill")
    except Exception:
        pass

    # Step 2: Detect section type and extract text to polish
    section_type = _detect_section_type(query)
    text_to_polish = _extract_text(query)

    if not text_to_polish:
        return {
            "handler_answer": "请提供需要润色的文本内容。",
            "handler_contexts": [],
            "handler_sources": [],
            "handler_used_tools": used_tools,
        }

    llm = get_llm()

    # Step 3: Polish
    polish_response = llm.invoke([
        SystemMessage(content=_POLISH_SYSTEM),
        HumanMessage(content=f"Section type: {section_type}\n\nText to polish:\n\n{text_to_polish}"),
    ])
    used_tools.append("polish_academic_text")
    polished = polish_response.content if hasattr(polish_response, "content") else str(polish_response)

    # Step 4: Peer review if requested
    wants_review = bool(re.search(r"review|评审|检查|审查|改进建议", query.lower()))
    if wants_review:
        review_response = llm.invoke([
            SystemMessage(content=_PEER_REVIEW_SYSTEM),
            HumanMessage(content=f"Section type: {section_type}\n\nDraft:\n\n{text_to_polish}"),
        ])
        used_tools.append("peer_review_draft")
        review_text = review_response.content if hasattr(review_response, "content") else str(review_response)
        answer = f"### 润色结果\n\n{polished}\n\n### 同行评审\n\n{review_text}"
    else:
        answer = polished

    return {
        "handler_answer": answer,
        "handler_contexts": [],
        "handler_sources": [],
        "handler_used_tools": used_tools,
    }


def _extract_text(query: str) -> str:
    """Extract the text to polish from a query. Remove common command prefixes."""
    text = query.strip()
    # Remove common command patterns
    prefixes = [
        r"^润色[：:]?\s*", r"^改写[：:]?\s*", r"^polish[：:]?\s*",
        r"^帮我改[：:]?\s*", r"^帮我润色[：:]?\s*", r"^修改[：:]?\s*",
        r"^优化文字[：:]?\s*", r"^review[：:]?\s*",
    ]
    for prefix in prefixes:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE).strip()

    if len(text) < 20:
        return text if len(text) >= 3 else ""
    return text
