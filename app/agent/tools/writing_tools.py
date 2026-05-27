"""Paper writing helper tools.

These tools produce lightweight draft artifacts from caller-provided evidence.
They do not replace RAG evidence retrieval; agents should call retrieval/search
tools first, then pass grounded notes here.
"""

from __future__ import annotations

from app.agent.tools.contracts import error_response, json_response, truncate
from app.agent.tools.decorators import tool


def _split_notes(notes: str) -> list[str]:
    """Convert loose notes into compact bullet strings."""
    parts = [
        item.strip(" -\t")
        for item in notes.replace("\r\n", "\n").split("\n")
        if item.strip(" -\t")
    ]
    if parts:
        return parts[:10]
    return [truncate(notes, 500)] if notes.strip() else []


@tool
def draft_review_outline(topic: str, evidence_notes: str = "") -> str:
    """Draft a literature-review outline from retrieved evidence notes."""
    tool_name = "draft_review_outline"
    try:
        notes = _split_notes(evidence_notes)
        outline = [
            f"1. 研究背景：界定 {topic} 的问题范围与应用场景。",
            "2. 方法脉络：按核心技术路线归纳代表性论文。",
            "3. 数据与实验：总结常用数据集、指标和实验设置。",
            "4. 对比分析：比较方法优势、局限与适用条件。",
            "5. 未来方向：基于证据提出仍未解决的问题。",
        ]
        return json_response(
            tool_name=tool_name,
            query=topic,
            results=[{"topic": topic, "outline": outline, "evidence_notes": notes}],
            confidence=0.8 if notes else 0.4,
        )
    except Exception as exc:  # noqa: BLE001
        return error_response(tool_name, topic, exc)


@tool
def draft_review_section(section_title: str, evidence_notes: str) -> str:
    """Draft one review section from caller-provided evidence notes."""
    tool_name = "draft_review_section"
    try:
        notes = _split_notes(evidence_notes)
        if notes:
            body = (
                f"{section_title} 可以围绕以下证据展开："
                + "；".join(notes[:5])
                + "。写作时应保留对应论文来源并避免无依据扩展。"
            )
        else:
            body = f"{section_title} 缺少可用证据，当前只能生成占位结构。"
        return json_response(
            tool_name=tool_name,
            query=section_title,
            results=[{"section_title": section_title, "draft": body, "evidence_notes": notes}],
            confidence=0.75 if notes else 0.2,
        )
    except Exception as exc:  # noqa: BLE001
        return error_response(tool_name, section_title, exc)


@tool
def rewrite_academic_paragraph(paragraph: str, style: str = "concise") -> str:
    """Rewrite a paragraph in a more academic style without adding new facts."""
    tool_name = "rewrite_academic_paragraph"
    try:
        text = paragraph.strip()
        if not text:
            rewritten = ""
        elif style == "formal":
            rewritten = f"从已有证据来看，{text}"
        else:
            rewritten = text
        return json_response(
            tool_name=tool_name,
            query=style,
            results=[{"style": style, "rewritten": truncate(rewritten, 1200)}],
            confidence=0.6 if rewritten else 0.0,
            metadata={"adds_new_facts": False},
        )
    except Exception as exc:  # noqa: BLE001
        return error_response(tool_name, style, exc)


WRITING_TOOLS = [
    draft_review_outline,
    draft_review_section,
    rewrite_academic_paragraph,
]
