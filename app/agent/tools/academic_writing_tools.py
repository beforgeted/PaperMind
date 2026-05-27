"""Academic writing tools backed by nature-skills and ARS references.

Nature-style polishing, structured review outline generation, and peer review
are powered by LLM with curated reference rules loaded as context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.agent.tools.contracts import error_response, json_response
from app.agent.tools.decorators import tool
from app.services.llm_service import get_llm

_REFERENCES_DIR = Path(__file__).resolve().parent / "references"

_NATURE_POLISH_RULES = (_REFERENCES_DIR / "nature_polishing_rules.md").read_text(encoding="utf-8")
_NATURE_WRITING_RULES = (_REFERENCES_DIR / "nature_writing_patterns.md").read_text(encoding="utf-8")
_LIT_REVIEW_STRUCTURE = (_REFERENCES_DIR / "literature_review_structure.md").read_text(encoding="utf-8")

_POLISH_SYSTEM = f"""You are an academic writing editor trained on Nature-journal editorial standards.

Below are the editing rules you must follow:

{_NATURE_POLISH_RULES}

Apply these rules to the text provided by the user. Return ONLY the polished text plus revision notes.
Do not invent citations, data, or claims. Preserve the author's core argument while improving structure, clarity, and academic tone."""

_REVIEW_OUTLINE_SYSTEM = f"""You are a literature review strategist. Use the following structure template:

{_LIT_REVIEW_STRUCTURE}

{_NATURE_WRITING_RULES}

Given a topic and available evidence, generate a structured literature review outline.
Include specific theme titles based on the evidence provided.
Each theme should map to evidence from the knowledge base.
Return JSON with the outline and a brief explanation of the thematic organization."""

_PEER_REVIEW_SYSTEM = f"""You are a constructive peer reviewer for academic drafts. Apply these review standards:

{_NATURE_POLISH_RULES}

Review the provided draft section and return a structured review:
1. Overall assessment (score 0-100)
2. What works well (2-3 points)
3. What needs improvement (2-3 points, prioritized)
4. Specific suggestions for revision
5. Overclaim or unsupported claim check
Be specific and actionable. Reference the style rules when relevant."""


def _llm_generate(system_prompt: str, user_prompt: str) -> str:
    llm = get_llm()
    from langchain_core.messages import HumanMessage, SystemMessage

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return response.content if hasattr(response, "content") else str(response)


@tool
def polish_academic_text(
    text: str,
    section_type: str = "unknown",
    style: str = "nature",
) -> str:
    """Polish academic text using Nature-journal editorial standards.

    Args:
        text: The text to polish (paragraph, section, or abstract).
        section_type: One of introduction, results, discussion, conclusion, abstract, methods.
        style: nature (default) for Nature standards.
    """
    tool_name = "polish_academic_text"
    query = f"{section_type}/{style}"
    try:
        if not text.strip():
            return error_response(tool_name, query, ValueError("text is empty"))

        user_prompt = (
            f"Section type: {section_type}\nStyle: {style}\n\n"
            f"Please polish the following academic text:\n\n{text}"
        )
        polished = _llm_generate(_POLISH_SYSTEM, user_prompt)
        return json_response(
            tool_name=tool_name,
            query=query,
            results=[{"original": text[:500], "polished": polished, "section_type": section_type}],
            confidence=0.85,
        )
    except Exception as exc:
        return error_response(tool_name, query, exc)


@tool
def generate_review_outline(
    topic: str,
    evidence_notes: str = "",
    num_themes: int = 3,
) -> str:
    """Generate a structured literature review outline following academic standards.

    Uses ARS literature review structure and Nature writing patterns.

    Args:
        topic: The review topic.
        evidence_notes: Retrieved evidence from the knowledge base (paper titles, key findings).
        num_themes: Number of thematic sections (3-5 recommended).
    """
    tool_name = "generate_review_outline"
    try:
        if not topic.strip():
            return error_response(tool_name, topic, ValueError("topic is empty"))

        user_prompt = (
            f"Topic: {topic}\n"
            f"Number of themes: {num_themes}\n\n"
            f"Available evidence from knowledge base:\n{evidence_notes or '(no evidence provided — generate a scaffold outline with placeholders)'}\n\n"
            "Generate a structured literature review outline. Return JSON with:\n"
            '{"title": "review title", "abstract_outline": "...", '
            '"themes": [{"title": "...", "sub_themes": ["..."]}], '
            '"cross_cutting": ["..."], "gaps": ["..."], "explanation": "..."}'
        )
        result = _llm_generate(_REVIEW_OUTLINE_SYSTEM, user_prompt)
        return json_response(
            tool_name=tool_name,
            query=topic,
            results=[{"topic": topic, "outline": result, "evidence_based": bool(evidence_notes.strip())}],
            confidence=0.8 if evidence_notes.strip() else 0.4,
        )
    except Exception as exc:
        return error_response(tool_name, topic, exc)


@tool
def generate_review_section(
    section_title: str,
    evidence_notes: str,
    section_type: str = "body",
) -> str:
    """Draft a literature review section with Nature-journal writing standards.

    Args:
        section_title: Title of the review section.
        evidence_notes: Evidence from the knowledge base to ground the section.
        section_type: body (thematic synthesis), introduction, discussion, or conclusion.
    """
    tool_name = "generate_review_section"
    try:
        if not section_title.strip():
            return error_response(tool_name, "", ValueError("section_title is empty"))

        user_prompt = (
            f"Section title: {section_title}\n"
            f"Section type: {section_type}\n\n"
            f"Evidence from knowledge base:\n{evidence_notes or '(no evidence)'}\n\n"
            f"Draft this section. Synthesize thematically (NOT paper-by-paper). "
            f"Include citations to the evidence. Mark any claims that lack evidence with [NEEDS EVIDENCE]."
        )
        draft = _llm_generate(_REVIEW_OUTLINE_SYSTEM, user_prompt)
        return json_response(
            tool_name=tool_name,
            query=section_title,
            results=[{"section_title": section_title, "draft": draft, "evidence_notes": evidence_notes[:500]}],
            confidence=0.75 if evidence_notes.strip() else 0.2,
            metadata={"adds_new_facts": False},
        )
    except Exception as exc:
        return error_response(tool_name, section_title, exc)


@tool
def peer_review_draft(
    draft_text: str,
    section_type: str = "body",
) -> str:
    """Self-review a draft section using academic peer review standards.

    Args:
        draft_text: The draft text to review.
        section_type: introduction, results, discussion, conclusion, abstract, body.
    """
    tool_name = "peer_review_draft"
    query = section_type
    try:
        if not draft_text.strip():
            return error_response(tool_name, query, ValueError("draft_text is empty"))

        user_prompt = (
            f"Section type: {section_type}\n\n"
            f"Draft to review:\n\n{draft_text}\n\n"
            "Provide a structured peer review."
        )
        review = _llm_generate(_PEER_REVIEW_SYSTEM, user_prompt)
        return json_response(
            tool_name=tool_name,
            query=query,
            results=[{"section_type": section_type, "review": review, "draft_excerpt": draft_text[:300]}],
            confidence=0.8,
        )
    except Exception as exc:
        return error_response(tool_name, query, exc)


ACADEMIC_WRITING_TOOLS = [
    polish_academic_text,
    generate_review_outline,
    generate_review_section,
    peer_review_draft,
]
