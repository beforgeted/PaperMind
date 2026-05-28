"""Literature review (summary) handler — search → retrieve → outline → draft → review → polish."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.state import AgentState
from app.shared.serializers import chunk_to_result, paper_search_result_to_result, result_to_source
from app.services.llm import get_llm
from app.services.papers.search import search_papers_by_query
from app.services.retrieval import hybrid_retrieve
from app.services.skills import get_skill_loader

_OUTLINE_SYSTEM = """You are a literature review strategist. Generate a structured review outline.

For the given topic and available evidence, create:
1. A working title
2. An abstract scaffold
3. 3-5 thematic sections (with sub-themes)
4. Cross-cutting synthesis points
5. Identified research gaps

IMPORTANT:
- Organize by THEME, not by paper.
- Each theme title should be specific to the evidence.
- Output clean JSON-like structure that can be used for drafting."""

_DRAFT_SYSTEM = """You are an academic writer. Draft ONE section of a literature review.

Rules:
- Synthesize thematically, NOT paper-by-paper.
- Use evidence from the provided notes. Cite sources.
- Mark unsupported claims with [NEEDS EVIDENCE].
- Follow Nature writing standards: one paragraph = one controlling idea.
- Keep sentences under 30 words.
- Write in English (or Chinese if evidence is Chinese)."""

_POLISH_SYSTEM = """You are an academic editor. Polish the provided literature review draft.

Apply:
- Nature-journal editorial standards
- Remove AI-typical terms (delve, tapestry, landscape)
- No em dashes
- Vary paragraph length and sentence rhythm
- Every claim needs evidence support
- Add revision notes (3-5 bullets)"""


async def summary_handler(state: AgentState) -> dict[str, Any]:
    query = state.get("enriched_query") or state.get("query", "")
    task_id = state.get("task_id")
    used_tools: list[str] = []
    all_contexts: list[dict] = []

    # Step 1: Load skills
    try:
        loader = get_skill_loader()
        loader.load_skill("academic-paper")
        loader.load_skill("nature-polishing")
        used_tools.append("load_skill")
    except Exception:
        pass

    # Step 2: Search for papers on the topic
    search_results = await search_papers_by_query(query=query, task_id=task_id)
    used_tools.append("search_paper_profiles")

    if len(search_results) < 2:
        return {
            "handler_answer": "当前知识库中论文数量不足，无法生成有意义的综述。请先上传更多相关论文。",
            "handler_contexts": [],
            "handler_sources": [],
            "handler_used_tools": used_tools,
        }

    # Step 3: Retrieve evidence for top papers
    top_papers = search_results[:8]
    evidence_parts: list[str] = []
    for paper in top_papers:
        chunks = await hybrid_retrieve(query=query, task_id=paper.paper_id, top_k=3)
        used_tools.append("retrieve_evidence")
        for c in chunks:
            all_contexts.append(chunk_to_result(c))
        paper_text = "\n".join(c.parent_text[:300] for c in chunks if c.parent_text)
        evidence_parts.append(
            f"### {paper.title or paper.source_file} ({paper.paper_id})\n"
            f"Main task: {paper.main_task}\n{paper.abstract_summary}\n{paper.method_summary}\n\n{paper_text}"
        )

    evidence_notes = "\n\n".join(evidence_parts)
    llm = get_llm()

    # Step 4: Generate outline
    outline_resp = await llm.ainvoke([
        SystemMessage(content=_OUTLINE_SYSTEM),
        HumanMessage(content=f"Topic: {query}\n\nEvidence:\n{evidence_notes[:8000]}"),
    ])
    used_tools.append("generate_review_outline")
    outline = outline_resp.content if hasattr(outline_resp, "content") else str(outline_resp)

    # Step 5: Draft themes (generate up to 3 sections)
    # Extract theme titles from outline (simple heuristic)
    sections_drafted: list[str] = []
    for line in outline.split("\n"):
        line = line.strip()
        if line.startswith("##") or (line.startswith("#") and "theme" in line.lower()):
            section_title = line.lstrip("#").strip()
            draft_resp = await llm.ainvoke([
                SystemMessage(content=_DRAFT_SYSTEM),
                HumanMessage(content=f"Topic: {query}\nSection: {section_title}\nEvidence:\n{evidence_notes[:6000]}"),
            ])
            sections_drafted.append(f"## {section_title}\n\n{draft_resp.content}")
            if len(sections_drafted) >= 3:
                break
    used_tools.append("generate_review_section")

    if not sections_drafted:
        sections_drafted = ["(未能生成综述段落，请提供更具体的主题。)"]

    # Step 6: Polish the full draft
    full_draft = f"# Literature Review: {query}\n\n{outline}\n\n" + "\n\n".join(sections_drafted)
    polish_resp = await llm.ainvoke([
        SystemMessage(content=_POLISH_SYSTEM),
        HumanMessage(content=full_draft[:10000]),
    ])
    used_tools.append("polish_academic_text")
    polished = polish_resp.content if hasattr(polish_resp, "content") else str(polish_resp)

    return {
        "handler_answer": polished,
        "handler_contexts": all_contexts,
        "handler_sources": _dedup([result_to_source(c) for c in all_contexts if isinstance(c, dict)]),
        "handler_used_tools": list(set(used_tools)),
    }


def _dedup(sources: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    result: list[dict] = []
    for src in sources:
        key = (src.get("paper_id"), src.get("title"))
        if key not in seen:
            seen.add(key)
            result.append(src)
    return result
