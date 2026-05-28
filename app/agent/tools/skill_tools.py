"""Skill management tools — load and use external Claude Code skills.

Installed skills provide curated writing rules, review standards, and domain
knowledge. The agent should load a skill before applying its rules.
"""

from __future__ import annotations

from app.agent.tools.contracts import error_response, json_response
from app.agent.tools.decorators import tool
from app.services.skills import SkillContent, get_skill_loader


def _content_to_result(content: SkillContent) -> dict:
    return {
        "name": content.meta.name,
        "vendor": content.meta.vendor,
        "version": content.meta.version,
        "description": content.meta.description,
        "rules": content.rules,
        "references": content.references,
    }


@tool
def list_available_skills() -> str:
    """List all installed skill packs available for the agent to use.

    Returns metadata for each skill: name, vendor, version, description.
    Use this before load_skill to discover what skills are available.
    """
    tool_name = "list_available_skills"
    try:
        loader = get_skill_loader()
        skills = loader.list_skills()
        results = [
            {
                "name": meta.name,
                "vendor": meta.vendor,
                "version": meta.version,
                "description": meta.description,
            }
            for meta in skills
        ]
        return json_response(
            tool_name=tool_name,
            query="",
            results=results,
            confidence=1.0 if results else 0.0,
            metadata={"count": len(results)},
        )
    except Exception as exc:
        return error_response(tool_name, "", exc)


@tool
def load_skill(skill_name: str) -> str:
    """Load a skill's full rules into context.

    Args:
        skill_name: The skill name (e.g. 'nature-polishing', 'academic-paper').

    After loading, apply the skill's rules to your current task.
    Use list_available_skills first to discover what skills are installed.
    """
    tool_name = "load_skill"
    try:
        loader = get_skill_loader()
        content = loader.load_skill(skill_name)
        if content is None:
            return json_response(
                tool_name=tool_name,
                query=skill_name,
                results=[],
                error=f"Skill '{skill_name}' not found. Use list_available_skills to see installed skills.",
                confidence=0.0,
            )
        return json_response(
            tool_name=tool_name,
            query=skill_name,
            results=[_content_to_result(content)],
            confidence=1.0,
        )
    except Exception as exc:
        return error_response(tool_name, skill_name, exc)


@tool
def use_skill_reference(skill_name: str, reference_path: str) -> str:
    """Load a specific reference file from an installed skill.

    Use this when you need detailed rules for a specific task within a skill.
    Reference paths are listed in the skill's 'references' field returned by load_skill.

    Args:
        skill_name: The skill name (e.g. 'nature-polishing').
        reference_path: Reference file path (e.g. 'references/section-moves.md').
    """
    tool_name = "use_skill_reference"
    query = f"{skill_name}/{reference_path}"
    try:
        loader = get_skill_loader()
        text = loader.load_reference(skill_name, reference_path)
        if text is None:
            return json_response(
                tool_name=tool_name,
                query=query,
                results=[],
                error=f"Reference '{reference_path}' not found in skill '{skill_name}'. Check available references with load_skill.",
                confidence=0.0,
            )
        return json_response(
            tool_name=tool_name,
            query=query,
            results=[{"skill": skill_name, "reference": reference_path, "content": text[:5000]}],
            confidence=1.0,
        )
    except Exception as exc:
        return error_response(tool_name, query, exc)


SKILL_TOOLS = [list_available_skills, load_skill, use_skill_reference]
