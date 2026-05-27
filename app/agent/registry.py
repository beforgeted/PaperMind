"""Tool registry for PaperMind agents."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Literal

ToolProfile = Literal["basic", "research", "writing", "workflow", "all"]


@dataclass(frozen=True)
class ToolFactory:
    """Lazy reference to a LangChain-compatible tool."""

    name: str
    module: str
    attribute: str

    def load(self):
        """Import and return the underlying tool object."""
        return getattr(import_module(self.module), self.attribute)


_FACTORIES: dict[str, ToolFactory] = {
    "retrieve_evidence": ToolFactory(
        "retrieve_evidence", "app.agent.tools.rag_tools", "retrieve_evidence"
    ),
    "answer_with_rag": ToolFactory(
        "answer_with_rag", "app.agent.tools.rag_tools", "answer_with_rag"
    ),
    "search_paper_chunks": ToolFactory(
        "search_paper_chunks", "app.agent.tools.rag_tools", "search_paper_chunks"
    ),
    "search_paper_profiles": ToolFactory(
        "search_paper_profiles", "app.agent.tools.paper_search_tools", "search_paper_profiles"
    ),
    "deep_search_papers": ToolFactory(
        "deep_search_papers", "app.agent.tools.paper_search_tools", "deep_search_papers"
    ),
    "get_paper_profile": ToolFactory(
        "get_paper_profile", "app.agent.tools.paper_search_tools", "get_paper_profile"
    ),
    "get_task_status": ToolFactory(
        "get_task_status", "app.agent.tools.task_tools", "get_task_status"
    ),
    "remember_fact": ToolFactory(
        "remember_fact", "app.agent.tools.memory_tools", "remember_fact"
    ),
    "recall_memory": ToolFactory(
        "recall_memory", "app.agent.tools.memory_tools", "recall_memory"
    ),
    "update_workspace_state": ToolFactory(
        "update_workspace_state", "app.agent.tools.memory_tools", "update_workspace_state"
    ),
    "get_workspace_state": ToolFactory(
        "get_workspace_state", "app.agent.tools.memory_tools", "get_workspace_state"
    ),
    "draft_review_outline": ToolFactory(
        "draft_review_outline", "app.agent.tools.writing_tools", "draft_review_outline"
    ),
    "draft_review_section": ToolFactory(
        "draft_review_section", "app.agent.tools.writing_tools", "draft_review_section"
    ),
    "rewrite_academic_paragraph": ToolFactory(
        "rewrite_academic_paragraph", "app.agent.tools.writing_tools", "rewrite_academic_paragraph"
    ),
}

_PROFILES: dict[ToolProfile, tuple[str, ...]] = {
    "basic": (
        "retrieve_evidence",
        "answer_with_rag",
        "search_paper_profiles",
        "deep_search_papers",
        "get_paper_profile",
        "get_task_status",
    ),
    "research": (
        "retrieve_evidence",
        "answer_with_rag",
        "search_paper_profiles",
        "deep_search_papers",
        "get_paper_profile",
        "get_task_status",
        "remember_fact",
        "recall_memory",
        "update_workspace_state",
        "get_workspace_state",
    ),
    "writing": (
        "retrieve_evidence",
        "search_paper_profiles",
        "get_paper_profile",
        "remember_fact",
        "recall_memory",
        "draft_review_outline",
        "draft_review_section",
        "rewrite_academic_paragraph",
    ),
    "workflow": (
        "retrieve_evidence",
        "search_paper_profiles",
        "deep_search_papers",
        "get_paper_profile",
        "remember_fact",
        "recall_memory",
        "update_workspace_state",
        "get_workspace_state",
        "draft_review_outline",
        "draft_review_section",
    ),
    "all": tuple(_FACTORIES.keys()),
}


def available_profiles() -> list[str]:
    """Return the registered tool profile names."""
    return list(_PROFILES.keys())


def get_tool_factories(profile: ToolProfile = "research") -> list[ToolFactory]:
    """Return lazy tool factories for a profile."""
    names = _PROFILES[profile]
    return [_FACTORIES[name] for name in names]


def get_tools(profile: ToolProfile = "research") -> list:
    """Load concrete LangChain-compatible tools for a profile."""
    return [factory.load() for factory in get_tool_factories(profile)]
