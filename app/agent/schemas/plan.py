"""Structured agent plans — executable parameters, not free-form step descriptions."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.agent.schemas.types import PlanStep

AgentTaskType = Literal[
    "paper_comparison",
    "paper_qa",
    "paper_profile",
    "literature_summary",
    "writing",
    "chat",
]

AspectName = Literal[
    "method",
    "architecture",
    "module",
    "loss",
    "experiment",
    "dataset",
    "metric",
    "innovation",
    "limitation",
]

_OUTPUT_FORMAT = Literal["table", "summary", "report"]


class PaperTarget(BaseModel):
    alias: str = Field(description="A/B/C or user-given label")
    query: str = Field(description="Title, short name, or keywords for search")
    required: bool = True


class CompareAspect(BaseModel):
    name: AspectName
    evidence_query: str = ""
    section_types: list[str] = Field(default_factory=list)


class ComparisonPlan(BaseModel):
    task_type: Literal["paper_comparison"] = "paper_comparison"
    targets: list[PaperTarget] = Field(default_factory=list)
    aspects: list[CompareAspect] = Field(default_factory=list)
    output_format: _OUTPUT_FORMAT = "table"
    discovery_query: str = ""


class AgentPlan(BaseModel):
    task_type: AgentTaskType
    summary: str = ""
    targets: list[PaperTarget] = Field(default_factory=list)
    aspects: list[CompareAspect] = Field(default_factory=list)
    output_format: _OUTPUT_FORMAT = "table"
    discovery_query: str = ""
    steps: list[dict[str, Any]] = Field(default_factory=list)

    def model_dump_plan(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


DEFAULT_METHOD_ASPECT = CompareAspect(
    name="method",
    evidence_query="method architecture framework module proposed method 方法 网络结构 核心模块",
    section_types=["method", "approach", "model", "proposed_method"],
)

_ASPECT_PRESETS: dict[str, CompareAspect] = {
    "method": DEFAULT_METHOD_ASPECT,
    "experiment": CompareAspect(
        name="experiment",
        evidence_query="experiment result metric evaluation dataset 实验 结果 指标",
        section_types=["experiment", "results", "evaluation"],
    ),
    "innovation": CompareAspect(
        name="innovation",
        evidence_query="contribution novelty innovation 贡献 创新",
        section_types=["introduction", "method", "contribution"],
    ),
    "dataset": CompareAspect(
        name="dataset",
        evidence_query="dataset benchmark data 数据集",
        section_types=["experiment", "dataset"],
    ),
    "metric": CompareAspect(
        name="metric",
        evidence_query="metric PSNR SSIM performance 指标",
        section_types=["experiment", "results"],
    ),
    "loss": CompareAspect(
        name="loss",
        evidence_query="loss function objective 损失函数",
        section_types=["method", "approach"],
    ),
    "architecture": CompareAspect(
        name="architecture",
        evidence_query="architecture network structure 架构 网络结构",
        section_types=["method", "approach", "model"],
    ),
    "module": CompareAspect(
        name="module",
        evidence_query="module block component 模块",
        section_types=["method", "approach"],
    ),
    "limitation": CompareAspect(
        name="limitation",
        evidence_query="limitation discussion future work 局限",
        section_types=["discussion", "conclusion"],
    ),
}


def aspect_from_name(name: str) -> CompareAspect:
    normalized = name.strip().lower()
    if normalized in _ASPECT_PRESETS:
        return _ASPECT_PRESETS[normalized].model_copy()
    return CompareAspect(name="method", evidence_query=name, section_types=[])


def intent_to_task_type(intent: str) -> AgentTaskType:
    mapping: dict[str, AgentTaskType] = {
        "retrieval": "paper_qa",
        "comparison": "paper_comparison",
        "summary": "literature_summary",
        "profile": "paper_profile",
        "writing": "writing",
        "chat": "chat",
    }
    return mapping.get(intent, "paper_qa")


def minimal_agent_plan(intent: str, query: str) -> AgentPlan:
    task_type = intent_to_task_type(intent)
    if intent == "comparison":
        targets = extract_comparison_targets_from_query(query)
        return AgentPlan(
            task_type="paper_comparison",
            summary=f"Auto comparison plan for: {query[:80]}",
            targets=targets,
            aspects=[DEFAULT_METHOD_ASPECT.model_copy()],
            discovery_query=query if not targets else "",
        )
    steps = _minimal_steps(intent, query)
    return AgentPlan(
        task_type=task_type,
        summary=f"Auto-generated {intent} plan",
        steps=[dict(s) for s in steps],
    )


def _minimal_steps(intent: str, query: str) -> list[PlanStep]:
    if intent == "retrieval":
        return [
            {"step": 1, "action": "search_papers", "description": "Search for papers matching the query", "params": {"query": query}},
            {"step": 2, "action": "retrieve_evidence", "description": "Retrieve detailed evidence from matched papers", "params": {"query": query}},
        ]
    if intent == "summary":
        return [
            {"step": 1, "action": "search_papers", "description": "Search for papers", "params": {"query": query}},
            {"step": 2, "action": "retrieve_evidence", "description": "Retrieve detailed evidence", "params": {}},
            {"step": 3, "action": "generate_outline", "description": "Generate review outline", "params": {}},
        ]
    if intent == "writing":
        return [{"step": 1, "action": "polish", "description": "Polish the provided text", "params": {}}]
    if intent == "profile":
        return [{"step": 1, "action": "search_papers", "description": "Search paper profiles", "params": {"query": query}}]
    return [{"step": 1, "action": "chat", "description": "Respond to user", "params": {}}]


_COMPARISON_SPLIT_RE = re.compile(r"\s+(?:和|与|vs\.?|versus|对比|比较)\s+", re.IGNORECASE)


def extract_comparison_targets_from_query(query: str) -> list[PaperTarget]:
    text = query.strip()
    for sep in ("的方法", "的实验", "方法", "实验", "区别", "差异"):
        if text.endswith(sep):
            text = text[: -len(sep)].strip()
        elif sep in text and sep not in ("方法", "实验"):
            text = text.split(sep)[0].strip()

    parts = _COMPARISON_SPLIT_RE.split(text)
    if len(parts) < 2:
        return []

    cleaned: list[str] = []
    for part in parts:
        p = part.strip(" ，,、. ")
        p = re.sub(r"^(比较|对比|请)", "", p).strip()
        if len(p) >= 2:
            cleaned.append(p[:120])

    if len(cleaned) < 2:
        return []

    aliases = ["A", "B", "C", "D", "E"]
    return [PaperTarget(alias=aliases[i], query=cleaned[i], required=True) for i in range(min(len(cleaned), 5))]


def _infer_aspects_from_query(query: str) -> list[CompareAspect]:
    normalized = query.lower()
    aspects: list[CompareAspect] = []
    keywords: list[tuple[tuple[str, ...], str]] = [
        (("实验", "结果", "指标", "experiment", "metric"), "experiment"),
        (("数据集", "dataset", "benchmark"), "dataset"),
        (("创新", "贡献", "innovation", "contribution"), "innovation"),
        (("损失", "loss"), "loss"),
        (("架构", "网络", "architecture"), "architecture"),
        (("方法", "method", "模块", "module"), "method"),
    ]
    for terms, name in keywords:
        if any(t in normalized for t in terms):
            aspects.append(aspect_from_name(name))
    if not aspects:
        aspects.append(DEFAULT_METHOD_ASPECT.model_copy())
    return aspects


def parse_plan_from_llm(intent: str, raw: dict[str, Any], query: str) -> AgentPlan:
    try:
        if intent == "comparison":
            return _parse_comparison_plan(raw, query)
        task_type = intent_to_task_type(intent)
        steps = raw.get("steps") or raw.get("plan") or []
        if not isinstance(steps, list):
            steps = []
        return AgentPlan(task_type=raw.get("task_type", task_type), summary=str(raw.get("summary", "")), steps=steps)
    except Exception:
        return minimal_agent_plan(intent, query)


def _parse_comparison_plan(raw: dict[str, Any], query: str) -> AgentPlan:
    targets_raw = raw.get("targets") or []
    aspects_raw = raw.get("aspects") or []
    targets: list[PaperTarget] = []
    for i, item in enumerate(targets_raw):
        if not isinstance(item, dict):
            continue
        alias = str(item.get("alias") or ["A", "B", "C", "D", "E"][min(i, 4)])
        q = str(item.get("query") or "").strip()
        if q:
            targets.append(PaperTarget(alias=alias, query=q, required=bool(item.get("required", True))))

    if not targets:
        targets = extract_comparison_targets_from_query(query)

    aspects: list[CompareAspect] = []
    for item in aspects_raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "method").lower()
        try:
            aspects.append(CompareAspect(name=name, evidence_query=str(item.get("evidence_query") or ""), section_types=list(item.get("section_types") or [])))
        except Exception:
            aspects.append(aspect_from_name(name))

    if not aspects:
        aspects = _infer_aspects_from_query(query)

    for asp in aspects:
        if not asp.evidence_query:
            preset = aspect_from_name(asp.name)
            asp.evidence_query = preset.evidence_query
        if not asp.section_types:
            preset = aspect_from_name(asp.name)
            asp.section_types = list(preset.section_types)

    output_format = raw.get("output_format", "table")
    if output_format not in ("table", "summary", "report"):
        output_format = "table"

    return AgentPlan(task_type="paper_comparison", summary=str(raw.get("summary", "")), targets=targets, aspects=aspects, output_format=output_format, discovery_query=str(raw.get("discovery_query") or ""))


def plan_steps_from_agent_plan(plan: AgentPlan | dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(plan, AgentPlan):
        return list(plan.steps)
    if isinstance(plan, dict):
        steps = plan.get("steps")
        if isinstance(steps, list):
            return steps
    return []


def get_plan_task_type(plan: AgentPlan | dict[str, Any] | list | None) -> str:
    if plan is None:
        return "paper_qa"
    if isinstance(plan, dict):
        return str(plan.get("task_type") or "paper_qa")
    return "paper_qa"
