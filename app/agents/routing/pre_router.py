"""Enhanced rule-based query pre-router with confidence scoring.

Routes queries into 5 categories and assigns a confidence score 0-1.
High-confidence routes (>= threshold) skip the LLM router entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class PreRouteResult:
    route: str
    confidence: float
    reason: str


# ---- Keyword hint groups ----

_TASK_HINTS: Tuple[str, ...] = (
    "任务状态", "解析状态", "解析任务", "解析", "处理进度", "处理状态",
    "task status", "task_id",
    "任务", "上传", "入库", "文件", "文档", "切分", "索引", "paper",
)

_STATUS_HINTS: Tuple[str, ...] = (
    "状态", "进度", "完成", "完了吗", "失败", "成功", "status",
    "入库", "向量化", "好了吗", "处理", "检查", "查一下",
)

_PAPER_PROFILE_HINTS: Tuple[str, ...] = (
    "这篇论文", "介绍一下", "论文摘要", "论文概述", "论文画像",
    "paper profile", "介绍这篇", "概述一下", "讲了什么",
    "主要贡献", "核心思想", "创新点", "创新之处",
    "论文的贡献", "论文的作者", "发表信息",
)

_PAPER_SEARCH_HINTS: Tuple[str, ...] = (
    "哪些", "有没有", "有哪些", "知识库中", "多少篇", "多少",
    "列出", "筛选", "相关论文", "相关方法", "有什么论文",
    "盘点", "统计", "几篇",
)

_DEEP_SEARCH_HINTS: Tuple[str, ...] = (
    "哪些", "有没有", "有哪些", "知识库", "相关论文", "相关方法",
    "列出", "找出", "推荐", "盘点",
)

_EVIDENCE_HINTS: Tuple[str, ...] = (
    "依据", "证据", "理由", "解释", "分析", "为什么",
    "论证", "证明", "说明理由", "说明原因", "说明",
    "论证了什么", "有效性", "必要性",
)


@dataclass(frozen=True)
class _RouteRule:
    route: str
    base_confidence: float
    hints: Tuple[str, ...] = ()
    require_dual_group: bool = False  # e.g. task_status needs task+status match


_CONFIDENCE_RULES: Tuple[_RouteRule, ...] = (
    _RouteRule("task_status", 0.95, (), require_dual_group=True),
    _RouteRule("paper_profile", 0.75, _PAPER_PROFILE_HINTS),
    _RouteRule("paper_search", 0.65, _PAPER_SEARCH_HINTS),
    _RouteRule("paper_deep_search", 0.55, _DEEP_SEARCH_HINTS),
    _RouteRule("chunk_qa", 0.40, ()),  # always matches — default fallback
)


class EnhancedRuleRouter:
    """Rule-based pre-router with confidence scoring."""

    @staticmethod
    def route(query: str) -> PreRouteResult:
        normalized = query.strip().lower()
        candidates: List[Tuple[str, float, str]] = []

        for rule in _CONFIDENCE_RULES:
            if rule.require_dual_group:
                # task_status: need at least one task hint AND one status hint
                has_task = any(h in normalized for h in _TASK_HINTS)
                has_status = any(h in normalized for h in _STATUS_HINTS)
                if has_task and has_status:
                    candidates.append(
                        (rule.route, rule.base_confidence, "matched_task_status_keywords")
                    )
                continue

            if not rule.hints:
                # chunk_qa — always a candidate but lowest priority
                candidates.append(
                    (rule.route, rule.base_confidence, "default_fallback")
                )
                continue

            matched = sum(1 for h in rule.hints if h in normalized)
            if matched == 0:
                continue

            match_ratio = matched / len(rule.hints)
            confidence = min(rule.base_confidence + match_ratio * 0.25, 0.95)
            candidates.append(
                (rule.route, confidence, f"matched_{matched}_hints")
            )

        # Special case: deep_search needs both inventory AND evidence hints
        has_evidence = any(h in normalized for h in _EVIDENCE_HINTS)
        has_deep_inventory = any(h in normalized for h in _DEEP_SEARCH_HINTS)
        if has_deep_inventory and has_evidence:
            # Boost paper_deep_search confidence
            candidates = [
                (r, c, reason)
                for r, c, reason in candidates
                if r != "paper_deep_search"
            ]
            candidates.append(
                ("paper_deep_search", 0.85, "matched_inventory_with_evidence")
            )

        # Sort by confidence descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        if not candidates:
            return PreRouteResult(route="chunk_qa", confidence=0.40, reason="default")

        best = candidates[0]
        # Apply ambiguity penalty when top-2 are close
        if len(candidates) > 1 and (best[1] - candidates[1][1]) < 0.10:
            return PreRouteResult(
                route=best[0],
                confidence=round(best[1] * 0.85, 4),
                reason=f"ambiguous: {best[0]} vs {candidates[1][0]} ({best[2]})",
            )

        return PreRouteResult(route=best[0], confidence=round(best[1], 4), reason=best[2])
