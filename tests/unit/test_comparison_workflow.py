"""Unit tests for plan-driven comparison workflow."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class ComparisonPlanSchemaTests(unittest.TestCase):
    def test_extract_targets_lcdnet_vs_ushape(self):
        from app.agent.schemas.plan import extract_comparison_targets_from_query

        targets = extract_comparison_targets_from_query("比较 LCDNet 和 U-shape 的方法")
        self.assertGreaterEqual(len(targets), 2)
        queries = {t.query for t in targets}
        self.assertTrue(any("LCDNet" in q for q in queries))
        self.assertTrue(any("U-shape" in q or "U shape" in q for q in queries))

    def test_minimal_comparison_plan_has_targets(self):
        from app.agent.schemas.plan import minimal_agent_plan

        plan = minimal_agent_plan("comparison", "对比 LCDNet 与 U-shape 方法")
        self.assertEqual(plan.task_type, "paper_comparison")
        self.assertGreaterEqual(len(plan.targets), 2)
        self.assertGreaterEqual(len(plan.aspects), 1)


class PaperSearchNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_separate_search_per_target(self):
        from app.agent.workflows.comparison.nodes import paper_search_node

        calls: list[str] = []

        async def _fake_search(query: str, task_id=None):
            calls.append(query)
            return []

        with patch(
            "app.agent.workflows.comparison.nodes.search_papers_by_query",
            new=_fake_search,
        ):
            await paper_search_node(
                {
                    "target": {"alias": "A", "query": "LCDNet"},
                    "task_id": None,
                }
            )
            await paper_search_node(
                {
                    "target": {"alias": "B", "query": "U-shape"},
                    "task_id": None,
                }
            )

        self.assertEqual(calls, ["LCDNet", "U-shape"])


class ResolvePapersTests(unittest.IsolatedAsyncioTestCase):
    async def test_not_found_when_no_candidates(self):
        from app.agent.workflows.comparison.nodes import resolve_papers_node

        result = await resolve_papers_node(
            {
                "plan": {"targets": [{"alias": "X", "query": "NonExistPaper"}]},
                "paper_candidates": {"X": []},
            }
        )
        self.assertEqual(result["resolved_papers"]["X"]["status"], "not_found")


class PlanValidateDegradationTests(unittest.TestCase):
    def test_degrade_when_no_targets_and_no_discovery(self):
        from app.agent.planning.validator import _degrade_comparison_if_unexecutable
        from app.agent.schemas.plan import AgentPlan

        plan = AgentPlan(
            task_type="paper_comparison",
            summary="empty",
            targets=[],
            aspects=[],
            discovery_query="",
        )
        degraded, reason = _degrade_comparison_if_unexecutable(plan, "比较")
        self.assertEqual(degraded.task_type, "paper_qa")
        self.assertEqual(reason, "comparison_degraded_to_paper_qa")


class CoverageCheckTests(unittest.IsolatedAsyncioTestCase):
    async def test_insufficient_evidence_triggers_missing(self):
        from app.agent.workflows.comparison.nodes import coverage_check_node

        result = await coverage_check_node(
            {
                "plan": {
                    "aspects": [{"name": "method", "evidence_query": "method"}],
                },
                "resolved_papers": {
                    "A": {"status": "resolved", "paper_id": "p1", "title": "Paper A"},
                },
                "evidence_blocks": [
                    {
                        "alias": "A",
                        "aspect": "method",
                        "paper_id": "p1",
                        "parent_id": "c1",
                    }
                ],
            }
        )
        report = result["coverage_report"]
        self.assertFalse(report["ok"])
        self.assertTrue(any(m.get("reason") == "insufficient_evidence" for m in report["missing"]))

    async def test_route_after_coverage_retry_once(self):
        from app.agent.workflows.comparison.routing import route_after_coverage

        self.assertEqual(
            route_after_coverage({"coverage_report": {"ok": False}, "retry_count": 0}),
            "query_rewrite_node",
        )
        self.assertEqual(
            route_after_coverage({"coverage_report": {"ok": False}, "retry_count": 1}),
            "compare_node",
        )


class DispatchEvidenceTests(unittest.TestCase):
    def test_expands_paper_times_aspect(self):
        from app.agent.workflows.comparison.dispatch import dispatch_evidence_retrieval
        from langgraph.types import Send

        sends = dispatch_evidence_retrieval(
            {
                "plan": {
                    "aspects": [
                        {"name": "method", "evidence_query": "method"},
                        {"name": "experiment", "evidence_query": "experiment"},
                    ],
                },
                "resolved_papers": {
                    "A": {"status": "resolved", "paper_id": "p1", "title": "A"},
                    "B": {"status": "resolved", "paper_id": "p2", "title": "B"},
                    "C": {"status": "resolved", "paper_id": "p3", "title": "C"},
                },
            }
        )
        self.assertEqual(len(sends), 6)
        self.assertTrue(all(isinstance(s, Send) for s in sends))

    def test_retry_only_missing_pairs(self):
        from app.agent.workflows.comparison.dispatch import dispatch_evidence_retrieval
        from langgraph.types import Send

        sends = dispatch_evidence_retrieval(
            {
                "retry_count": 1,
                "plan": {
                    "aspects": [
                        {"name": "method", "evidence_query": "method expanded"},
                        {"name": "experiment", "evidence_query": "experiment"},
                    ],
                },
                "resolved_papers": {
                    "A": {"status": "resolved", "paper_id": "p1", "title": "A"},
                    "B": {"status": "resolved", "paper_id": "p2", "title": "B"},
                },
                "coverage_report": {
                    "ok": False,
                    "missing": [
                        {
                            "alias": "B",
                            "aspect": "method",
                            "reason": "insufficient_evidence",
                            "count": 1,
                        },
                    ],
                },
            }
        )
        self.assertEqual(len(sends), 1)
        self.assertTrue(isinstance(sends[0], Send))
        payload = sends[0].arg
        self.assertEqual(payload.get("alias"), "B")
        self.assertEqual(payload.get("aspect", {}).get("name"), "method")


class RetrievalSectionTypesTests(unittest.TestCase):
    def test_build_retrieval_intent_override(self):
        from app.services.retrieval import _build_retrieval_intent

        intent = _build_retrieval_intent("generic query", ["method", "approach"])
        self.assertIn("method", intent.section_types)
        self.assertIn("approach", intent.section_types)


if __name__ == "__main__":
    unittest.main()
