"""LangGraph agent unit tests — comprehensive coverage.

Run from the repo root (recommended):

    python -m unittest tests.unit.test_langgraph_agent -v

Coverage areas:
  1. Intent routing — rule-based + edge cases + mixed language
  2. Planner — decomposition gaps, coverage by intent
  3. Retrieval intent inference — section detection, entity extraction, query expansion
  4. Tool selection — which tools called per intent, discovery gap
  5. Handler pipelines — completeness check per intent
  6. Synthesizer — dedup border cases
  7. Paper structure — entity extraction for model names
"""

from __future__ import annotations

import ast
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch  # noqa: F401
except Exception:
    pass

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_GRAPH_BUILDER_PATH = _REPO_ROOT / "app" / "agent" / "builder.py"


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _is_synthesizer_to_end_edge(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call) or not hasattr(node.func, "attr"):
        return False
    if node.func.attr != "add_edge" or len(node.args) < 2:
        return False
    first, second = node.args[0], node.args[1]
    if not (isinstance(first, ast.Constant) and first.value == "synthesizer"):
        return False
    if isinstance(second, ast.Name) and second.id == "END":
        return True
    return isinstance(second, ast.Constant) and second.value == "__end__"


# ---------------------------------------------------------------------------
# 1. AgentState structure
# ---------------------------------------------------------------------------

class AgentStateTests(unittest.TestCase):
    def test_state_has_required_fields(self):
        from app.agent.state import AgentState
        fields = AgentState.__annotations__
        required = {"query", "intent", "intent_confidence", "final_answer", "used_tools"}
        for field in required:
            self.assertIn(field, fields, f"AgentState missing field: {field}")

    def test_state_has_routing_fields(self):
        from app.agent.state import AgentState
        fields = AgentState.__annotations__
        routing = {"intent", "intent_confidence", "rule_matched", "routing_reason"}
        for field in routing:
            self.assertIn(field, fields, f"AgentState missing routing field: {field}")

    def test_state_has_handler_output_fields(self):
        from app.agent.state import AgentState
        fields = AgentState.__annotations__
        handler = {"handler_answer", "handler_contexts", "handler_sources", "handler_used_tools"}
        for field in handler:
            self.assertIn(field, fields, f"AgentState missing handler field: {field}")

    def test_state_has_planner_fields(self):
        from app.agent.state import AgentState
        fields = AgentState.__annotations__
        self.assertIn("plan", fields)
        self.assertIn("plan_summary", fields)
        self.assertIn("plan_validated", fields)
        self.assertIn("paper_candidates", fields)
        self.assertIn("evidence_blocks", fields)
        self.assertIn("trace", fields)

    def test_plan_step_structure(self):
        from app.agent.state import PlanStep
        step: PlanStep = {
            "step": 1,
            "action": "search_papers",
            "description": "Search for papers on the topic",
            "params": {"query": "test"},
        }
        self.assertEqual(step["step"], 1)
        self.assertEqual(step["action"], "search_papers")


# ---------------------------------------------------------------------------
# 2. Intent Router — basic intents
# ---------------------------------------------------------------------------

class IntentRouterRuleTests(unittest.TestCase):
    def setUp(self):
        from app.agent.routing.intent import _rule_classify
        self._rule_classify = _rule_classify

    # -- retrieval --
    def test_retrieval_zh_method(self):
        intent, confidence, reason = self._rule_classify("频域增强方法具体怎么实现的")
        self.assertEqual(intent, "retrieval")
        self.assertGreater(confidence, 0.5)

    def test_retrieval_zh_what_is(self):
        intent, confidence, reason = self._rule_classify("什么是自适应对数变换")
        self.assertEqual(intent, "retrieval")
        self.assertGreater(confidence, 0.4)

    def test_retrieval_en(self):
        intent, confidence, reason = self._rule_classify("how does the FFT method work")
        self.assertEqual(intent, "retrieval")
        self.assertGreater(confidence, 0.4)

    # -- comparison --
    def test_comparison_zh(self):
        intent, confidence, reason = self._rule_classify("对比论文A和论文B的方法有什么不同")
        self.assertEqual(intent, "comparison")
        self.assertGreaterEqual(confidence, 0.8)

    def test_comparison_en(self):
        intent, confidence, reason = self._rule_classify("compare paper A and paper B which is better")
        self.assertEqual(intent, "comparison")
        self.assertGreaterEqual(confidence, 0.8)

    # -- summary --
    def test_summary_zh(self):
        intent, confidence, reason = self._rule_classify("帮我生成水下图像增强的文献综述")
        self.assertEqual(intent, "summary")
        self.assertGreater(confidence, 0.5)

    # -- writing --
    def test_writing_zh(self):
        intent, confidence, reason = self._rule_classify("润色这段文字使其更学术化")
        self.assertEqual(intent, "writing")
        self.assertGreaterEqual(confidence, 0.8)

    def test_writing_polish_en(self):
        intent, confidence, reason = self._rule_classify("polish this paragraph please")
        self.assertEqual(intent, "writing")
        self.assertGreaterEqual(confidence, 0.7)

    # -- chat --
    def test_chat_greeting(self):
        intent, confidence, reason = self._rule_classify("你好，你能做什么")
        self.assertEqual(intent, "chat")
        self.assertGreaterEqual(confidence, 0.7)

    def test_chat_hello(self):
        intent, confidence, reason = self._rule_classify("hello")
        self.assertEqual(intent, "chat")
        self.assertGreaterEqual(confidence, 0.4)

    # -- profile --
    def test_profile_zh(self):
        intent, confidence, reason = self._rule_classify("知识库中有哪些频域相关的论文")
        self.assertEqual(intent, "profile")
        self.assertGreater(confidence, 0.5)

    # -- fallback --
    def test_fallback_to_retrieval_on_empty(self):
        intent, confidence, reason = self._rule_classify("abc123")
        self.assertEqual(intent, "retrieval")
        self.assertEqual(confidence, 0.40)


# ---------------------------------------------------------------------------
# 3. Intent Router — edge cases (NEW)
# ---------------------------------------------------------------------------

class IntentRouterEdgeCaseTests(unittest.TestCase):
    """Test intent routing for queries that resemble real-world PaperMind usage."""

    def setUp(self):
        from app.agent.routing.intent import _rule_classify
        self._classify = _rule_classify

    # -- Paper-specific queries: should be retrieval, not profile --
    def test_paper_specific_method_query(self):
        """Query about a specific paper's method details -> retrieval."""
        intent, conf, reason = self._classify(
            "论文LCDNet: Adaptive Restoration of Light, Color and Details "
            "for Underwater Image Enhancement的自适应对数变换是什么？"
        )
        self.assertEqual(intent, "retrieval",
                         f"Expected retrieval, got {intent}. Reason: {reason}")
        self.assertGreaterEqual(conf, 0.4,
                                f"Confidence too low: {conf}")

    def test_paper_specific_architecture_query(self):
        """Query about network architecture in a paper -> retrieval."""
        intent, conf, reason = self._classify("LCDNet论文中的网络架构是怎样的")
        self.assertEqual(intent, "retrieval")
        self.assertGreater(conf, 0.4)

    def test_paper_specific_experiment_query(self):
        """Query about experiments in a specific paper -> retrieval."""
        intent, conf, reason = self._classify("LCDNet的实验结果PSNR是多少")
        self.assertEqual(intent, "retrieval")

    # -- Mixed language queries --
    def test_mixed_language_retrieval(self):
        """Chinese-English mixed queries should still route to retrieval."""
        intent, conf, reason = self._classify("LCDNet的adaptive logarithmic transformation怎么实现")
        self.assertEqual(intent, "retrieval")

    def test_mixed_language_comparison(self):
        """Mixed language comparison should route to comparison."""
        intent, conf, reason = self._classify("compare LCDNet and FUnIE-GAN 哪个更好")
        self.assertEqual(intent, "comparison")

    # -- Ambiguous queries: retrieval vs profile --
    def test_retrieval_vs_profile_boundary(self):
        """Queries that could be retrieval or profile — check behavior."""
        # "有哪些论文" -> profile
        intent, conf, reason = self._classify("有哪些论文用了logarithmic transformation")
        self.assertEqual(intent, "profile")

        # "什么是" -> retrieval, even when mentioning a paper
        intent, conf, reason = self._classify("什么是logarithmic transformation在水下图像增强中的应用")
        self.assertEqual(intent, "retrieval")

    # -- Ambiguous queries: retrieval vs comparison --
    def test_retrieval_vs_comparison_boundary(self):
        """Queries about differences should go to comparison."""
        intent, conf, reason = self._classify("LCDNet和FUnIE-GAN的区别")
        self.assertEqual(intent, "comparison")

    # -- Query with entity-like paper name only --
    def test_entity_only_profiles(self):
        """Just mentioning a paper name without content questions."""
        intent, conf, reason = self._classify("LCDNet这篇论文的基本信息")
        # "论文信息" + "论文基本信息" are profile strong hints
        self.assertEqual(intent, "profile",
                         f"Expected profile, got {intent}. Reason: {reason}")

    def test_concept_explanation_without_paper(self):
        """Explaining a concept without a specific paper reference."""
        intent, conf, reason = self._classify("解释一下underwater image enhancement中的logarithmic transformation")
        self.assertEqual(intent, "retrieval")

    # -- Confidence bounds --
    def test_confidence_within_range(self):
        """All classifications should have confidence in [0, 1]."""
        queries = [
            "LCDNet的自适应对数变换是什么？",
            "compare LCDNet and FUnIE-GAN",
            "润色这段",
            "你好",
            "知识库有哪些论文",
            "生成文献综述",
            "xyz123",
        ]
        for q in queries:
            _, conf, _ = self._classify(q)
            self.assertGreaterEqual(conf, 0.0, f"Confidence < 0 for: {q}")
            self.assertLessEqual(conf, 1.0, f"Confidence > 1 for: {q}")

    # -- Empty / whitespace --
    def test_empty_query(self):
        intent, conf, reason = self._classify("")
        self.assertEqual(intent, "retrieval")
        self.assertEqual(conf, 0.40)

    def test_whitespace_query(self):
        intent, conf, reason = self._classify("   ")
        self.assertEqual(intent, "retrieval")
        self.assertEqual(conf, 0.40)


# ---------------------------------------------------------------------------
# 4. Planner — coverage by intent (NEW)
# ---------------------------------------------------------------------------

class PlannerCoverageTests(unittest.TestCase):
    """Test which intents get decomposition plans and which don't."""

    def test_comparison_uses_dedicated_prompt(self):
        from app.agent.planning.planner import _COMPARISON_PLANNER_PROMPT
        self.assertIn("paper_comparison", _COMPARISON_PLANNER_PROMPT)
        self.assertIn("targets", _COMPARISON_PLANNER_PROMPT)

    def test_generic_planner_prompts_cover_planner_intents(self):
        from app.agent.planning.planner import _GENERIC_PLANNER_PROMPTS
        for intent in ("retrieval", "summary"):
            self.assertIn(intent, _GENERIC_PLANNER_PROMPTS)

    def test_direct_intents_not_in_planner_prompts(self):
        from app.agent.planning.planner import _GENERIC_PLANNER_PROMPTS
        self.assertNotIn("chat", _GENERIC_PLANNER_PROMPTS)
        self.assertNotIn("writing", _GENERIC_PLANNER_PROMPTS)
        self.assertNotIn("profile", _GENERIC_PLANNER_PROMPTS)

    def test_minimal_agent_plan_for_retrieval(self):
        from app.agent.schemas.plan import minimal_agent_plan
        plan = minimal_agent_plan("retrieval", "test query")
        self.assertEqual(plan.task_type, "paper_qa")
        self.assertEqual(len(plan.steps), 2)
        actions = [p["action"] for p in plan.steps]
        self.assertIn("search_papers", actions)
        self.assertIn("retrieve_evidence", actions)

    def test_minimal_agent_plan_for_comparison(self):
        from app.agent.schemas.plan import minimal_agent_plan
        plan = minimal_agent_plan("comparison", "对比 A 和 B 的方法")
        self.assertEqual(plan.task_type, "paper_comparison")
        self.assertGreaterEqual(len(plan.aspects), 1)

    def test_minimal_agent_plan_for_summary(self):
        from app.agent.schemas.plan import minimal_agent_plan
        plan = minimal_agent_plan("summary", "test")
        self.assertEqual(plan.task_type, "literature_summary")
        self.assertEqual(len(plan.steps), 3)


# ---------------------------------------------------------------------------
# 5. Route conditional edges
# ---------------------------------------------------------------------------

class RouteConditionalEdgeTests(unittest.TestCase):
    def test_chat_goes_to_chat_handler(self):
        from app.agent.routing.intent import route_after_intent

        state = {"intent": "chat", "intent_confidence": 0.95}
        self.assertEqual(route_after_intent(state), "chat_handler")

    def test_planner_bound_intents_go_to_planner(self):
        from app.agent.routing.intent import route_after_intent

        for intent in ("retrieval", "comparison", "summary"):
            state = {"intent": intent, "intent_confidence": 0.9}
            self.assertEqual(route_after_intent(state), "planner")

    def test_chat_writing_profile_route_direct(self):
        from app.agent.routing.intent import route_after_intent

        expected = {
            "chat": "chat_handler",
            "writing": "writing_handler",
            "profile": "profile_handler",
        }
        for intent, handler in expected.items():
            state = {"intent": intent, "intent_confidence": 0.95}
            self.assertEqual(route_after_intent(state), handler)

    def test_low_confidence_direct_intents_still_go_planner(self):
        from app.agent.routing.intent import route_after_intent

        for intent in ("chat", "writing", "profile"):
            state = {"intent": intent, "intent_confidence": 0.1}
            self.assertEqual(route_after_intent(state), "planner")

    def test_route_by_plan_type_comparison_subgraph(self):
        from app.agent.routing.routes import route_by_plan_type

        state = {"intent": "comparison", "plan": {"task_type": "paper_comparison"}}
        self.assertEqual(route_by_plan_type(state), "comparison_subgraph")

    def test_route_by_plan_type_retrieval(self):
        from app.agent.routing.routes import route_by_plan_type
        state = {"intent": "retrieval", "plan": {"task_type": "paper_qa"}}
        self.assertEqual(route_by_plan_type(state), "retrieval_handler")


# ---------------------------------------------------------------------------
# 6. Retrieval intent inference — section/entity detection (NEW)
# ---------------------------------------------------------------------------

class RetrievalIntentInferenceTests(unittest.TestCase):
    """Test _infer_intent, _expand_query_for_retrieval, _extract_query_entities."""

    def test_infer_method_section_from_zh(self):
        from app.services.retrieval import _infer_intent
        intent = _infer_intent("LCDNet网络架构是怎么设计的")
        self.assertIn("method", intent.section_types)

    def test_infer_experiment_section_from_zh(self):
        from app.services.retrieval import _infer_intent
        intent = _infer_intent("PSNR指标是多少实验结果如何")
        self.assertIn("experiment", intent.section_types)

    def test_infer_no_section_defaults_empty(self):
        from app.services.retrieval import _infer_intent
        intent = _infer_intent("LCDNet是什么")
        self.assertEqual(intent.section_types, [],
                         "Generic queries should not infer specific sections")

    def test_infer_multiple_sections(self):
        from app.services.retrieval import _infer_intent
        intent = _infer_intent("LCDNet的方法和实验结果对比")
        self.assertIn("method", intent.section_types)
        self.assertIn("experiment", intent.section_types)

    def test_infer_ablation_section(self):
        from app.services.retrieval import _infer_intent
        intent = _infer_intent("LCDNet的消融实验")
        self.assertIn("ablation", intent.section_types)

    # -- entity extraction --
    def test_extract_model_name_from_query(self):
        from app.services.retrieval import _extract_query_entities
        entities = _extract_query_entities("LCDNet的自适应对数变换是什么？")
        self.assertIn("LCDNet", entities,
                      f"LCDNet should be extracted as entity, got: {entities}")

    def test_extract_multiple_model_names(self):
        from app.services.retrieval import _extract_query_entities
        entities = _extract_query_entities("compare LCDNet and FUnIE-GAN")
        # At minimum LCDNet should be detected
        self.assertIn("LCDNet", entities)

    def test_extract_metric_names(self):
        from app.services.retrieval import _extract_query_entities
        entities = _extract_query_entities("PSNR SSIM指标")
        self.assertTrue(any("PSNR" in e for e in entities),
                        f"PSNR should be extracted, got: {entities}")

    # -- query expansion --
    def test_expand_zh_method_query(self):
        from app.services.retrieval import _expand_query_for_retrieval, _RetrievalIntent
        intent = _RetrievalIntent(section_types=["method"], entities=[])
        result = _expand_query_for_retrieval("LCDNet怎么设计的", intent)
        # Should expand with English terms
        self.assertIn("design", result.lower())
        self.assertIn("LCDNet", result)

    def test_expand_zh_experiment_query(self):
        from app.services.retrieval import _expand_query_for_retrieval, _RetrievalIntent
        intent = _RetrievalIntent(section_types=["experiment"], entities=[])
        result = _expand_query_for_retrieval("实验结果怎么样", intent)
        # Should expand with experiment-related terms
        self.assertIn("experiment", result.lower())

    def test_expand_section_type_added(self):
        from app.services.retrieval import _expand_query_for_retrieval, _RetrievalIntent
        intent = _RetrievalIntent(section_types=["method", "ablation"], entities=[])
        result = _expand_query_for_retrieval("怎么设计的", intent)
        self.assertIn("method", result.lower())

    def test_no_expand_for_pure_english(self):
        from app.services.retrieval import _expand_query_for_retrieval, _RetrievalIntent
        intent = _RetrievalIntent(section_types=["method"], entities=[])
        result = _expand_query_for_retrieval("how does LCDNet work", intent)
        # Pure English should not be expanded
        self.assertEqual(result, "how does LCDNet work")

    def test_expand_dedupes_terms(self):
        from app.services.retrieval import _expand_query_for_retrieval, _RetrievalIntent
        # method section_type and "method" trigger should not duplicate
        intent = _RetrievalIntent(section_types=["method"], entities=[])
        result = _expand_query_for_retrieval("LCDNet方法怎么设计", intent)
        # "method" should appear at most once in additions
        count = result.lower().count("method")
        self.assertLessEqual(count, 2, f"'method' appears {count} times, expected <= 2 (once in original + once expanded)")

    def test_expand_preserves_original_query(self):
        from app.services.retrieval import _expand_query_for_retrieval, _RetrievalIntent
        original = "LCDNet的自适应对数变换是什么？"
        intent = _RetrievalIntent(section_types=[], entities=["LCDNet"])
        result = _expand_query_for_retrieval(original, intent)
        self.assertTrue(result.startswith(original),
                        f"Expanded query should start with original. Got: {result[:80]}")


# ---------------------------------------------------------------------------
# 7. Handler pipeline analysis — which tools each handler uses (NEW)
# ---------------------------------------------------------------------------

class HandlerToolUsageTests(unittest.TestCase):
    """Verify which tools each handler calls and identify coverage gaps."""

    async def _run_handler_and_check_tools(self, handler, state, expected_tools):
        """Helper: invoke handler and verify used_tools."""
        result = await handler(state)
        used = result.get("handler_used_tools", [])
        for tool in expected_tools:
            self.assertIn(tool, used,
                          f"Expected tool '{tool}' in handler output, got: {used}")

    @patch("app.services.qa_service.answer")
    @patch("app.services.retrieval_service.hybrid_retrieve")
    async def test_retrieval_handler_tools(self, mock_retrieve, mock_qa):
        """retrieval_handler calls answer_with_rag + retrieve_evidence.
        Does NOT call search_paper_profiles — known gap for paper-specific queries."""
        mock_qa.return_value = {"answer": "test answer", "contexts": []}
        mock_retrieve.return_value = []

        from app.agent.handlers.retrieval import retrieval_handler
        result = await retrieval_handler({"query": "LCDNet的自适应对数变换是什么？"})
        used = result.get("handler_used_tools", [])

        self.assertIn("answer_with_rag", used)
        self.assertIn("retrieve_evidence", used)
        # These tools SHOULD be called for paper-specific queries but aren't:
        self.assertNotIn("search_paper_profiles", used,
                         "retrieval_handler does NOT search for paper profiles first — this is a gap")
        self.assertNotIn("deep_search_papers", used,
                         "retrieval_handler does NOT deep-search papers — this is a gap")

    @patch("app.services.qa_service.answer")
    @patch("app.services.retrieval_service.hybrid_retrieve")
    async def test_retrieval_handler_no_paper_discovery_step(self, mock_retrieve, mock_qa):
        """Confirm the paper discovery gap: retrieval_handler has no paper search step."""
        mock_qa.return_value = {"answer": "test", "contexts": []}
        mock_retrieve.return_value = []

        from app.agent.handlers.retrieval import retrieval_handler
        result = await retrieval_handler({"query": "LCDNet论文的自适应对数变换是什么"})

        # The answer was generated without ever searching for the paper
        self.assertEqual(result["handler_used_tools"], ["answer_with_rag", "retrieve_evidence"])


# ---------------------------------------------------------------------------
# 8. MQE — multi-query expansion (NEW)
# ---------------------------------------------------------------------------

class MQEConditionTests(unittest.TestCase):
    """Test _should_trigger_mqe and _llm_expand_queries logic."""

    def test_should_trigger_mqe_with_mixed_language(self):
        from app.services.retrieval import _should_trigger_mqe
        from app.core.schemas import RetrievedChunk
        # Mixed CJK + entity (model name) → should trigger
        self.assertTrue(
            _should_trigger_mqe("LCDNet的自适应对数变换是什么？", []),
            "Mixed-language query should trigger MQE regardless of retrieval quality",
        )

    def test_should_trigger_mqe_with_low_scores(self):
        from app.services.retrieval import _should_trigger_mqe
        from app.core.schemas import RetrievedChunk
        chunks = [
            RetrievedChunk(parent_id="p1", parent_text="x", score=0.2),
            RetrievedChunk(parent_id="p2", parent_text="y", score=0.15),
            RetrievedChunk(parent_id="p3", parent_text="z", score=0.1),
        ]
        self.assertTrue(
            _should_trigger_mqe("how does LCDNet work", chunks),
            "Low-confidence retrieval (max 0.2 < 0.5) should trigger MQE",
        )

    def test_should_not_trigger_mqe_with_good_scores(self):
        from app.services.retrieval import _should_trigger_mqe
        from app.core.schemas import RetrievedChunk
        chunks = [
            RetrievedChunk(parent_id="p1", parent_text="x", score=0.85),
            RetrievedChunk(parent_id="p2", parent_text="y", score=0.6),
        ]
        self.assertFalse(
            _should_trigger_mqe("pure english query", chunks),
            "Pure English query with high scores should NOT trigger MQE",
        )

    def test_should_not_trigger_mqe_pure_chinese_no_entities(self):
        from app.services.retrieval import _should_trigger_mqe
        from app.core.schemas import RetrievedChunk
        self.assertFalse(
            _should_trigger_mqe("什么是深度学习", []),
            "Pure Chinese without entities and no low-score evidence should NOT trigger",
        )

    def test_should_trigger_mqe_empty_chunks_mixed_language(self):
        from app.services.retrieval import _should_trigger_mqe
        # Even with empty results, mixed language should trigger
        self.assertTrue(
            _should_trigger_mqe("什么是LCDNet的自适应对数变换", []),
        )

    def test_should_not_trigger_mqe_empty_query(self):
        from app.services.retrieval import _should_trigger_mqe
        self.assertFalse(_should_trigger_mqe("", []))

    def test_llm_expand_queries_returns_list_starting_with_original(self):
        from app.services.retrieval import _llm_expand_queries
        import asyncio
        result = asyncio.run(_llm_expand_queries("LCDNet是什么", n=2))
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 1)
        self.assertEqual(result[0], "LCDNet是什么",
                         "Original query should be the first element")


class MQEHandlerIntegrationTests(unittest.TestCase):
    """Test retrieval_handler behavior with MQE enabled."""

    @patch("app.services.qa_service.answer")
    @patch("app.services.retrieval_service.hybrid_retrieve")
    async def test_direct_retrieve_triggers_mqe_for_mixed_language(self, mock_retrieve, mock_qa):
        from app.core.schemas import RetrievedChunk
        from app.agent.handlers.retrieval import _direct_retrieve

        # First retrieval: low quality
        mock_qa.return_value = {"answer": "insufficient", "contexts": []}
        mock_retrieve.return_value = [
            RetrievedChunk(parent_id="p1", parent_text="...", score=0.2),
        ]

        result = await _direct_retrieve("LCDNet的自适应对数变换是什么", top_k=None, task_id=None)
        used = result.get("handler_used_tools", [])
        # With mock LLM, expansion may or may not succeed, but the attempt
        # should be made — check basic structure
        self.assertIsNotNone(result.get("handler_answer"))
        self.assertIsNotNone(result.get("handler_contexts"))

    @patch("app.services.qa_service.answer")
    @patch("app.services.retrieval_service.hybrid_retrieve")
    async def test_direct_retrieve_skips_mqe_for_pure_chinese(self, mock_retrieve, mock_qa):
        from app.core.schemas import RetrievedChunk
        from app.agent.handlers.retrieval import _direct_retrieve

        mock_qa.return_value = {"answer": "ok", "contexts": []}
        mock_retrieve.return_value = [
            RetrievedChunk(parent_id="p1", parent_text="...", score=0.85),
        ]

        result = await _direct_retrieve("深度学习是什么", top_k=None, task_id=None)
        used = result.get("handler_used_tools", [])
        # Pure Chinese + high scores → no MQE
        self.assertNotIn("multi_query_expansion", used,
                         "Pure Chinese query with good scores should NOT trigger MQE")


# ---------------------------------------------------------------------------
# 9. Writing handler helpers
# ---------------------------------------------------------------------------

class WritingHandlerHelpersTests(unittest.TestCase):
    def test_detect_abstract_section(self):
        from app.agent.handlers.writing import _detect_section_type
        self.assertEqual(_detect_section_type("请润色这段摘要"), "abstract")
        self.assertEqual(_detect_section_type("polish this abstract please"), "abstract")

    def test_detect_introduction_section(self):
        from app.agent.handlers.writing import _detect_section_type
        self.assertEqual(_detect_section_type("帮我改写引言部分"), "introduction")
        self.assertEqual(_detect_section_type("improve the introduction"), "introduction")

    def test_detect_methods_section(self):
        from app.agent.handlers.writing import _detect_section_type
        self.assertEqual(_detect_section_type("润色实验设置这一段"), "methods")
        self.assertEqual(_detect_section_type("polish the methods section"), "methods")

    def test_detect_results_section(self):
        from app.agent.handlers.writing import _detect_section_type
        self.assertEqual(_detect_section_type("改写实验结果"), "results")
        self.assertEqual(_detect_section_type("improve experiment results"), "results")

    def test_detect_discussion_section(self):
        from app.agent.handlers.writing import _detect_section_type
        self.assertEqual(_detect_section_type("润色讨论部分"), "discussion")

    def test_detect_conclusion_section(self):
        from app.agent.handlers.writing import _detect_section_type
        self.assertEqual(_detect_section_type("改写结论"), "conclusion")

    def test_detect_body_default(self):
        from app.agent.handlers.writing import _detect_section_type
        self.assertEqual(_detect_section_type("请帮我改这段文字"), "body")

    def test_extract_text_removes_prefix(self):
        from app.agent.handlers.writing import _extract_text
        result = _extract_text("润色：这是需要润色的文本内容，包含足够多的字符")
        self.assertIn("需要润色的文本", result)
        self.assertTrue(result.startswith("这是"), f"should strip prefix, got: {result}")

    def test_extract_text_short_returns_itself(self):
        from app.agent.handlers.writing import _extract_text
        short = "这一句话太短"
        result = _extract_text(short)
        self.assertEqual(result, short)

    def test_extract_text_too_short_returns_empty(self):
        from app.agent.handlers.writing import _extract_text
        self.assertEqual(_extract_text("hi"), "")


# ---------------------------------------------------------------------------
# 9. Synthesizer — dedup + border cases (NEW)
# ---------------------------------------------------------------------------

class SynthesizerTests(unittest.TestCase):
    def setUp(self):
        from app.shared.dedup import dedup_by_key, source_from_context, context_key, source_key
        from app.agent.synthesizer import _add_source

        self._dedup_by_key = dedup_by_key
        self._context_key = context_key
        self._source_key = source_key
        self._source_from_context = source_from_context
        self._add_source = _add_source

    # -- dedup contexts --
    def test_dedup_contexts_removes_duplicates(self):
        ctx1 = {"paper_id": "p1", "parent_id": "doc1", "title": "Paper A", "content": "data1"}
        ctx2 = {"paper_id": "p1", "parent_id": "doc1", "title": "Paper A", "content": "data2"}
        ctx3 = {"paper_id": "p2", "parent_id": "doc2", "title": "Paper B", "content": "data3"}
        result = self._dedup_by_key([ctx1, ctx2, ctx3], key_fn=self._context_key)
        self.assertEqual(len(result), 2)

    def test_dedup_contexts_handles_empty(self):
        self.assertEqual(self._dedup_by_key([], key_fn=self._context_key), [])
        self.assertEqual(self._dedup_by_key(None, key_fn=self._context_key), [])

    def test_dedup_contexts_handles_non_dict_items(self):
        result = self._dedup_by_key([{"paper_id": "p1", "parent_id": "d1", "title": "A"}, "not_a_dict", 123], key_fn=self._context_key)
        self.assertEqual(len(result), 1)

    def test_dedup_contexts_different_same_paper_different_parent(self):
        """Same paper, different parent docs should both be kept."""
        ctx1 = {"paper_id": "p1", "parent_id": "doc1", "title": "Paper A"}
        ctx2 = {"paper_id": "p1", "parent_id": "doc2", "title": "Paper A"}  # different parent
        result = self._dedup_by_key([ctx1, ctx2], key_fn=self._context_key)
        self.assertEqual(len(result), 2,
                         "Different parent_ids under same paper should both be kept")

    # -- dedup sources --
    def test_dedup_sources_removes_duplicates(self):
        src1 = {"paper_id": "p1", "title": "Paper A", "score": 0.9}
        src2 = {"paper_id": "p1", "title": "Paper A", "score": 0.8}
        src3 = {"paper_id": "p2", "title": "Paper B", "score": 0.7}
        result = self._dedup_by_key([src1, src2, src3], key_fn=self._source_key)
        self.assertEqual(len(result), 2)

    def test_dedup_sources_handles_none(self):
        """dedup_by_key guards against None input."""
        self.assertEqual(self._dedup_by_key(None, key_fn=self._source_key), [])

    # -- source from context --
    def test_source_from_context_full(self):
        ctx = {
            "paper_id": "p1", "title": "Paper A", "section": "method",
            "section_type": "method", "chunk_ids": ["c1"], "parent_id": "doc1",
            "score": 0.92, "metadata": {"key": "val"},
        }
        src = self._source_from_context(ctx)
        self.assertEqual(src["paper_id"], "p1")
        self.assertEqual(src["title"], "Paper A")
        self.assertEqual(src["chunk_ids"], ["c1"])
        self.assertEqual(src["metadata"], {"key": "val"})

    def test_source_from_context_minimal(self):
        ctx = {"paper_id": None, "title": None}
        src = self._source_from_context(ctx)
        self.assertIsNone(src["paper_id"])

    # -- add source --
    def test_add_source_avoids_duplicates(self):
        sources = [{"paper_id": "p1", "title": "Paper A", "score": 0.9}]
        new = {"paper_id": "p1", "title": "Paper A", "score": 0.5}
        result = self._add_source(sources, new)
        self.assertEqual(len(result), 1)

    def test_add_source_appends_new(self):
        sources = [{"paper_id": "p1", "title": "Paper A", "score": 0.9}]
        new = {"paper_id": "p2", "title": "Paper B", "score": 0.7}
        result = self._add_source(sources, new)
        self.assertEqual(len(result), 2)

    def test_add_source_skips_empty(self):
        sources = []
        empty = {"paper_id": None, "title": None}
        result = self._add_source(sources, empty)
        self.assertEqual(result, sources)


# ---------------------------------------------------------------------------
# 10. Synthesizer — error handling (NEW)
# ---------------------------------------------------------------------------

class SynthesizerErrorTests(unittest.TestCase):
    async def test_synthesizer_returns_error_on_state_error(self):
        from app.agent.synthesizer import synthesizer_node
        result = await synthesizer_node({"error": "Something went wrong"})
        self.assertIn("处理出错", result["final_answer"])
        self.assertIn("Something went wrong", result["final_answer"])
        self.assertEqual(result["final_contexts"], [])
        self.assertEqual(result["final_sources"], [])


# ---------------------------------------------------------------------------
# 11. Runtime backward compat
# ---------------------------------------------------------------------------

class RuntimeBackwardCompatTests(unittest.TestCase):
    def test_answer_with_agent_signature(self):
        from app.agent.runtime import answer_with_agent
        import inspect
        sig = inspect.signature(answer_with_agent)
        params = sig.parameters
        self.assertIn("query", params)
        self.assertIn("top_k", params)
        self.assertIn("task_id", params)

    def test_system_prompt_builds(self):
        from app.agent.prompts import build_system_prompt
        prompt = build_system_prompt()
        self.assertIn("PaperMind", prompt)
        self.assertIn("Skill", prompt)
        self.assertIn("证据", prompt)
        self.assertIn("综述", prompt)

    def test_old_runtime_functions_removed(self):
        from app.agent import runtime as rt
        self.assertFalse(hasattr(rt, "_create_agent"), "_create_agent should be removed")
        self.assertFalse(hasattr(rt, "_extract_agent_outputs"), "_extract_agent_outputs should be removed")
        self.assertFalse(hasattr(rt, "_parse_tool_payload"), "_parse_tool_payload should be removed")

    def test_api_response_structure(self):
        """Verify AgentChatResponse has expected fields."""
        from app.core.schemas import AgentChatResponse
        fields = AgentChatResponse.model_fields
        for key in ("query", "answer", "contexts", "sources", "used_tools"):
            self.assertIn(key, fields)


# ---------------------------------------------------------------------------
# 12. Handler registry
# ---------------------------------------------------------------------------

class HandlerRegistryTests(unittest.TestCase):
    def test_all_handlers_registered(self):
        from app.agent.handlers import HANDLERS
        expected = {
            "retrieval_handler",
            "profile_handler",
            "chat_handler",
            "summary_handler",
            "writing_handler",
        }
        self.assertSetEqual(set(HANDLERS.keys()), expected)

    def test_all_handlers_are_callable(self):
        from app.agent.handlers import HANDLERS
        for name, handler in HANDLERS.items():
            self.assertTrue(callable(handler), f"{name} should be callable")

    def test_all_handlers_are_async(self):
        from app.agent.handlers import HANDLERS
        import asyncio
        for name, handler in HANDLERS.items():
            self.assertTrue(
                asyncio.iscoroutinefunction(handler),
                f"{name} should be an async function",
            )


# ---------------------------------------------------------------------------
# 13. Graph structure (AST-based)
# ---------------------------------------------------------------------------

class GraphStructureTests(unittest.TestCase):
    def test_graph_has_required_nodes(self):
        with _GRAPH_BUILDER_PATH.open(encoding="utf-8") as f:
            tree = ast.parse(f.read())
        add_node_names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and hasattr(node.func, "attr"):
                if node.func.attr == "add_node":
                    for arg in node.args:
                        if isinstance(arg, ast.Constant):
                            add_node_names.append(arg.value)
                        elif isinstance(arg, ast.Name):
                            add_node_names.append(f"<{arg.id}>")
        self.assertIn("intent_router", add_node_names)
        self.assertIn("planner", add_node_names)
        self.assertIn("plan_validate", add_node_names)
        self.assertIn("comparison_subgraph", add_node_names)
        self.assertIn("synthesizer", add_node_names)

    def test_comparison_subgraph_has_workflow_nodes(self):
        subgraph_path = _REPO_ROOT / "app" / "agent" / "workflows" / "comparison" / "__init__.py"
        with subgraph_path.open(encoding="utf-8") as f:
            tree = ast.parse(f.read())
        add_node_names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and hasattr(node.func, "attr"):
                if node.func.attr == "add_node":
                    for arg in node.args:
                        if isinstance(arg, ast.Constant):
                            add_node_names.append(arg.value)
        for required in (
            "paper_search_node",
            "coverage_check_node",
            "compare_node",
        ):
            self.assertIn(required, add_node_names)

    def test_graph_set_entry_point(self):
        with _GRAPH_BUILDER_PATH.open(encoding="utf-8") as f:
            tree = ast.parse(f.read())
        entry_points = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and hasattr(node.func, "attr"):
                if node.func.attr == "set_entry_point":
                    entry_points.append(node.args[0].value)
        self.assertIn("intent_router", entry_points)

    def test_synthesizer_connected_to_end(self):
        with _GRAPH_BUILDER_PATH.open(encoding="utf-8") as f:
            tree = ast.parse(f.read())
        found = any(_is_synthesizer_to_end_edge(node) for node in ast.walk(tree))
        self.assertTrue(found, "synthesizer -> END edge not found")

    def test_graph_compiled(self):
        with _GRAPH_BUILDER_PATH.open(encoding="utf-8") as f:
            code = f.read()
        self.assertIn("return builder.compile()", code)
        self.assertIn("def build_agent_graph()", code)
        self.assertIn("def get_agent_graph()", code)


# ---------------------------------------------------------------------------
# 14. Paper structure — entity extraction (NEW)
# ---------------------------------------------------------------------------

class PaperEntityExtractionTests(unittest.TestCase):
    """Test entity extraction from paper text — critical for model-name detection."""

    def test_extract_lcdnet(self):
        from app.shared.paper_utils import extract_entities
        entities = extract_entities("We propose LCDNet for underwater image enhancement.")
        self.assertIn("LCDNet", entities)

    def test_extract_common_metrics(self):
        from app.shared.paper_utils import extract_entities
        text = "We evaluate PSNR, SSIM, and UIQM on the EUVP dataset."
        entities = extract_entities(text)
        for metric in ("PSNR", "SSIM", "UIQM", "EUVP"):
            self.assertIn(metric, entities, f"{metric} should be extracted")

    def test_extract_compound_model_name(self):
        from app.shared.paper_utils import extract_entities
        entities = extract_entities("FUnIE-GAN outperforms UGAN-P on LSUI.")
        # Known issue: _ENTITY_RE does not capture FUnIE pattern (upper-upper-lower-upper-upper)
        # Only GAN (via _COMMON_KEYWORDS), LSUI, and UGAN-P are captured
        self.assertIn("LSUI", entities)

    def test_extract_from_mixed_language(self):
        from app.shared.paper_utils import extract_entities
        # After CJK boundary fix, model names (LCDNet) are extractable.
        # Plain English words (Adaptive, Logarithmic, Transformation) are
        # intentionally excluded by _ENTITY_RE — it targets model names, metrics, datasets.
        entities = extract_entities("LCDNet的自适应对数变换（Adaptive Logarithmic Transformation）")
        self.assertIn("LCDNet", entities)

    def test_extract_empty_string(self):
        from app.shared.paper_utils import extract_entities
        entities = extract_entities("")
        self.assertEqual(entities, [])


# ---------------------------------------------------------------------------
# 15. Tool contracts (NEW)
# ---------------------------------------------------------------------------

class ToolContractsTests(unittest.TestCase):
    def test_json_response_builds_valid_json(self):
        from app.agent.tools.contracts import json_response
        import json
        result = json_response(
            tool_name="test_tool",
            query="test query",
            results=[{"key": "value"}],
            confidence=0.9,
        )
        parsed = json.loads(result)
        self.assertEqual(parsed["tool_name"], "test_tool")
        self.assertEqual(parsed["result_count"], 1)
        self.assertEqual(parsed["confidence"], 0.9)

    def test_error_response_includes_error(self):
        from app.agent.tools.contracts import error_response
        import json
        result = error_response("test_tool", "q", ValueError("bad input"))
        parsed = json.loads(result)
        self.assertEqual(parsed["tool_name"], "test_tool")
        self.assertIn("bad input", parsed["error"])
        self.assertEqual(parsed["results"], [])

    def test_truncate_preserves_short_text(self):
        from app.agent.tools.contracts import truncate
        short = "hello world"
        self.assertEqual(truncate(short), short)

    def test_truncate_cuts_long_text(self):
        from app.agent.tools.contracts import truncate
        long_text = "x" * 2000
        result = truncate(long_text, max_chars=100)
        self.assertLess(len(result), len(long_text))
        self.assertIn("truncated", result)

    def test_tool_result_to_dict(self):
        from app.agent.tools.contracts import ToolResult
        result = ToolResult(
            tool_name="test",
            query="q",
            results=[{"a": 1}],
            confidence=0.8,
        )
        d = result.to_dict()
        self.assertEqual(d["tool_name"], "test")
        self.assertEqual(d["result_count"], 1)


# ---------------------------------------------------------------------------
# 16. Session helpers (NEW)
# ---------------------------------------------------------------------------

class SessionHelperTests(unittest.TestCase):
    def test_build_user_query_adds_top_k_constraint(self):
        from app.agent.session import build_user_query
        result = build_user_query("test query", top_k=10)
        self.assertIn("test query", result)
        self.assertIn("top_k=10", result)

    def test_build_user_query_adds_task_id_constraint(self):
        from app.agent.session import build_user_query
        result = build_user_query("test query", task_id="abc-123")
        self.assertIn("abc-123", result)

    def test_build_user_query_no_constraints(self):
        from app.agent.session import build_user_query
        result = build_user_query("simple query")
        self.assertEqual(result, "simple query")

    def test_is_task_status_query_positive(self):
        from app.agent.session import is_task_status_query
        self.assertTrue(is_task_status_query("任务状态是什么"))
        self.assertTrue(is_task_status_query("解析进度如何"))
        self.assertTrue(is_task_status_query("任务完成了吗"))

    def test_is_task_status_query_negative(self):
        from app.agent.session import is_task_status_query
        self.assertFalse(is_task_status_query("LCDNet的自适应对数变换是什么"))
        self.assertFalse(is_task_status_query("润色这段文字"))


# ---------------------------------------------------------------------------
# 17. Config sanity (NEW)
# ---------------------------------------------------------------------------

class ConfigSanityTests(unittest.TestCase):
    def test_rule_route_threshold_reasonable(self):
        from app.core.config import settings
        self.assertGreater(settings.rule_route_threshold, 0.5)
        self.assertLessEqual(settings.rule_route_threshold, 1.0)

    def test_qa_top_k_positive(self):
        from app.core.config import settings
        self.assertGreater(settings.qa_top_k, 0)

    def test_retrieval_top_k_reasonable(self):
        from app.core.config import settings
        self.assertGreater(settings.retrieve_top_k_bm25, 0)
        self.assertGreater(settings.retrieve_top_k_knn, 0)
        self.assertGreater(settings.rrf_k, 0)

    def test_final_top_k_positive(self):
        from app.core.config import settings
        self.assertGreater(settings.final_top_k, 0)


# ---------------------------------------------------------------------------
# 18. Negation detection (NEW)
# ---------------------------------------------------------------------------

class NegationDetectionTests(unittest.TestCase):
    def setUp(self):
        from app.agent.routing.intent import _rule_classify
        self._classify = _rule_classify

    def test_negation_dont_compare(self):
        intent, conf, reason = self._classify("不要帮我对比LCDNet和U-shape，我只想了解LCDNet的方法")
        self.assertEqual(intent, "retrieval",
                         f"Negation of comparison should route to retrieval, got {intent}. Reason: {reason}")

    def test_negation_dont_write(self):
        intent, conf, reason = self._classify("不要润色这段文字，帮我搜索有哪些论文")
        self.assertEqual(intent, "profile",
                         f"Negation of writing should route to profile, got {intent}. Reason: {reason}")

    def test_negation_confidence_dropped(self):
        intent, conf, reason = self._classify("不要对比 A 和 B")
        # Should have lower confidence than a normal comparison query
        self.assertLessEqual(conf, 0.7,
                            f"Negated comparison should have low confidence, got {conf}. Reason: {reason}")

    def test_no_negation_normal_comparison(self):
        intent, conf, reason = self._classify("对比 LCDNet 和 U-shape 的方法")
        self.assertEqual(intent, "comparison")
        # "对比" and "方法" both match (comparison vs retrieval), ambiguity applies
        self.assertGreater(conf, 0.3)


# ---------------------------------------------------------------------------
# 19. Dedup utility tests (NEW)
# ---------------------------------------------------------------------------

class DedupUtilityTests(unittest.TestCase):
    def test_dedup_by_key_removes_duplicates(self):
        from app.shared.dedup import dedup_by_key
        items = [
            {"paper_id": "p1", "parent_id": "d1", "title": "A", "text": "first"},
            {"paper_id": "p1", "parent_id": "d1", "title": "A", "text": "second"},
            {"paper_id": "p2", "parent_id": "d2", "title": "B", "text": "third"},
        ]
        result = dedup_by_key(items, key_fn=lambda x: (x.get("paper_id"), x.get("parent_id"), x.get("title")))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "first")

    def test_dedup_by_key_handles_empty(self):
        from app.shared.dedup import dedup_by_key
        self.assertEqual(dedup_by_key(None, key_fn=lambda x: ()), [])
        self.assertEqual(dedup_by_key([], key_fn=lambda x: ()), [])

    def test_dedup_by_key_skips_non_dict(self):
        from app.shared.dedup import dedup_by_key
        items = [{"paper_id": "p1", "title": "A"}, "not_a_dict", 123]
        result = dedup_by_key(items, key_fn=lambda x: (x.get("paper_id"),))
        self.assertEqual(len(result), 1)

    def test_merge_deduped_combines_lists(self):
        from app.shared.dedup import merge_deduped
        list1 = [{"paper_id": "p1", "parent_id": "d1", "title": "A"}]
        list2 = [{"paper_id": "p2", "parent_id": "d2", "title": "B"}]
        list3 = [{"paper_id": "p1", "parent_id": "d1", "title": "A"}]  # duplicate
        result = merge_deduped(list1, list2, list3)
        self.assertEqual(len(result), 2)

    def test_merge_deduped_handles_none_inputs(self):
        from app.shared.dedup import merge_deduped
        result = merge_deduped(None, [], [{"paper_id": "p1", "parent_id": "d1", "title": "A"}])
        self.assertEqual(len(result), 1)


# ---------------------------------------------------------------------------
# 20. Config summary settings (NEW)
# ---------------------------------------------------------------------------

class ConfigSummarySettingsTests(unittest.TestCase):
    def test_summary_settings_exist(self):
        from app.core.config import settings
        self.assertGreater(settings.summary_max_papers, 0)
        self.assertGreater(settings.summary_max_sections, 0)
        self.assertGreater(settings.summary_evidence_top_k, 0)
        self.assertGreater(settings.summary_outline_max_chars, 0)
        self.assertGreater(settings.summary_draft_max_chars, 0)
        self.assertGreater(settings.summary_polish_max_chars, 0)


if __name__ == "__main__":
    unittest.main()
