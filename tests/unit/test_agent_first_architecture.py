import json
import tempfile
import unittest
from pathlib import Path


class ToolContractTests(unittest.TestCase):
    def test_tool_result_json_has_stable_shape(self):
        from app.agent.tools.contracts import ToolResult, ToolSource

        result = ToolResult(
            tool_name="retrieve_evidence",
            query="frequency domain enhancement",
            results=[{"paper_id": "p1", "content": "evidence"}],
            sources=[ToolSource(paper_id="p1", title="Paper 1", score=0.9)],
            confidence=0.8,
        )

        payload = json.loads(result.to_json())

        self.assertEqual(payload["tool_name"], "retrieve_evidence")
        self.assertEqual(payload["query"], "frequency domain enhancement")
        self.assertEqual(payload["result_count"], 1)
        self.assertEqual(payload["results"][0]["paper_id"], "p1")
        self.assertEqual(payload["sources"][0]["title"], "Paper 1")
        self.assertIsNone(payload["error"])
        self.assertEqual(payload["confidence"], 0.8)

    def test_error_response_preserves_contract_shape(self):
        from app.agent.tools.contracts import error_response

        payload = json.loads(error_response("memory", "recall", RuntimeError("boom")))

        self.assertEqual(payload["tool_name"], "memory")
        self.assertEqual(payload["query"], "recall")
        self.assertEqual(payload["result_count"], 0)
        self.assertEqual(payload["results"], [])
        self.assertEqual(payload["sources"], [])
        self.assertEqual(payload["error"], "boom")


class MemoryStoreTests(unittest.TestCase):
    def test_memory_scopes_are_isolated_and_queryable(self):
        from app.services.memory_store import JsonMemoryStore

        with tempfile.TemporaryDirectory() as tmp:
            store = JsonMemoryStore(Path(tmp) / "memory.json")
            store.remember("session", "selected_papers", ["p1", "p2"])
            store.remember("user", "language", "中文")

            session_hits = store.recall("session", "paper")
            user_hits = store.recall("user", "paper")

        self.assertEqual(session_hits[0]["key"], "selected_papers")
        self.assertEqual(session_hits[0]["value"], ["p1", "p2"])
        self.assertEqual(user_hits, [])

    def test_workspace_state_updates_do_not_remove_other_keys(self):
        from app.services.memory_store import JsonMemoryStore

        with tempfile.TemporaryDirectory() as tmp:
            store = JsonMemoryStore(Path(tmp) / "memory.json")
            store.update_workspace_state("topic", "underwater enhancement")
            store.update_workspace_state("stage", "outline")

            state = store.get_workspace_state()

        self.assertEqual(state["topic"], "underwater enhancement")
        self.assertEqual(state["stage"], "outline")


class ToolRegistryTests(unittest.TestCase):
    def test_registry_profiles_are_lazy_and_named(self):
        from app.agent.registry import available_profiles, get_tool_factories

        profiles = available_profiles()
        factories = get_tool_factories("research")

        self.assertIn("basic", profiles)
        self.assertIn("research", profiles)
        self.assertIn("workflow", profiles)
        self.assertIn("retrieve_evidence", [factory.name for factory in factories])
        self.assertIn("recall_memory", [factory.name for factory in factories])


class AgentEntrypointTests(unittest.TestCase):
    def test_agent_prompt_includes_evidence_memory_and_workflow_rules(self):
        from app.agent.prompts import build_system_prompt

        prompt = build_system_prompt()

        self.assertIn("工具获取论文知识库中的证据", prompt)
        self.assertIn("记忆", prompt)
        self.assertIn("综述", prompt)

    def test_user_query_builder_preserves_top_k_and_task_scope(self):
        from app.agent.session import build_user_query

        query = build_user_query("总结这篇论文", top_k=3, task_id="task-1")

        self.assertIn("总结这篇论文", query)
        self.assertIn("top_k=3", query)
        self.assertIn("task_id=task-1", query)


if __name__ == "__main__":
    unittest.main()
