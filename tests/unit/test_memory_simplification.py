"""Tests for the simplified Redis + Episodic archive + Semantic memory design."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from app.services.memory.config import MemoryConfig
from app.services.memory.models import MemoryItem, MemoryType


class _FakeIndices:
    def __init__(self) -> None:
        self.created: dict[str, dict] = {}

    def exists(self, index: str) -> bool:
        return index in self.created

    def create(self, index: str, body: dict) -> None:
        self.created[index] = body


class _FakeES:
    def __init__(self) -> None:
        self.indices = _FakeIndices()
        self.indexed: list[dict] = []

    def index(self, *, index: str, id: str, body: dict, refresh: bool = False) -> None:
        self.indexed.append({"index": index, "id": id, "body": body, "refresh": refresh})

    def count(self, index: str) -> dict:
        return {"count": len(self.indexed)}


class _FakeEmbeddings:
    dim = 3

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class MemoryManagerContractTests(unittest.TestCase):
    def test_recall_uses_episodic_archive_contract_without_working_layer(self):
        from app.services.memory.manager import MemoryManager

        class FakeSemantic:
            async def recall(self, query: str, **kwargs):
                return [
                    MemoryItem(
                        memory_type=MemoryType.SEMANTIC,
                        key="preferred_language",
                        content="用户偏好中文回答",
                        scope="user",
                    )
                ]

        class FakeEpisodic:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def recall(self, query: str = "", session_id: str | None = None, limit: int = 10):
                self.calls.append({"query": query, "session_id": session_id, "limit": limit})
                return [{"role": "user", "content": "历史问题", "session_id": session_id}]

        episodic = FakeEpisodic()
        manager = MemoryManager(
            semantic=FakeSemantic(),
            episodic=episodic,
            session_store=None,
            config=MemoryConfig(),
        )

        result = asyncio.run(
            manager.recall(
                "中文回答",
                scope="user",
                include_semantic=True,
                include_episodic=True,
                session_id="sess-1",
            )
        )

        self.assertNotIn("working", result)
        self.assertEqual(episodic.calls, [{"query": "中文回答", "session_id": "sess-1", "limit": 10}])
        self.assertEqual(result["semantic"][0]["key"], "preferred_language")
        self.assertEqual(result["episodic"][0]["content"], "历史问题")


class SemanticLongTermMemoryTests(unittest.TestCase):
    def test_upsert_long_term_memory_indexes_user_project_and_source_fields(self):
        from app.services.memory.semantic import SemanticMemory

        fake_es = _FakeES()
        semantic = SemanticMemory(
            config=MemoryConfig(semantic_embedding_dims=3),
            es_client=fake_es,
            embedding_model=_FakeEmbeddings(),
        )

        item = asyncio.run(
            semantic.upsert_long_term_memory(
                key="preferred_language",
                content="用户偏好使用中文回答。",
                memory_kind="preference",
                user_id="user-1",
                project_id="project-1",
                source_session_ids=["sess-1"],
                confidence=0.9,
                importance=0.8,
            )
        )

        body = fake_es.indexed[-1]["body"]
        self.assertEqual(item.memory_type, MemoryType.SEMANTIC)
        self.assertEqual(body["memory_kind"], "preference")
        self.assertEqual(body["user_id"], "user-1")
        self.assertEqual(body["project_id"], "project-1")
        self.assertEqual(body["source_session_ids"], ["sess-1"])
        self.assertEqual(body["confidence"], 0.9)
        self.assertIn("last_evidence_at", body)


class MemoryConsolidatorTests(unittest.TestCase):
    def test_consolidate_session_extracts_candidates_and_marks_archive_processed(self):
        from app.services.memory.consolidation import MemoryConsolidator

        class FakeEpisodic:
            def __init__(self) -> None:
                self.marked: list[tuple[str, int]] = []

            def get_session_doc(self, session_id: str):
                return {
                    "session_id": session_id,
                    "user_id": "user-1",
                    "project_id": "project-1",
                    "turn_count": 2,
                    "turns": [
                        {"role": "user", "content": "以后请用中文回答"},
                        {"role": "assistant", "content": "好的，我会用中文回答。"},
                    ],
                }

            def mark_consolidated(self, session_id: str, turn_count: int) -> None:
                self.marked.append((session_id, turn_count))

        class FakeSemantic:
            def __init__(self) -> None:
                self.upserts: list[dict] = []

            async def upsert_long_term_memory(self, **kwargs):
                self.upserts.append(kwargs)
                return MemoryItem(memory_type=MemoryType.SEMANTIC, key=kwargs["key"], content=kwargs["content"])

        async def extractor(session_doc: dict) -> list[dict]:
            return [
                {
                    "key": "preferred_language",
                    "content": "用户偏好中文回答。",
                    "memory_kind": "preference",
                    "confidence": 0.9,
                    "importance": 0.8,
                }
            ]

        episodic = FakeEpisodic()
        semantic = FakeSemantic()
        consolidator = MemoryConsolidator(episodic=episodic, semantic=semantic, extractor=extractor)

        count = asyncio.run(consolidator.consolidate_session("sess-1"))

        self.assertEqual(count, 1)
        self.assertEqual(semantic.upserts[0]["source_session_ids"], ["sess-1"])
        self.assertEqual(semantic.upserts[0]["user_id"], "user-1")
        self.assertEqual(episodic.marked, [("sess-1", 2)])


class MemoryRecallIntegrationTests(unittest.TestCase):
    def test_memory_recall_combines_redis_history_and_semantic_long_term_context(self):
        from app.agent.routing import memory_recall

        store = MagicMock()
        store.get_history.return_value = [{"role": "user", "content": "你好"}]
        store.build_context_for_prompt.return_value = "## 对话历史\n[1] 用户: 你好"

        class FakeManager:
            async def recall(self, *args, **kwargs):
                return {"semantic": [{"key": "preferred_language", "content": "用户偏好中文回答"}]}

            def build_context_prompt(self, recalled, max_items=5):
                return "## 长期记忆\n- [preferred_language] 用户偏好中文回答"

        with (
            patch.object(memory_recall, "get_session_store", return_value=store),
            patch.object(memory_recall, "get_memory_manager", return_value=FakeManager()),
        ):
            result = asyncio.run(memory_recall.memory_recall_node({"query": "你记得我的偏好吗", "session_id": "sess-1"}))

        self.assertIn("对话历史", result["memory_context"])
        self.assertIn("长期记忆", result["memory_context"])
        self.assertEqual(result["recalled_memories"]["semantic"][0]["key"], "preferred_language")


if __name__ == "__main__":
    unittest.main()
