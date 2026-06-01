"""Unit tests for the memory system data models and config."""

from __future__ import annotations

import unittest

from app.services.memory.models import MemoryItem, MemoryKind, MemoryType
from app.services.memory.config import MemoryConfig


class MemoryItemTests(unittest.TestCase):
    def test_item_creation_defaults(self):
        item = MemoryItem(key="test", content="hello world")
        self.assertEqual(item.memory_type, MemoryType.WORKING)
        self.assertEqual(item.scope, "session")
        self.assertEqual(item.importance, 0.5)
        self.assertGreater(len(item.memory_id), 0)
        self.assertGreater(len(item.created_at), 0)

    def test_item_touch_updates_access_count(self):
        item = MemoryItem(key="test", content="data")
        self.assertEqual(item.access_count, 0)
        item.touch()
        self.assertEqual(item.access_count, 1)
        item.touch()
        self.assertEqual(item.access_count, 2)
        self.assertGreater(len(item.last_accessed_at), 0)

    def test_item_is_expired(self):
        item = MemoryItem(key="test", content="data", ttl_seconds=0)
        item.expires_at = "2020-01-01T00:00:00+00:00"  # force past
        self.assertTrue(item.is_expired())

        item2 = MemoryItem(key="test2", content="data", ttl_seconds=86400)
        self.assertFalse(item2.is_expired())

    def test_item_summary(self):
        item = MemoryItem(key="user_pref", content="prefer Chinese", importance=0.9)
        summary = item.summary()
        self.assertIn("working", summary)
        self.assertIn("user_pref", summary)
        self.assertIn("0.9", summary)

    def test_item_long_term_fields(self):
        item = MemoryItem(
            key="preferred_language",
            content="用户偏好中文回答",
            memory_type=MemoryType.SEMANTIC,
            memory_kind="preference",
            user_id="user-1",
            project_id="project-1",
            source_session_ids=["sess-1"],
            confidence=0.9,
            importance=0.8,
        )
        self.assertEqual(item.memory_kind, "preference")
        self.assertEqual(item.user_id, "user-1")
        self.assertEqual(item.confidence, 0.9)
        self.assertEqual(item.source_session_ids, ["sess-1"])


class MemoryConfigTests(unittest.TestCase):
    def test_defaults(self):
        cfg = MemoryConfig()
        self.assertEqual(cfg.semantic_max_items, 10000)
        self.assertEqual(cfg.semantic_retrieval_top_k, 5)
        self.assertEqual(cfg.consolidation_min_importance, 0.7)
        self.assertEqual(cfg.forget_max_age_days, 90)

    def test_custom_values(self):
        cfg = MemoryConfig(semantic_max_items=100, consolidation_min_importance=0.5)
        self.assertEqual(cfg.semantic_max_items, 100)
        self.assertEqual(cfg.consolidation_min_importance, 0.5)

    def test_no_working_memory_fields(self):
        """Verify WorkingMemory dead fields have been removed."""
        cfg = MemoryConfig()
        self.assertFalse(hasattr(cfg, "working_max_items"))
        self.assertFalse(hasattr(cfg, "working_default_ttl_seconds"))
        self.assertFalse(hasattr(cfg, "working_cleanup_interval"))
        self.assertFalse(hasattr(cfg, "auto_recall_working"))

    def test_consolidation_use_llm_exists(self):
        cfg = MemoryConfig()
        self.assertTrue(hasattr(cfg, "consolidation_use_llm"))
        self.assertIs(cfg.consolidation_use_llm, True)


class MemoryItemSerializationTests(unittest.TestCase):
    def test_model_dump(self):
        item = MemoryItem(key="test", content="hello")
        d = item.model_dump(mode="json")
        self.assertEqual(d["key"], "test")
        self.assertEqual(d["memory_type"], "working")
        self.assertIn("memory_id", d)

    def test_semantic_type(self):
        item = MemoryItem(key="knowledge", content="PSNR is a common metric", memory_type=MemoryType.SEMANTIC)
        self.assertEqual(item.memory_type, MemoryType.SEMANTIC)

    def test_episodic_metadata(self):
        item = MemoryItem(
            key="interaction:2026-05-28",
            content="Q: what is LCDNet",
            memory_type=MemoryType.EPISODIC,
            metadata={"query": "what is LCDNet", "intent": "retrieval", "used_tools": ["retrieve_evidence"]},
        )
        self.assertEqual(item.metadata["intent"], "retrieval")
        self.assertEqual(item.metadata["used_tools"], ["retrieve_evidence"])

    def test_memory_kind_default(self):
        item = MemoryItem(key="test", content="x")
        self.assertEqual(item.memory_kind, "fact")


class MemoryKindTests(unittest.TestCase):
    def test_valid_kinds(self):
        """Verify MemoryKind type includes all expected categories."""
        valid = {"preference", "research_interest", "project_state", "writing_style", "fact"}
        for kind in valid:
            item = MemoryItem(key=kind, content="test", memory_kind=kind)  # type: ignore[arg-type]
            self.assertEqual(item.memory_kind, kind)


if __name__ == "__main__":
    unittest.main()
