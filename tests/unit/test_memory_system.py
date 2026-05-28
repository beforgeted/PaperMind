"""Unit tests for the three-layer memory system."""

from __future__ import annotations

import time
import unittest

from app.services.memory.models import MemoryItem, MemoryType
from app.services.memory.config import MemoryConfig
from app.services.memory.working import WorkingMemory


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
        import time
        time.sleep(0.1)
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


class MemoryConfigTests(unittest.TestCase):
    def test_defaults(self):
        cfg = MemoryConfig()
        self.assertEqual(cfg.working_max_items, 50)
        self.assertEqual(cfg.working_default_ttl_seconds, 3600)
        self.assertEqual(cfg.consolidation_min_importance, 0.7)
        self.assertEqual(cfg.forget_max_age_days, 90)

    def test_custom_values(self):
        cfg = MemoryConfig(working_max_items=100, consolidation_min_importance=0.5)
        self.assertEqual(cfg.working_max_items, 100)
        self.assertEqual(cfg.consolidation_min_importance, 0.5)


class WorkingMemoryTests(unittest.TestCase):
    def setUp(self):
        self.wm = WorkingMemory()

    def test_store_and_recall(self):
        item = MemoryItem(key="topic", content="underwater image enhancement", scope="user", importance=0.8)
        self.wm.store(item)

        results = self.wm.recall(query="underwater", scope="user")
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].key, "topic")

    def test_recall_no_match(self):
        item = MemoryItem(key="topic", content="underwater image enhancement", scope="user")
        self.wm.store(item)

        results = self.wm.recall(query="quantum physics")
        self.assertEqual(len(results), 0)

    def test_recall_scope_filter(self):
        item1 = MemoryItem(key="a", content="data", scope="session")
        item2 = MemoryItem(key="b", content="data", scope="user")
        self.wm.store(item1)
        self.wm.store(item2)

        results = self.wm.recall(query="data", scope="session")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].key, "a")

    def test_get_by_id(self):
        item = MemoryItem(key="test", content="value")
        self.wm.store(item)

        found = self.wm.get(item.memory_id)
        self.assertIsNotNone(found)
        self.assertEqual(found.key, "test")

        not_found = self.wm.get("nonexistent")
        self.assertIsNone(not_found)

    def test_forget(self):
        item = MemoryItem(key="temp", content="delete me")
        self.wm.store(item)

        self.assertTrue(self.wm.forget(item.memory_id))
        self.assertIsNone(self.wm.get(item.memory_id))
        self.assertFalse(self.wm.forget(item.memory_id))  # already gone

    def test_forget_expired(self):
        # Create an expired item
        item = MemoryItem(key="old", content="stale data", ttl_seconds=-1)  # already expired
        item.expires_at = "2020-01-01T00:00:00+00:00"
        self.wm._items[item.memory_id] = item  # bypass store to avoid TTL reset

        removed = self.wm.forget_expired()
        self.assertEqual(removed, 1)
        self.assertEqual(self.wm.size, 0)

    def test_capacity_eviction(self):
        cfg = MemoryConfig(working_max_items=3)
        wm = WorkingMemory(config=cfg)

        for i in range(5):
            item = MemoryItem(key=f"item_{i}", content=f"data_{i}", importance=0.1 * i)
            wm.store(item)

        self.assertLessEqual(wm.size, 3)

    def test_consolidation_candidates(self):
        # Low importance, low access — not a candidate
        item1 = MemoryItem(key="low", content="low", importance=0.3, access_count=0)
        self.wm.store(item1)

        # High importance, high access — candidate
        item2 = MemoryItem(key="high", content="high", importance=0.9, access_count=5)
        self.wm.store(item2)

        candidates = self.wm.get_consolidation_candidates()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].key, "high")

    def test_stats(self):
        item = MemoryItem(key="a", content="data", importance=0.7)
        self.wm.store(item)

        stats = self.wm.stats()
        self.assertEqual(stats["total_items"], 1)
        self.assertGreater(stats["avg_importance"], 0)
        self.assertEqual(stats["expired"], 0)


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


if __name__ == "__main__":
    unittest.main()
