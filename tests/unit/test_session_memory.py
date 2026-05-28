"""Unit tests for Redis-based session memory (dual-key sliding window).

Uses fakeredis for Redis simulation — no real Redis needed.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.services.memory.session_store import SessionStore, TurnRecord


class TurnRecordTests(unittest.TestCase):
    def test_create_and_serialize(self):
        turn = TurnRecord(role="user", content="hello", intent="chat")
        d = turn.to_dict()
        self.assertEqual(d["role"], "user")
        self.assertEqual(d["content"], "hello")
        self.assertEqual(d["intent"], "chat")
        self.assertGreater(len(d["turn_id"]), 0)
        self.assertGreater(len(d["timestamp"]), 0)

    def test_roundtrip(self):
        original = TurnRecord(
            role="assistant",
            content="The answer is 42.",
            intent="retrieval",
            used_tools=["retrieve_evidence", "answer_with_rag"],
        )
        restored = TurnRecord.from_dict(original.to_dict())
        self.assertEqual(restored.role, "assistant")
        self.assertEqual(restored.content, "The answer is 42.")
        self.assertEqual(restored.used_tools, ["retrieve_evidence", "answer_with_rag"])

    def test_content_truncation(self):
        long_content = "x" * 5000
        turn = TurnRecord(role="user", content=long_content)
        self.assertLessEqual(len(turn.content), 4000)

    def test_from_dict_missing_fields(self):
        d = {"role": "user"}
        turn = TurnRecord.from_dict(d)
        self.assertEqual(turn.role, "user")
        self.assertEqual(turn.content, "")
        self.assertEqual(turn.used_tools, [])


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        # Use a mock Redis (dict-based simulation)
        self._data: dict[str, str] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._lists: dict[str, list[str]] = {}
        self._expires: dict[str, int] = {}

        mock_redis = MagicMock()

        def fake_exists(key):
            return key in self._hashes or key in self._lists

        def fake_hgetall(key):
            return dict(self._hashes.get(key, {}))

        def fake_hset(key, field, value):
            if key not in self._hashes:
                self._hashes[key] = {}
            self._hashes[key][field] = value

        def fake_hincrby(key, field, amount):
            current = int(self._hashes.get(key, {}).get(field, 0))
            new_val = current + amount
            if key not in self._hashes:
                self._hashes[key] = {}
            self._hashes[key][field] = str(new_val)
            return new_val

        def fake_hget(key, field):
            return self._hashes.get(key, {}).get(field)

        def fake_lpush(key, value):
            if key not in self._lists:
                self._lists[key] = []
            self._lists[key].insert(0, value)

        def fake_ltrim(key, start, end):
            if key in self._lists:
                self._lists[key] = self._lists[key][start:end + 1]

        def fake_lrange(key, start, end):
            lst = self._lists.get(key, [])
            return lst[start:end + 1]

        def fake_delete(*keys):
            for k in keys:
                self._hashes.pop(k, None)
                self._lists.pop(k, None)

        def fake_expire(key, ttl):
            self._expires[key] = ttl

        def fake_pipeline():
            pipe = MagicMock()
            pipe.lpush = fake_lpush
            pipe.ltrim = fake_ltrim
            pipe.execute = lambda: None
            return pipe

        mock_redis.exists = fake_exists
        mock_redis.hgetall = fake_hgetall
        mock_redis.hset = fake_hset
        mock_redis.hget = fake_hget
        mock_redis.hincrby = fake_hincrby
        mock_redis.lpush = fake_lpush
        mock_redis.ltrim = fake_ltrim
        mock_redis.lrange = fake_lrange
        mock_redis.delete = fake_delete
        mock_redis.expire = fake_expire
        mock_redis.pipeline = fake_pipeline
        mock_redis.scan = lambda cursor=0, match="", count=50: (0, [])

        self._data = self._data
        self._hashes = self._hashes
        self._lists = self._lists
        self._expires = self._expires

        self.store = SessionStore(
            redis_client=mock_redis,
            window_size=10,
            session_ttl=86400,
        )

    # ── session lifecycle ──────────────────────────────────────────────

    def test_ensure_session_creates_new(self):
        meta = self.store.ensure_session("sess-001", user_id="user-1")
        self.assertEqual(meta["status"], "active")
        self.assertEqual(meta["user_id"], "user-1")
        self.assertEqual(meta["turn_count"], "0")
        self.assertIn("created_at", meta)

    def test_ensure_session_idempotent(self):
        meta1 = self.store.ensure_session("sess-001")
        meta2 = self.store.ensure_session("sess-001")
        self.assertEqual(meta1["created_at"], meta2["created_at"])
        self.assertIn("last_active", meta2)

    def test_get_session_returns_none_for_unknown(self):
        self.assertIsNone(self.store.get_session("nonexistent"))

    def test_end_session_marks_expired(self):
        self.store.ensure_session("sess-002")
        self.store.end_session("sess-002")
        meta = self.store.get_session("sess-002")
        self.assertEqual(meta["status"], "expired")

    def test_delete_session_removes_all(self):
        self.store.ensure_session("sess-003")
        self.store.append_turn("sess-003", TurnRecord(role="user", content="hello"))
        self.store.delete_session("sess-003")
        self.assertIsNone(self.store.get_session("sess-003"))
        self.assertEqual(len(self.store.get_history("sess-003", as_messages=True)), 0)

    # ── turn history ───────────────────────────────────────────────────

    def test_append_and_get_history(self):
        self.store.ensure_session("sess-004")
        self.store.append_turn("sess-004", TurnRecord(role="user", content="Q1: what is LCDNet"))
        self.store.append_turn("sess-004", TurnRecord(role="assistant", content="A1: LCDNet is..."))

        history = self.store.get_history("sess-004", as_messages=True)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "Q1: what is LCDNet")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[1]["content"], "A1: LCDNet is...")

    def test_turn_count(self):
        self.store.ensure_session("sess-005")
        self.store.append_turn("sess-005", TurnRecord(role="user", content="hello"))
        self.store.append_turn("sess-005", TurnRecord(role="assistant", content="hi"))
        self.assertEqual(self.store.get_turn_count("sess-005"), 2)

    def test_sliding_window_trims_oldest(self):
        store = SessionStore(redis_client=self.store._redis, window_size=3, session_ttl=86400)
        store.ensure_session("sess-006")
        for i in range(10):
            store.append_turn("sess-006", TurnRecord(role="user", content=f"msg_{i}"))

        history = store.get_history("sess-006", as_messages=True)
        self.assertEqual(len(history), 3)
        # Should keep the 3 newest: msg_7, msg_8, msg_9
        self.assertIn("msg_7", history[0]["content"])
        self.assertIn("msg_9", history[2]["content"])

    # ── context builder ──────────────────────────────────────────────

    def test_build_context_for_prompt(self):
        self.store.ensure_session("sess-007")
        self.store.append_turn("sess-007", TurnRecord(role="user", content="what is PSNR"))
        self.store.append_turn("sess-007", TurnRecord(role="assistant", content="PSNR is a metric for image quality."))

        context = self.store.build_context_for_prompt("sess-007", "how to calculate it")
        self.assertIn("对话历史", context)
        self.assertIn("what is PSNR", context)
        self.assertIn("## 当前问题", context)
        self.assertIn("how to calculate it", context)

    def test_build_context_empty_session(self):
        context = self.store.build_context_for_prompt("empty-session", "test query")
        self.assertEqual(context, "")

    # ── key format ───────────────────────────────────────────────────

    def test_session_key_format(self):
        self.assertEqual(SessionStore.session_key("abc-123"), "session:abc-123")

    def test_history_key_format(self):
        self.assertEqual(SessionStore.history_key("abc-123"), "history:abc-123")

    # ── dual-key decoupling ──────────────────────────────────────────

    def test_dual_key_separation(self):
        """Session metadata and history are in different Redis keys."""
        self.store.ensure_session("sess-008")
        self.store.append_turn("sess-008", TurnRecord(role="user", content="test"))

        skey = SessionStore.session_key("sess-008")
        hkey = SessionStore.history_key("sess-008")

        self.assertIn(skey, self._hashes)
        self.assertIn(hkey, self._lists)
        # They're independent
        self.assertGreater(len(self._hashes[skey]), 0)

    def test_history_survives_connection_drop(self):
        """Simulate: write turns → 'disconnect' → restore from session_id."""
        self.store.ensure_session("sess-009")
        self.store.append_turn("sess-009", TurnRecord(role="user", content="step 1"))
        self.store.append_turn("sess-009", TurnRecord(role="assistant", content="result 1"))
        self.store.append_turn("sess-009", TurnRecord(role="user", content="step 2"))

        # Simulate reconnect: re-fetch from same session_id
        history = self.store.get_history("sess-009", as_messages=True)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]["content"], "step 1")
        self.assertEqual(history[2]["content"], "step 2")

        context = self.store.build_context_for_prompt("sess-009", "step 3")
        self.assertIn("step 1", context)
        self.assertIn("step 3", context)


if __name__ == "__main__":
    unittest.main()
