"""Tests for session schemas and episodic per-session doc storage (mocked ES)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class SessionRecordModelTests(unittest.TestCase):
    def test_session_record_defaults(self):
        from app.core.schemas import SessionRecord
        s = SessionRecord(session_id="abc", title="test")
        self.assertEqual(s.session_id, "abc")
        self.assertEqual(s.title, "test")
        self.assertEqual(s.turn_count, 0)
        self.assertEqual(s.status, "active")

    def test_session_list_response(self):
        from app.core.schemas import SessionListResponse, SessionRecord
        resp = SessionListResponse(sessions=[
            SessionRecord(session_id="a", title="A"),
            SessionRecord(session_id="b", title="B"),
        ])
        self.assertEqual(len(resp.sessions), 2)

    def test_agent_chat_request_with_session(self):
        from app.core.schemas import AgentChatRequest
        req = AgentChatRequest(query="hello", session_id="sess-123")
        self.assertEqual(req.session_id, "sess-123")

    def test_agent_chat_request_without_session(self):
        from app.core.schemas import AgentChatRequest
        req = AgentChatRequest(query="hello")
        self.assertEqual(req.session_id, "")


class SessionHistoryPairingTests(unittest.TestCase):
    def test_pair_user_assistant_turns(self):
        from app.api.v1.sessions import _pair_turns_to_messages

        turns = [
            {"turn_id": "u1", "role": "user", "content": "你好", "timestamp": "2026-01-01T10:00:00Z"},
            {"turn_id": "a1", "role": "assistant", "content": "你好，我是 PaperMind", "used_tools": ["chat"], "timestamp": "2026-01-01T10:00:01Z"},
            {"turn_id": "u2", "role": "user", "content": "第二问", "timestamp": "2026-01-01T10:01:00Z"},
            {"turn_id": "a2", "role": "assistant", "content": "第二答", "timestamp": "2026-01-01T10:01:05Z"},
        ]
        messages = _pair_turns_to_messages(turns)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].question, "你好")
        self.assertEqual(messages[0].answer, "你好，我是 PaperMind")
        self.assertEqual(messages[0].used_tools, ["chat"])
        self.assertEqual(messages[1].question, "第二问")
        self.assertEqual(messages[1].answer, "第二答")


class EpisodicSessionDocTests(unittest.TestCase):
    """Mocked ES tests for episodic per-session storage."""

    def setUp(self):
        self.mock_es = MagicMock()
        self.mock_es.indices.exists.return_value = True

        patcher = patch("app.services.memory.episodic.get_es_client", return_value=self.mock_es)
        self.mock_get_es = patcher.start()
        self.addCleanup(patcher.stop)

    def _make_episodic(self):
        from app.services.memory.episodic import EpisodicMemory
        return EpisodicMemory()

    def test_create_session_doc(self):
        episodic = self._make_episodic()
        # indices.exists called on init
        self.mock_es.indices.exists.assert_called()

    def test_append_turn_existing_doc(self):
        episodic = self._make_episodic()

        new_turn = {"turn_id": "t3", "role": "user", "content": "what is PSNR?", "intent": "retrieval", "timestamp": "..."}
        count = episodic.append_turn("sess-002", new_turn)

        self.assertEqual(count, 0)  # mock ES returns no doc, so turn_count = 0
        self.mock_es.update.assert_called_once()
        call_kwargs = self.mock_es.update.call_args[1]
        self.assertEqual(call_kwargs["id"], "sess-002")

    def test_append_turn_creates_new_doc_if_not_found(self):
        episodic = self._make_episodic()

        turn = {"turn_id": "t1", "role": "user", "content": "first message", "intent": "chat", "timestamp": "..."}
        count = episodic.append_turn("new-session", turn)

        self.assertEqual(count, 0)  # mock ES returns nothing
        self.mock_es.update.assert_called_once()

    def test_append_turn_preserves_full_content(self):
        """Content should be original text, not summarized."""
        episodic = self._make_episodic()

        rich_content = "LCDNet 采用自适应对数变换来增强水下图像对比度。核心公式为 I_out = log(1 + α·I_in) / log(1 + α)，其中 α 是自适应参数。该方法相比传统方法在 PSNR 指标上提升了 2.3dB。"
        turn = {"turn_id": "tx", "role": "assistant", "content": rich_content, "intent": "retrieval", "timestamp": "..."}
        episodic.append_turn("s", turn)

        self.mock_es.update.assert_called_once()
        upsert_doc = self.mock_es.update.call_args[1]["upsert"]
        stored = upsert_doc["turns"][0]["content"]
        self.assertIn("LCDNet", stored)
        self.assertIn("自适应对数变换", stored)
        self.assertIn("PSNR", stored)
        self.assertIn("2.3dB", stored)
        self.assertGreater(len(stored), 100)

    def test_list_sessions(self):
        self.mock_es.search.return_value = {
            "hits": {
                "hits": [
                    {"_source": {"session_id": "a", "title": "Session A", "turn_count": 3, "updated_at": "..."}},
                    {"_source": {"session_id": "b", "title": "Session B", "turn_count": 1, "updated_at": "..."}},
                ]
            }
        }

        episodic = self._make_episodic()

        sessions = episodic.list_sessions()
        self.assertEqual(len(sessions), 2)
        self.assertEqual(sessions[0]["title"], "Session A")

    def test_get_session_doc_found(self):
        expected = {"session_id": "sess-003", "title": "found", "turns": [{"turn_id": "t1", "content": "data", "role": "user"}]}
        self.mock_es.get.return_value = {"_source": expected}

        episodic = self._make_episodic()

        doc = episodic.get_session_doc("sess-003")
        self.assertIsNotNone(doc)
        self.assertEqual(doc["title"], "found")
        self.assertEqual(len(doc["turns"]), 1)

    def test_delete_session(self):
        episodic = self._make_episodic()

        result = episodic.delete_session("sess-to-delete")
        self.assertTrue(result)
        self.mock_es.delete.assert_called_once()


if __name__ == "__main__":
    unittest.main()
