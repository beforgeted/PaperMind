"""Tests for next-parent expansion on hybrid retrieve."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.services.retrieval import _append_next_parent_text, _load_next_parent_docs


class RetrievalNextParentTests(unittest.TestCase):
    def test_append_next_parent_concatenates_text(self):
        base = Document(
            page_content="part one",
            metadata={"next_parent_id": "next-1"},
        )
        next_doc = Document(page_content="part two", metadata={})
        expanded = _append_next_parent_text(
            base,
            next_by_id={"next-1": next_doc},
        )
        self.assertIn("part one", expanded.page_content)
        self.assertIn("part two", expanded.page_content)
        self.assertEqual(expanded.metadata.get("_included_next_parent_id"), "next-1")

    @patch("app.services.retrieval.get_docstore")
    def test_load_next_parent_docs_batch(self, mock_get_docstore):
        store = MagicMock()
        mock_get_docstore.return_value = store
        parents = [
            Document(page_content="a", metadata={"next_parent_id": "n1"}),
            Document(page_content="b", metadata={"next_parent_id": "n1"}),
        ]
        store.mget.return_value = [Document(page_content="next", metadata={})]
        loaded = _load_next_parent_docs(parents)
        self.assertIn("n1", loaded)
        store.mget.assert_called_once_with(["n1"])


if __name__ == "__main__":
    unittest.main()
