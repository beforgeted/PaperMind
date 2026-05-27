"""Unit tests for ordered parent/child chunk metadata."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from app.services.chunk_indexing import build_parent_child_batches


class ChunkIndexingMetadataTests(unittest.TestCase):
    @patch("app.services.chunk_indexing.parent_text_splitter")
    @patch("app.services.chunk_indexing.child_text_splitter")
    def test_parent_chain_and_child_indices(self, mock_child_splitter, mock_parent_splitter):
        section = Document(
            page_content="Section body for indexing tests.",
            metadata={
                "task_id": "task-1",
                "section_index": 3,
                "section_title": "3.3 Qualitative Analysis",
            },
        )
        parents = [
            Document(page_content="parent-A", metadata=dict(section.metadata)),
            Document(page_content="parent-B", metadata=dict(section.metadata)),
        ]
        mock_parent_splitter.return_value.split_documents.return_value = parents

        def split_children(parent_doc):
            return [
                Document(
                    page_content=f"{parent_doc.page_content}-c0",
                    metadata=dict(parent_doc.metadata),
                ),
                Document(
                    page_content=f"{parent_doc.page_content}-c1",
                    metadata=dict(parent_doc.metadata),
                ),
            ]

        mock_child_splitter.return_value.split_documents.side_effect = split_children

        parent_pairs, children = build_parent_child_batches([section])

        self.assertEqual(len(parent_pairs), 2)
        self.assertEqual(len(children), 4)

        p0_id, p0 = parent_pairs[0]
        p1_id, p1 = parent_pairs[1]
        self.assertEqual(p0.metadata["parent_index"], 0)
        self.assertEqual(p1.metadata["parent_index"], 1)
        self.assertEqual(p0.metadata["parent_count"], 2)
        self.assertNotIn("prev_parent_id", p0.metadata)
        self.assertEqual(p0.metadata["next_parent_id"], p1_id)
        self.assertEqual(p1.metadata["prev_parent_id"], p0_id)
        self.assertNotIn("next_parent_id", p1.metadata)

        by_parent: dict[str, list[Document]] = {}
        for child in children:
            pid = child.metadata["doc_id"]
            by_parent.setdefault(pid, []).append(child)

        self.assertEqual(len(by_parent[p0_id]), 2)
        self.assertEqual(by_parent[p0_id][0].metadata["child_index"], 0)
        self.assertEqual(by_parent[p0_id][1].metadata["child_index"], 1)
        self.assertEqual(by_parent[p0_id][0].metadata["child_count"], 2)


if __name__ == "__main__":
    unittest.main()
