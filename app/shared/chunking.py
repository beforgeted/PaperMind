"""Markdown -> paper-aware LangChain `Document`s for ParentDocumentRetriever.

Docling gives us Markdown, but papers should be chunked around their structure
before the parent/child splitters run. Each section-level `Document` carries
metadata that will be propagated to parent and child chunks, allowing retrieval
to prefer method, experiment, ablation, caption, and reference sections.
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document

from app.core.schemas import ParsedDocument
from app.shared.paper_utils import parse_markdown_sections


def parsed_to_documents(parsed: ParsedDocument) -> List[Document]:
    """Build section-level documents with paper-aware metadata."""
    if not parsed.markdown or not parsed.markdown.strip():
        return []

    documents: list[Document] = []
    sections = parse_markdown_sections(parsed.markdown, parsed.title)
    for section_index, section in enumerate(sections):
        metadata = {
            "paper_id": parsed.task_id,
            "task_id": parsed.task_id,
            "original_filename": parsed.original_filename,
            "title": parsed.title,
            "section_title": section.section_title,
            "section_type": section.section_type,
            "subsection": section.subsection,
            "page": section.page,
            "keywords": section.keywords,
            "entities": section.entities,
            "content_type": section.content_type,
            "section_index": section_index,
        }
        documents.append(Document(page_content=section.text, metadata=metadata))

    if documents:
        return documents

    metadata = {
        "paper_id": parsed.task_id,
        "task_id": parsed.task_id,
        "original_filename": parsed.original_filename,
        "title": parsed.title,
        "section_title": parsed.title or "Document",
        "section_type": "introduction",
        "subsection": None,
        "page": None,
        "keywords": [],
        "entities": [],
        "content_type": "text",
        "section_index": 0,
    }
    return [Document(page_content=parsed.markdown, metadata=metadata)]


__all__ = ["parsed_to_documents"]
