"""LCEL question-answering chain over hybrid-retrieved parent contexts.

Pipeline:
    query -> ParentDocumentRetriever (hybrid + parent backtrack)
          -> format docs into a single context block
          -> ChatPromptTemplate
          -> ChatTongyi (Qwen)
          -> StrOutputParser

`answer()` returns both the generated answer and the retrieved contexts so the
caller can show citations to the user.
"""

from __future__ import annotations

import os
from typing import AsyncIterator, List, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch  # noqa: F401
except Exception:
    pass

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import settings
from app.core.schemas import RetrievedChunk
from app.services.llm_client import get_llm, get_streaming_llm
from app.services.retrieval import retrieve_parent_documents


_SYSTEM_PROMPT = (
    "你是科研论文问答助手。基于下方检索到的论文上下文回答用户问题，"
    "在引用论据时使用 [parent_id] 标注来源；若上下文不足以回答，请明确说明。"
    "回答应严谨、忠实于上下文，不要编造未在上下文中出现的事实。"
    "请使用 Markdown 格式组织回答，合理使用标题、列表、表格等结构，提高可读性。"
)


_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "上下文：\n{context}\n\n问题：{question}"),
    ]
)


def _citation_label(md: dict, fallback: str) -> str:
    """Build a human-readable citation: '<filename>#<short-id>' when possible."""
    doc_id = md.get("doc_id") or fallback
    filename = md.get("original_filename")
    short = doc_id[:8] if isinstance(doc_id, str) else fallback
    return f"{filename}#{short}" if filename else short


def _format_docs(docs: List[Document]) -> str:
    if not docs:
        return "(无相关上下文)"
    blocks = []
    for i, d in enumerate(docs):
        md = d.metadata or {}
        label = _citation_label(md, f"rank-{i}")
        section = md.get("section_title") or "Unknown section"
        section_type = md.get("section_type") or "unknown"
        blocks.append(f"[{label} | {section_type} | {section}] {d.page_content}")
    return "\n\n".join(blocks)


def _to_retrieved_chunks(docs: List[Document]) -> List[RetrievedChunk]:
    out: List[RetrievedChunk] = []
    for i, d in enumerate(docs):
        md = d.metadata or {}
        out.append(
            RetrievedChunk(
                parent_id=md.get("doc_id") or f"rank-{i}",
                parent_text=d.page_content,
                child_ids=[],
                score=1.0 / (i + 1),
                metadata={
                    "paper_id": md.get("paper_id") or md.get("task_id"),
                    "task_id": md.get("task_id"),
                    "original_filename": md.get("original_filename"),
                    "title": md.get("title"),
                    "section_title": md.get("section_title"),
                    "section_type": md.get("section_type"),
                    "subsection": md.get("subsection"),
                    "page": md.get("page"),
                    "keywords": md.get("keywords") or [],
                    "entities": md.get("entities") or [],
                    "content_type": md.get("content_type"),
                    "citation": _citation_label(md, f"rank-{i}"),
                },
            )
        )
    return out


async def answer(
    query: str,
    top_k: Optional[int] = None,
    task_id: Optional[str] = None,
) -> dict:
    """Run the full retrieve->generate pipeline. Returns {answer, contexts}."""
    final_top_k = top_k or settings.qa_top_k
    docs: List[Document] = await retrieve_parent_documents(
        query=query,
        top_k=final_top_k,
        task_id=task_id,
    )

    chain = _PROMPT | get_llm() | StrOutputParser()
    text = await chain.ainvoke(
        {"context": _format_docs(docs), "question": query}
    )

    return {"answer": text, "contexts": _to_retrieved_chunks(docs)}


async def stream_answer(
    query: str,
    top_k: Optional[int] = None,
    task_id: Optional[str] = None,
) -> AsyncIterator[dict]:
    """Run retrieve -> streaming generate and yield NDJSON-ready events."""
    final_top_k = top_k or settings.qa_top_k
    docs: List[Document] = await retrieve_parent_documents(
        query=query,
        top_k=final_top_k,
        task_id=task_id,
    )
    contexts = _to_retrieved_chunks(docs)
    yield {
        "type": "metadata",
        "query": query,
        "contexts": [item.model_dump() for item in contexts],
    }

    chain = _PROMPT | get_streaming_llm() | StrOutputParser()
    async for token in chain.astream(
        {"context": _format_docs(docs), "question": query}
    ):
        if token:
            yield {"type": "delta", "text": token}

    yield {"type": "done"}
