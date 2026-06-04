"""PaperMind 检索工具实现 — 供 MCP Server 与测试复用。"""

from __future__ import annotations

import asyncio
from typing import Optional

from app.services.retrieval import hybrid_retrieve, retrieve_parent_documents
from app.services.qa import answer as qa_answer
from app.services.papers.index import get_paper_profile as fetch_paper_profile
from app.services.papers.search import deep_search_papers_by_query, search_papers_by_query


async def retrieve_evidence(
    query: str,
    top_k: int = 5,
    task_id: Optional[str] = None,
) -> str:
    """混合检索已索引论文片段。"""
    try:
        docs = await retrieve_parent_documents(
            query=query,
            top_k=top_k,
            task_id=task_id,
        )
        if not docs:
            return "未找到相关论文片段。请尝试调整查询关键词。"

        lines = []
        for i, doc in enumerate(docs):
            title = doc.metadata.get("original_filename", "未知来源")
            section = doc.metadata.get("section_title") or "未知章节"
            lines.append(f"[{i + 1}] 来源: {title} | {section}")
            lines.append(doc.page_content[:800])
            lines.append("---")
        return "\n".join(lines)
    except Exception as exc:
        return f"检索执行失败: {str(exc)}"


async def retrieve_with_mqe(
    query: str,
    top_k: int = 5,
    task_id: Optional[str] = None,
) -> str:
    """多查询扩展混合检索。"""
    try:
        chunks = await hybrid_retrieve(
            query=query,
            top_k=top_k,
            task_id=task_id,
        )
        if not chunks:
            return "未找到相关论文片段。"

        lines = []
        for i, chunk in enumerate(chunks):
            title = chunk.metadata.get("original_filename", "未知来源")
            section = chunk.metadata.get("section_title") or "未知章节"
            lines.append(f"[{i + 1}] 来源: {title} | {section}")
            lines.append(chunk.parent_text[:800])
            lines.append("---")
        return "\n".join(lines)
    except Exception as exc:
        return f"多查询检索执行失败: {str(exc)}"


async def answer_with_rag(
    question: str,
    top_k: int = 5,
) -> str:
    """RAG 问答。"""
    try:
        result = await qa_answer(
            query=question,
            top_k=top_k,
        )
        answer_text = result.get("answer", "")
        sources = result.get("contexts", [])
        if sources:
            src_lines = ["", "## 引用来源", ""]
            for i, src in enumerate(sources[:5]):
                meta = getattr(src, "metadata", None) or {}
                parent_id = getattr(src, "parent_id", "") or ""
                filename = str(
                    meta.get("original_filename")
                    or meta.get("source_file")
                    or meta.get("title")
                    or "未知来源"
                )
                section = str(meta.get("section_title") or "")
                src_label = f"[{i + 1}] {filename}"
                if parent_id:
                    src_label += f" #{parent_id[:12]}"
                if section:
                    src_label += f" | {section}"
                src_lines.append(f"- {src_label}")
            answer_text += "\n".join(src_lines)
        return answer_text or "无法生成答案。"
    except Exception as exc:
        return f"RAG 问答失败: {str(exc)}"


async def search_papers(
    query: str,
    task_id: Optional[str] = None,
) -> str:
    """论文级搜索。"""
    if not query or not query.strip():
        try:
            from app.services.storage.es import get_es_client
            from app.core.config import settings

            es = get_es_client()
            body = {
                "query": {"match_all": {}},
                "size": 20,
                "sort": [{"_score": "desc"}],
            }
            resp = await asyncio.to_thread(
                es.search,
                index=settings.es_index_papers,
                body=body,
            )
            hits = resp.get("hits", {}).get("hits", [])
            total = resp.get("hits", {}).get("total", {})
            total_value = total.get("value", 0) if isinstance(total, dict) else total
            if not hits:
                return (
                    f"知识库当前为空，未找到任何已索引的论文"
                    f"（ES 索引 {settings.es_index_papers} 存在但无文档）。"
                )
            lines = [
                f"知识库中共有 {total_value} 篇已索引论文，以下是前 {len(hits)} 篇：",
                "",
            ]
            for i, hit in enumerate(hits):
                src = hit.get("_source", {})
                lines.append(
                    f"[{i + 1}] **{src.get('title') or src.get('source_file') or src.get('paper_id')}**"
                )
                if src.get("main_task"):
                    lines.append(f"    主要任务: {src['main_task']}")
                if src.get("abstract_summary"):
                    lines.append(f"    摘要: {src['abstract_summary'][:300]}")
                lines.append("")
            return "\n".join(lines)
        except Exception as exc:
            return f"检索知识库统计失败: {str(exc)}"

    try:
        results = await search_papers_by_query(query=query, task_id=task_id)
        if not results:
            return "未找到相关论文。请尝试使用更宽泛的关键词。"

        lines = [f"共检索到 {len(results)} 篇相关论文：", ""]
        for i, r in enumerate(results[:10]):
            lines.append(f"[{i + 1}] **{r.title or r.source_file or r.paper_id}**")
            if r.main_task:
                lines.append(f"    主要任务: {r.main_task}")
            if r.method_tags:
                lines.append(f"    核心方法: {', '.join(r.method_tags[:4])}")
            if r.abstract_summary:
                lines.append(f"    摘要: {r.abstract_summary[:300]}")
            if r.reason:
                lines.append(f"    匹配原因: {r.reason}")
            if r.evidence_chunks:
                lines.append(f"    证据片段: {r.evidence_chunks[0][:200]}")
            lines.append("")
        return "\n".join(lines)
    except Exception as exc:
        return f"论文搜索失败: {str(exc)}"


async def deep_search_papers(
    query: str,
    task_id: Optional[str] = None,
) -> str:
    """深度论文检索。"""
    try:
        results = await deep_search_papers_by_query(query=query, task_id=task_id)
        if not results:
            return "未找到相关论文。请尝试调整查询关键词。"

        lines = [f"深度检索到 {len(results)} 篇高度相关论文：", ""]
        for i, r in enumerate(results[:10]):
            lines.append(f"[{i + 1}] **{r.title or r.source_file or r.paper_id}**")
            if r.main_task:
                lines.append(f"    主要任务: {r.main_task}")
            if r.method_tags:
                lines.append(f"    核心方法: {', '.join(r.method_tags[:4])}")
            if r.abstract_summary:
                lines.append(f"    摘要: {r.abstract_summary[:300]}")
            if r.reason:
                lines.append(f"    匹配原因: {r.reason}")
            if r.method_summary:
                lines.append(f"    方法简述: {r.method_summary[:200]}")
            if r.evidence_chunks:
                for chunk in r.evidence_chunks[:2]:
                    lines.append(f"    证据: {chunk[:200]}")
            lines.append("")
        return "\n".join(lines)
    except Exception as exc:
        return f"深度论文搜索失败: {str(exc)}"


async def get_paper_profile(paper_id: str) -> str:
    """获取单篇论文画像。"""
    try:
        profile = await asyncio.to_thread(fetch_paper_profile, paper_id=paper_id)
        if profile is None:
            return f"未找到论文 {paper_id} 的画像。"

        parts = []
        if profile.title:
            parts.append(f"## 标题\n{profile.title}")
        if profile.authors:
            parts.append(f"## 作者\n{', '.join(profile.authors)}")
        if profile.year:
            parts.append(f"## 年份\n{profile.year}")
        if profile.research_problem:
            parts.append(f"## 研究问题\n{profile.research_problem}")
        if profile.method_name or profile.method_summary:
            parts.append(f"## 方法\n{profile.method_name}\n{profile.method_summary}")
        if profile.method_tags:
            parts.append(f"## 方法标签\n{', '.join(profile.method_tags)}")
        if profile.main_task:
            parts.append(f"## 主要任务\n{profile.main_task}")
        if profile.contribution_summary:
            parts.append(f"## 贡献\n{profile.contribution_summary}")
        if profile.abstract_summary or profile.abstract:
            parts.append(f"## 摘要\n{profile.abstract_summary or profile.abstract[:500]}")
        if profile.summary:
            parts.append(f"## 总结\n{profile.summary}")
        if profile.source_file:
            parts.append(f"## 来源文件\n{profile.source_file}")

        return "\n\n".join(parts) if parts else f"论文 {paper_id} 的画像数据不完整。"
    except Exception as exc:
        return f"获取论文画像失败: {str(exc)}"
