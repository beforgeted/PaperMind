"""papermind-retrieval MCP Server — 暴露本地 ES/RAG 检索工具。"""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from app.mcp_servers.papermind_retrieval import impl

mcp = FastMCP("papermind-retrieval")


@mcp.tool()
async def retrieve_evidence(
    query: str,
    top_k: int = 5,
    task_id: Optional[str] = None,
) -> str:
    """在已索引论文知识库中执行混合检索（BM25 + 向量 + RRF），返回相关片段。"""
    return await impl.retrieve_evidence(query=query, top_k=top_k, task_id=task_id)


@mcp.tool()
async def retrieve_with_mqe(
    query: str,
    top_k: int = 5,
    task_id: Optional[str] = None,
) -> str:
    """多查询扩展混合检索，提高召回率。"""
    return await impl.retrieve_with_mqe(query=query, top_k=top_k, task_id=task_id)


@mcp.tool()
async def answer_with_rag(question: str, top_k: int = 5) -> str:
    """基于知识库的 RAG 问答，返回答案与引用来源。"""
    return await impl.answer_with_rag(question=question, top_k=top_k)


@mcp.tool()
async def search_papers(query: str, task_id: Optional[str] = None) -> str:
    """搜索知识库中的论文列表及元数据。"""
    return await impl.search_papers(query=query, task_id=task_id)


@mcp.tool()
async def deep_search_papers(query: str, task_id: Optional[str] = None) -> str:
    """深度检索：论文级初筛 + chunk 精检 + RRF 排序。"""
    return await impl.deep_search_papers(query=query, task_id=task_id)


@mcp.tool()
async def get_paper_profile(paper_id: str) -> str:
    """获取单篇论文的结构化画像。"""
    return await impl.get_paper_profile(paper_id=paper_id)
