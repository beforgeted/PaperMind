"""MCP Server 目录与 Agent 工具白名单（模式 A）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    description: str
    tool_names: Tuple[str, ...]


PAPERMIND_RETRIEVAL_SERVER = "papermind-retrieval"
PAPER_SEARCH_SERVER = "paper-search"

PAPER_SEARCH_MCP_SERVER = MCPServerConfig(
    name=PAPER_SEARCH_SERVER,
    description=(
        "外部论文搜索 MCP Server，支持 arXiv / PubMed / bioRxiv / "
        "medRxiv / Google Scholar 等多源检索与下载"
    ),
    tool_names=(
        "search_arxiv",
        "search_pubmed",
        "search_biorxiv",
        "search_medrxiv",
        "search_google_scholar",
        "download_arxiv",
        "download_pubmed",
        "download_biorxiv",
        "download_medrxiv",
        "read_arxiv_paper",
        "read_pubmed_paper",
        "read_biorxiv_paper",
        "read_medrxiv_paper",
    ),
)

INTERNAL_RETRIEVAL_SERVER = MCPServerConfig(
    name=PAPERMIND_RETRIEVAL_SERVER,
    description="PaperMind 本地知识库检索工具集（ES BM25 + kNN + RRF）",
    tool_names=(
        "retrieve_evidence",
        "retrieve_with_mqe",
        "answer_with_rag",
        "search_papers",
        "deep_search_papers",
        "get_paper_profile",
    ),
)

ALL_MCP_SERVERS: tuple[MCPServerConfig, ...] = (
    PAPER_SEARCH_MCP_SERVER,
    INTERNAL_RETRIEVAL_SERVER,
)

TOOLS_WITH_TASK_ID: frozenset[str] = frozenset(
    {
        "retrieve_evidence",
        "retrieve_with_mqe",
        "search_papers",
        "deep_search_papers",
    }
)

TOOLS_WITH_TOP_K: frozenset[str] = frozenset(
    {
        "retrieve_evidence",
        "retrieve_with_mqe",
        "answer_with_rag",
    }
)

AGENT_TOOL_ALLOWLIST: Dict[str, Tuple[str, ...]] = {
    "retrieval-agent": (
        *INTERNAL_RETRIEVAL_SERVER.tool_names,
        "search_arxiv",
        "search_pubmed",
        "search_biorxiv",
        "search_google_scholar",
        "download_arxiv",
        "download_pubmed",
        "read_arxiv_paper",
        "read_pubmed_paper",
    ),
    "profile-agent": (
        "search_papers",
        "deep_search_papers",
        "get_paper_profile",
    ),
    "summary-agent": (
        "retrieve_evidence",
        "answer_with_rag",
    ),
    "writing-agent": (
        "retrieve_evidence",
        "answer_with_rag",
    ),
    "chat-agent": (
        "retrieve_evidence",
        "answer_with_rag",
    ),
}


class MCPServerCatalog:
    """MCP Server 元数据目录（供 HTTP / 文档使用）。"""

    def list_servers(self) -> list[dict]:
        return [
            {
                "name": server.name,
                "description": server.description,
                "tool_names": list(server.tool_names),
            }
            for server in ALL_MCP_SERVERS
        ]

    def get(self, name: str) -> Optional[MCPServerConfig]:
        for server in ALL_MCP_SERVERS:
            if server.name == name:
                return server
        return None


mcp_server_registry = MCPServerCatalog()


def get_tool_names_for_agent(agent_id: str) -> Tuple[str, ...]:
    """返回某 Agent 允许绑定的 MCP 工具名列表。"""
    return AGENT_TOOL_ALLOWLIST.get(agent_id, ())
