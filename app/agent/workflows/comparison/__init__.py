"""Plan-driven paper comparison subgraph (Send fan-out + coverage loop)."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agent.workflows.comparison.dispatch import dispatch_evidence_retrieval, dispatch_paper_search
from app.agent.workflows.comparison.nodes import (
    compare_node,
    comparison_entry_node,
    coverage_check_node,
    dispatch_evidence_node,
    dispatch_paper_search_node,
    paper_search_node,
    query_rewrite_node,
    resolve_papers_node,
    retrieve_evidence_node,
)
from app.agent.workflows.comparison.routing import route_after_coverage
from app.agent.state import AgentState


def build_comparison_subgraph():
    """Build and compile the comparison workflow subgraph."""
    builder = StateGraph(AgentState)  # type: ignore[arg-type]

    builder.add_node("comparison_entry", comparison_entry_node)
    builder.add_node("dispatch_paper_search_node", dispatch_paper_search_node)
    builder.add_node("paper_search_node", paper_search_node)
    builder.add_node("resolve_papers_node", resolve_papers_node)
    builder.add_node("dispatch_evidence_node", dispatch_evidence_node)
    builder.add_node("retrieve_evidence_node", retrieve_evidence_node)
    builder.add_node("coverage_check_node", coverage_check_node)
    builder.add_node("query_rewrite_node", query_rewrite_node)
    builder.add_node("compare_node", compare_node)

    builder.set_entry_point("comparison_entry")
    builder.add_edge("comparison_entry", "dispatch_paper_search_node")

    builder.add_conditional_edges(
        "dispatch_paper_search_node", dispatch_paper_search,
        ["paper_search_node", "resolve_papers_node"],
    )
    builder.add_edge("paper_search_node", "resolve_papers_node")

    builder.add_edge("resolve_papers_node", "dispatch_evidence_node")
    builder.add_conditional_edges(
        "dispatch_evidence_node", dispatch_evidence_retrieval,
        ["retrieve_evidence_node", "coverage_check_node"],
    )
    builder.add_edge("retrieve_evidence_node", "coverage_check_node")

    builder.add_conditional_edges(
        "coverage_check_node", route_after_coverage,
        {"compare_node": "compare_node", "query_rewrite_node": "query_rewrite_node"},
    )
    builder.add_edge("query_rewrite_node", "dispatch_evidence_node")
    builder.add_edge("compare_node", END)

    return builder.compile()
