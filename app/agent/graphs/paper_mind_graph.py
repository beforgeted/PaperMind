"""PaperMind 学术研究多 Agent 执行的 LangGraph 工作流。"""

from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph

from app.agent.agents.agent_registry import CANONICAL_AGENT_ORDER
from app.agent.context import build_conversation_context
from app.agent.graphs.graph_state import PaperMindState, OrchestrateFn, RunAgentFn, SummarizeFn


MAIN_AGENT_NODE = "main_agent_node"
EXECUTE_SUB_AGENT_NODE = "execute_sub_agent_node"
SUSPEND_NODE = "suspend_node"
SUMMARIZE_NODE = "summarize_node"
DIRECT_ANSWER_NODE = "direct_answer_node"

ORCHESTRATE_NODE = MAIN_AGENT_NODE
NODE_AGENT_IDS: Dict[str, str] = {}


def _resolve_agent(agent_id: str, agents: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for agent in agents:
        if str(agent.get("agent_id")) == agent_id:
            return agent
    return None


def _selected_agents(decision: Dict[str, Any]) -> List[str]:
    target_agents = [str(agent_id) for agent_id in decision.get("target_agents") or []]
    selected = [agent_id for agent_id in CANONICAL_AGENT_ORDER if agent_id in target_agents]
    selected.extend(agent_id for agent_id in target_agents if agent_id not in selected)
    return selected


def _completed_agent_ids(state: PaperMindState) -> set[str]:
    return {
        str(result.get("agent_id") or "")
        for result in state.get("agent_results") or []
        if str(result.get("agent_id") or "")
    }


def _next_agent_id(state: PaperMindState) -> Optional[str]:
    completed = _completed_agent_ids(state)
    for agent_id in state.get("selected_agents") or []:
        if agent_id not in completed:
            return agent_id
    return None


def create_paper_mind_graph(
    *,
    orchestrate: OrchestrateFn,
    run_agent: RunAgentFn,
    summarize: SummarizeFn,
):
    graph = StateGraph(PaperMindState)

    async def main_agent_node(state: PaperMindState) -> Dict[str, Any]:
        agents = state.get("agents") or []
        decision = state.get("decision") or await orchestrate(state=state, agents=agents)
        history = list(state.get("history_messages") or [])
        conversation_context = build_conversation_context(
            query=str(state.get("query") or ""),
            decision=decision,
            history_messages=history,
        )
        return {
            "decision": decision,
            "conversation_context": conversation_context,
            "selected_agents": _selected_agents(decision),
            "agent_results": state.get("agent_results") or [],
            "evidence_packets": state.get("evidence_packets") or [],
            "pending_task_id": state.get("pending_task_id"),
            "pending_step_id": state.get("pending_step_id"),
        }

    def route_after_main_agent(state: PaperMindState) -> str:
        decision = state.get("decision") or {}
        if decision.get("action") != "dispatch":
            return DIRECT_ANSWER_NODE
        if state.get("pending_task_id"):
            return SUSPEND_NODE
        return EXECUTE_SUB_AGENT_NODE if _next_agent_id(state) else SUMMARIZE_NODE

    async def execute_sub_agent_node(state: PaperMindState) -> Dict[str, Any]:
        agent_id = _next_agent_id(state)
        prior_results = state.get("agent_results") or []
        prior_evidence = list(state.get("evidence_packets") or [])
        if not agent_id:
            return {"agent_results": prior_results}

        agents = state.get("agents") or []
        agent = _resolve_agent(agent_id, agents)
        if not agent:
            return {
                "last_agent_id": agent_id,
                "agent_results": [
                    *prior_results,
                    {
                        "agent_id": agent_id,
                        "agent_name": agent_id,
                        "task": "",
                        "content": "",
                        "error": "未找到目标 Agent 配置",
                    },
                ],
            }

        result = await run_agent(agent=agent, agent_id=agent_id, state=state)
        new_evidence = list(result.pop("evidence_packets", None) or [])
        update: Dict[str, Any] = {
            "last_agent_id": agent_id,
            "agent_results": [*prior_results, result],
            "evidence_packets": [*prior_evidence, *new_evidence],
        }
        if result.get("pending_task_id"):
            update["pending_task_id"] = result.get("pending_task_id")
            update["pending_step_id"] = result.get("pending_step_id")
        return update

    def route_after_execute_sub_agent(state: PaperMindState) -> str:
        if state.get("pending_task_id"):
            return SUSPEND_NODE
        return EXECUTE_SUB_AGENT_NODE if _next_agent_id(state) else SUMMARIZE_NODE

    async def suspend_node(state: PaperMindState) -> Dict[str, Any]:
        task_id = str(state.get("pending_task_id") or "")
        return {
            "final_report": f"工作流已挂起，等待外部计算任务完成：{task_id}",
            "pending_task_id": task_id,
            "pending_step_id": state.get("pending_step_id"),
        }

    async def summarize_node(state: PaperMindState) -> Dict[str, Any]:
        agent_results = state.get("agent_results") or []
        if not agent_results:
            decision = state.get("decision") or {}
            return {"final_report": str(decision.get("reason") or "抱歉，本次没有生成有效回答。")}

        content = await summarize(state=state)
        return {"final_report": content or "抱歉，本次没有生成有效回答。"}

    async def direct_answer_node(state: PaperMindState) -> Dict[str, Any]:
        decision = state.get("decision") or {}
        return {"final_report": str(decision.get("reason") or "当前没有匹配的子 Agent。")}

    graph.add_node(MAIN_AGENT_NODE, main_agent_node)
    graph.add_node(EXECUTE_SUB_AGENT_NODE, execute_sub_agent_node)
    graph.add_node(SUSPEND_NODE, suspend_node)
    graph.add_node(SUMMARIZE_NODE, summarize_node)
    graph.add_node(DIRECT_ANSWER_NODE, direct_answer_node)

    graph.add_edge(START, MAIN_AGENT_NODE)
    graph.add_conditional_edges(
        MAIN_AGENT_NODE,
        route_after_main_agent,
        {
            EXECUTE_SUB_AGENT_NODE: EXECUTE_SUB_AGENT_NODE,
            SUSPEND_NODE: SUSPEND_NODE,
            SUMMARIZE_NODE: SUMMARIZE_NODE,
            DIRECT_ANSWER_NODE: DIRECT_ANSWER_NODE,
        },
    )
    graph.add_conditional_edges(
        EXECUTE_SUB_AGENT_NODE,
        route_after_execute_sub_agent,
        {
            EXECUTE_SUB_AGENT_NODE: EXECUTE_SUB_AGENT_NODE,
            SUSPEND_NODE: SUSPEND_NODE,
            SUMMARIZE_NODE: SUMMARIZE_NODE,
        },
    )
    graph.add_edge(SUSPEND_NODE, END)
    graph.add_edge(SUMMARIZE_NODE, END)
    graph.add_edge(DIRECT_ANSWER_NODE, END)

    return graph.compile()


__all__ = [
    "CANONICAL_AGENT_ORDER",
    "DIRECT_ANSWER_NODE",
    "EXECUTE_SUB_AGENT_NODE",
    "MAIN_AGENT_NODE",
    "NODE_AGENT_IDS",
    "ORCHESTRATE_NODE",
    "SUMMARIZE_NODE",
    "SUSPEND_NODE",
    "PaperMindState",
    "create_paper_mind_graph",
]
