"""PaperMind 学术研究多 Agent 执行的 LangGraph 工作流。"""

from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph

from app.agent.agents.agent_registry import CANONICAL_AGENT_ORDER
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


def _format_upstream_results(agent_results: List[Dict[str, str]]) -> str:
    if not agent_results:
        return "暂无上游 Agent 输出。"

    blocks = []
    for index, result in enumerate(agent_results, 1):
        status = result.get("error") or result.get("content") or "无有效输出"
        blocks.append(
            "\n".join(
                [
                    f"{index}. {result.get('agent_name') or result.get('agent_id')}",
                    f"任务：{result.get('task') or ''}",
                    f"输出：\n{status}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _serial_task_query(
    *,
    query: str,
    task_query: str,
    agent_results: List[Dict[str, str]],
) -> str:
    return "\n\n".join(
        [
            f"用户原始需求：\n{query}",
            f"当前阶段任务：\n{task_query}",
            f"上游阶段输出：\n{_format_upstream_results(agent_results)}",
            "请基于用户需求和上游输出完成本阶段任务；不要重复执行其他阶段职责。",
        ]
    )


def create_paper_mind_graph(
    *,
    orchestrate: OrchestrateFn,
    run_agent: RunAgentFn,
    summarize: SummarizeFn,
):
    graph = StateGraph(PaperMindState)

    async def main_agent_node(state: PaperMindState) -> Dict[str, Any]:
        agents = state.get("agents") or []
        decision = state.get("decision") or await orchestrate(query=state["query"], agents=agents)
        return {
            "decision": decision,
            "selected_agents": _selected_agents(decision),
            "agent_results": state.get("agent_results") or [],
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

        decision = state.get("decision") or {}
        queries_per_agent = decision.get("queries_per_agent") or {}
        default_query = str(decision.get("query_for_agent") or state["query"])
        base_task = str(queries_per_agent.get(agent_id) or default_query)
        task_query = _serial_task_query(
            query=state["query"],
            task_query=base_task,
            agent_results=prior_results,
        )
        result = await run_agent(agent=agent, task_query=task_query)
        update: Dict[str, Any] = {
            "last_agent_id": agent_id,
            "agent_results": [*prior_results, result],
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

        content = await summarize(
            query=state["query"],
            decision=state.get("decision") or {},
            sub_agent_results=agent_results,
        )
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
