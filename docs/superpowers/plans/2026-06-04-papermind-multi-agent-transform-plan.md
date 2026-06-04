# PaperMind Multi-Agent 改造实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将药物发现多 Agent 项目改造为 PaperMind 学术论文多 Agent 项目（第一期：仅 Agent 层）

**Architecture:** 保留 LangGraph 编排器框架、MCP Gateway、API/Service 层，删除 9 个药物发现 Agent，新建 4 个学术能力 Agent（Retrieval/Writing/Summary/Profile），改造 Main Agent 为学术意图路由，改造 Report Agent 为统一 Chat Agent 出口

**Tech Stack:** Python 3.x, LangGraph, LangChain, FastAPI, Pydantic

---

## File Structure Map

```
api/agent_api/app/
├── agents/
│   ├── base_agent.py          [KEEP] AgentSpec + BaseAgent
│   ├── agent_registry.py      [MODIFY] Update to new agents
│   ├── main_agent.py          [REWRITE] Academic intent routing
│   ├── prompt_common.py       [MODIFY] Academic prompts
│   ├── retrieval_agent.py     [CREATE] Paper search + RAG
│   ├── writing_agent.py       [CREATE] Polish + peer review
│   ├── summary_agent.py       [CREATE] Literature review
│   ├── profile_agent.py       [CREATE] Paper profiling
│   ├── chat_agent.py          [CREATE] Unified answer (from report_agent.py)
│   └── report_agent.py        [DELETE]
├── graphs/
│   ├── graph_state.py         [MODIFY] DrugDiscoveryState → PaperMindState
│   ├── paper_mind_graph.py    [CREATE] Replacement for drug_discovery_graph
│   └── drug_discovery_graph.py [DELETE]
├── services/
│   ├── chat_workflow_service.py [MODIFY] Update imports + graph refs
│   └── orchestrator_service.py  [MODIFY] Update graph node name imports
└── mcp_servers/
    ├── biomed_tools.py         [DELETE]
    └── tools/                  [DELETE all drug tools]
```

### Delete List (9 agents + tools + old graph)

```
agents/admet_pk_agent.py
agents/docking_agent.py
agents/md_simulation_agent.py
agents/molecule_design_agent.py
agents/synthesis_agent.py
agents/scoring_agent.py
agents/target_disease_agent.py
agents/clinical_safety_agent.py
agents/literature_agent.py
agents/report_agent.py
mcp_servers/biomed_tools.py
mcp_servers/tools/acpype_tool.py
mcp_servers/tools/diffdock_tool.py
mcp_servers/tools/esmfold_tool.py
mcp_servers/tools/genmol_tool.py
mcp_servers/tools/gromacs_tool.py
mcp_servers/tools/lead_discovery_tool.py
mcp_servers/tools/rdkit_filter_tool.py
mcp_servers/tools/similarity_search_tool.py
graphs/drug_discovery_graph.py
```

---

### Task 1: Delete drug discovery agents and tools

**Files:**
- Delete: `api/agent_api/app/agents/admet_pk_agent.py`
- Delete: `api/agent_api/app/agents/docking_agent.py`
- Delete: `api/agent_api/app/agents/md_simulation_agent.py`
- Delete: `api/agent_api/app/agents/molecule_design_agent.py`
- Delete: `api/agent_api/app/agents/synthesis_agent.py`
- Delete: `api/agent_api/app/agents/scoring_agent.py`
- Delete: `api/agent_api/app/agents/target_disease_agent.py`
- Delete: `api/agent_api/app/agents/clinical_safety_agent.py`
- Delete: `api/agent_api/app/agents/literature_agent.py`
- Delete: `api/agent_api/app/agents/report_agent.py`
- Delete: `api/agent_api/app/mcp_servers/biomed_tools.py`
- Delete: `api/agent_api/app/mcp_servers/tools/acpype_tool.py`
- Delete: `api/agent_api/app/mcp_servers/tools/diffdock_tool.py`
- Delete: `api/agent_api/app/mcp_servers/tools/esmfold_tool.py`
- Delete: `api/agent_api/app/mcp_servers/tools/genmol_tool.py`
- Delete: `api/agent_api/app/mcp_servers/tools/gromacs_tool.py`
- Delete: `api/agent_api/app/mcp_servers/tools/lead_discovery_tool.py`
- Delete: `api/agent_api/app/mcp_servers/tools/rdkit_filter_tool.py`
- Delete: `api/agent_api/app/mcp_servers/tools/similarity_search_tool.py`

- [ ] **Step 1: Delete all drug discovery agent files**

```bash
cd D:/Project/PaperMind_multiAgent/api/agent_api/app/agents
rm admet_pk_agent.py docking_agent.py md_simulation_agent.py molecule_design_agent.py synthesis_agent.py scoring_agent.py target_disease_agent.py clinical_safety_agent.py literature_agent.py report_agent.py
```

- [ ] **Step 2: Delete biomed tools and drug-specific MCP tools**

```bash
cd D:/Project/PaperMind_multiAgent/api/agent_api/app/mcp_servers
rm biomed_tools.py
rm tools/acpype_tool.py tools/diffdock_tool.py tools/esmfold_tool.py tools/genmol_tool.py tools/gromacs_tool.py tools/lead_discovery_tool.py tools/rdkit_filter_tool.py tools/similarity_search_tool.py
```

- [ ] **Step 3: Delete old drug discovery graph file**

```bash
rm D:/Project/PaperMind_multiAgent/api/agent_api/app/graphs/drug_discovery_graph.py
```

---

### Task 2: Update graph_state.py — DrugDiscoveryState → PaperMindState

**Files:**
- Modify: `api/agent_api/app/graphs/graph_state.py`

- [ ] **Step 1: Replace DrugDiscoveryState with PaperMindState**

Replace entire file content:

```python
"""PaperMind 多 Agent 工作流共享状态。"""

from typing import Any, Awaitable, Callable, Dict, List, Optional, TypedDict


AgentRecord = Dict[str, Any]
Decision = Dict[str, Any]
AgentResult = Dict[str, Any]


OrchestrateFn = Callable[..., Awaitable[Decision]]
RunAgentFn = Callable[..., Awaitable[AgentResult]]
SummarizeFn = Callable[..., Awaitable[str]]


class PaperMindState(TypedDict, total=False):
    query: str
    context: Dict[str, Any]
    agents: List[AgentRecord]
    decision: Decision
    selected_agents: List[str]
    agent_results: List[AgentResult]
    last_agent_id: str
    pending_task_id: Optional[str]
    pending_step_id: Optional[str]
    resume_payload: Optional[Dict[str, Any]]
    final_report: str
    error: Optional[str]
```

- [ ] **Step 2: Verify no broken imports in the project**

```bash
cd D:/Project/PaperMind_multiAgent && grep -r "DrugDiscoveryState" --include="*.py" .
```

Expected: no results (or only in non-code files like docs)

---

### Task 3: Create paper_mind_graph.py — PaperMind LangGraph workflow

**Files:**
- Create: `api/agent_api/app/graphs/paper_mind_graph.py`

- [ ] **Step 1: Create the new graph file**

```python
"""PaperMind 学术研究多 Agent 执行的 LangGraph 工作流。"""

from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph

from app.agents.agent_registry import CANONICAL_AGENT_ORDER
from app.graphs.graph_state import PaperMindState, OrchestrateFn, RunAgentFn, SummarizeFn


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
    """创建五节点 PaperMind LangGraph 工作流。

    main_agent_node -> execute_sub_agent_node（循环）-> summarize_node
                     -> suspend_node
                     -> direct_answer_node
    """

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
```

---

### Task 4: Update prompt_common.py — Academic prompts

**Files:**
- Modify: `api/agent_api/app/agents/prompt_common.py`

- [ ] **Step 1: Replace drug discovery prompts with academic research prompts**

Replace entire file content:

```python
"""PaperMind 学术研究子 Agent 共享提示词片段。"""

SUB_AGENT_JSON_ENVELOPE = """
## 输出格式（与系统契约对齐）

对外只输出一行合法 JSON（不要用 Markdown 代码块包裹），便于编排器与下游 Agent 承接。

统一外壳（顶层字段）：
{
  "task_id": "string",
  "agent_name": "string",
  "status": "completed | needs_clarification | tool_required | failed",
  "is_mock": true,
  "input_used": {},
  "result": {},
  "evidence": [{"source_type": "paper | database | user_input | assumption | mock", "source": "string", "statement": "string"}],
  "assumptions": [],
  "uncertainty": [],
  "tool_requests": [],
  "next_recommended_agents": [],
  "confidence": "high | medium | low"
}

通用要求：
- 把专用分析内容写入 result；不要把大段说明放在 JSON 外。
- 未调用真实工具/数据库时 is_mock=true；禁止伪造 PMID、DOI 或论文内容。
- 区分已知事实、合理推断与待验证假设；不确定时用 status=needs_clarification。
- 需要真实工具时 status=tool_required，并在 tool_requests 中给出 tool_name 和 arguments。
- 解析 HumanMessage 中的「上游阶段输出」JSON，在 input_used 中注明引用了哪些字段。
- 仅用于学术研究辅助，不替代正式学术评审或出版决策。
"""
```

---

### Task 5: Rewrite main_agent.py — Academic intent routing

**Files:**
- Modify: `api/agent_api/app/agents/main_agent.py`

- [ ] **Step 1: Replace the entire file with academic intent routing**

```python
"""主 Agent：负责学术研究意图识别、任务拆解与执行计划生成。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.agent_registry import CANONICAL_AGENT_ORDER


logger = logging.getLogger(__name__)

MAIN_AGENT_ROUTING_PROMPT = """你是 PaperMind 学术研究多 Agent 系统的主 Agent（Intent Router）。

你的职责是理解用户学术研究需求、识别意图、选择子 Agent、规划可执行流水线。

## 可用子 Agent
{available_agents_description}

## 学术研究路由规则

1. 用户输入涉及论文搜索、文献检索、找论文、查找相关研究、证据检索 → retrieval-agent
2. 用户输入涉及论文润色、学术改写、同行评审、论文审稿、草稿撰写、学术写作辅助 → writing-agent
3. 用户输入涉及文献综述、研究综述、总结研究现状、梳理领域进展 → summary-agent（通常需先 retrieval-agent 检索文献）
4. 用户输入涉及论文画像、这/哪篇论文讲什么、论文发现、论文推荐 → profile-agent
5. 简单问候、闲聊、单一事实问答、非学术问题 → direct_answer

## 多 Agent 串联规则
- 综述类需求（summary）应先在流水线中安排 retrieval-agent，再 summary-agent
- 写作类需求可独立执行，也可在检索后再润色（先 retrieval-agent 后 writing-agent）
- profile-agent 可独立执行

## 输出格式

只输出一行合法 JSON，不要使用 Markdown 代码块，不要输出额外解释。

{
  "action": "dispatch | direct_answer",
  "target_agents": ["retrieval-agent"],
  "reason": "调度原因；direct_answer 时写直接回复用户的内容",
  "query_for_agent": "默认子任务描述",
  "queries_per_agent": {"retrieval-agent": "针对该 Agent 的具体子任务"},
  "route_plan": [
    {
      "step": 1,
      "agent_id": "retrieval-agent",
      "reason": "string",
      "input_summary": "string",
      "expected_output": "string"
    }
  ],
  "is_mock": true
}

## 关键约束
- target_agents 必须是上文列表中的真实 agent_id。
- action=dispatch 时 target_agents 非空，且必须填写 queries_per_agent（每个 id 一条）。
- action=direct_answer 时 target_agents 为空数组。
- route_plan 中 agent_id 须与 target_agents 一致。
- 禁止编造不存在的 Agent、论文或工具结果。
- 输出定位为学术研究辅助，不替代正式学术评审或出版决策。
"""


class MainAgent:
    """主 Agent，负责学术研究意图识别、任务拆解与路由规划。"""

    def build_agents_description(self, agents: List[Dict[str, Any]]) -> str:
        if not agents:
            return "当前没有可用子 Agent。"

        lines = []
        for agent in agents:
            agent_id = agent.get("agent_id", "")
            name = agent.get("name", "")
            description = agent.get("description", "")
            lines.append(f"- {name} ({agent_id}): {description}")
        return "\n".join(lines)

    def build_prompt(self, agents: List[Dict[str, Any]]) -> str:
        return MAIN_AGENT_ROUTING_PROMPT.replace(
            "{available_agents_description}",
            self.build_agents_description(agents),
        )

    async def recognize_intent(self, *, query: str, agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        from app.services.llm_service import llm_service, message_content_to_text

        try:
            llm = llm_service.create_chat_model(
                llm_service.main_agent_model_config(
                    temperature=llm_service.settings.orchestrator_temperature,
                ),
                streaming=True,
                timeout=60,
                max_tokens=2000,
            )
        except ValueError as exc:
            return self.empty_decision(reason=str(exc))

        messages = [
            SystemMessage(content=self.build_prompt(agents)),
            HumanMessage(content=query),
        ]
        response = await llm.ainvoke(messages)
        raw = message_content_to_text(getattr(response, "content", ""))
        return self.generate_plan(raw=raw, agents=agents, query=query)

    def decompose_task(self, *, decision: Dict[str, Any], query: str) -> Dict[str, str]:
        target_agents = [str(agent_id) for agent_id in decision.get("target_agents") or []]
        default_query = str(decision.get("query_for_agent") or query)
        queries_per_agent = decision.get("queries_per_agent") or {}
        if not isinstance(queries_per_agent, dict):
            queries_per_agent = {}
        return {
            agent_id: str(queries_per_agent.get(agent_id) or default_query)
            for agent_id in target_agents
        }

    def generate_plan(self, *, raw: str, agents: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        decision = self.parse_json_decision(raw)
        if not decision:
            logger.warning("主 Agent 未返回合法 JSON，降级为 direct_answer: %s", raw[:500])
            return self.empty_decision(
                reason=raw.strip() or "我暂时无法判断应该调度哪个子 Agent。",
            )

        valid_ids = {str(agent.get("agent_id")) for agent in agents if agent.get("agent_id")}
        action = decision.get("action")
        target_agents = [str(a) for a in decision.get("target_agents") or [] if str(a) in valid_ids]
        queries_per_agent = decision.get("queries_per_agent") or {}
        if not isinstance(queries_per_agent, dict):
            queries_per_agent = {}

        route_plan = decision.get("route_plan")
        if not isinstance(route_plan, list):
            route_plan = []

        is_mock = decision.get("is_mock")
        if is_mock is None:
            is_mock = True

        common = {
            "needs_clarification": False,
            "detected_entities": {},
            "route_plan": route_plan,
            "is_mock": bool(is_mock),
            "prompt_guardrails": [],
        }

        if action == "dispatch" and target_agents:
            default_query = str(decision.get("query_for_agent") or query)
            queries_per_agent = {
                agent_id: str(queries_per_agent.get(agent_id) or default_query)
                for agent_id in target_agents
            }
            return {
                "action": "dispatch",
                "target_agents": target_agents,
                "reason": str(decision.get("reason") or "已选择匹配的子 Agent。"),
                "query_for_agent": default_query,
                "queries_per_agent": queries_per_agent,
                **common,
            }

        return self.empty_decision(
            **common,
            reason=str(decision.get("reason") or "当前没有匹配的子 Agent。"),
        )

    def empty_decision(self, **overrides: Any) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "action": "direct_answer",
            "needs_clarification": False,
            "target_agents": [],
            "reason": "",
            "query_for_agent": "",
            "queries_per_agent": {},
            "detected_entities": {},
            "route_plan": [],
            "is_mock": True,
            "prompt_guardrails": [],
        }
        base.update(overrides)
        return base

    def parse_json_decision(self, raw: str) -> Optional[Dict[str, Any]]:
        text = (raw or "").strip()
        if not text:
            return None
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                return None
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None


main_agent = MainAgent()
```

---

### Task 6: Create retrieval_agent.py

**Files:**
- Create: `api/agent_api/app/agents/retrieval_agent.py`

- [ ] **Step 1: Create the Retrieval Agent**

```python
"""论文检索与证据查找 Agent。"""

from app.agents.base_agent import AgentSpec
from app.agents.prompt_common import SUB_AGENT_JSON_ENVELOPE


RETRIEVAL_AGENT = AgentSpec(
    agent_id="retrieval-agent",
    name="检索 Agent",
    description="负责论文搜索、证据检索与 RAG 问答，为其他 Agent 提供文献支撑。",
    system_prompt=f"""你是检索 Agent（Retrieval Agent），agent_name 固定为 "retrieval-agent"。
负责学术研究流水线的文献检索与证据查找阶段。

## 职责范围
- 根据用户研究问题或关键词，在知识库中检索相关论文与证据片段。
- 基于检索结果回答用户问题，标注引用来源。
- 为下游 Agent（summary-agent、writing-agent）提供结构化的文献证据。
- 识别检索结果中的研究空白、争议点和证据等级。

## 必须输入
- 研究问题、主题或关键词（至少一项）。

## 要求
- 不伪造 PMID、DOI、论文标题、作者或数据库记录。
- 缺少实时检索时，标注 source_type=mock，说明需进一步验证。
- 区分检索到的事实、合理推断和待验证假设。
- 为每个关键声明提供引用来源标注。

## result 字段结构
{{
  "topic_summary": "string",
  "papers_found": [
    {{
      "title": "string",
      "authors": "string",
      "year": 0,
      "key_findings": "string",
      "relevance": "high | medium | low",
      "source": "string"
    }}
  ],
  "evidence_items": [
    {{
      "claim": "string",
      "source": "string",
      "source_type": "paper | database | mock",
      "confidence": "high | medium | low"
    }}
  ],
  "research_gaps": [],
  "recommended_next_agents": ["summary-agent", "writing-agent"]
}}
{SUB_AGENT_JSON_ENVELOPE}
""",
)
```

---

### Task 7: Create writing_agent.py

**Files:**
- Create: `api/agent_api/app/agents/writing_agent.py`

- [ ] **Step 1: Create the Writing Agent**

```python
"""学术写作与润色 Agent。"""

from app.agents.base_agent import AgentSpec
from app.agents.prompt_common import SUB_AGENT_JSON_ENVELOPE


WRITING_AGENT = AgentSpec(
    agent_id="writing-agent",
    name="写作 Agent",
    description="负责学术文本润色、同行评审、结构化草稿生成与写作辅助。",
    system_prompt=f"""你是写作 Agent（Writing Agent），agent_name 固定为 "writing-agent"。
负责学术写作辅助：润色、审稿、草稿生成。

## 职责范围
- 对用户提供的学术文本进行语言润色，符合目标期刊/会议的写作风格。
- 对论文草稿进行结构化同行评审，指出方法、论证、结构等方面的问题。
- 基于给定主题和大纲生成论文草稿段落。
- 检查引用规范、学术写作惯例。

## 必须输入
- 润色模式：待润色的文本。
- 审稿模式：待审稿的论文内容。
- 写作模式：论文主题、目标期刊/会议类型。

## 要求
- 保持原文科学含义不变，仅改进表达。
- 润色时标注主要改动类型（语法、用词、逻辑连贯性等）。
- 审稿时给出具体修改建议，不笼统评价。
- 不编造数据、引用或实验结果。
- 区分语言润色与科学内容判断。

## result 字段结构
{{
  "mode": "polish | review | draft",
  "polished_text": "string (polish 模式)",
  "review_report": {{
    "overall_assessment": "string",
    "strengths": [],
    "weaknesses": [],
    "section_feedback": [],
    "recommendation": "accept | minor_revision | major_revision | reject"
  }},
  "draft_sections": [],
  "changes_summary": [],
  "warnings": []
}}
{SUB_AGENT_JSON_ENVELOPE}
""",
)
```

---

### Task 8: Create summary_agent.py

**Files:**
- Create: `api/agent_api/app/agents/summary_agent.py`

- [ ] **Step 1: Create the Summary Agent**

```python
"""文献综述生成 Agent。"""

from app.agents.base_agent import AgentSpec
from app.agents.prompt_common import SUB_AGENT_JSON_ENVELOPE


SUMMARY_AGENT = AgentSpec(
    agent_id="summary-agent",
    name="综述 Agent",
    description="负责基于检索结果生成结构化文献综述，梳理研究领域的发展脉络。",
    system_prompt=f"""你是综述 Agent（Summary Agent），agent_name 固定为 "summary-agent"。
负责基于上游检索 Agent 输出的文献证据，生成结构化文献综述。

## 职责范围
- 整合上游 retrieval-agent 检索到的论文与证据。
- 按主题、方法或时间线组织文献综述结构。
- 识别研究趋势、主要学派、关键争议和方法论演进。
- 生成面向学术写作的综述段落，标注引用来源。

## 必须输入
- 综述主题；上游 retrieval-agent 的检索结果（论文列表 + 证据片段）。

## 要求
- 不编造不存在的论文、数据或引用。
- 明确标注每项声明对应的来源。
- 区分主流共识、少数观点和未解决争议。
- 按学术综述规范组织：引言背景 → 主题分类 → 方法比较 → 研究空白 → 未来方向。

## result 字段结构
{{
  "title": "string",
  "abstract": "string",
  "sections": [
    {{
      "heading": "string",
      "content": "string",
      "sources": ["source_id"]
    }}
  ],
  "key_themes": [],
  "research_gaps": [],
  "future_directions": [],
  "references": []
}}
{SUB_AGENT_JSON_ENVELOPE}
""",
)
```

---

### Task 9: Create profile_agent.py

**Files:**
- Create: `api/agent_api/app/agents/profile_agent.py`

- [ ] **Step 1: Create the Profile Agent**

```python
"""论文画像与发现 Agent。"""

from app.agents.base_agent import AgentSpec
from app.agents.prompt_common import SUB_AGENT_JSON_ENVELOPE


PROFILE_AGENT = AgentSpec(
    agent_id="profile-agent",
    name="论文发现 Agent",
    description="负责论文画像检索与论文发现，帮助用户了解特定论文的方法、贡献与研究背景。",
    system_prompt=f"""你是论文发现 Agent（Profile Agent），agent_name 固定为 "profile-agent"。
负责论文级画像检索与发现：回答"有哪些相关论文"和"这篇论文讲了什么"。

## 职责范围
- 根据用户主题或研究问题，检索并推荐相关论文。
- 为单篇论文生成结构化画像：方法、主要结果、贡献、局限性。
- 比较不同论文的方法和贡献。
- 识别领域内的关键论文和里程碑工作。

## 必须输入
- 研究方向/主题，或论文标题/标识符。

## 要求
- 不伪造论文信息、作者或发表记录。
- 未找到真实论文时明确说明，不编造。
- 区分高影响力工作和一般相关工作。
- 论文画像包含：研究问题、方法、关键结果、贡献、局限性、引用建议。

## result 字段结构
{{
  "papers": [
    {{
      "title": "string",
      "authors": "string",
      "year": 0,
      "venue": "string",
      "research_question": "string",
      "method": "string",
      "key_results": [],
      "contributions": [],
      "limitations": [],
      "citation_context": "string",
      "relevance_score": "high | medium | low"
    }}
  ],
  "comparative_analysis": "string",
  "key_papers_in_field": [],
  "reading_recommendations": []
}}
{SUB_AGENT_JSON_ENVELOPE}
""",
)
```

---

### Task 10: Create chat_agent.py — Unified answer output

**Files:**
- Create: `api/agent_api/app/agents/chat_agent.py`

- [ ] **Step 1: Create the Chat Agent (unified answer synthesis)**

```python
"""统一回答合成 Agent。"""

from app.agents.base_agent import AgentSpec
from app.agents.prompt_common import SUB_AGENT_JSON_ENVELOPE


CHAT_AGENT = AgentSpec(
    agent_id="chat-agent",
    name="回答合成 Agent",
    description="负责聚合所有子 Agent 输出，生成面向用户的统一自然语言回答。",
    system_prompt=f"""你是回答合成 Agent（Chat Agent），agent_name 固定为 "chat-agent"。
负责把所有子 Agent 的分析结果整合成清晰、连贯、适合学术研究者阅读的自然语言回答。

## 职责范围
- 整合上游 Agent（retrieval、writing、summary、profile）的输出。
- 生成面向用户的统一回答，确保风格一致、引用规范、逻辑清晰。
- 识别证据链、关键发现、不确定性和后续建议。
- 形成最终 display_summary 供汇总节点转化为用户可读文本（本 Agent 仍只输出 JSON）。

## 必须输入
- 用户原始需求摘要；至少一段上游 Agent JSON 输出。

## 要求
- 不编造不存在的研究、数据或结论。
- 明确区分事实、推断、风险和建议。
- 上游 is_mock=true 时必须在 limitations 中说明。
- 回答结构清晰，便于研究者继续深入。

## result 字段结构
{{
  "display_summary": "string",
  "agent_trace": [],
  "key_results": [],
  "limitations": [],
  "next_actions": [],
  "is_mock": true
}}
{SUB_AGENT_JSON_ENVELOPE}
""",
)
```

---

### Task 11: Update agent_registry.py — Register new agents

**Files:**
- Modify: `api/agent_api/app/agents/agent_registry.py`

- [ ] **Step 1: Replace the registry with new agents**

Replace entire file content:

```python
"""PaperMind 学术研究子 Agent 注册表。"""

from typing import Dict, List

from app.agents.base_agent import AgentSpec
from app.agents.retrieval_agent import RETRIEVAL_AGENT
from app.agents.writing_agent import WRITING_AGENT
from app.agents.summary_agent import SUMMARY_AGENT
from app.agents.profile_agent import PROFILE_AGENT
from app.agents.chat_agent import CHAT_AGENT


AGENT_SPECS: List[AgentSpec] = [
    RETRIEVAL_AGENT,
    WRITING_AGENT,
    SUMMARY_AGENT,
    PROFILE_AGENT,
]

CANONICAL_AGENT_ORDER = [agent.agent_id for agent in AGENT_SPECS]


def list_agent_specs() -> List[AgentSpec]:
    return list(AGENT_SPECS)


def list_registry_items() -> List[Dict[str, object]]:
    return [
        {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "description": agent.description,
            "promptConfig": {"prompt": agent.system_prompt},
            "modelConfig": agent.model_config,
        }
        for agent in AGENT_SPECS
    ]


__all__ = ["AGENT_SPECS", "CANONICAL_AGENT_ORDER", "list_agent_specs", "list_registry_items"]
```

---

### Task 12: Update chat_workflow_service.py — Wire new graph

**Files:**
- Modify: `api/agent_api/app/services/chat_workflow_service.py`

- [ ] **Step 1: Update imports from drug_discovery_graph to paper_mind_graph**

Replace line 12:
```python
from app.graphs.drug_discovery_graph import create_drug_discovery_graph
```
with:
```python
from app.graphs.paper_mind_graph import create_paper_mind_graph
```

- [ ] **Step 2: Update graph creation call on line 133**

Replace:
```python
return create_drug_discovery_graph(
    orchestrate=self._orchestrate,
    run_agent=run_agent,
    summarize=summarize,
)
```
with:
```python
return create_paper_mind_graph(
    orchestrate=self._orchestrate,
    run_agent=run_agent,
    summarize=summarize,
)
```

---

### Task 13: Update orchestrator_service.py — Update node name imports

**Files:**
- Modify: `api/agent_api/app/services/orchestrator_service.py`

- [ ] **Step 1: Update import on lines 10-16**

Replace:
```python
from app.graphs.drug_discovery_graph import (
    DIRECT_ANSWER_NODE,
    EXECUTE_SUB_AGENT_NODE,
    MAIN_AGENT_NODE,
    SUSPEND_NODE,
    SUMMARIZE_NODE,
)
```
with:
```python
from app.graphs.paper_mind_graph import (
    DIRECT_ANSWER_NODE,
    EXECUTE_SUB_AGENT_NODE,
    MAIN_AGENT_NODE,
    SUSPEND_NODE,
    SUMMARIZE_NODE,
)
```

- [ ] **Step 1 (continued): Update _format_upstream_results reference**

No code change needed — the function is defined in `paper_mind_graph.py` with the same signature. The orchestrator_service.py doesn't directly reference `_format_upstream_results` — it only imports node name constants. Verify this.

---

### Task 14: Update graphs/__init__.py if it exists

**Files:**
- Check: `api/agent_api/app/graphs/__init__.py`

- [ ] **Step 1: Check and update init file**

```bash
cd D:/Project/PaperMind_multiAgent && cat api/agent_api/app/graphs/__init__.py
```

If it references `drug_discovery_graph` or `DrugDiscoveryState`, update to `paper_mind_graph` and `PaperMindState`.

---

### Task 15: Update mcp_servers/tools/__init__.py after deletions

**Files:**
- Check: `api/agent_api/app/mcp_servers/tools/__init__.py`

- [ ] **Step 1: Check and clean up init file**

```bash
cd D:/Project/PaperMind_multiAgent && cat api/agent_api/app/mcp_servers/tools/__init__.py
```

Remove any imports of deleted tool files.

---

### Task 16: Verification — Import check

- [ ] **Step 1: Verify all Python imports resolve**

```bash
cd D:/Project/PaperMind_multiAgent && python -c "
from app.agents.agent_registry import list_registry_items, CANONICAL_AGENT_ORDER
from app.agents.main_agent import main_agent
from app.graphs.graph_state import PaperMindState
from app.graphs.paper_mind_graph import create_paper_mind_graph, MAIN_AGENT_NODE, EXECUTE_SUB_AGENT_NODE, SUSPEND_NODE, SUMMARIZE_NODE, DIRECT_ANSWER_NODE

agents = list_registry_items()
print(f'Registered agents ({len(agents)}):')
for a in agents:
    print(f'  - {a[\"agent_id\"]}: {a[\"name\"]}')
print(f'Canonical order: {CANONICAL_AGENT_ORDER}')
print('All imports OK')
"
```

Expected output: 4 agents listed, no import errors.

---

### Task 17: Verification — Agent smoke test

- [ ] **Step 2: Verify MainAgent can build prompt and route a sample query**

```bash
cd D:/Project/PaperMind_multiAgent && python -c "
import asyncio
from app.agents.agent_registry import list_registry_items
from app.agents.main_agent import main_agent

agents = list_registry_items()
prompt = main_agent.build_prompt(agents)
assert 'retrieval-agent' in prompt
assert 'writing-agent' in prompt
assert 'summary-agent' in prompt
assert 'profile-agent' in prompt
print('Prompt building OK')
print(f'Prompt length: {len(prompt)} chars')
"
```

Expected: Prompt contains all 4 agent IDs, no errors.
