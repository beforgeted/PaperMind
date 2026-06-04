"""主 Agent：负责学术研究意图识别、任务拆解与执行计划生成。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.agents.agent_registry import CANONICAL_AGENT_ORDER


logger = logging.getLogger(__name__)

MAIN_AGENT_ROUTING_PROMPT = """你是 PaperMind 学术研究多 Agent 系统的主 Agent（Intent Router）。

你的职责是理解用户学术研究需求、识别意图、选择子 Agent、规划可执行流水线。

## 可用子 Agent
{available_agents_description}

## 学术研究路由规则

1. 用户输入涉及论文搜索、文献检索、找论文、查找相关研究、证据检索、查询知识库、知识库里有哪些论文/文献 → retrieval-agent
2. 用户输入涉及论文润色、学术改写、同行评审、论文审稿、草稿撰写、学术写作辅助 → writing-agent
3. 用户输入涉及文献综述、研究综述、总结研究现状、梳理领域进展 → summary-agent（需先 retrieval-agent）
4. 用户输入涉及论文画像、这/哪篇论文讲什么、论文发现、论文推荐 → profile-agent
5. 简单问候、闲聊、与学术研究无关的问题 → direct_answer

## 重要：当用户询问知识库中有什么内容、有多少论文等需要访问知识库才能回答的问题时，必须 dispatch 到 retrieval-agent，不要走 direct_answer。

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
  ]
}

## 关键约束
- target_agents 必须是上文列表中的真实 agent_id。
- action=dispatch 时 target_agents 非空，且必须填写 queries_per_agent（每个 id 一条）。
- action=direct_answer 时 target_agents 为空数组。
- route_plan 中 agent_id 须与 target_agents 一致。
- 禁止编造不存在的 Agent、论文或工具结果。
- 子 Agent 已配备真实检索工具（ES），所有论文数据均从知识库获取，无需标注 mock。
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

        common = {
            "needs_clarification": False,
            "detected_entities": {},
            "route_plan": route_plan,
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
