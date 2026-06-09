"""GraphState → LLM messages 的上下文装配器。"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.observability.events import CONTEXT_BUILT

logger = logging.getLogger(__name__)

from app.agent.context.formatters import (
    build_sub_agent_human_content,
    build_summary_user_content,
)
from app.agent.context.policies import (
    DEFAULT_SUMMARY_POLICY,
    merge_agent_policy,
)
from app.agent.context.types import (
    ROUTER_HISTORY_TURNS,
    AgentContextPolicy,
    ContextStage,
    ConversationContext,
    SummaryContextPolicy,
)
from app.agent.graphs.graph_state import PaperMindState
from app.services.memory.message_utils import history_to_langchain_messages


SUB_AGENT_DEFAULT_PROMPT = """你是"{agent_name}"。

职责范围：
{agent_description}

## 工具使用规则

你有可用的检索工具。对于需要事实性信息的问题，你必须调用工具获取真实数据，
不得编造或猜测论文内容、作者、发表年份等具体信息。
如果工具返回了结果，请基于工具返回的真实数据回答用户。
如果工具调用失败或未找到数据，如实说明，不要编造。

请只处理主 Agent 分配给你的任务。回答要清晰、直接。
"""

TOOL_SYSTEM_PROMPT_SUFFIX = """
## 关键规则

1. **必须使用工具**: 当问题涉及论文检索、文献证据、论文信息时，必须调用对应工具获取真实数据
2. **禁止编造**: 不得伪造论文标题、作者、年份、PMID、DOI 或任何研究结果
3. **引用来源**: 基于工具返回的真实数据回答，标注来源
4. **工具失败时如实说明**: 如果工具返回空结果或失败，直接告知用户，不要编造
5. **格式化输出**: 使用 Markdown 组织回答
6. **直接给答案**: 不要输出你的思考过程、分析步骤或"本阶段任务""职责履行完毕"等元描述。只输出用户需要的最终答案
"""

RETRIEVAL_TOOL_NAMES = frozenset(
    {
        "search_papers",
        "deep_search_papers",
        "retrieve_evidence",
        "retrieve_with_mqe",
        "answer_with_rag",
        "get_paper_profile",
    }
)


def build_conversation_context(
    *,
    query: str,
    decision: dict[str, Any],
    history_messages: list[dict[str, str]],
) -> ConversationContext:
    """从主 Agent decision 派生 conversation_context。"""
    standalone = str(decision.get("standalone_query") or query).strip() or query
    resolved = decision.get("resolved_entities") or {}
    if not isinstance(resolved, dict):
        resolved = {}
    requirements = decision.get("context_requirements") or {}
    if not isinstance(requirements, dict):
        requirements = {}

    return ConversationContext(
        standalone_query=standalone,
        depends_on_history=bool(history_messages),
        resolved_entities={str(k): str(v) for k, v in resolved.items()},
        context_requirements=requirements,
    )


def extract_evidence_packet(
    *,
    tool_name: str,
    content: str,
    agent_id: str,
) -> dict[str, Any] | None:
    """从检索类工具返回中提取 evidence packet。"""
    if tool_name not in RETRIEVAL_TOOL_NAMES:
        return None
    text = (content or "").strip()
    if not text:
        return None
    return {
        "tool_name": tool_name,
        "agent_id": agent_id,
        "content": text[:4000],
    }


class ContextBuilder:
    """根据 GraphState 与阶段策略现场组装 LLM messages。"""

    def build(
        self,
        stage: ContextStage,
        state: PaperMindState,
        **kwargs: Any,
    ) -> list[BaseMessage]:
        if stage == ContextStage.ROUTER:
            return self.build_for_router(state, agents=kwargs.get("agents") or [])
        if stage == ContextStage.SUB_AGENT:
            return self.build_for_sub_agent(
                state,
                agent_id=str(kwargs.get("agent_id") or ""),
                agent=kwargs.get("agent") or {},
            )
        if stage == ContextStage.SUMMARY:
            return self.build_for_summary(state)
        raise ValueError(f"未知 ContextStage: {stage}")

    def build_for_router(
        self,
        state: PaperMindState,
        *,
        agents: list[dict[str, Any]],
    ) -> list[BaseMessage]:
        """路由主 Agent：System + 近 10 轮历史 + 当前问题。"""
        from app.agent.agents.main_agent import main_agent

        messages: list[BaseMessage] = [
            SystemMessage(content=main_agent.build_prompt(agents)),
        ]
        history = self._history_slice(state, ROUTER_HISTORY_TURNS)
        if history:
            messages.extend(history_to_langchain_messages(history))
        messages.append(HumanMessage(content=str(state.get("query") or "")))
        logger.debug(
            "context_built",
            extra={
                "event": CONTEXT_BUILT,
                "stage": "router",
                "message_count": len(messages),
                "history_turns": ROUTER_HISTORY_TURNS,
            },
        )
        return messages

    def build_for_sub_agent(
        self,
        state: PaperMindState,
        *,
        agent_id: str,
        agent: dict[str, Any],
    ) -> list[BaseMessage]:
        """子 Agent：System + 策略裁剪历史 + 格式化任务 Human。"""
        conv = state.get("conversation_context") or {}
        requirements = conv.get("context_requirements") or {}
        override = requirements.get(agent_id) if isinstance(requirements, dict) else None
        if not isinstance(override, dict):
            override = {}
        policy = merge_agent_policy(agent_id, override)

        messages: list[BaseMessage] = [
            SystemMessage(content=self._sub_agent_system_prompt(agent, state)),
        ]
        history = self._history_slice(state, policy.history_turns)
        if history:
            messages.extend(history_to_langchain_messages(history))

        human_content = self._build_sub_agent_human(state, agent_id, agent, policy)
        messages.append(HumanMessage(content=human_content))
        logger.debug(
            "context_built",
            extra={
                "event": CONTEXT_BUILT,
                "stage": "sub_agent",
                "agent_id": agent_id,
                "message_count": len(messages),
                "history_turns": policy.history_turns,
            },
        )
        return messages

    def build_for_summary(self, state: PaperMindState) -> list[BaseMessage]:
        """汇总 Agent：System + 策略裁剪历史 + 子 Agent 结果 Human。"""
        policy = DEFAULT_SUMMARY_POLICY
        decision = state.get("decision") or {}
        conv = state.get("conversation_context") or {}

        messages: list[BaseMessage] = [
            SystemMessage(content=self._summary_system_prompt(decision)),
        ]
        history = self._history_slice(state, policy.history_turns)
        if history:
            messages.extend(history_to_langchain_messages(history))

        human_content = build_summary_user_content(
            query=str(state.get("query") or ""),
            standalone_query=str(conv.get("standalone_query") or state.get("query") or ""),
            sub_agent_results=list(state.get("agent_results") or []),
            evidence_packets=list(state.get("evidence_packets") or []),
            include_evidence=policy.include_evidence_packets,
        )
        messages.append(HumanMessage(content=human_content))
        logger.debug(
            "context_built",
            extra={
                "event": CONTEXT_BUILT,
                "stage": "summary",
                "message_count": len(messages),
                "history_turns": policy.history_turns,
            },
        )
        return messages

    def get_sub_agent_human_content(
        self,
        state: PaperMindState,
        *,
        agent_id: str,
        agent: dict[str, Any],
    ) -> str:
        """返回子 Agent Human 正文（供 result.task 记录）。"""
        conv = state.get("conversation_context") or {}
        requirements = conv.get("context_requirements") or {}
        override = requirements.get(agent_id) if isinstance(requirements, dict) else None
        if not isinstance(override, dict):
            override = {}
        policy = merge_agent_policy(agent_id, override)
        return self._build_sub_agent_human(state, agent_id, agent, policy)

    def _history_slice(
        self,
        state: PaperMindState,
        max_turns: int,
    ) -> list[dict[str, str]]:
        if max_turns <= 0:
            return []
        history = list(state.get("history_messages") or [])
        if not history:
            ctx = state.get("context") or {}
            history = list(ctx.get("history_messages") or [])
        if not history:
            return []
        return history[-max_turns:]

    def _sub_agent_system_prompt(
        self,
        agent: dict[str, Any],
        state: PaperMindState,
    ) -> str:
        prompt_config = agent.get("promptConfig") or {}
        prompt = prompt_config.get("prompt") if isinstance(prompt_config, dict) else ""
        if prompt:
            base_prompt = str(prompt)
        else:
            base_prompt = SUB_AGENT_DEFAULT_PROMPT.format(
                agent_name=agent.get("name", "子 Agent"),
                agent_description=agent.get("description", "未配置职责描述"),
            )
        base_prompt += TOOL_SYSTEM_PROMPT_SUFFIX

        ctx = state.get("context") or {}
        task_id = ctx.get("task_id") or ""
        top_k = ctx.get("top_k")
        if task_id or top_k is not None:
            hints: list[str] = []
            if task_id:
                hints.append(
                    f'- 当前任务上下文 task_id = "{task_id}"，调用检索工具时请传入此 task_id'
                )
            if top_k is not None:
                hints.append(f"- 检索数量 top_k = {top_k}，调用检索工具时请传入此值")
            if hints:
                base_prompt += (
                    "\n\n## 当前检索参数\n"
                    + "\n".join(hints)
                    + "\n调用任意检索工具时，必须使用上述参数值。"
                )
        return base_prompt

    def _summary_system_prompt(self, decision: dict[str, Any]) -> str:
        return (
            "你是 PaperMind 学术研究多智能体系统的主 Agent。各子 Agent 已按流水线执行完毕，"
            "请基于其输出给用户一份统一、清晰的中文最终回答。\n\n"
            "## 汇总要求\n"
            "- 最终回答必须优先呈现真实工具执行结果；不要只写概述。\n"
            "- 如果子智能体结果中包含 external_task / tool_result，必须明确列出工具名、执行状态、入参摘要、输出路径、文件大小、错误信息等可用字段。\n"
            "- 工具返回的原始结果不可丢弃；可以先给结构化执行结果，再给解释和后续建议。\n"
            "- 子智能体结果多为 JSON：请解析 result、evidence、assumptions、uncertainty、is_mock、status。\n"
            "- 只基于子智能体结果作答，不要编造未提供的分数、文献或实验结论。\n"
            "- 若 is_mock 为 true 或 status 为 tool_required，必须在回答中明确说明当前为模拟/待工具结果。\n"
            "- 整合多阶段结论时去重；保留各阶段关键发现、风险与下一步建议。\n"
            "- 存在执行失败的子智能体时，简要说明并尽量保留成功阶段结果。\n"
            "- 不要暴露 agent_id、route_plan 等内部调度细节，除非用户明确要求。\n"
            "- 不提供诊疗建议、人体剂量或监管结论。\n"
            "- 输出中文。\n\n"
            f"## 调度原因\n{decision.get('reason') or ''}"
        )

    def _build_sub_agent_human(
        self,
        state: PaperMindState,
        agent_id: str,
        agent: dict[str, Any],
        policy: AgentContextPolicy,
    ) -> str:
        conv = state.get("conversation_context") or {}
        decision = state.get("decision") or {}
        queries_per_agent = decision.get("queries_per_agent") or {}
        default_query = str(decision.get("query_for_agent") or state.get("query") or "")
        stage_task = str(queries_per_agent.get(agent_id) or default_query)

        return build_sub_agent_human_content(
            query=str(state.get("query") or ""),
            standalone_query=str(conv.get("standalone_query") or state.get("query") or ""),
            stage_task=stage_task,
            policy_use_standalone=policy.use_standalone_query,
            include_upstream=policy.include_upstream,
            agent_results=list(state.get("agent_results") or []),
            include_evidence=policy.include_evidence_packets,
            evidence_packets=list(state.get("evidence_packets") or []),
            resolved_entities=dict(conv.get("resolved_entities") or {}),
        )


context_builder = ContextBuilder()
