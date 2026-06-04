"""子 Agent 执行服务 -- 支持 tool-calling 真实检索。"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.agents.main_agent import main_agent
from app.agent.tools.registry import get_tools_for_agent
from app.services.mcp_server_service import TOOLS_WITH_TASK_ID, TOOLS_WITH_TOP_K
from app.services.llm_service import llm_service, message_content_to_text
from app.services.orchestrator_service import orchestrator


logger = logging.getLogger(__name__)

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

MAX_TOOL_ROUNDS = 5


class AgentExecutorService:
    """执行子 Agent，支持 tool-calling 循环。"""

    def parse_result_content(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        content = str(result.get("content") or "")
        return main_agent.parse_json_decision(content)

    def sub_agent_prompt(
        self, agent: Dict[str, Any], context: Optional[Dict[str, Any]] = None
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

        # Issue #7: inject task_id / top_k from request context so the LLM
        # passes them to retrieval tools (retrieve_evidence, search_papers, etc.)
        ctx = context or {}
        task_id = ctx.get("task_id") or ""
        top_k = ctx.get("top_k")
        if task_id or top_k is not None:
            hints = []
            if task_id:
                hints.append(f"- 当前任务上下文 task_id = \"{task_id}\"，调用检索工具时请传入此 task_id")
            if top_k is not None:
                hints.append(f"- 检索数量 top_k = {top_k}，调用检索工具时请传入此值")
            if hints:
                base_prompt += (
                    "\n\n## 当前检索参数\n"
                    + "\n".join(hints)
                    + "\n调用任意检索工具时，必须使用上述参数值。"
                )
        return base_prompt

    async def _execute_tool_calls(
        self,
        response: AIMessage,
        messages: list,
        tools: list,
        agent_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Execute all tool calls from a single AI response and append ToolMessages."""
        messages.append(response)
        for tool_call in response.tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id", "")
            logger.info(
                "Agent %s calling tool: %s(%s)",
                agent_id,
                tool_name,
                str(tool_args)[:100],
            )

            # Inject default task_id/top_k from context if not provided by model
            ctx = context or {}
            if tool_name in TOOLS_WITH_TASK_ID:
                if "task_id" not in tool_args or not tool_args.get("task_id"):
                    default_task_id = ctx.get("task_id") or ""
                    if default_task_id:
                        tool_args = dict(tool_args)
                        tool_args["task_id"] = default_task_id
            if tool_name in TOOLS_WITH_TOP_K:
                if "top_k" not in tool_args or not tool_args.get("top_k"):
                    default_top_k = ctx.get("top_k")
                    if default_top_k is not None:
                        tool_args = dict(tool_args)
                        tool_args["top_k"] = int(default_top_k)

            tool_result = "工具未找到"
            for tool in tools:
                if tool.name == tool_name:
                    try:
                        result = await tool.ainvoke(tool_args)
                        tool_result = str(result)
                    except Exception as exc:
                        tool_result = f"工具调用失败: {str(exc)}"
                    break

            messages.append(
                ToolMessage(
                    content=tool_result[:4000],
                    tool_call_id=tool_call_id,
                )
            )

    async def stream_sub_agent(
        self,
        *,
        agent: Dict[str, Any],
        task_query: str,
        tools: Optional[list] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """执行子 Agent，支持可选的 tool-calling 循环。

        当 tools 非空且 agent 需要事实性信息时，先通过非流式调用完成
        tool-calling 循环，然后流式输出最终回答。
        无 tools 时直接流式调用 LLM。
        """
        agent_id = str(agent.get("agent_id") or "")
        llm = llm_service.create_chat_model(
            llm_service.merge_sub_agent_model_config(agent.get("modelConfig")),
            streaming=True,
            timeout=120,
            max_tokens=4096,
        )

        system_prompt = self.sub_agent_prompt(agent, context=context)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=task_query),
        ]

        if tools:
            # Tool-calling non-streaming loop
            llm_with_tools = llm.bind_tools(tools)
            response = await llm_with_tools.ainvoke(messages)
            rounds = 0
            while (
                hasattr(response, "tool_calls")
                and response.tool_calls
                and rounds < MAX_TOOL_ROUNDS
            ):
                rounds += 1
                await self._execute_tool_calls(response, messages, tools, agent_id, context=context)
                response = await llm_with_tools.ainvoke(messages)

            # Issue #11: stream final answer WITHOUT bind_tools to prevent
            # the model from triggering additional tool_calls during streaming.
            if hasattr(response, "content") and response.content:
                # Append final response so the stream context is complete
                messages.append(response)
                final_llm_no_tools = llm_service.create_chat_model(
                    llm_service.merge_sub_agent_model_config(agent.get("modelConfig")),
                    streaming=True,
                    timeout=120,
                    max_tokens=4096,
                )
                async for chunk in final_llm_no_tools.astream(messages):
                    text = message_content_to_text(getattr(chunk, "content", ""))
                    if text:
                        yield text
            else:
                yield "该子 Agent 未返回有效内容。"
        else:
            # No tools -- pure LLM streaming
            async for chunk in llm.astream(messages):
                text = message_content_to_text(getattr(chunk, "content", ""))
                if text:
                    yield text

    async def run_sub_agent(
        self,
        *,
        agent: Dict[str, Any],
        task_query: str,
        chunk_callback: Optional[Callable[[str], Awaitable[None]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        agent_id = str(agent.get("agent_id") or "")
        agent_name = str(agent.get("name") or agent_id or "子 Agent")
        tools = await get_tools_for_agent(agent_id)

        try:
            chunks: list[str] = []
            chunk_count = 0
            async for text in self.stream_sub_agent(
                agent=agent,
                task_query=task_query,
                tools=tools,
                context=context,
            ):
                chunk_count += 1
                if chunk_count == 1:
                    logger.info(
                        "子 Agent 首个 chunk: agent=%s has_tools=%s",
                        agent_id,
                        bool(tools),
                    )
                chunks.append(text)
                if chunk_callback is not None:
                    await chunk_callback(text)
            content = "".join(chunks).strip()
            logger.info(
                "子 Agent 模型输出完成: agent=%s chunks=%s content_len=%s",
                agent_id,
                chunk_count,
                len(content),
            )
            return {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "task": task_query,
                "content": content or "该子 Agent 未返回有效内容。",
                "error": "",
            }
        except Exception as exc:
            logger.exception("子 Agent 执行失败: agent=%s", agent_id)
            return {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "task": task_query,
                "content": "",
                "error": str(exc),
            }

    async def run_agent_for_run(
        self,
        *,
        run_id: Optional[str],
        agent: Dict[str, Any],
        task_query: str,
        chunk_callback: Optional[Callable[[str], Awaitable[None]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result = await self.run_sub_agent(
            agent=agent,
            task_query=task_query,
            chunk_callback=chunk_callback,
            context=context,
        )
        # Check for MCP tool requests (external async compute, not LangChain tools)
        parsed = self.parse_result_content(result)
        if not parsed or not run_id:
            return result

        tool_requests = parsed.get("tool_requests") or []
        if isinstance(tool_requests, list) and tool_requests:
            first = tool_requests[0] if tool_requests else {}
            if isinstance(first, dict) and first.get("tool_name"):
                task = await orchestrator.submit_mcp_task(
                    run_id=run_id,
                    agent=agent,
                    task_query=task_query,
                    result=result,
                    tool_name=str(first["tool_name"]),
                    arguments=first.get("arguments") or {},
                )
                if task:
                    result = dict(result)
                    result["pending_task_id"] = task.task_id
                    result["pending_step_id"] = task.step_id

        return result


agent_executor_service = AgentExecutorService()
