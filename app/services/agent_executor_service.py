"""子 Agent 执行服务 -- 支持 tool-calling 真实检索。"""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional

from app.core.log_context import log_bind
from app.observability.events import (
    AGENT_EXECUTION_COMPLETED,
    AGENT_EXECUTION_STARTED,
    TOOL_CALL_COMPLETED,
)

from langchain_core.messages import AIMessage, ToolMessage

from app.agent.agents.main_agent import main_agent
from app.agent.context import context_builder, extract_evidence_packet
from app.agent.graphs.graph_state import PaperMindState
from app.agent.tools.registry import get_tools_for_agent
from app.services.mcp_server_service import TOOLS_WITH_TASK_ID, TOOLS_WITH_TOP_K
from app.services.llm_service import llm_service, message_content_to_text
from app.services.orchestrator_service import orchestrator


logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 5


class AgentExecutorService:
    """执行子 Agent，支持 tool-calling 循环。"""

    def parse_result_content(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        content = str(result.get("content") or "")
        return main_agent.parse_json_decision(content)

    async def _execute_tool_calls(
        self,
        response: AIMessage,
        messages: list,
        tools: list,
        agent_id: str,
        state: PaperMindState,
        evidence_packets: List[Dict[str, Any]],
    ) -> None:
        """Execute all tool calls from a single AI response and append ToolMessages."""
        messages.append(response)
        ctx = state.get("context") or {}
        for tool_call in response.tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id", "")
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
            tool_start = time.perf_counter()
            for tool in tools:
                if tool.name == tool_name:
                    try:
                        result = await tool.ainvoke(tool_args)
                        tool_result = str(result)
                    except Exception as exc:
                        tool_result = f"工具调用失败: {str(exc)}"
                    break

            logger.info(
                "tool_call_completed",
                extra={
                    "event": TOOL_CALL_COMPLETED,
                    "tool_name": tool_name,
                    "args_preview": str(tool_args)[:200],
                    "result_len": len(tool_result),
                    "latency_ms": round((time.perf_counter() - tool_start) * 1000),
                    "success": "工具调用失败" not in tool_result and tool_result != "工具未找到",
                },
            )

            packet = extract_evidence_packet(
                tool_name=tool_name,
                content=tool_result,
                agent_id=agent_id,
            )
            if packet:
                evidence_packets.append(packet)

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
        state: PaperMindState,
        tools: Optional[list] = None,
        evidence_packets: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        """执行子 Agent，支持可选的 tool-calling 循环。"""
        agent_id = str(agent.get("agent_id") or "")
        exec_start = time.perf_counter()
        tool_rounds = 0
        with log_bind(agent_id=agent_id):
            logger.info(
                "Agent execution started",
                extra={"event": AGENT_EXECUTION_STARTED, "has_tools": bool(tools)},
            )
        llm = llm_service.create_chat_model(
            llm_service.merge_sub_agent_model_config(agent.get("modelConfig")),
            streaming=True,
            timeout=120,
            max_tokens=4096,
        )

        messages = context_builder.build_for_sub_agent(
            state,
            agent_id=agent_id,
            agent=agent,
        )
        collected_evidence = evidence_packets if evidence_packets is not None else []

        if tools:
            llm_with_tools = llm.bind_tools(tools)
            response = await llm_with_tools.ainvoke(messages)
            rounds = 0
            while (
                hasattr(response, "tool_calls")
                and response.tool_calls
                and rounds < MAX_TOOL_ROUNDS
            ):
                rounds += 1
                tool_rounds = rounds
                await self._execute_tool_calls(
                    response,
                    messages,
                    tools,
                    agent_id,
                    state,
                    collected_evidence,
                )
                response = await llm_with_tools.ainvoke(messages)

            if hasattr(response, "content") and response.content:
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
            async for chunk in llm.astream(messages):
                text = message_content_to_text(getattr(chunk, "content", ""))
                if text:
                    yield text

        with log_bind(agent_id=agent_id):
            logger.info(
                "Agent execution completed",
                extra={
                    "event": AGENT_EXECUTION_COMPLETED,
                    "tool_rounds": tool_rounds,
                    "latency_ms": round((time.perf_counter() - exec_start) * 1000),
                },
            )

    async def run_sub_agent(
        self,
        *,
        agent: Dict[str, Any],
        agent_id: str,
        state: PaperMindState,
        chunk_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        agent_name = str(agent.get("name") or agent_id or "子 Agent")
        tools = await get_tools_for_agent(agent_id)
        task_text = context_builder.get_sub_agent_human_content(
            state,
            agent_id=agent_id,
            agent=agent,
        )
        evidence_packets: List[Dict[str, Any]] = []

        try:
            chunks: list[str] = []
            chunk_count = 0
            async for text in self.stream_sub_agent(
                agent=agent,
                state=state,
                tools=tools,
                evidence_packets=evidence_packets,
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
                "子 Agent 模型输出完成: agent=%s chunks=%s content_len=%s evidence=%s",
                agent_id,
                chunk_count,
                len(content),
                len(evidence_packets),
            )
            return {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "task": task_text,
                "content": content or "该子 Agent 未返回有效内容。",
                "error": "",
                "evidence_packets": evidence_packets,
            }
        except Exception as exc:
            logger.exception("子 Agent 执行失败: agent=%s", agent_id)
            return {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "task": task_text,
                "content": "",
                "error": str(exc),
                "evidence_packets": evidence_packets,
            }

    async def run_agent_for_run(
        self,
        *,
        run_id: Optional[str],
        agent: Dict[str, Any],
        agent_id: str,
        state: PaperMindState,
        chunk_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        result = await self.run_sub_agent(
            agent=agent,
            agent_id=agent_id,
            state=state,
            chunk_callback=chunk_callback,
        )
        task_text = str(result.get("task") or "")

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
                    task_query=task_text,
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
