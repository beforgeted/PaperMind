"""Orchestrator 编排服务，负责运行状态、步骤状态、挂起与恢复。"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional

from app.agent.graphs.paper_mind_graph import (
    DIRECT_ANSWER_NODE,
    EXECUTE_SUB_AGENT_NODE,
    MAIN_AGENT_NODE,
    SUSPEND_NODE,
    SUMMARIZE_NODE,
)
from app.core.log_context import bind, log_bind
from app.observability.events import (
    GRAPH_NODE_UPDATED,
    MAIN_AGENT_DECISION,
    MCP_TASK_SUBMITTED,
    RUN_CREATED,
    SUB_AGENT_STEP_COMPLETED,
)
from app.schemas.runtime_schema import AgentRunRecord, ComputeTaskRecord
from app.services.run_service import agent_run_service
from app.services.task_service import compute_task_service
from app.task.runners.mcp_tool_runner import McpToolRunner
from app.task.storage_adapter import ComputeTaskStorageAdapter
from app.task.task_manager import TaskManager


logger = logging.getLogger(__name__)

LoadAgentsFn = Callable[[], Awaitable[List[Dict[str, Any]]]]
CreateGraphFn = Callable[..., Any]
GRAPH_NODE_NAMES = {
    MAIN_AGENT_NODE,
    EXECUTE_SUB_AGENT_NODE,
    SUSPEND_NODE,
    SUMMARIZE_NODE,
    DIRECT_ANSWER_NODE,
}


def json_line(data: Dict[str, Any]) -> str:
    """把事件对象序列化成一行 JSON，供 SSE 流式返回使用。"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


class Orchestrator:
    """中心编排器，统一管理 AgentRun、WorkflowStep、上下文、挂起与恢复。"""

    def __init__(self) -> None:
        """绑定运行态存储服务和外部计算任务服务。"""
        self.agent_runs = agent_run_service
        self.compute_tasks = compute_task_service

    async def create_agent_run(
        self,
        *,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentRunRecord:
        """创建一次 AgentRun，并立即切换到 running 状态。"""
        run = await self.agent_runs.create_run(query=query, context=context)
        bind(run_id=run.run_id)
        logger.info(
            "AgentRun created",
            extra={
                "event": RUN_CREATED,
                "query_len": len(query or ""),
                "context_keys": sorted((context or {}).keys()),
            },
        )
        await self.agent_runs.mark_running(run.run_id)
        refreshed = await self.agent_runs.get_run(run.run_id)
        logger.info("AgentRun 进入 running: run_id=%s", run.run_id)
        return refreshed or run

    async def create_pending_task(
        self,
        *,
        run_id: str,
        agent: Dict[str, Any],
        task_query: str,
        result: Dict[str, Any],
    ) -> Optional[ComputeTaskRecord]:
        """为需要外部工具或计算的子 Agent 输出创建 pending 任务。"""
        run = await self.agent_runs.get_run(run_id)
        if not run:
            return None

        agent_id = str(agent.get("agent_id") or result.get("agent_id") or "")
        step = next((item for item in run.steps if item.agent_id == agent_id), None)
        if not step:
            return None

        return await self.compute_tasks.create_task(
            run_id=run_id,
            step_id=step.step_id,
            agent_id=agent_id,
            kind="external_compute",
            input_data={
                "agent_id": agent_id,
                "agent_name": str(agent.get("name") or result.get("agent_name") or agent_id),
                "task_query": task_query,
                "agent_result": result,
            },
        )

    async def submit_mcp_task(
        self,
        *,
        run_id: str,
        agent: Dict[str, Any],
        task_query: str,
        result: Dict[str, Any],
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Optional[ComputeTaskRecord]:
        """把子 Agent 的 MCP 工具请求提交给 TaskManager 异步执行。"""
        logger.info(
            "准备提交 MCP 异步任务: run_id=%s agent=%s tool=%s arg_keys=%s",
            run_id,
            str(agent.get("agent_id") or result.get("agent_id") or ""),
            tool_name,
            sorted(arguments.keys()),
        )
        run = await self.agent_runs.get_run(run_id)
        if not run:
            logger.warning("提交 MCP 异步任务失败，AgentRun 不存在: run_id=%s tool=%s", run_id, tool_name)
            return None

        agent_id = str(agent.get("agent_id") or result.get("agent_id") or "")
        step = next((item for item in run.steps if item.agent_id == agent_id), None)
        if not step:
            logger.warning("提交 MCP 异步任务失败，未找到 step: run_id=%s agent=%s tool=%s", run_id, agent_id, tool_name)
            return None

        async def resume_when_suspended(task_info) -> None:
            """等待当前 run 挂起后自动恢复，避免工具太快完成时抢在 suspend_node 前执行。"""
            logger.info("MCP 任务成功，等待工作流挂起后恢复: run_id=%s task_id=%s", run_id, task_info.task_id)
            for _ in range(30):
                current = await self.agent_runs.get_run(run_id)
                if current and current.pending_task_id == task_info.task_id:
                    try:
                        from app.services.chat_workflow_service import chat_workflow_service

                        logger.info("开始从 MCP 任务恢复工作流: run_id=%s task_id=%s", run_id, task_info.task_id)
                        await chat_workflow_service.resume_from_task(task_id=task_info.task_id)
                        logger.info("MCP 任务恢复工作流完成: run_id=%s task_id=%s", run_id, task_info.task_id)
                    except Exception as exc:
                        logger.exception("MCP 异步任务自动恢复失败: task_id=%s error=%s", task_info.task_id, exc)
                    return
                await asyncio.sleep(0.5)
            logger.warning("MCP 任务完成后等待挂起超时，未自动恢复: run_id=%s task_id=%s", run_id, task_info.task_id)

        async def fail_run_from_task(task_info) -> None:
            """异步工具任务失败时，让 AgentRun 进入 failed，避免长期 suspended。"""
            message = task_info.error or f"MCP 异步任务失败: {tool_name}"
            logger.warning("MCP 异步任务失败，标记 AgentRun failed: run_id=%s task_id=%s error=%s", run_id, task_info.task_id, message)
            await self.agent_runs.fail_run(run_id=run_id, error=message)

        manager = TaskManager(
            storage=ComputeTaskStorageAdapter(
                self.compute_tasks,
                run_id=run_id,
                step_id=step.step_id,
                agent_id=agent_id,
                kind=tool_name,
            ),
            runner_registry={McpToolRunner.tool_name: McpToolRunner()},
            on_task_succeeded=resume_when_suspended,
            on_task_failed=fail_run_from_task,
        )
        task = await manager.submit(
            tool_name=McpToolRunner.tool_name,
            params={
                "tool_name": tool_name,
                "arguments": arguments,
                "agent_id": agent_id,
                "agent_name": str(agent.get("name") or result.get("agent_name") or agent_id),
                "task_query": task_query,
            },
        )
        logger.info(
            "MCP task submitted",
            extra={
                "event": MCP_TASK_SUBMITTED,
                "agent_id": agent_id,
                "tool_name": tool_name,
                "task_id": task.task_id,
            },
        )
        return await self.compute_tasks.get_task(task.task_id)

    async def build_resume_state(
        self,
        *,
        run_id: str,
        load_agents: LoadAgentsFn,
    ) -> Dict[str, Any]:
        """根据持久化的 AgentRun 重新构造 LangGraph 恢复执行所需状态。"""
        run = await self.agent_runs.get_run(run_id)
        if not run:
            raise ValueError(f"AgentRun 不存在: {run_id}")
        agents = await load_agents()
        from app.agent.context import build_conversation_context

        ctx = dict(run.context or {})
        history_messages = list(ctx.get("history_messages") or [])
        decision = run.decision or {}
        conversation_context = (
            build_conversation_context(
                query=str(run.query or ""),
                decision=decision,
                history_messages=history_messages,
            )
            if decision
            else {}
        )
        return {
            "query": run.query,
            "history_messages": history_messages,
            "conversation_context": conversation_context,
            "context": ctx,
            "agents": agents,
            "decision": decision,
            "selected_agents": run.selected_agents,
            "agent_results": run.agent_results,
            "evidence_packets": list(ctx.get("evidence_packets") or []),
            "pending_task_id": None,
            "pending_step_id": None,
        }

    async def apply_graph_update(
        self,
        *,
        run_id: str,
        agents: List[Dict[str, Any]],
        state: Dict[str, Any],
        node_name: str,
        node_update: Dict[str, Any],
        emit_events: bool,
    ) -> List[str]:
        """应用单个 LangGraph 节点更新，持久化状态并按需生成 SSE 事件。"""
        with log_bind(run_id=run_id):
            logger.info(
                "LangGraph node updated",
                extra={
                    "event": GRAPH_NODE_UPDATED,
                    "node_name": node_name,
                    "keys": sorted(node_update.keys()),
                },
            )
        events: List[str] = []
        state.update(node_update)
        await self.agent_runs.save_checkpoint(
            run_id=run_id,
            node_name=node_name,
            state=state,
            node_update=node_update,
        )

        if node_name == MAIN_AGENT_NODE:
            decision = node_update.get("decision") or {}
            selected_agents = node_update.get("selected_agents") or []
            await self.agent_runs.set_decision(
                run_id=run_id,
                decision=decision,
                selected_agents=selected_agents,
                agents=agents,
            )
            await self.agent_runs.complete_step(
                run_id=run_id,
                node_name="main_agent_node",
                result={"decision": decision},
            )
            if emit_events:
                events.append(json_line({"type": "thinking", "content": json.dumps(decision, ensure_ascii=False)}))
                if decision.get("action") == "dispatch":
                    events.append(
                        json_line(
                            {
                                "type": "status",
                                "content": f"LangGraph 已规划 {len(selected_agents)} 个 Agent 串行执行",
                            }
                        )
            )
            logger.info(
                "MainAgent decision completed",
                extra={
                    "event": MAIN_AGENT_DECISION,
                    "action": decision.get("action"),
                    "selected_agents": selected_agents,
                },
            )
            return events

        if node_name == EXECUTE_SUB_AGENT_NODE:
            result = (node_update.get("agent_results") or [])[-1]
            agent_id = str(result.get("agent_id") or node_update.get("last_agent_id") or "")
            task_id = result.get("pending_task_id")
            await self.agent_runs.complete_agent_step(
                run_id=run_id,
                agent_id=agent_id,
                result=result,
                task_id=task_id,
            )
            if emit_events:
                events.append(
                    json_line(
                        {
                            "type": "status",
                            "content": f"LangGraph 节点完成：{result.get('agent_name') or agent_id}",
                        }
                    )
                )
                events.append(
                    json_line(
                        {
                            "type": "sub_agent_result",
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
                )
                if task_id:
                    events.append(
                        json_line(
                            {
                                "type": "compute_task",
                                "content": json.dumps(
                                    {
                                        "run_id": run_id,
                                        "task_id": task_id,
                                        "status": "pending",
                                    },
                                    ensure_ascii=False,
                                ),
                            }
                        )
                    )
            with log_bind(agent_id=agent_id):
                logger.info(
                    "Sub agent step completed",
                    extra={
                        "event": SUB_AGENT_STEP_COMPLETED,
                        "content_len": len(str(result.get("content") or "")),
                        "pending_task_id": task_id or "",
                        "error": bool(result.get("error")),
                    },
                )
            return events

        if node_name == DIRECT_ANSWER_NODE:
            final_report = str(node_update.get("final_report") or "")
            await self.agent_runs.complete_step(
                run_id=run_id,
                node_name="direct_answer_node",
                result={"content": final_report},
            )
            await self.agent_runs.complete_run(run_id=run_id, final_report=final_report)
            state["final_report"] = final_report
            if emit_events:
                events.append(json_line({"type": "content", "content": final_report}))
            logger.info("Direct answer 完成: run_id=%s final_len=%s", run_id, len(final_report))
            return events

        if node_name == SUMMARIZE_NODE:
            final_report = str(node_update.get("final_report") or "")
            await self.agent_runs.complete_step(
                run_id=run_id,
                node_name=SUMMARIZE_NODE,
                result={"content": final_report},
            )
            await self.agent_runs.complete_run(
                run_id=run_id,
                final_report=final_report,
                agent_results=state.get("agent_results") or [],
            )
            state["final_report"] = final_report
            if emit_events:
                events.append(json_line({"type": "status", "content": "LangGraph 汇总节点完成"}))
                events.append(json_line({"type": "content", "content": final_report}))
            logger.info("汇总节点完成: run_id=%s final_len=%s", run_id, len(final_report))
            return events

        if node_name == SUSPEND_NODE:
            final_report = str(node_update.get("final_report") or "")
            task_id = str(node_update.get("pending_task_id") or "")
            step_id = str(node_update.get("pending_step_id") or "")
            if task_id and step_id:
                await self.agent_runs.suspend_run(run_id=run_id, task_id=task_id, step_id=step_id)
            state["final_report"] = final_report
            if emit_events:
                events.append(json_line({"type": "status", "content": "LangGraph 工作流已挂起"}))
                events.append(json_line({"type": "content", "content": final_report}))
            logger.info("工作流挂起: run_id=%s task_id=%s step_id=%s", run_id, task_id, step_id)
            return events

        return events

    async def stream_graph_events(
        self,
        *,
        run_id: str,
        graph: Any,
        state: Dict[str, Any],
        agents: List[Dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """执行 LangGraph 工作流，并持续产出可直接返回给前端的 SSE 事件。"""
        logger.info("开始 stream_graph_events: run_id=%s", run_id)
        async for update in graph.astream(state, stream_mode="updates"):
            if not isinstance(update, dict):
                continue
            has_node_update = any(key in GRAPH_NODE_NAMES for key in update.keys())
            if not has_node_update:
                state.update(update)
                logger.info("LangGraph 状态快照更新: run_id=%s keys=%s", run_id, sorted(update.keys()))
                continue
            for node_name, node_update in update.items():
                if not isinstance(node_update, dict):
                    continue
                events = await self.apply_graph_update(
                    run_id=run_id,
                    agents=agents,
                    state=state,
                    node_name=node_name,
                    node_update=node_update,
                    emit_events=True,
                )
                for event in events:
                    yield event
        logger.info("结束 stream_graph_events: run_id=%s", run_id)

    async def run_graph_to_completion(
        self,
        *,
        run_id: str,
        graph: Any,
        state: Dict[str, Any],
        agents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """执行 LangGraph 工作流直到结束，但不向前端发送 SSE 事件。"""
        async for update in graph.astream(state, stream_mode="updates"):
            if not isinstance(update, dict):
                continue
            has_node_update = any(key in GRAPH_NODE_NAMES for key in update.keys())
            if not has_node_update:
                state.update(update)
                logger.info("LangGraph 状态快照更新: run_id=%s keys=%s", run_id, sorted(update.keys()))
                continue
            for node_name, node_update in update.items():
                if isinstance(node_update, dict):
                    await self.apply_graph_update(
                        run_id=run_id,
                        agents=agents,
                        state=state,
                        node_name=node_name,
                        node_update=node_update,
                        emit_events=False,
                    )
        return state

    async def resume_from_task(
        self,
        *,
        task_id: str,
        load_agents: LoadAgentsFn,
        create_graph: CreateGraphFn,
    ) -> Dict[str, Any]:
        """外部计算任务成功后恢复对应 AgentRun，并继续执行剩余工作流。"""
        logger.info("尝试从计算任务恢复工作流: task_id=%s", task_id)
        task = await self.compute_tasks.get_task(task_id)
        if not task:
            raise ValueError(f"计算任务不存在: {task_id}")
        if task.status != "succeeded":
            raise ValueError(f"计算任务尚未成功，当前状态: {task.status}")

        run = await self.agent_runs.get_run(task.run_id)
        if not run:
            raise ValueError(f"AgentRun 不存在: {task.run_id}")
        if run.pending_task_id and run.pending_task_id != task_id:
            raise ValueError(f"AgentRun 当前等待的任务是 {run.pending_task_id}，不是 {task_id}")

        await self.agent_runs.resume_from_task_result(
            run_id=task.run_id,
            task_id=task_id,
            result=task.result,
        )
        state = await self.build_resume_state(run_id=task.run_id, load_agents=load_agents)
        agents = state.get("agents") or []
        graph = create_graph(run_id=task.run_id)
        await self.run_graph_to_completion(
            run_id=task.run_id,
            graph=graph,
            state=state,
            agents=agents,
        )
        final_report = str(state.get("final_report") or "").strip()
        if final_report:
            logger.info("恢复后检测到 final_report，标记 AgentRun completed: run_id=%s final_len=%s", task.run_id, len(final_report))
            await self.agent_runs.complete_run(
                run_id=task.run_id,
                final_report=final_report,
                agent_results=state.get("agent_results") or [],
            )
        else:
            logger.warning("恢复后没有 final_report，AgentRun 暂不标记 completed: run_id=%s task_id=%s", task.run_id, task_id)
        final_run = await self.agent_runs.get_run(task.run_id)
        if not final_run:
            raise ValueError(f"AgentRun 不存在: {task.run_id}")
        logger.info("从计算任务恢复工作流完成: run_id=%s task_id=%s status=%s", task.run_id, task_id, final_run.status)
        return final_run.model_dump()

    async def fail_run(self, *, run_id: str, error: str) -> None:
        """把指定 AgentRun 标记为失败并记录错误信息。"""
        await self.agent_runs.fail_run(run_id=run_id, error=error)


orchestrator = Orchestrator()
