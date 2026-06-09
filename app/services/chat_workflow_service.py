"""聊天工作流门面，负责通过 SSE 执行 PaperMind Agent 图。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional

from app.agent.agents.agent_registry import list_registry_items
from app.core.log_context import log_bind
from app.observability.events import (
    CHAT_STREAM_COMPLETED,
    CHAT_STREAM_FAILED,
    CHAT_STREAM_STARTED,
)
from app.agent.agents.main_agent import main_agent
from app.agent.graphs.paper_mind_graph import create_paper_mind_graph
from app.services.agent_executor_service import agent_executor_service
from app.services.orchestrator_service import orchestrator
from app.services.session_title_service import (
    generate_session_title,
    is_default_session_title,
)
from app.services.summary_service import summary_service


logger = logging.getLogger(__name__)


def json_line(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


class ChatWorkflowService:
    """串联 MainAgent、Orchestrator、LangGraph 和 SSE 的应用服务。"""

    def error_chunk(self, message: str) -> str:
        return json_line({"type": "error", "content": message})

    async def _maybe_update_session_title(
        self,
        *,
        session_id: str,
        user_query: str,
        is_first_turn: bool,
    ) -> Optional[str]:
        """On the first turn, summarize the user query into a session title."""
        if not session_id or not is_first_turn:
            return None

        from app.services.memory.episodic import EpisodicMemory
        from app.services.memory.session_store import get_session_store

        store = get_session_store()
        current_title = store.get_title(session_id)
        if not is_default_session_title(current_title):
            return None

        raw_query = (user_query or "").strip()
        if not raw_query:
            return None

        title = await generate_session_title(raw_query)
        store.set_title(session_id, title)

        try:
            episodic = EpisodicMemory()
            doc = episodic.get_session_doc(session_id)
            if doc:
                doc["title"] = title
                episodic._es.index(
                    index=episodic.config.episodic_es_index,
                    id=session_id,
                    body=doc,
                    refresh=True,
                )
        except Exception as exc:
            logger.warning("更新 ES 会话标题失败: session_id=%s error=%s", session_id, exc)

        logger.info("会话标题已更新: session_id=%s title=%s", session_id, title[:40])
        return title

    def _persist_session_turns(
        self,
        *,
        session_id: str,
        user_query: str,
        assistant_content: str,
        context: Dict[str, Any],
    ) -> None:
        """Write user/assistant turns to Redis once; mirror to ES episodic."""
        if not session_id or not assistant_content.strip():
            return
        if context.get("__history_saved"):
            return

        from app.services.memory.episodic import EpisodicMemory
        from app.services.memory.session_store import TurnRecord, get_session_store

        store = get_session_store()
        store.ensure_session(session_id)
        user_text = (user_query or "").strip()
        assistant_text = assistant_content.strip()
        store.append_turn(session_id, TurnRecord(role="user", content=user_text))
        store.append_turn(
            session_id,
            TurnRecord(role="assistant", content=assistant_text, allow_long=True),
        )

        try:
            episodic = EpisodicMemory()
            episodic.append_turn(
                session_id,
                {"role": "user", "content": user_text},
            )
            episodic.append_turn(
                session_id,
                {"role": "assistant", "content": assistant_text},
            )
        except Exception as exc:
            logger.warning("ES 会话轮次同步失败: session_id=%s error=%s", session_id, exc)

        context["__history_saved"] = True
        logger.debug("会话轮次已写回: session_id=%s", session_id)

    def _extract_tool_results(self, agent_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tool_results: List[Dict[str, Any]] = []
        for result in agent_results:
            external_task = result.get("external_task")
            if not isinstance(external_task, dict):
                continue
            task_result = external_task.get("result")
            if not isinstance(task_result, dict):
                task_result = {}
            tool_result = task_result.get("tool_result")
            if not isinstance(tool_result, dict):
                tool_result = {}
            tool_results.append(
                {
                    "agent_id": result.get("agent_id"),
                    "agent_name": result.get("agent_name"),
                    "task_id": external_task.get("task_id"),
                    "status": external_task.get("status"),
                    "tool_name": task_result.get("tool_name"),
                    "arguments": task_result.get("arguments"),
                    "ok": tool_result.get("ok"),
                    "content": tool_result.get("content"),
                    "error": tool_result.get("error"),
                    "raw": external_task,
                }
            )
        return tool_results

    async def _load_enabled_agents(self) -> List[Dict[str, Any]]:
        return list_registry_items()

    async def _orchestrate(
        self,
        *,
        state: Dict[str, Any],
        agents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        history_messages = list(state.get("history_messages") or [])
        logger.info(
            "MainAgent 开始识别意图: query_len=%s agents=%s history_turns=%s",
            len(str(state.get("query") or "")),
            len(agents),
            len(history_messages),
        )
        return await main_agent.recognize_intent(state=state, agents=agents)

    def _create_graph(
        self,
        *,
        run_id: Optional[str] = None,
        event_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Build the LangGraph with callbacks wired for SSE streaming.

        The optional *context* dict carries per-request metadata (session_id,
        task_id, top_k) that gets forwarded to sub-agent execution so the LLM
        sees the right retrieval parameters.
        """
        _ctx = dict(context or {})

        async def run_agent(
            agent: Dict[str, Any],
            agent_id: str,
            state: Dict[str, Any],
        ) -> Dict[str, Any]:
            agent_name = str(agent.get("name") or agent_id or "子智能体")
            logger.info("子智能体开始执行: agent=%s", agent_id)

            if event_callback is not None:
                await event_callback(
                    {
                        "type": "status",
                        "content": f"正在执行：{agent_name}…",
                    }
                )

            # 子 Agent 全文仅进入 state.agent_results 供汇总节点使用；
            # 不向 SSE 推送 sub_agent_delta，避免与 summary_delta 重复展示。
            result = await agent_executor_service.run_agent_for_run(
                run_id=run_id,
                agent=agent,
                agent_id=agent_id,
                state=state,
                chunk_callback=None,
            )
            logger.info(
                "子智能体执行结束: agent=%s content_len=%s error=%s pending_task=%s",
                agent_id,
                len(str(result.get("content") or "")),
                bool(result.get("error")),
                result.get("pending_task_id") or "",
            )
            return result

        async def summarize(state: Dict[str, Any]) -> str:
            agent_results = list(state.get("agent_results") or [])
            history_messages = list(state.get("history_messages") or [])
            logger.info(
                "最终汇总开始: sub_agent_results=%s history_turns=%s",
                len(agent_results),
                len(history_messages),
            )

            async def on_chunk(text: str) -> None:
                if event_callback is None:
                    return
                logger.debug("最终汇总流式 chunk: chunk_len=%s", len(text or ""))
                await event_callback(
                    {
                        "type": "summary_delta",
                        "channel": "final",
                        "phase": "summary",
                        "content": text,
                    }
                )

            content = await summary_service.summarize_results(
                state=state,
                chunk_callback=on_chunk if event_callback is not None else None,
            )
            logger.info("最终汇总完成: content_len=%s", len(content or ""))
            return content

        return create_paper_mind_graph(
            orchestrate=self._orchestrate,
            run_agent=run_agent,
            summarize=summarize,
        )

    async def resume_from_task(self, *, task_id: str) -> Dict[str, Any]:
        return await orchestrator.resume_from_task(
            task_id=task_id,
            load_agents=self._load_enabled_agents,
            create_graph=self._create_graph,
        )

    async def _wait_for_run_terminal(self, *, run_id: str) -> AsyncGenerator[str, None]:
        """异步任务挂起后继续等待 run 终态，并把最终结果返回给 SSE 客户端。"""
        last_status = ""
        last_task_progress = ""
        emitted_tool_result_keys: set[str] = set()
        heartbeat_index = 0
        while True:
            run = await orchestrator.agent_runs.get_run(run_id)
            if not run:
                yield json_line({"type": "error", "content": f"AgentRun 不存在: {run_id}"})
                return

            if run.pending_task_id:
                task = await orchestrator.compute_tasks.get_task(run.pending_task_id)
                if task:
                    progress = task.input.get("progress", 0.0)
                    current_stage = str(task.input.get("current_stage") or "")
                    task_progress_key = f"{task.task_id}:{task.status}:{progress}:{current_stage}"
                    if task_progress_key != last_task_progress:
                        last_task_progress = task_progress_key
                        yield json_line(
                            {
                                "type": "tool_progress",
                                "content": current_stage or f"工具任务状态：{task.status}",
                                "task_id": task.task_id,
                                "status": task.status,
                                "progress": progress,
                            }
                        )

            tool_results = self._extract_tool_results(run.agent_results)
            for item in tool_results:
                key = str(item.get("task_id") or "")
                if key and key not in emitted_tool_result_keys:
                    emitted_tool_result_keys.add(key)
                    yield json_line(
                        {
                            "type": "tool_result",
                            "channel": "execution",
                            **item,
                        }
                    )

            if run.status != last_status:
                last_status = run.status
                yield json_line({"type": "status", "content": f"AgentRun 状态：{run.status}"})
            else:
                heartbeat_index += 1
                yield json_line(
                    {
                        "type": "heartbeat",
                        "content": f"等待异步任务完成：{run.status}",
                        "index": heartbeat_index,
                    }
                )

            if run.status == "completed":
                writeback_sid = (run.context or {}).get("session_id") or ""
                writeback_response = run.final_report or ""
                writeback_query = (run.context or {}).get("__original_query") or run.query or ""
                ctx = dict(run.context or {})
                try:
                    self._persist_session_turns(
                        session_id=writeback_sid,
                        user_query=writeback_query,
                        assistant_content=writeback_response,
                        context=ctx,
                    )
                except Exception:
                    pass
                yield json_line({"type": "content", "content": run.final_report})
                yield json_line({"type": "done", "content": run.final_report})
                return

            if run.status == "failed":
                message = run.error or "异步任务失败，工作流已终止。"
                yield json_line({"type": "error", "content": message})
                yield json_line({"type": "done", "content": message})
                return

            await asyncio.sleep(2)

    async def process_chat_stream(
        self,
        *,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        stream_start = time.perf_counter()
        _ctx = dict(context or {})
        session_id = str(_ctx.get("session_id") or "")

        with log_bind(session_id=session_id):
            logger.info(
                "Chat stream started",
                extra={
                    "event": CHAT_STREAM_STARTED,
                    "query_len": len(query or ""),
                    "context_keys": sorted(_ctx.keys()),
                },
            )
            yield json_line({"type": "status", "content": "SSE 连接已建立，开始处理聊天请求"})

            # Store the original (raw) user query so write-back paths always use
            # it for the user turn — never the enriched version that contains
            # injected conversation history.
            _ctx["__original_query"] = query

            # 加载近 10 轮 role/content 历史：Redis 优先，缺失时回源 ES 并回填 Redis。
            is_first_turn = False
            if session_id:
                try:
                    from app.services.memory.session_store import (
                        DEFAULT_WINDOW_SIZE,
                        get_session_store,
                    )

                    store = get_session_store()
                    store.ensure_session(session_id)
                    history_messages = store.load_history_messages(
                        session_id,
                        max_turns=DEFAULT_WINDOW_SIZE,
                    )
                    is_first_turn = len(history_messages) == 0
                    if history_messages:
                        _ctx["history_messages"] = history_messages
                        logger.info(
                            "会话历史注入成功: session_id=%s turns=%s",
                            session_id,
                            len(history_messages),
                        )
                    else:
                        logger.info("无会话历史: session_id=%s", session_id)
                except Exception as exc:
                    logger.warning("加载会话历史失败: session_id=%s error=%s", session_id, exc)

            run = await orchestrator.create_agent_run(query=query, context=_ctx)

            with log_bind(run_id=run.run_id):
                logger.info("AgentRun 已创建: run_id=%s", run.run_id)
                agents = await self._load_enabled_agents()
                logger.info("已加载子智能体: run_id=%s count=%s", run.run_id, len(agents))
                queue: asyncio.Queue[object] = asyncio.Queue()
                sentinel = object()

                async def emit_event(data: Dict[str, Any]) -> None:
                    await queue.put(json_line(data))

                graph = self._create_graph(
                    run_id=run.run_id, event_callback=emit_event, context=_ctx
                )
                state: Dict[str, Any] = {
                    "query": query,
                    "history_messages": list(_ctx.get("history_messages") or []),
                    "conversation_context": {},
                    "context": _ctx,
                    "agents": agents,
                    "agent_results": [],
                    "evidence_packets": [],
                }

                yield json_line({"type": "run", "content": json.dumps(run.model_dump(), ensure_ascii=False)})
                yield json_line({"type": "status", "content": "LangGraph 工作流启动：主智能体正在识别意图"})

                async def run_workflow() -> None:
                    try:
                        logger.info("LangGraph 工作流开始: run_id=%s", run.run_id)
                        async for event in orchestrator.stream_graph_events(
                            run_id=run.run_id,
                            graph=graph,
                            state=state,
                            agents=agents,
                        ):
                            await queue.put(event)

                        pending_task_id = state.get("pending_task_id")
                        if pending_task_id:
                            logger.info(
                                "工作流出现 pending_task: run_id=%s task_id=%s",
                                run.run_id,
                                pending_task_id,
                            )
                            pending_task = await orchestrator.compute_tasks.get_task(
                                str(pending_task_id)
                            )
                            if pending_task and pending_task.kind == "external_compute":
                                response_content = str(
                                    state.get("final_report")
                                    or "工作流已挂起，等待外部计算任务完成。"
                                )
                                logger.info(
                                    "工作流挂起等待外部计算: run_id=%s task_id=%s",
                                    run.run_id,
                                    pending_task_id,
                                )
                                await queue.put(
                                    json_line({"type": "done", "content": response_content})
                                )
                                return
                            async for event in self._wait_for_run_terminal(run_id=run.run_id):
                                await queue.put(event)
                            return

                        response_content = str(state.get("final_report") or "").strip()
                        if not response_content:
                            response_content = "抱歉，本次没有生成有效回答。"
                            await queue.put(
                                json_line({"type": "content", "content": response_content})
                            )

                        tool_results = self._extract_tool_results(
                            state.get("agent_results") or []
                        )
                        for item in tool_results:
                            await queue.put(
                                json_line(
                                    {
                                        "type": "tool_result",
                                        "channel": "execution",
                                        **item,
                                    }
                                )
                            )
                        sid = _ctx.get("session_id") or ""
                        raw_user_query = str(_ctx.get("__original_query") or query)
                        if sid and response_content:
                            try:
                                self._persist_session_turns(
                                    session_id=sid,
                                    user_query=raw_user_query,
                                    assistant_content=response_content,
                                    context=_ctx,
                                )
                            except Exception:
                                pass
                        if sid and response_content:
                            try:
                                new_title = await self._maybe_update_session_title(
                                    session_id=sid,
                                    user_query=raw_user_query,
                                    is_first_turn=is_first_turn,
                                )
                                if new_title:
                                    await queue.put(
                                        json_line(
                                            {
                                                "type": "status",
                                                "phase": "title",
                                                "title": new_title,
                                            }
                                        )
                                    )
                            except Exception as exc:
                                logger.warning(
                                    "会话标题更新失败: session_id=%s error=%s", sid, exc
                                )
                        logger.info(
                            "LangGraph 工作流结束: run_id=%s final_len=%s",
                            run.run_id,
                            len(response_content or ""),
                        )
                        logger.info(
                            "Chat stream completed",
                            extra={
                                "event": CHAT_STREAM_COMPLETED,
                                "latency_ms": round(
                                    (time.perf_counter() - stream_start) * 1000
                                ),
                                "final_len": len(response_content or ""),
                            },
                        )
                        await queue.put(
                            json_line({"type": "done", "content": response_content})
                        )
                    except Exception as exc:
                        logger.exception(
                            "Chat stream failed",
                            extra={
                                "event": CHAT_STREAM_FAILED,
                                "latency_ms": round(
                                    (time.perf_counter() - stream_start) * 1000
                                ),
                            },
                        )
                        await orchestrator.fail_run(run_id=run.run_id, error=str(exc))
                        await queue.put(self.error_chunk(str(exc)))
                    finally:
                        await queue.put(sentinel)

                workflow_task = asyncio.create_task(run_workflow())
                try:
                    while True:
                        event = await queue.get()
                        if event is sentinel:
                            break
                        yield str(event)
                finally:
                    if not workflow_task.done():
                        logger.info(
                            "客户端断开或生成器结束，取消工作流: run_id=%s", run.run_id
                        )
                        workflow_task.cancel()
                        try:
                            await workflow_task
                        except asyncio.CancelledError:
                            pass


chat_workflow_service = ChatWorkflowService()
