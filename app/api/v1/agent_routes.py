"""Agent 对话路由 — 挂载到 /api/v1/agent/* 以匹配前端期望的端点。

提供两个端点:
  POST /agent/chat         — 非流式 JSON 响应
  POST /agent/chat/stream  — SSE 流式响应
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.log_context import log_bind
from app.observability.events import (
    AGENT_CHAT_COMPLETED,
    AGENT_CHAT_FAILED,
    AGENT_CHAT_STARTED,
)
from app.services.chat_workflow_service import chat_workflow_service, json_line

logger = logging.getLogger(__name__)

agent_router = APIRouter(prefix="/agent", tags=["Agent 对话"])


def _extract_sources_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract paper citation info from a tool_result SSE payload.

    Looks into several possible sub-structures:
      - payload["content"]  (stringified JSON from tool output)
      - payload["raw"]["result"]["tool_result"]["content"]
    """
    sources: List[Dict[str, Any]] = []
    content_value = payload.get("content")
    if isinstance(content_value, str):
        try:
            parsed = json.loads(content_value)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            content_value = parsed
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    sources.append(item)
            return sources
    if isinstance(content_value, dict):
        sources.append(content_value)
    return sources


_CITATION_RE = __import__("re").compile(
    r"\[parent_id\s*[:\-]?\s*([^\]]+)\]|\[来源\s*[:\-]?\s*([^\]]+)\]|source\s*[:\-]\s*(\S+)",
    __import__("re").IGNORECASE,
)


def _extract_sources_from_text(answer_text: str) -> List[Dict[str, Any]]:
    """Fallback: extract citation markers from answer text when structured
    tool_results are empty.  Scans for patterns like [parent_id: ...],
    [来源: ...], source: ... and also attempts to extract stand-alone URLs.
    """
    sources: List[Dict[str, Any]] = []
    for m in _CITATION_RE.finditer(answer_text):
        value = next((g for g in m.groups() if g), "").strip()
        if value:
            label = "citation"
            if "http" in value or "doi" in value.lower():
                label = "url"
            sources.append({"type": label, "value": value})
    return sources


# ---------------------------------------------------------------------------
# 并发控制 (复用 chat_api 的全局信号量)
# ---------------------------------------------------------------------------
from app.core.config import settings as app_settings

_AGENT_SEMAPHORE = asyncio.Semaphore(max(1, int(app_settings.chat_max_concurrent_tasks)))
_STREAM_QUEUE_MAX = max(1, int(app_settings.chat_stream_queue_max_size))
_STREAM_QUEUE_TIMEOUT = max(0.1, float(app_settings.chat_stream_queue_put_timeout))


class AgentChatRequest(BaseModel):
    """前端 AgentChatRequest 字段对齐。"""
    query: str = Field(..., description="用户输入")
    top_k: Optional[int] = Field(default=None, description="检索结果数（预留）")
    task_id: Optional[str] = Field(default=None, description="关联的论文任务 ID")
    session_id: Optional[str] = Field(default=None, description="对话会话 ID")


# ---------------------------------------------------------------------------
# POST /agent/chat — 非流式请求
# ---------------------------------------------------------------------------
@agent_router.post("/chat")
async def agent_chat(request: AgentChatRequest):
    """非流式 Agent 对话：收集完整 SSE 流后返回 JSON 响应。"""
    try:
        await asyncio.wait_for(_AGENT_SEMAPHORE.acquire(), timeout=0.05)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="聊天服务繁忙，请稍后重试",
        )

    start = time.perf_counter()
    try:
        with log_bind(session_id=request.session_id or ""):
            logger.info(
                "Agent chat started",
                extra={
                    "event": AGENT_CHAT_STARTED,
                    "query_len": len(request.query or ""),
                    "stream": False,
                    "task_id": request.task_id or "",
                },
            )
            answer_parts: List[str] = []
            tool_results: List[Dict[str, Any]] = []
            used_tools: List[str] = []
            sources: List[Dict[str, Any]] = []

            async for raw_line in chat_workflow_service.process_chat_stream(
                query=request.query,
                context={
                    "session_id": request.session_id,
                    "task_id": request.task_id,
                    "top_k": request.top_k,
                },
            ):
                line = str(raw_line).strip()
                if not line.startswith("data: "):
                    continue
                try:
                    payload = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                event_type = payload.get("type")
                content = payload.get("content", "")
                channel = payload.get("channel")

                if event_type == "content" and isinstance(content, str):
                    answer_parts.append(content)
                elif event_type in ("sub_agent_delta", "summary_delta"):
                    delta_text = str(payload.get("content", ""))
                    if delta_text:
                        answer_parts.append(delta_text)
                elif event_type == "done" and isinstance(content, str):
                    if content not in answer_parts:
                        answer_parts.append(content)
                elif event_type == "tool_result" and isinstance(payload, dict):
                    tool_results.append(payload)
                    if channel:
                        used_tools.append(channel)
                    extracted = _extract_sources_from_payload(payload)
                    sources.extend(extracted)
                elif event_type == "error":
                    if content and isinstance(content, str):
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=content,
                        )

            final_answer = "\n\n".join(filter(None, answer_parts))

            if not sources:
                sources.extend(_extract_sources_from_text(final_answer))

            logger.info(
                "Agent chat completed",
                extra={
                    "event": AGENT_CHAT_COMPLETED,
                    "latency_ms": round((time.perf_counter() - start) * 1000),
                    "answer_len": len(final_answer),
                    "stream": False,
                },
            )
            return {
                "query": request.query,
                "answer": final_answer,
                "contexts": tool_results,
                "sources": sources,
                "used_tools": list(dict.fromkeys(used_tools)),
            }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Agent chat failed",
            extra={
                "event": AGENT_CHAT_FAILED,
                "stream": False,
                "latency_ms": round((time.perf_counter() - start) * 1000),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"对话处理失败: {str(exc)}",
        )
    finally:
        _AGENT_SEMAPHORE.release()


# ---------------------------------------------------------------------------
# POST /agent/chat/stream — SSE 流式
# ---------------------------------------------------------------------------
@agent_router.post("/chat/stream")
async def agent_chat_stream(request: AgentChatRequest):
    """SSE 流式 Agent 对话。

    Translates internal backend events to the frontend SSE protocol:
      - sub_agent_delta / summary_delta → type: "delta"  with "text" field
      - done                         → type: "done"  with answer/contexts/sources/used_tools
    """
    try:
        await asyncio.wait_for(_AGENT_SEMAPHORE.acquire(), timeout=0.05)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="聊天服务繁忙，请稍后重试",
        )

    stream_start = time.perf_counter()

    sentinel = object()
    client_disconnected = asyncio.Event()
    queue: asyncio.Queue[object] = asyncio.Queue(maxsize=_STREAM_QUEUE_MAX)

    async def enqueue(item: object) -> None:
        while not client_disconnected.is_set():
            try:
                await asyncio.wait_for(queue.put(item), timeout=_STREAM_QUEUE_TIMEOUT)
                return
            except asyncio.TimeoutError:
                logger.warning("Agent 流式队列积压，等待客户端消费")

    async def run_task() -> None:
        try:
            with log_bind(session_id=request.session_id or ""):
                logger.info(
                    "Agent chat started",
                    extra={
                        "event": AGENT_CHAT_STARTED,
                        "query_len": len(request.query or ""),
                        "stream": True,
                        "task_id": request.task_id or "",
                    },
                )
                async for chunk in chat_workflow_service.process_chat_stream(
                    query=request.query,
                    context={
                        "session_id": request.session_id,
                        "task_id": request.task_id,
                        "top_k": request.top_k,
                    },
                ):
                    await enqueue(str(chunk))
        except Exception as exc:
            logger.exception(
                "Agent chat failed",
                extra={
                    "event": AGENT_CHAT_FAILED,
                    "stream": True,
                    "latency_ms": round((time.perf_counter() - stream_start) * 1000),
                },
            )
            await enqueue(json_line({"type": "error", "content": str(exc)}))
        finally:
            try:
                await enqueue(sentinel)
            finally:
                _AGENT_SEMAPHORE.release()

    asyncio.get_running_loop().create_task(run_task())

    async def generate():
        """Read raw SSE lines from the queue, translate to frontend protocol."""
        answer_parts: List[str] = []
        tool_results: List[Dict[str, Any]] = []
        used_tools: List[str] = []
        sources: List[Dict[str, Any]] = []
        done_sent = False
        streamed_delta = False

        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break

                raw = str(item).strip()
                if not raw.startswith("data: "):
                    continue

                try:
                    payload = json.loads(raw[6:])
                except json.JSONDecodeError:
                    continue

                event_type = payload.get("type", "")
                content = payload.get("content", "")
                channel = payload.get("channel", "")

                # ── streaming delta (仅最终汇总；子 Agent 中间结果不展示) ──
                if event_type == "summary_delta":
                    text = str(payload.get("content", ""))
                    if text:
                        streamed_delta = True
                        yield json_line({"type": "delta", "text": text})

                elif event_type == "sub_agent_delta":
                    pass

                # ── content blocks (done 时兜底；勿重复整段 delta) ───
                elif event_type == "content" and isinstance(content, str):
                    answer_parts.append(content)
                    if content.strip() and not streamed_delta:
                        yield json_line({"type": "delta", "text": content})

                # ── tool results (collect for done payload) ─────────
                elif event_type == "tool_result" and isinstance(payload, dict):
                    tool_results.append(payload)
                    tool_name = payload.get("tool_name") or channel or ""
                    if tool_name:
                        used_tools.append(tool_name)
                    # Extract paper sources from tool result content
                    extracted_sources = _extract_sources_from_payload(payload)
                    sources.extend(extracted_sources)

                # ── done -- emit with full answer fields ────────────
                elif event_type == "done":
                    if not done_sent:
                        done_sent = True
                        if isinstance(content, str) and content.strip():
                            final_answer = content.strip()
                        else:
                            final_answer = "\n\n".join(filter(None, answer_parts))
                        # Fallback: extract citation markers from answer text
                        # if no structured tool_results provided sources
                        if not sources:
                            sources.extend(_extract_sources_from_text(final_answer))
                        logger.info(
                            "Agent chat completed",
                            extra={
                                "event": AGENT_CHAT_COMPLETED,
                                "latency_ms": round((time.perf_counter() - stream_start) * 1000),
                                "answer_len": len(final_answer),
                                "stream": True,
                            },
                        )
                        yield json_line(
                            {
                                "type": "done",
                                "answer": final_answer,
                                "contexts": tool_results,
                                "sources": sources,
                                "used_tools": list(dict.fromkeys(used_tools)),
                            }
                        )

                # ── title update (explicit shape for frontend) ───────
                elif event_type == "status" and payload.get("phase") == "title":
                    yield json_line(
                        {
                            "type": "status",
                            "phase": "title",
                            "title": str(payload.get("title") or ""),
                        }
                    )

                # ── pass-through events ─────────────────────────────
                elif event_type in ("error", "status", "thinking", "heartbeat", "run"):
                    yield raw

                else:
                    # Unknown event types: pass through
                    yield raw

        finally:
            client_disconnected.set()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
