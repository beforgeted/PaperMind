"""PaperMind research agent entrypoint."""

from __future__ import annotations

import inspect
import json
import os
from typing import Any, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch  # noqa: F401
except Exception:
    pass

from app.agent.prompts import build_system_prompt
from app.agent.session import build_user_query, is_task_status_query
from app.agent.registry import get_tools
from app.services.llm_service import get_llm

SYSTEM_PROMPT = build_system_prompt()

_agent: Optional[Any] = None
_agent_prompt_injected = False


def _create_agent() -> Any:
    global _agent, _agent_prompt_injected
    if _agent is not None:
        return _agent

    from langchain.agents import create_agent  # type: ignore[import-not-found]

    llm = get_llm()
    tools = get_tools("all")
    params = inspect.signature(create_agent).parameters
    if "system_prompt" in params:
        _agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
        _agent_prompt_injected = True
    elif "prompt" in params:
        _agent = create_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)
        _agent_prompt_injected = True
    elif "state_modifier" in params:
        _agent = create_agent(model=llm, tools=tools, state_modifier=SYSTEM_PROMPT)
        _agent_prompt_injected = True
    else:
        _agent = create_agent(model=llm, tools=tools)
        _agent_prompt_injected = False
    return _agent


def _message_type(message: Any) -> str:
    explicit_type = getattr(message, "type", None)
    if explicit_type:
        return str(explicit_type)
    return message.__class__.__name__


def _message_content(message: Any) -> Any:
    return getattr(message, "content", "")


def _message_name(message: Any) -> Optional[str]:
    name = getattr(message, "name", None)
    if name:
        return str(name)
    additional_kwargs = getattr(message, "additional_kwargs", None) or {}
    name = additional_kwargs.get("name")
    return str(name) if name else None


def _raw_message(message: Any) -> dict:
    raw = {
        "type": _message_type(message),
        "name": _message_name(message),
        "content": _message_content(message),
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        raw["tool_calls"] = tool_calls
    return raw


def _parse_tool_payload(content: Any) -> Optional[dict]:
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _source_from_result(result: dict) -> dict:
    return {
        "paper_id": result.get("paper_id"),
        "title": result.get("title"),
        "section": result.get("section"),
        "section_type": result.get("section_type"),
        "chunk_ids": result.get("chunk_ids") or [],
        "parent_id": result.get("parent_id"),
        "score": result.get("score"),
        "metadata": result.get("metadata") or {},
    }


def _source_key(result: dict) -> Optional[tuple]:
    paper_id = result.get("paper_id")
    title = result.get("title")
    parent_id = result.get("parent_id")
    if paper_id and parent_id:
        return ("paper_parent", paper_id, parent_id)
    if paper_id:
        return ("paper", paper_id)
    if title:
        return ("title", title)
    return None


def _extract_agent_outputs(
    messages: list[Any],
) -> tuple[str, list[dict], list[dict], list[str]]:
    answer = ""
    contexts: list[dict] = []
    sources: list[dict] = []
    used_tools: list[str] = []
    seen_contexts: set[tuple] = set()
    seen_sources: set[tuple] = set()

    for message in messages:
        message_type = _message_type(message)
        content = _message_content(message)
        if message_type in {"ai", "AIMessage"} and content:
            answer = str(content)

        tool_name = _message_name(message)
        if message_type in {"tool", "ToolMessage"}:
            if tool_name and tool_name not in used_tools:
                used_tools.append(tool_name)
            payload = _parse_tool_payload(content)
            if not payload:
                continue
            payload_tool_name = payload.get("tool_name")
            if payload_tool_name and payload_tool_name not in used_tools:
                used_tools.append(str(payload_tool_name))
            if payload.get("error"):
                continue

            for source in payload.get("sources") or []:
                if not isinstance(source, dict):
                    continue
                key = _source_key(source) or ("source", len(sources))
                if key not in seen_sources:
                    sources.append(source)
                    seen_sources.add(key)

            for item in payload.get("results") or []:
                if not isinstance(item, dict):
                    continue
                for context in item.get("contexts") or [item]:
                    if not isinstance(context, dict):
                        continue
                    key = _source_key(context)
                    context_key = key or ("context", len(contexts))
                    if context_key not in seen_contexts:
                        contexts.append(context)
                        seen_contexts.add(context_key)
                    if key and key not in seen_sources:
                        sources.append(_source_from_result(context))
                        seen_sources.add(key)

        for call in getattr(message, "tool_calls", None) or []:
            call_name = call.get("name") if isinstance(call, dict) else None
            if call_name and call_name not in used_tools:
                used_tools.append(str(call_name))

    return answer, contexts, sources, used_tools


async def answer_with_agent(
    query: str,
    top_k: int | None = None,
    task_id: str | None = None,
) -> dict:
    """Answer a PaperMind question with the research agent."""
    if is_task_status_query(query) and not task_id:
        return {
            "answer": "请提供 task_id 后再查询论文解析状态。",
            "contexts": [],
            "sources": [],
            "used_tools": [],
            "raw_messages": [],
        }
    try:
        agent = _create_agent()
        user_query = build_user_query(query, top_k, task_id)
        if _agent_prompt_injected:
            payload = {"messages": [("human", user_query)]}
        else:
            payload = {"messages": [("system", SYSTEM_PROMPT), ("human", user_query)]}
        result = await agent.ainvoke(payload)

        messages = result.get("messages", []) if isinstance(result, dict) else []
        answer, contexts, sources, used_tools = _extract_agent_outputs(messages)
        return {
            "answer": answer,
            "contexts": contexts,
            "sources": sources,
            "used_tools": used_tools,
            "raw_messages": [_raw_message(message) for message in messages],
        }
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        return {
            "answer": f"Agent 调用失败：{error}",
            "contexts": [],
            "sources": [],
            "used_tools": [],
            "raw_messages": [],
            "error": error,
        }
