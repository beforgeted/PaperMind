"""PaperMind LangGraph ReAct Agent entrypoint."""

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

from app.agents.paper_tools import get_paper_tools
from app.services.llm_service import get_llm

SYSTEM_PROMPT = """
你是 PaperMind 论文知识库智能助手，负责论文检索、论文阅读、论文总结、论文对比和基于知识库的问答。

你必须通过工具获取论文知识库中的证据后再回答。除非用户只是问系统使用说明，否则不要直接凭常识回答论文内容问题。
所有论文内容、论文数量、数据集、实验指标、方法结论都必须来自工具返回的 JSON。

你拥有以下工具：

1. search_paper_chunks
用于细粒度论文内容检索。
适合回答方法细节、实验结果、数据集、指标、消融实验、模块设计、损失函数、结论等问题。

2. search_paper_profiles
用于论文级检索、筛选、推荐和盘点。
适合回答“有哪些论文”“找几篇论文”“哪些论文用了某方法/数据集/任务”等问题。

3. deep_search_papers
用于论文级检索并补充证据。
适合回答“找相关论文并说明依据”“哪些论文用了某方法，证据是什么”“推荐论文并解释原因”等问题。

4. get_paper_profile
用于获取单篇论文画像。
适合回答某篇论文的标题、摘要、研究任务、方法标签、数据集、贡献点等问题。

5. get_task_status
用于查询论文上传、解析、切分、向量化、写入 Elasticsearch 的任务状态。
适合回答“论文解析完了吗”“为什么还不能问答”“任务现在是什么状态”等问题。

工具选择规则：
- 找论文、筛选论文、推荐论文：优先使用 search_paper_profiles。
- 找论文并要求给出依据：优先使用 deep_search_papers。
- 问具体论文内容：优先使用 search_paper_chunks。
- 问单篇论文概况：优先使用 get_paper_profile，必要时再调用 search_paper_chunks。
- 问上传、解析、入库状态：使用 get_task_status。
- 调用 get_task_status 前必须确认用户提供了明确的 task_id；没有 task_id 时不要猜测、不要使用 default。
- 如果工具结果为空或相关性较低，必须说明当前知识库中没有找到足够依据。
- 不要把不同论文的结论混在一起。
- 不要编造论文标题、作者、实验指标、数据集、结论。
- 不要输出“常被迁移至某任务”“强适配某场景”“被多项工作复用”等没有工具 content 或 metadata 明确支持的推断。

回答要求：
1. 回答必须基于工具返回的 content、title、section、metadata 等信息。
2. 尽量给出来源，包括 paper_id、title、section、parent_id、chunk_ids、score。
3. 如果证据不足，明确说明“当前检索结果不足以回答”。
4. 只能报告工具实际返回的结果数量。工具 JSON 中的 result_count 或 results 长度就是实际数量。
5. 如果用户要求找几篇但工具只返回 k 篇，应回答“当前知识库中检索到 k 篇”，不能为了凑数量补充无来源论文。
6. 推荐论文时，每篇论文必须对应一个真实 source；source 数量少于用户期望数量时，明确说明当前只检索到这些结果。
7. 如果是标签匹配，只能说“该论文因 task_tags/method_tags/dataset_tags 与查询相关而被召回”，不要扩展为模型推测。
8. 不得承诺“我可以继续检索”“我可立即执行扩展检索”“是否需要我继续”等后续动作；检索结果不足时直接说明不足。
9. 多论文对比优先使用 Markdown 表格。
"""

_agent: Optional[Any] = None
_agent_prompt_injected = False


def _build_user_query(query: str, top_k: Optional[int], task_id: Optional[str]) -> str:
    constraints: list[str] = []
    if top_k is not None:
        constraints.append(
            f"检索工具如支持 top_k，请优先使用 top_k={top_k}；top_k 只是检索上限，不代表必须回答 {top_k} 条。"
        )
        constraints.append("最终结果数量必须以工具返回的 result_count 或 results 实际长度为准。")
    if task_id:
        constraints.append(
            f"当前问题限定在 task_id={task_id} 对应的任务或论文上传结果范围内。"
        )
    if not constraints:
        return query
    return f"{query}\n\n约束：\n" + "\n".join(f"- {item}" for item in constraints)


def _create_agent() -> Any:
    global _agent, _agent_prompt_injected
    if _agent is not None:
        return _agent

    from langchain.agents import create_agent  # type: ignore[import-not-found]

    llm = get_llm()
    tools = get_paper_tools()
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


def _is_task_status_query(query: str) -> bool:
    normalized = query.strip().lower()
    return any(
        term in normalized
        for term in (
            "任务状态",
            "解析状态",
            "解析任务",
            "处理进度",
            "task status",
            "task_id",
            "任务",
            "上传",
            "入库",
            "向量化",
        )
    ) and any(
        term in normalized
        for term in (
            "状态",
            "进度",
            "完成",
            "完了吗",
            "失败",
            "成功",
            "status",
            "入库",
            "向量化",
        )
    )
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
            for item in payload.get("results") or []:
                if not isinstance(item, dict):
                    continue
                key = _source_key(item)
                context_key = key or ("context", len(contexts))
                if context_key not in seen_contexts:
                    contexts.append(item)
                    seen_contexts.add(context_key)
                if key and key not in seen_sources:
                    sources.append(_source_from_result(item))
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
    """Answer a PaperMind question with the LangGraph ReAct Agent."""
    if _is_task_status_query(query) and not task_id:
        return {
            "answer": "请提供 task_id 后再查询论文解析状态。",
            "contexts": [],
            "sources": [],
            "used_tools": [],
            "raw_messages": [],
        }
    try:
        agent = _create_agent()
        user_query = _build_user_query(query, top_k, task_id)
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
