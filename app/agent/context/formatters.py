"""上下文块格式化工具。"""

from __future__ import annotations

import json
from typing import Any


def format_upstream_results(agent_results: list[dict[str, Any]]) -> str:
    """格式化上游子 Agent 输出块。"""
    if not agent_results:
        return "暂无上游 Agent 输出。"

    blocks: list[str] = []
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


def format_resolved_entities(resolved_entities: dict[str, str]) -> str:
    """格式化指代消解结果。"""
    if not resolved_entities:
        return ""
    lines = [f"- {key} → {value}" for key, value in resolved_entities.items()]
    return "## 已消解指代\n" + "\n".join(lines)


def format_evidence_packets(evidence_packets: list[dict[str, Any]]) -> str:
    """格式化检索证据包。"""
    if not evidence_packets:
        return "暂无检索证据包。"

    blocks: list[str] = []
    for index, packet in enumerate(evidence_packets, 1):
        tool_name = packet.get("tool_name") or "unknown"
        agent_id = packet.get("agent_id") or ""
        content = str(packet.get("content") or "")[:3000]
        blocks.append(
            "\n".join(
                [
                    f"{index}. 工具：{tool_name}（来自 {agent_id}）",
                    f"结果：\n{content}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_sub_agent_human_content(
    *,
    query: str,
    standalone_query: str,
    stage_task: str,
    policy_use_standalone: bool,
    include_upstream: bool,
    agent_results: list[dict[str, Any]],
    include_evidence: bool,
    evidence_packets: list[dict[str, Any]],
    resolved_entities: dict[str, str],
) -> str:
    """组装子 Agent 的 Human 消息正文。"""
    effective_query = standalone_query if policy_use_standalone and standalone_query else query
    parts: list[str] = [f"用户原始需求：\n{effective_query}"]

    entities_block = format_resolved_entities(resolved_entities)
    if entities_block:
        parts.append(entities_block)

    parts.append(f"当前阶段任务：\n{stage_task}")

    if include_upstream:
        parts.append(f"上游阶段输出：\n{format_upstream_results(agent_results)}")

    if include_evidence:
        parts.append(f"检索证据：\n{format_evidence_packets(evidence_packets)}")

    parts.append("请基于用户需求和上游输出完成本阶段任务；不要重复执行其他阶段职责。")
    return "\n\n".join(parts)


def build_summary_user_content(
    *,
    query: str,
    standalone_query: str,
    sub_agent_results: list[dict[str, Any]],
    evidence_packets: list[dict[str, Any]],
    include_evidence: bool,
) -> str:
    """组装汇总 Agent 的 Human 消息正文。"""
    blocks = [
        f"用户原始问题：\n{query}",
        f"独立问题（消解指代后）：\n{standalone_query or query}",
        "子智能体执行结果：",
    ]
    for index, result in enumerate(sub_agent_results, 1):
        error = result.get("error") or ""
        content = result.get("content") or ""
        status = f"执行失败：{error}" if error else content
        external_task = result.get("external_task")
        if external_task:
            status = "\n\n".join(
                [
                    status,
                    "真实工具执行结果（external_task）：",
                    json_text(external_task),
                ]
            )
        blocks.append(
            "\n".join(
                [
                    f"{index}. 子智能体：{result.get('agent_name')}",
                    f"任务：{result.get('task')}",
                    f"结果：\n{status}",
                ]
            )
        )

    if include_evidence and evidence_packets:
        blocks.append(f"论文证据包：\n{format_evidence_packets(evidence_packets)}")

    return "\n\n".join(blocks)


def json_text(value: Any) -> str:
    """JSON 安全序列化。"""
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        return str(value)
