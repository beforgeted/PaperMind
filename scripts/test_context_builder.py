#!/usr/bin/env python3
"""ContextBuilder 单元测试（无需 Redis/ES）。

用法:
    conda run -n papermind python scripts/test_context_builder.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _banner(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def test_router_ten_turns() -> None:
    _banner("1. Router 近 10 轮历史")
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from app.agent.context import context_builder

    history = []
    for i in range(10):
        history.append({"role": "user", "content": f"用户问题{i}"})
        history.append({"role": "assistant", "content": f"助手回答{i}"})

    state = {
        "query": "继续上次的话题",
        "history_messages": history,
        "agents": [{"agent_id": "retrieval-agent", "name": "检索", "description": "x"}],
    }
    messages = context_builder.build_for_router(state, agents=state["agents"])
    assert isinstance(messages[0], SystemMessage)
    assert messages[-1].content == "继续上次的话题"
    assert isinstance(messages[-1], HumanMessage)
    history_msgs = [m for m in messages[1:-1] if isinstance(m, (HumanMessage, AIMessage))]
    assert len(history_msgs) == 10
    _ok(f"Router messages={len(messages)}（1 System + 10 历史 + 1 Human）")


def test_retrieval_no_history() -> None:
    _banner("2. Retrieval Agent 默认无历史")
    from langchain_core.messages import HumanMessage, SystemMessage

    from app.agent.context import context_builder

    history = [
        {"role": "user", "content": "帮我总结 LCDNet"},
        {"role": "assistant", "content": "LCDNet 是..."},
    ]
    state = {
        "query": "它的方法有什么不足？",
        "history_messages": history,
        "conversation_context": {
            "standalone_query": "LCDNet 的方法有什么不足？",
            "resolved_entities": {"它": "LCDNet"},
            "context_requirements": {},
        },
        "decision": {
            "queries_per_agent": {"retrieval-agent": "检索 LCDNet 方法局限"},
            "query_for_agent": "检索 LCDNet 方法局限",
        },
        "agent_results": [],
        "evidence_packets": [],
        "context": {},
    }
    agent = {
        "agent_id": "retrieval-agent",
        "name": "检索 Agent",
        "description": "检索",
        "promptConfig": {"prompt": "你是检索 Agent"},
    }
    messages = context_builder.build_for_sub_agent(
        state,
        agent_id="retrieval-agent",
        agent=agent,
    )
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    human = messages[1]
    assert isinstance(human, HumanMessage)
    assert "LCDNet 的方法有什么不足？" in human.content
    assert "它" not in human.content.split("已消解指代")[0]
    assert "LCDNet" in human.content
    _ok("Retrieval 使用 standalone_query，无历史 messages")


def test_standalone_via_requirements() -> None:
    _banner("3. context_requirements 覆盖策略")
    from langchain_core.messages import HumanMessage

    from app.agent.context import context_builder

    state = {
        "query": "它的不足？",
        "history_messages": [{"role": "user", "content": "x"}],
        "conversation_context": {
            "standalone_query": "Transformer 的不足",
            "resolved_entities": {},
            "context_requirements": {
                "writing-agent": {"history_turns": 0, "use_standalone_query": True},
            },
        },
        "decision": {
            "queries_per_agent": {"writing-agent": "润色分析"},
            "query_for_agent": "润色分析",
        },
        "agent_results": [],
        "evidence_packets": [],
        "context": {},
    }
    agent = {
        "agent_id": "writing-agent",
        "name": "写作",
        "description": "写作",
        "promptConfig": {"prompt": "写作 Agent"},
    }
    messages = context_builder.build_for_sub_agent(
        state,
        agent_id="writing-agent",
        agent=agent,
    )
    assert len(messages) == 2
    human = messages[1]
    assert isinstance(human, HumanMessage)
    assert "Transformer 的不足" in human.content
    _ok("writing-agent history_turns=0 时仅 System + Human")


def test_upstream_in_second_agent() -> None:
    _banner("4. 上游输出传递给后续子 Agent")
    from langchain_core.messages import HumanMessage

    from app.agent.context import context_builder

    state = {
        "query": "写综述",
        "history_messages": [],
        "conversation_context": {
            "standalone_query": "写综述",
            "resolved_entities": {},
            "context_requirements": {},
        },
        "decision": {
            "queries_per_agent": {"summary-agent": "生成综述"},
            "query_for_agent": "生成综述",
        },
        "agent_results": [
            {
                "agent_id": "retrieval-agent",
                "agent_name": "检索",
                "task": "检索",
                "content": "找到 5 篇相关论文",
                "error": "",
            }
        ],
        "evidence_packets": [],
        "context": {},
    }
    agent = {
        "agent_id": "summary-agent",
        "name": "综述",
        "description": "综述",
        "promptConfig": {"prompt": "综述 Agent"},
    }
    messages = context_builder.build_for_sub_agent(
        state,
        agent_id="summary-agent",
        agent=agent,
    )
    human = messages[1]
    assert isinstance(human, HumanMessage)
    assert "找到 5 篇相关论文" in human.content
    assert "上游阶段输出" in human.content
    _ok("summary-agent 默认 include_upstream=true")


def test_summary_evidence_packets() -> None:
    _banner("5. 汇总阶段包含 evidence_packets")
    from langchain_core.messages import HumanMessage

    from app.agent.context import context_builder

    state = {
        "query": "问题",
        "history_messages": [],
        "conversation_context": {"standalone_query": "问题", "resolved_entities": {}},
        "decision": {"reason": "测试"},
        "agent_results": [
            {"agent_name": "检索", "task": "t", "content": "结果", "error": ""},
        ],
        "evidence_packets": [
            {"tool_name": "retrieve_evidence", "agent_id": "retrieval-agent", "content": "证据片段"},
        ],
        "context": {},
    }
    messages = context_builder.build_for_summary(state)
    human = messages[-1]
    assert isinstance(human, HumanMessage)
    assert "证据片段" in human.content
    assert "论文证据包" in human.content
    _ok("Summary Human 含 evidence_packets")


def test_build_conversation_context() -> None:
    _banner("6. conversation_context 派生")
    from app.agent.context import build_conversation_context

    ctx = build_conversation_context(
        query="它的不足？",
        decision={
            "standalone_query": "LCDNet 的不足",
            "resolved_entities": {"它": "LCDNet"},
            "context_requirements": {"retrieval-agent": {"history_turns": 0}},
        },
        history_messages=[{"role": "user", "content": "hi"}],
    )
    assert ctx["standalone_query"] == "LCDNet 的不足"
    assert ctx["depends_on_history"] is True
    assert ctx["resolved_entities"]["它"] == "LCDNet"
    _ok("build_conversation_context 字段正确")


def main() -> int:
    try:
        test_router_ten_turns()
        test_retrieval_no_history()
        test_standalone_via_requirements()
        test_upstream_in_second_agent()
        test_summary_evidence_packets()
        test_build_conversation_context()
        _banner("全部通过")
        print("ContextBuilder 实现正常。")
        return 0
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
