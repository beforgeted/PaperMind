#!/usr/bin/env python3
"""记忆注入能力集成测试：Redis 热缓存、ES 回填、role/content 完整字段。

用法（项目根目录）:
    conda run -n papermind python scripts/test_memory_injection.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _banner(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    raise AssertionError(msg)


def test_message_utils() -> None:
    _banner("1. message_utils 转换")
    from app.services.memory.message_utils import (
        history_to_langchain_messages,
        turns_to_history_messages,
    )
    from app.services.memory.session_store import TurnRecord
    from langchain_core.messages import AIMessage, HumanMessage

    turns = [
        TurnRecord(role="user", content="第一轮问题"),
        TurnRecord(role="assistant", content="第一轮回答", allow_long=True),
    ]
    msgs = turns_to_history_messages(turns)
    assert msgs[0] == {"role": "user", "content": "第一轮问题"}
    assert msgs[1]["role"] == "assistant"

    lc = history_to_langchain_messages(msgs)
    assert isinstance(lc[0], HumanMessage)
    assert isinstance(lc[1], AIMessage)
    _ok("role/content → LangChain messages")


def test_redis_load(session_id: str) -> None:
    _banner("2. Redis 写入与 load_history_messages")
    from app.services.memory.session_store import DEFAULT_WINDOW_SIZE, TurnRecord, get_session_store

    store = get_session_store()
    store.delete_session(session_id)
    store.ensure_session(session_id)

    long_user = "U" * 600
    long_assistant = "A" * 2500
    store.append_turn(session_id, TurnRecord(role="user", content=long_user))
    store.append_turn(
        session_id,
        TurnRecord(role="assistant", content=long_assistant, allow_long=True),
    )

    history = store.load_history_messages(session_id, max_turns=DEFAULT_WINDOW_SIZE)
    assert len(history) == 2, f"期望 2 条，实际 {len(history)}"
    assert history[0]["role"] == "user"
    assert len(history[0]["content"]) == 600, "user content 应完整保留（非 200 字截断）"
    assert len(history[1]["content"]) == 2500, "assistant content 应完整保留"
    _ok(f"Redis 加载 {len(history)} 条，长文本未截断")


def test_es_fallback_and_hydrate(session_id: str) -> None:
    _banner("3. ES 回填 Redis（Redis 空 → ES 有 → 回填）")
    from app.services.memory.episodic import EpisodicMemory
    from app.services.memory.session_store import DEFAULT_WINDOW_SIZE, get_session_store

    store = get_session_store()
    episodic = EpisodicMemory()

    store.delete_session(session_id)
    try:
        episodic.delete_session(session_id)
    except Exception:
        pass

    now = "2026-06-08T12:00:00+00:00"
    episodic.create_session_doc(session_id, "测试会话", now)
    episodic.append_turn(
        session_id,
        {
            "role": "user",
            "content": "ES 中的用户问题",
            "turn_id": "t1",
            "timestamp": now,
        },
    )
    episodic.append_turn(
        session_id,
        {
            "role": "assistant",
            "content": "ES 中的助手回答，包含较长正文：" + "X" * 800,
            "turn_id": "t2",
            "timestamp": now,
            "used_tools": ["search_papers"],
        },
    )

    assert not store.has_redis_history(session_id), "Redis 应无历史"

    history = store.load_history_messages(session_id, max_turns=DEFAULT_WINDOW_SIZE)
    assert len(history) == 2, f"应从 ES 加载 2 条，实际 {len(history)}"
    assert history[0]["content"] == "ES 中的用户问题"
    assert len(history[1]["content"]) > 800
    assert store.has_redis_history(session_id), "ES 命中后应回填 Redis"
    _ok("ES → Redis 回填成功，content 完整")

    history_again = store.load_history_messages(session_id, max_turns=DEFAULT_WINDOW_SIZE)
    assert history_again == history, "第二次应从 Redis 读取相同数据"
    _ok("二次读取与 ES 回填结果一致")


def test_empty_history(session_id: str) -> None:
    _banner("4. 无历史时返回空列表")
    from app.services.memory.episodic import EpisodicMemory
    from app.services.memory.session_store import get_session_store

    store = get_session_store()
    episodic = EpisodicMemory()
    empty_id = f"{session_id}-empty"
    store.delete_session(empty_id)
    try:
        episodic.delete_session(empty_id)
    except Exception:
        pass

    history = store.load_history_messages(empty_id)
    assert history == [], f"无历史应返回 []，实际 {history!r}"
    _ok("Redis/ES 均无数据时返回空列表")


def test_main_agent_message_build(session_id: str) -> None:
    _banner("5. 主 Agent messages 组装（含 history_messages）")
    from app.agent.context import context_builder
    from app.services.memory.session_store import TurnRecord, get_session_store
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    store = get_session_store()
    store.delete_session(session_id)
    store.ensure_session(session_id)
    store.append_turn(session_id, TurnRecord(role="user", content="之前问过什么？"))
    store.append_turn(
        session_id,
        TurnRecord(role="assistant", content="之前回答过检索相关。", allow_long=True),
    )
    history = store.load_history_messages(session_id)

    agents = [{"agent_id": "retrieval-agent", "name": "检索", "description": "检索论文"}]
    state = {
        "query": "继续上次的主题",
        "history_messages": history,
    }
    messages = context_builder.build_for_router(state, agents=agents)
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "之前问过什么？"
    assert isinstance(messages[2], AIMessage)
    assert messages[2].content == "之前回答过检索相关。"
    assert isinstance(messages[3], HumanMessage)
    assert messages[3].content == "继续上次的主题"
    _ok(f"System + {len(history)} 轮历史 + 当前问题，共 {len(messages)} 条 messages")


def test_connectivity() -> None:
    _banner("0. 连接检查 Redis / ES")
    from app.core.config import settings
    from app.services.memory.episodic import EpisodicMemory
    from app.services.memory.session_store import get_session_store

    print(f"  redis_url={settings.redis_url}")
    print(f"  es_hosts={settings.es_hosts}")

    store = get_session_store()
    pong = store._redis.ping()
    assert pong is True
    _ok("Redis PING")

    episodic = EpisodicMemory()
    count = episodic.size
    _ok(f"ES episodic 索引可访问（文档数={count}）")


def main() -> int:
    session_id = f"memtest-{uuid.uuid4().hex[:8]}"
    print(f"测试 session_id: {session_id}")

    try:
        test_connectivity()
        test_message_utils()
        test_redis_load(session_id)
        test_es_fallback_and_hydrate(session_id)
        test_empty_history(session_id)
        test_main_agent_message_build(session_id)

        _banner("全部通过")
        print("记忆注入实现正常：Redis 热读、ES 回填、完整 role/content、主 Agent 多轮组装。")
        return 0
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        try:
            from app.services.memory.episodic import EpisodicMemory
            from app.services.memory.session_store import get_session_store

            get_session_store().delete_session(session_id)
            EpisodicMemory().delete_session(session_id)
            get_session_store().delete_session(f"{session_id}-empty")
            print(f"\n已清理测试数据: {session_id}")
        except Exception as cleanup_exc:
            print(f"\n清理测试数据时警告: {cleanup_exc}")


if __name__ == "__main__":
    raise SystemExit(main())
