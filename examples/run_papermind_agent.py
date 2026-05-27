"""Minimal local smoke test for the PaperMind ReAct Agent.

Run from the project root:
    python examples/run_papermind_agent.py

Optionally scope the task-status question:
    $env:PAPERMIND_AGENT_TASK_ID="your-task-id"
    python examples/run_papermind_agent.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.runtime import answer_with_agent  # noqa: E402

QUESTIONS = [
    "帮我找几篇水下图像增强相关论文",
    "哪些论文使用了 UIEBD 数据集？",
    "这篇论文的方法流程是什么？",
    "我的论文解析任务现在是什么状态？",
]


async def _run() -> None:
    task_id = os.getenv("PAPERMIND_AGENT_TASK_ID")
    for index, question in enumerate(QUESTIONS, start=1):
        result = await answer_with_agent(
            question,
            top_k=5,
            task_id=task_id if "任务" in question else None,
        )
        print(f"\n=== Question {index} ===")
        print(f"query: {question}")
        print("\nanswer:")
        print(result.get("answer", ""))
        print("\nused_tools:")
        print(result.get("used_tools", []))
        print("\nsources_count:")
        print(len(result.get("sources", [])))


if __name__ == "__main__":
    asyncio.run(_run())
