"""最终回答汇总服务。"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.services.llm_service import llm_service, message_content_to_text


class SummaryService:
    """把子智能体输出汇总成面向用户的最终回答。"""

    def build_summary_prompt(self, decision: Dict[str, Any]) -> str:
        return (
            "你是 PaperMind 学术研究多智能体系统的主 Agent。各子 Agent 已按流水线执行完毕，"
            "请基于其输出给用户一份统一、清晰的中文最终回答。\n\n"
            "## 汇总要求\n"
            "- 最终回答必须优先呈现真实工具执行结果；不要只写概述。\n"
            "- 如果子智能体结果中包含 external_task / tool_result，必须明确列出工具名、执行状态、入参摘要、输出路径、文件大小、错误信息等可用字段。\n"
            "- 工具返回的原始结果不可丢弃；可以先给结构化执行结果，再给解释和后续建议。\n"
            "- 子智能体结果多为 JSON：请解析 result、evidence、assumptions、uncertainty、is_mock、status。\n"
            "- 只基于子智能体结果作答，不要编造未提供的分数、文献或实验结论。\n"
            "- 若 is_mock 为 true 或 status 为 tool_required，必须在回答中明确说明当前为模拟/待工具结果。\n"
            "- 整合多阶段结论时去重；保留各阶段关键发现、风险与下一步建议。\n"
            "- 存在执行失败的子智能体时，简要说明并尽量保留成功阶段结果。\n"
            "- 不要暴露 agent_id、route_plan 等内部调度细节，除非用户明确要求。\n"
            "- 不提供诊疗建议、人体剂量或监管结论。\n"
            "- 输出中文。\n\n"
            f"## 调度原因\n{decision.get('reason') or ''}"
        )

    def build_summary_user_message(self, *, query: str, sub_agent_results: List[Dict[str, str]]) -> str:
        blocks = [f"用户原始问题：\n{query}", "子智能体执行结果："]
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
                        self._json_text(external_task),
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
        return "\n\n".join(blocks)

    def _json_text(self, value: Any) -> str:
        import json

        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except TypeError:
            return str(value)

    async def stream_summary(
        self,
        *,
        query: str,
        decision: Dict[str, Any],
        sub_agent_results: List[Dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        llm = llm_service.create_chat_model(
            llm_service.main_agent_model_config(),
            streaming=True,
            timeout=120,
            max_tokens=4096,
        )
        messages = [
            SystemMessage(content=self.build_summary_prompt(decision)),
            HumanMessage(content=self.build_summary_user_message(query=query, sub_agent_results=sub_agent_results)),
        ]
        async for chunk in llm.astream(messages):
            text = message_content_to_text(getattr(chunk, "content", ""))
            if text:
                yield text

    async def summarize_results(
        self,
        *,
        query: str,
        decision: Dict[str, Any],
        sub_agent_results: List[Dict[str, str]],
        chunk_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> str:
        chunks = []
        async for text in self.stream_summary(
            query=query,
            decision=decision,
            sub_agent_results=sub_agent_results,
        ):
            chunks.append(text)
            if chunk_callback is not None:
                await chunk_callback(text)
        return "".join(chunks).strip()


summary_service = SummaryService()
