"""PaperMind 学术研究子 Agent 共享提示词片段。"""

SUB_AGENT_JSON_ENVELOPE = """
## 输出格式（与系统契约对齐）

对外只输出一行合法 JSON（不要用 Markdown 代码块包裹），便于编排器与下游 Agent 承接。

统一外壳（顶层字段）：
{
  "task_id": "string",
  "agent_name": "string",
  "status": "completed | needs_clarification | tool_required | failed",
  "input_used": {},
  "result": {},
  "evidence": [{"source_type": "paper | database | user_input | tool_result", "source": "string", "statement": "string"}],
  "assumptions": [],
  "uncertainty": [],
  "tool_requests": [],
  "next_recommended_agents": [],
  "confidence": "high | medium | low"
}

通用要求：
- 把专用分析内容写入 result；不要把大段说明放在 JSON 外。
- 所有论文信息必须来自工具调用结果或上游 Agent 真实输出，禁止编造。
- 区分已知事实、合理推断与待验证假设；不确定时用 status=needs_clarification。
- 需要真实工具时 status=tool_required，并在 tool_requests 中给出 tool_name 和 arguments。
- 解析 HumanMessage 中的「上游阶段输出」JSON，在 input_used 中注明引用了哪些字段。
- 仅用于学术研究辅助，不替代正式学术评审或出版决策。
"""
