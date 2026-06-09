"""PaperMind 多 Agent 工作流共享状态。"""

from typing import Any, Awaitable, Callable, Dict, List, Optional, TypedDict


AgentRecord = Dict[str, Any]
Decision = Dict[str, Any]
AgentResult = Dict[str, Any]
HistoryMessage = Dict[str, str]
ConversationContext = Dict[str, Any]
EvidencePacket = Dict[str, Any]


OrchestrateFn = Callable[..., Awaitable[Decision]]
RunAgentFn = Callable[..., Awaitable[AgentResult]]
SummarizeFn = Callable[..., Awaitable[str]]


class PaperMindState(TypedDict, total=False):
    """LangGraph 运行时状态黑板（存结构化事实，不存临时 messages）。"""

    query: str
    history_messages: List[HistoryMessage]
    conversation_context: ConversationContext
    context: Dict[str, Any]
    agents: List[AgentRecord]
    decision: Decision
    selected_agents: List[str]
    agent_results: List[AgentResult]
    evidence_packets: List[EvidencePacket]
    last_agent_id: str
    pending_task_id: Optional[str]
    pending_step_id: Optional[str]
    resume_payload: Optional[Dict[str, Any]]
    final_report: str
    error: Optional[str]
