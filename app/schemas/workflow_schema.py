"""工作流数据模型。"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class WorkflowRunRequest(BaseModel):
    query: str = Field(..., description="用户输入")


class AgentResult(BaseModel):
    agent_id: str
    agent_name: str
    task: str
    content: str = ""
    error: str = ""


class WorkflowRunResult(BaseModel):
    content: str
    agent_results: List[AgentResult] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowStepRecord(BaseModel):
    step_id: str
    run_id: str
    index: int
    node_name: str
    title: str
    status: str = "pending"
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    task: str = ""
    result: Dict[str, Any] = Field(default_factory=dict)
    task_id: Optional[str] = None
    error: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    updated_at: str


class AgentRunRecord(BaseModel):
    run_id: str
    query: str
    status: str = "created"
    decision: Dict[str, Any] = Field(default_factory=dict)
    selected_agents: List[str] = Field(default_factory=list)
    steps: List[WorkflowStepRecord] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    agent_results: List[Dict[str, Any]] = Field(default_factory=list)
    pending_task_id: Optional[str] = None
    final_report: str = ""
    error: str = ""
    created_at: str
    updated_at: str


class AgentRunListResponse(BaseModel):
    runs: List[AgentRunRecord] = Field(default_factory=list)


class AgentRunResumeRequest(BaseModel):
    task_id: Optional[str] = Field(default=None, description="需要恢复的计算任务ID；为空时使用当前 pending_task_id")


class AgentRunResumeResponse(BaseModel):
    run: AgentRunRecord
