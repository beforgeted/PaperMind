"""任务数据模型。"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class TaskRequest(BaseModel):
    query: str = Field(..., description="任务输入")


class TaskResult(BaseModel):
    status: str
    content: str = ""


class ComputeTaskRecord(BaseModel):
    task_id: str
    run_id: str
    step_id: str
    agent_id: Optional[str] = None
    kind: str = "external_compute"
    status: str = "pending"
    input: Dict[str, Any] = Field(default_factory=dict)
    result: Dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    created_at: str
    updated_at: str


class ComputeTaskCompleteRequest(BaseModel):
    result: Dict[str, Any] = Field(default_factory=dict, description="外部计算或人工回填结果")


class ComputeTaskFailRequest(BaseModel):
    error: str = Field(..., description="任务失败原因")


class ComputeTaskResponse(BaseModel):
    task: ComputeTaskRecord
