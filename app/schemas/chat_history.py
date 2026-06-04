
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List


# ==================== MESSAGE SCHEMAS ====================

class MessageCreate(BaseModel):
    """创建消息时的入参"""
    role: str = Field(..., description="角色: system, user, assistant, tool", examples=["user"])
    content: str = Field(..., description="消息内容或工具 JSON 字符串")
    token_count: int = Field(0, description="当前消息消耗的 Token 数")


class MessageResponse(BaseModel):
    """返回消息给前端或 Agent 时的结构"""
    id: str
    session_id: str
    role: str
    content: str
    token_count: int
    created_at: datetime

    class Config:
        from_attributes = True


# ==================== SESSION SCHEMAS ====================

class SessionCreate(BaseModel):
    """创建会话时的入参"""
    user_role_id: str = Field(..., description="当前登录用户在此公司下的 user_role_id")
    title: Optional[str] = Field(None, description="会话标题")


class SessionUpdate(BaseModel):
    """更新会话（例如修改标题）时的入参"""
    title: str = Field(..., description="新的会话标题")


class SessionResponse(BaseModel):
    """返回会话基础信息的结构"""
    id: str
    user_role_id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SessionDetailResponse(SessionResponse):
    """返回完整会话详情（包含该会话下的所有历史消息列表）"""
    messages: List[MessageResponse] = []

class SessionFileResponse(BaseModel):
    id: str
    session_id: str
    file_name: str
    file_type: Optional[str]
    minio_bucket: str
    minio_path: str
    file_size: int
    created_at: datetime

    class Config:
        from_attributes = True