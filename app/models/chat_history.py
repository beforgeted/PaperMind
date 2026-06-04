import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

def generate_uuid() -> str:
    return str(uuid.uuid4())

class SessionModel(Base):
    """会话模型"""
    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_role_id = Column(String(36), ForeignKey("user_roles.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255), nullable=True, comment="对话标题")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 级联关系：查询会话时可直接点出 messages，且删除会话时自动清理消息
    messages = relationship(
        "MessageModel", 
        back_populates="session", 
        cascade="all, delete-orphan",
        order_by="MessageModel.created_at"
    )


class MessageModel(Base):
    """消息模型"""
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False, comment="角色：system, user, assistant, tool")
    content = Column(Text, nullable=False, comment="具体对话内容或工具返回的 JSON")
    token_count = Column(Integer, default=0, comment="用于后续滑动窗口截断计算")
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("SessionModel", back_populates="messages")


class SessionFileModel(Base):
    """会话关联文件模型"""
    __tablename__ = "session_files"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=True)
    minio_bucket = Column(String(100), nullable=False)
    minio_path = Column(String(500), nullable=False)
    file_size = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.now)
