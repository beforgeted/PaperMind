"""聊天工作流服务的向后兼容导入入口。"""

from app.services.chat_workflow_service import ChatWorkflowService, chat_workflow_service


WorkflowService = ChatWorkflowService
workflow_service = chat_workflow_service


__all__ = ["ChatWorkflowService", "WorkflowService", "chat_workflow_service", "workflow_service"]
