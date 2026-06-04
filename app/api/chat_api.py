"""聊天接口。"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from fastapi import BackgroundTasks
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.llm_service import get_settings
from app.services.mysql_service import mysql_service
from app.services.chat_workflow_service import chat_workflow_service

from app.services.chat_history_service import ChatHistoryService
from app.schemas.chat_history import MessageCreate


router = APIRouter(prefix="/api/v1", tags=["多智能体对话"])
logger = logging.getLogger(__name__)

settings = get_settings()
_CHAT_TASK_SEMAPHORE = asyncio.Semaphore(max(1, int(settings.chat_max_concurrent_tasks)))
_CHAT_STREAM_QUEUE_MAX_SIZE = max(1, int(settings.chat_stream_queue_max_size))
_CHAT_STREAM_QUEUE_PUT_TIMEOUT = max(0.1, float(settings.chat_stream_queue_put_timeout))


def get_db() -> Any:
    """聊天历史数据库会话暂未接入，避免路由导入时引用未定义依赖。"""
    raise HTTPException(status_code=503, detail="聊天历史数据库会话暂未接入")


class ChatRequest(BaseModel):
    query: str = Field(..., description="用户输入")
    user_id: Optional[str] = Field(default=None, description="用户ID")
    session_id: Optional[str] = Field(default=None, description="会话ID，用于多轮对话")
    files: List[Dict[str, Any]] = Field(default_factory=list, description="前端已上传文件的元数据")


async def _validate_chat_user(chat_request: ChatRequest) -> str:
    user_id = (chat_request.user_id or "").strip()
    if not settings.chat_user_validation_enabled:
        return user_id

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少用户标识，请在请求体 user_id 中传入用户ID",
        )

    try:
        is_valid = await mysql_service.validate_user(user_id)
    except ValueError as exc:
        logger.exception("用户校验配置错误: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    except RuntimeError as exc:
        logger.exception("用户校验服务不可用: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="用户不存在或已被禁用",
        )
    return user_id


@router.post("/chat/completions")
async def chat(chat_request: ChatRequest):
    """流式返回多智能体聊天结果。"""
    try:
        user_id = await _validate_chat_user(chat_request)

        try:
            await asyncio.wait_for(_CHAT_TASK_SEMAPHORE.acquire(), timeout=0.05)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="聊天服务繁忙，请稍后重试",
            )

        sentinel = object()
        client_disconnected = asyncio.Event()
        queue: asyncio.Queue = asyncio.Queue(maxsize=_CHAT_STREAM_QUEUE_MAX_SIZE)

        async def put_stream_item(item: object) -> None:
            while not client_disconnected.is_set():
                try:
                    await asyncio.wait_for(queue.put(item), timeout=_CHAT_STREAM_QUEUE_PUT_TIMEOUT)
                    return
                except asyncio.TimeoutError:
                    logger.warning("聊天流式队列积压，等待客户端消费")

        async def run_task() -> None:
            try:
                async for chunk in chat_workflow_service.process_chat_stream(
                    query=chat_request.query,
                    context={
                        "user_id": user_id,
                        "session_id": chat_request.session_id,
                        "files": chat_request.files,
                    },
                ):
                    await put_stream_item(chunk)
            except Exception as task_err:
                logger.exception("聊天后台任务异常: %s", task_err)
                await put_stream_item(chat_workflow_service.error_chunk(str(task_err)))
            finally:
                try:
                    await put_stream_item(sentinel)
                finally:
                    _CHAT_TASK_SEMAPHORE.release()

        asyncio.get_event_loop().create_task(run_task())

        async def generate():
            try:
                while True:
                    item = await queue.get()
                    if item is sentinel:
                        break
                    yield item
            finally:
                client_disconnected.set()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("聊天接口错误: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"聊天处理失败: {str(exc)}",
        )





######################历史记录##################
@router.post("/chat/messages/append", tags=["多智能体对话历史"])
def append_chat_message(
    session_id: str, 
    payload: MessageCreate, 
    background_tasks: BackgroundTasks, # 👈 注入后台任务对象
    db: Any = Depends(get_db)
):
    try:
        # 1. 快速写入：这里面的 append_message 只做常规的规则截断，保证 1ms 内返回
        msg = ChatHistoryService.append_message(
            db, session_id=session_id, role=payload.role, content=payload.content
        )
        
        # 2. 异步智能总结：如果是工具大文本，丢给后台任务慢慢跑，不阻塞当前接口返回
        if payload.role == "tool" and len(payload.content) > 5000:
            background_tasks.add_task(
                ChatHistoryService.async_llm_summarize, # 👈 我们要在 Service 里写的异步模型总结函数
                message_id=msg.id,
                raw_content=payload.content # 原始的超长文本
            )

        return {"status": "success", "message_id": msg.id}
    except Exception as e:
        logger.exception("追加历史消息失败: %s", e)
        raise HTTPException(status_code=500, detail=f"保存失败: {str(e)}")

@router.get("/chat/sessions/{session_id}/context", tags=["多智能体对话历史"])
def get_llm_cleaned_context(
    session_id: str, 
    max_tokens: int = 6000, 
    db: Any = Depends(get_db)
):
    """
    【接口 2：取】获取经过滑动窗口安全裁剪后的标准大模型上下文数组。
    直接返回类似：[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    try:
        context = ChatHistoryService.get_sliced_context(
            db, 
            session_id=session_id, 
            max_tokens=max_tokens
        )
        return {"context": context}
    except Exception as e:
        logger.exception("获取裁剪上下文失败: %s", e)
        raise HTTPException(status_code=500, detail=f"获取上下文失败: {str(e)}")
        
class FileBindPayload(BaseModel):
    file_name: str
    minio_path: str
    file_size: int = 0

@router.post("/chat/sessions/{session_id}/files/bind", tags=["多智能体对话历史"])
def bind_uploaded_file(
    session_id: str, 
    payload: FileBindPayload, 
    db: Any = Depends(get_db)
):
    """
    【给团队伙伴/前端使用的接口】登记绑定一个已经存入 MinIO 的研发成果文件
    """
    try:
        # 直接调用你写好的 service 方法
        bound_file = ChatHistoryService.bind_file_to_session(
            db=db,
            session_id=session_id,
            file_name=payload.file_name,
            minio_path=payload.minio_path,
            file_size=payload.file_size
        )
        return {
            "status": "success",
            "file_id": bound_file.id,
            "inferred_type": bound_file.file_type
        }
    except Exception as e:
        logger.exception("会话绑定文件失败: %s", e)
        raise HTTPException(status_code=500, detail=f"文件登记失败: {str(e)}")