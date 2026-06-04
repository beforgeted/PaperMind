from sqlalchemy.orm import Session as DBSession
from typing import List, Optional
from app.models.chat_history import SessionFileModel,SessionModel, MessageModel
from app.schemas.chat_history import SessionCreate, MessageCreate

import tiktoken

def estimate_token_count(text: str, model_name: str = "gpt-4") -> int:
    """计算文本的 Token 数量"""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base") # 默认降级编码
    return len(encoding.encode(text))

class ChatHistoryService:
    
    @staticmethod
    def create_session(db: DBSession, obj_in: SessionCreate) -> SessionModel:
        """为特定用户的公司角色创建一个新会话"""
        db_session = SessionModel(
            user_role_id=obj_in.user_role_id,
            title=obj_in.title or "新学术研讨会话"
        )
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        return db_session

    @staticmethod
    def get_session(db: DBSession, session_id: str) -> Optional[SessionModel]:
        """获取单个会话，包含关联的 messages"""
        return db.query(SessionModel).filter(SessionModel.id == session_id).first()

    @staticmethod
    def list_user_sessions(db: DBSession, user_role_id: str) -> List[SessionModel]:
        """获取某个用户在某家公司下的所有会话列表（按时间倒序排列）"""
        return db.query(SessionModel)\
                 .filter(SessionModel.user_role_id == user_role_id)\
                 .order_by(SessionModel.updated_at.desc())\
                 .all()

    @staticmethod
    def append_message(db: DBSession, session_id: str, role: str, content: str) -> MessageModel:
        """追加消息，自动计算 Token，并对 Tool 的巨量文本进行防御性截断"""
        
        # 针对大文本输出的防御机制
        if role == "tool" and len(content) > 3000:
            # 尝试解析大 JSON 返回
            import json
            summary_content = ""
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    summary_content = f"【系统自动摘要】：工具返回了包含 {len(data)} 个分子的列表。前2个分子示例：{json.dumps(data[:2], ensure_ascii=False)} ... [其余数据已省略以节省 Token]"
                elif isinstance(data, dict):
                    summary_content = f"【系统自动摘要】：工具返回了复杂的 JSON 结果，键包含：{list(data.keys())}。部分关键信息：{str(data)[:300]}... [其余数据已省略]"
            except Exception:
                # 如果不是 JSON（比如是几万行的超大坐标或序列文本）
                lines = content.splitlines()
                summary_content = (
                    f"【系统自动摘要】：工具输出了超大文本（共 {len(lines)} 行，约 {len(content)} 字符）。\n"
                    f"开头前5行：\n{chr(10).join(lines[:5])}\n"
                    f"...\n"
                    f"末尾后5行：\n{chr(10).join(lines[-5:])}\n"
                    f"⚠️ [中间长文本已被系统自动省略，避免爆大模型上下文窗口，完整原始文件已通过底层存储管理]"
                )
            
            # 将 content 替换为精简后的摘要，防止后续 get_sliced_context 时爆掉
            content = summary_content

        # 重新计算安全后的 token
        token_count = estimate_token_count(content)
        
        db_msg = MessageModel(
            session_id=session_id,
            role=role,
            content=content,
            token_count=token_count
        )
        db.add(db_msg)
        
        # 触发表的更新时间
        db_session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if db_session:
            from datetime import datetime
            db_session.updated_at = datetime.now()
            
        db.commit()
        db.refresh(db_msg)
        return db_msg   

    @staticmethod
    def get_recent_context(db: DBSession, session_id: str, limit: int = 20) -> List[MessageModel]:
        """获取最近的 N 条消息，用于给大模型做滚动窗口上下文（后续可配合 Token 计算精细化截断）"""
        return db.query(MessageModel)\
                 .filter(MessageModel.session_id == session_id)\
                 .order_by(MessageModel.created_at.desc())\
                 .limit(limit)\
                 .all()[::-1]  # 倒序取出来后再正序排列返回给大模型

    @staticmethod
    def get_sliced_context(db: DBSession, session_id: str, max_tokens: int = 6000) -> list[dict]:
        """
        获取经过滑动窗口截断后的标准大模型上下文格式
        :param db: 数据库连接
        :param session_id: 会话ID
        :param max_tokens: 留给历史记录的最大 Token 限制（给新提问和回复留出空间）
        """
        # 1. 按时间正序查出该会话的所有消息
        messages = db.query(MessageModel)\
                     .filter(MessageModel.session_id == session_id)\
                     .order_by(MessageModel.created_at.asc())\
                     .all()
        
        if not messages:
            return []

        # 2. 分离出系统提示词（System Prompt 必须永远保留在最开头，不能被截断）
        system_msgs = [m for m in messages if m.role == "system"]
        chat_msgs = [m for m in messages if m.role != "system"]
        
        # 计算系统提示词占用的固定 Token
        system_tokens = sum(m.token_count for m in system_msgs)
        available_tokens = max_tokens - system_tokens
        
        # 3. 从后往前（从最新、最近的消息开始）滚动累加普通对话
        keep_chat_msgs = []
        current_tokens = 0
        
        for msg in reversed(chat_msgs):
            # 如果加上这条消息就超标了，说明更早的消息都不能要了，直接终止循环
            if current_tokens + msg.token_count > available_tokens:
                break
            # 插入到最前面，以保持原有的正序时间轴
            keep_chat_msgs.insert(0, msg)
            current_tokens += msg.token_count
            
        # 4. 重新组装：System 提示词永远在最前 + 截取后的历史记录
        final_msgs = system_msgs + keep_chat_msgs
        
        # 5. 转换为大模型通用的标准 List[dict] 格式
        return [{"role": m.role, "content": m.content} for m in final_msgs]

    @staticmethod
    def async_llm_summarize(message_id: str, raw_content: str):
        """【后台异步任务】让大模型在不阻塞主线程的情况下，智能总结工具的巨量输出"""
        from app.database import SessionLocal # 或者是你项目里的标准本地会话工厂
        from app.services.llm_service import call_fast_llm # 假设你们有一个调用轻量模型的方法
        
        db = SessionLocal()
        try:
            # 1. 组装一个非常明确的摘要 Prompt
            prompt = (
                "你是一个学术研究助手。请对以下工具返回的大量原始文本进行极简智能摘要。\n"
                "提取出：1. 工具运行是否成功；2. 关键核心数值（如亲和力/结合能 score、RMSD 结果、分子数量等）；\n"
                "3. 产生的文件路径或标识。字数严格控制在 300 字以内。\n\n"
                f"【工具原始输出片段（前5000字）】:\n{raw_content[:5000]}\n..."
            )
            
            # 2. 默默调用小模型进行总结
            summary = call_fast_llm(prompt)
            
            # 3. 总结完成后，更新数据库中该条消息的内容
            db_msg = db.query(MessageModel).filter(MessageModel.id == message_id).first()
            if db_msg:
                db_msg.content = f"【系统智能摘要】:\n{summary}\n⚠️ [原始巨量文本已归档至底层文件系统]"
                # 重新计算准确的 Token
                db_msg.token_count = estimate_token_count(db_msg.content)
                db.commit()
                
            logger.info("Message %s 后台智能摘要异步更新成功", message_id)
        except Exception as e:
            logger.error("后台异步摘要失败: %s", e)
        finally:
            db.close()

    @staticmethod
    def bind_file_to_session(
        db, 
        session_id: str, 
        file_name: str, 
        minio_path: str, 
        file_size: int = 0,
        minio_bucket: str = "papermind-files"
    ) -> SessionFileModel:
        """
        将已经上传到 MinIO 的学术研究文件与当前的聊天会话进行登记绑定
        """
        # 1. 自动提取文件后缀名作为类型（例如：paper_v1.pdf -> pdf）
        file_type = "unknown"
        if file_name and "." in file_name:
            # os.path.splitext(file_name)[1] 拿出来是 '.pdb'，用 lstrip('.') 去掉点，转小写保持规范
            file_type = os.path.splitext(file_name)[1].lstrip('.').lower()

        # 2. 构造数据库模型对象
        db_file = SessionFileModel(
            session_id=session_id,
            file_name=file_name,
            file_type=file_type,
            minio_bucket=minio_bucket,
            minio_path=minio_path,
            file_size=file_size,
            created_at=datetime.now()
        )
        
        db.add(db_file)

        # 3. 顺手触发表的更新时间（让含有最新文件的 Session 在前端历史列表中置顶）
        db_session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if db_session:
            db_session.updated_at = datetime.now()

        db.commit()
        db.refresh(db_file)
        return db_file