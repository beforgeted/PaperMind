"""请求级日志上下文，基于 contextvars 贯穿 HTTP → Agent → Tool 调用链。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Dict, Iterator, Optional
from uuid import uuid4

_CONTEXT_FIELDS = ("request_id", "session_id", "run_id", "step_id", "agent_id")
_CTX_TOKEN_KEY = "__ctx__"

_log_context: ContextVar[Dict[str, str]] = ContextVar("log_context", default={})


def generate_request_id() -> str:
    """生成 HTTP 请求关联 ID。"""
    return f"req_{uuid4().hex[:12]}"


def get_context() -> Dict[str, str]:
    """返回当前上下文副本（只读语义）。"""
    return dict(_log_context.get())


def bind(**fields: Optional[str]) -> Dict[str, Token]:
    """绑定上下文字段，返回 token 映射供 unbind 使用。"""
    current = dict(_log_context.get())
    changed = False
    for key, value in fields.items():
        if key not in _CONTEXT_FIELDS or value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        current[key] = text
        changed = True
    if not changed:
        return {}
    token = _log_context.set(current)
    return {_CTX_TOKEN_KEY: token}


def unbind(tokens: Dict[str, Token]) -> None:
    """按 token 还原 bind 之前的上下文。"""
    token = tokens.get(_CTX_TOKEN_KEY)
    if token is not None:
        _log_context.reset(token)


def clear() -> None:
    """清空当前上下文。"""
    _log_context.set({})


def copy_context() -> Dict[str, str]:
    """复制当前上下文，用于跨任务传递。"""
    return get_context()


@contextmanager
def log_bind(**fields: Optional[str]) -> Iterator[None]:
    """上下文管理器：在 with 块内绑定字段，退出时自动还原。"""
    tokens = bind(**fields)
    try:
        yield
    finally:
        if tokens:
            unbind(tokens)


def apply_context(ctx: Dict[str, Any]) -> Dict[str, Token]:
    """从字典批量绑定已知字段。"""
    filtered = {k: v for k, v in ctx.items() if k in _CONTEXT_FIELDS and v}
    return bind(**filtered)
