"""PaperMind 双通道日志：stdout 文本 + log/ 按日期分目录的 JSON 文件。"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import settings
from app.core.log_context import _CONTEXT_FIELDS, get_context
from app.observability.events import AGENT_BUSINESS_EVENTS

AUDIT_LOGGER_NAME = "papermind.audit"

_RESERVED_RECORD_ATTRS = frozenset(
    logging.LogRecord(
        name="",
        level=0,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    ).__dict__.keys()
) | frozenset(_CONTEXT_FIELDS) | frozenset({"event", "message"})

_NOISY_LOGGERS = ("httpx", "elasticsearch", "aiokafka", "urllib3", "httpcore", "elastic_transport")

_setup_done = False
_prune_lock = threading.Lock()
_last_prune_date: Optional[date] = None


class ContextInjectingFilter(logging.Filter):
    """将 log_context 字段注入 LogRecord。"""

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in get_context().items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class AgentBusinessFilter(logging.Filter):
    """仅放行 Agent 业务相关 event，用于 agent.json.log。"""

    def filter(self, record: logging.LogRecord) -> bool:
        event = getattr(record, "event", None)
        return bool(event and event in AGENT_BUSINESS_EVENTS)


class JsonFormatter(logging.Formatter):
    """单行 JSON 格式化器。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in _CONTEXT_FIELDS:
            value = getattr(record, key, None)
            if value:
                payload[key] = value
        event = getattr(record, "event", None)
        if event:
            payload["event"] = event
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_ATTRS or key.startswith("_"):
                continue
            if value is not None:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """人类可读文本格式，前缀附带关联 ID。"""

    def format(self, record: logging.LogRecord) -> str:
        parts = []
        for key in _CONTEXT_FIELDS:
            value = getattr(record, key, None)
            if value:
                parts.append(f"{key}={value}")
        prefix = f"[{' '.join(parts)}] " if parts else ""
        event = getattr(record, "event", None)
        event_part = f"({event}) " if event else ""
        base = super().format(record)
        return f"{prefix}{event_part}{base}"


class DateDirectoryFileHandler(logging.Handler):
    """按本地日期写入 log/YYYY/MM/DD/<basename>，跨日自动切换文件。"""

    def __init__(
        self,
        log_dir: Path,
        basename: str,
        formatter: logging.Formatter,
        *,
        level: int = logging.NOTSET,
        record_filter: Optional[logging.Filter] = None,
    ) -> None:
        super().__init__(level)
        self.log_dir = Path(log_dir)
        self.basename = basename
        self._current_date: Optional[date] = None
        self._stream = None
        self._write_lock = threading.Lock()
        self.setFormatter(formatter)
        self.addFilter(ContextInjectingFilter())
        if record_filter is not None:
            self.addFilter(record_filter)

    def _path_for(self, day: date) -> Path:
        if settings.log_date_subdirs:
            return (
                self.log_dir
                / day.strftime("%Y")
                / day.strftime("%m")
                / day.strftime("%d")
                / self.basename
            )
        return self.log_dir / self.basename

    def _ensure_stream(self) -> None:
        today = date.today()
        if self._current_date == today and self._stream is not None:
            return
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        path = self._path_for(today)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = open(path, "a", encoding="utf-8")
        self._current_date = today
        _maybe_prune_old_logs(self.log_dir)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with self._write_lock:
                self._ensure_stream()
                assert self._stream is not None
                self._stream.write(msg + "\n")
                self._stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        with self._write_lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            self._current_date = None
        super().close()


def resolve_log_path(basename: str, *, day: Optional[date] = None) -> Path:
    """返回指定日期（默认今天）的日志文件路径，供测试与运维脚本使用。"""
    log_dir = Path(settings.log_dir)
    target = day or date.today()
    if settings.log_date_subdirs:
        return (
            log_dir
            / target.strftime("%Y")
            / target.strftime("%m")
            / target.strftime("%d")
            / basename
        )
    return log_dir / basename


def _iter_date_log_dirs(log_dir: Path):
    """遍历 log/YYYY/MM/DD 日期目录。"""
    if not log_dir.is_dir():
        return
    for year_path in log_dir.iterdir():
        if not year_path.is_dir() or len(year_path.name) != 4 or not year_path.name.isdigit():
            continue
        for month_path in year_path.iterdir():
            if not month_path.is_dir() or len(month_path.name) != 2 or not month_path.name.isdigit():
                continue
            for day_path in month_path.iterdir():
                if not day_path.is_dir() or len(day_path.name) != 2 or not day_path.name.isdigit():
                    continue
                try:
                    yield date(int(year_path.name), int(month_path.name), int(day_path.name)), day_path
                except ValueError:
                    continue


def _prune_old_logs(log_dir: Path) -> None:
    """删除超过 log_retention_days 的日期目录。"""
    retention = int(settings.log_retention_days)
    if retention <= 0:
        return
    cutoff = date.today() - timedelta(days=retention)
    for day, day_path in _iter_date_log_dirs(log_dir):
        if day < cutoff:
            shutil.rmtree(day_path, ignore_errors=True)
    _prune_empty_parents(log_dir)


def _prune_empty_parents(log_dir: Path) -> None:
    """清理空的 YYYY/MM 目录。"""
    for year_path in sorted(log_dir.glob("*"), reverse=True):
        if not year_path.is_dir():
            continue
        for month_path in sorted(year_path.glob("*"), reverse=True):
            if month_path.is_dir() and not any(month_path.iterdir()):
                month_path.rmdir()
        if not any(year_path.iterdir()):
            year_path.rmdir()


def _maybe_prune_old_logs(log_dir: Path) -> None:
    """每天最多执行一次过期日志清理。"""
    global _last_prune_date
    today = date.today()
    with _prune_lock:
        if _last_prune_date == today:
            return
        _prune_old_logs(log_dir)
        _last_prune_date = today


def _build_file_handler(
    log_dir: Path,
    basename: str,
    formatter: logging.Formatter,
    *,
    level: int = logging.INFO,
    record_filter: Optional[logging.Filter] = None,
) -> DateDirectoryFileHandler:
    handler = DateDirectoryFileHandler(
        log_dir,
        basename,
        formatter,
        level=level,
        record_filter=record_filter,
    )
    return handler


def get_audit_logger() -> logging.Logger:
    """返回 MCP 审计专用 logger。"""
    return logging.getLogger(AUDIT_LOGGER_NAME)


def setup_logging() -> None:
    """初始化根 logger 与审计 logger（幂等）。"""
    global _setup_done
    if _setup_done:
        return

    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    _maybe_prune_old_logs(log_dir)

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    context_filter = ContextInjectingFilter()

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    if settings.log_to_stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(level)
        stdout_handler.setFormatter(
            TextFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        stdout_handler.addFilter(context_filter)
        root.addHandler(stdout_handler)

    if settings.log_json_to_file:
        json_formatter = JsonFormatter()
        root.addHandler(
            _build_file_handler(log_dir, "app.json.log", json_formatter, level=level)
        )
        root.addHandler(
            _build_file_handler(
                log_dir,
                "error.json.log",
                json_formatter,
                level=logging.ERROR,
            )
        )
        if settings.log_agent_business_file:
            root.addHandler(
                _build_file_handler(
                    log_dir,
                    "agent.json.log",
                    json_formatter,
                    level=level,
                    record_filter=AgentBusinessFilter(),
                )
            )

        audit_logger = get_audit_logger()
        audit_logger.handlers.clear()
        audit_logger.setLevel(logging.INFO)
        audit_logger.propagate = False
        audit_logger.addHandler(
            _build_file_handler(log_dir, "audit.json.log", JsonFormatter(), level=logging.INFO)
        )

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _setup_done = True
