"""基于 SQLAlchemy 的 MySQL 连接与用户校验服务。"""

from __future__ import annotations

import logging
import re
import asyncio
from typing import Any, Optional
from urllib.parse import quote_plus

from sqlalchemy import Boolean, Column, MetaData, String, Table, create_engine, select
from sqlalchemy.engine import Engine

from app.core.config import settings


logger = logging.getLogger(__name__)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(identifier: str) -> str:
    """校验来自配置的表名或字段名，避免动态标识符注入。"""
    if not _IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"MySQL 标识符不合法: {identifier}")
    return identifier


def _coerce_config_value(value: str) -> Any:
    text = str(value).strip()
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    if text.isdigit():
        return int(text)
    return text


class MySQLService:
    """基于 SQLAlchemy engine 的轻量 MySQL 服务。"""

    def __init__(self) -> None:
        self.settings = settings
        self._engine: Optional[Engine] = None

    @property
    def enabled(self) -> bool:
        return bool(self.settings.mysql_enabled)

    async def connect(self) -> None:
        """当 MYSQL_ENABLED=true 时创建 SQLAlchemy 连接引擎。"""
        if not self.enabled:
            logger.info("MySQL 连接未启用")
            return

        if self._engine is not None:
            return

        if not self.settings.mysql_user or not self.settings.mysql_database:
            raise RuntimeError("MySQL 配置缺失，请设置 MYSQL_USER 和 MYSQL_DATABASE")

        try:
            import pymysql  # noqa: F401
            import sqlalchemy  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("缺少数据库依赖，请安装 SQLAlchemy 和 PyMySQL") from exc

        pool_size = max(1, int(self.settings.mysql_pool_max_size))
        user = quote_plus(self.settings.mysql_user)
        password = quote_plus(self.settings.mysql_password or "")
        database = quote_plus(self.settings.mysql_database)
        charset = quote_plus(self.settings.mysql_charset)
        url = (
            f"mysql+pymysql://{user}:{password}"
            f"@{self.settings.mysql_host}:{int(self.settings.mysql_port)}/{database}"
            f"?charset={charset}"
        )

        self._engine = create_engine(
            url,
            pool_size=pool_size,
            pool_recycle=int(self.settings.mysql_pool_recycle_seconds),
            pool_pre_ping=True,
        )
        await asyncio.to_thread(self._ping)
        logger.info(
            "SQLAlchemy MySQL 引擎已初始化: host=%s port=%s database=%s",
            self.settings.mysql_host,
            self.settings.mysql_port,
            self.settings.mysql_database,
        )

    async def close(self) -> None:
        if self._engine is None:
            return
        await asyncio.to_thread(self._engine.dispose)
        self._engine = None
        logger.info("SQLAlchemy MySQL 引擎已关闭")

    async def validate_user(self, user_id: str) -> bool:
        """当配置的用户表中存在可用用户时返回 True。"""
        if not self.enabled:
            raise RuntimeError("MySQL 未启用，无法进行用户校验")
        table_name = _validate_identifier(self.settings.chat_user_table)
        id_column = _validate_identifier(self.settings.chat_user_id_column)
        active_column = self.settings.chat_user_active_column

        metadata = MetaData()
        columns = [Column(id_column, String(36), primary_key=True)]
        if active_column:
            active_column = _validate_identifier(active_column)
            columns.append(Column(active_column, Boolean))
        users = Table(table_name, metadata, *columns)

        query = select(users.c[id_column]).where(users.c[id_column] == user_id).limit(1)
        if active_column:
            active_value = _coerce_config_value(self.settings.chat_user_active_value)
            query = query.where(users.c[active_column] == active_value)

        return await asyncio.to_thread(self._validate_user_sync, query)

    async def require_engine(self) -> Engine:
        """返回可用的 SQLAlchemy 引擎；持久化不可用时立即失败。"""
        if not self.enabled:
            raise RuntimeError("MySQL 未启用，无法持久化工作流运行数据")
        if self._engine is None:
            await self.connect()
        if self._engine is None:
            raise RuntimeError("SQLAlchemy MySQL 引擎不可用")
        return self._engine

    async def execute(self, query: str, params: Optional[tuple[Any, ...]] = None) -> int:
        await self.require_engine()
        return await asyncio.to_thread(self._execute_sync, query, params or ())

    async def executemany(self, query: str, params: list[tuple[Any, ...]]) -> int:
        if not params:
            return 0
        await self.require_engine()
        return await asyncio.to_thread(self._executemany_sync, query, params)

    async def fetchone(self, query: str, params: Optional[tuple[Any, ...]] = None) -> Optional[dict[str, Any]]:
        await self.require_engine()
        return await asyncio.to_thread(self._fetchone_sync, query, params or ())

    async def fetchall(self, query: str, params: Optional[tuple[Any, ...]] = None) -> list[dict[str, Any]]:
        await self.require_engine()
        return await asyncio.to_thread(self._fetchall_sync, query, params or ())

    def _require_sync_engine(self) -> Engine:
        if self._engine is None:
            raise RuntimeError("SQLAlchemy MySQL 引擎不可用")
        return self._engine

    def _ping(self) -> None:
        engine = self._require_sync_engine()
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")

    def _validate_user_sync(self, query: Any) -> bool:
        engine = self._require_sync_engine()
        with engine.connect() as conn:
            result = conn.execute(query)
            return result.first() is not None

    def _execute_sync(self, query: str, params: tuple[Any, ...]) -> int:
        engine = self._require_sync_engine()
        with engine.begin() as conn:
            result = conn.exec_driver_sql(query, params)
            return int(result.rowcount)

    def _executemany_sync(self, query: str, params: list[tuple[Any, ...]]) -> int:
        engine = self._require_sync_engine()
        with engine.begin() as conn:
            result = conn.exec_driver_sql(query, params)
            return int(result.rowcount)

    def _fetchone_sync(self, query: str, params: tuple[Any, ...]) -> Optional[dict[str, Any]]:
        engine = self._require_sync_engine()
        with engine.connect() as conn:
            result = conn.exec_driver_sql(query, params)
            row = result.mappings().first()
            return dict(row) if row else None

    def _fetchall_sync(self, query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        engine = self._require_sync_engine()
        with engine.connect() as conn:
            result = conn.exec_driver_sql(query, params)
            return [dict(row) for row in result.mappings().all()]


mysql_service = MySQLService()
