"""LLM 配置与客户端工厂。"""

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """PaperMind 平台运行时配置。"""

    model_config = SettingsConfigDict(
        env_file=(".env", "/app/.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 18004
    debug: bool = False
    log_level: str = "INFO"

    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_model: str = "qwen-plus"
    main_agent_model: Optional[str] = None
    sub_agent_model: Optional[str] = None
    agent_temperature: float = 0.7
    orchestrator_temperature: float = 0.2

    chat_max_concurrent_tasks: int = 64
    chat_stream_queue_max_size: int = 256
    chat_stream_queue_put_timeout: float = 1.0

    mysql_enabled: bool = False
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: Optional[str] = None
    mysql_password: Optional[str] = None
    mysql_database: Optional[str] = None
    mysql_charset: str = "utf8mb4"
    mysql_pool_min_size: int = 1
    mysql_pool_max_size: int = 10
    mysql_pool_recycle_seconds: int = 3600

    chat_user_validation_enabled: bool = False
    chat_user_table: str = "users"
    chat_user_id_column: str = "id"
    chat_user_active_column: Optional[str] = "is_active"
    chat_user_active_value: str = "1"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


@dataclass(frozen=True)
class ResolvedModelConfig:
    provider: str
    model: str
    api_key: str
    base_url: Optional[str]
    temperature: float


def _as_dict(model_config: Any) -> Dict[str, Any]:
    if model_config is None:
        return {}
    if isinstance(model_config, BaseModel):
        return model_config.model_dump(exclude_none=True)
    if isinstance(model_config, dict):
        return model_config
    if isinstance(model_config, str):
        return {"name": model_config}
    return {}


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _first_float(default: float, *values: Any) -> float:
    for value in values:
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _read_named_setting(name: Any, settings: Settings) -> str:
    key = _first_text(name)
    if not key:
        return ""

    attr_name = key.strip().lower().replace("-", "_")
    value = getattr(settings, attr_name, None)
    if value not in (None, ""):
        return str(value)

    return os.getenv(key, "")


def resolve_model_config(model_config: Any = None, settings: Optional[Settings] = None) -> ResolvedModelConfig:
    settings = settings or get_settings()
    cfg = _as_dict(model_config)
    default_temperature = float(getattr(settings, "agent_temperature", 0.7))
    model = _first_text(
        cfg.get("name"),
        cfg.get("model"),
        _read_named_setting(cfg.get("model_env"), settings),
        getattr(settings, "openai_model", ""),
    )
    api_key = _first_text(
        cfg.get("apiKey"),
        cfg.get("api_key"),
        cfg.get("openaiApiKey"),
        cfg.get("openai_api_key"),
        _read_named_setting(cfg.get("api_key_env"), settings),
        getattr(settings, "openai_api_key", ""),
    )
    base_url = _first_text(
        cfg.get("baseUrl"),
        cfg.get("base_url"),
        cfg.get("openaiBaseUrl"),
        cfg.get("openai_base_url"),
        _read_named_setting(cfg.get("base_url_env"), settings),
        getattr(settings, "openai_base_url", ""),
    ) or None

    if not model or not api_key:
        raise ValueError("模型配置缺失，请先配置模型名称和 API Key")

    return ResolvedModelConfig(
        provider=_first_text(cfg.get("provider"), "openai-compatible"),
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=_first_float(
            default_temperature,
            cfg.get("temperature"),
            cfg.get("temp"),
            cfg.get("agentTemperature"),
        ),
    )


def mask_api_key(api_key: str) -> str:
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return "****"
    return f"{api_key[:4]}****{api_key[-4:]}"


def message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "".join(parts)
    return str(content)


class LLMService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def _base_model_config(self, *, model: str, temperature: float) -> Dict[str, Any]:
        return {
            "provider": "openai-compatible",
            "name": model,
            "model": model,
            "api_key": self.settings.openai_api_key or "",
            "apiKey": self.settings.openai_api_key or "",
            "base_url": self.settings.openai_base_url or "",
            "baseUrl": self.settings.openai_base_url or "",
            "temperature": float(temperature),
        }

    def main_agent_model_config(self, *, temperature: Optional[float] = None) -> Dict[str, Any]:
        model = self.settings.main_agent_model or self.settings.openai_model
        return self._base_model_config(
            model=model,
            temperature=float(self.settings.agent_temperature if temperature is None else temperature),
        )

    def sub_agent_model_config(self) -> Dict[str, Any]:
        model = self.settings.sub_agent_model or self.settings.openai_model
        return self._base_model_config(
            model=model,
            temperature=float(self.settings.agent_temperature),
        )

    def default_model_config(self) -> Dict[str, Any]:
        return self.main_agent_model_config()

    def merge_model_config(self, override: Any = None, *, base_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        config = dict(base_config or self.default_model_config())
        if isinstance(override, dict):
            for key, value in override.items():
                if value not in (None, ""):
                    config[key] = value
        return config

    def merge_sub_agent_model_config(self, override: Any = None) -> Dict[str, Any]:
        return self.merge_model_config(override, base_config=self.sub_agent_model_config())

    def create_chat_model(
        self,
        model_config: Any = None,
        *,
        streaming: bool,
        timeout: int = 120,
        max_tokens: int = 4096,
    ) -> ChatOpenAI:
        resolved = resolve_model_config(model_config, self.settings)
        logger.info(
            "创建LLM实例 - provider=%s model=%s base_url=%s api_key=%s streaming=%s",
            resolved.provider,
            resolved.model,
            resolved.base_url,
            mask_api_key(resolved.api_key),
            streaming,
        )
        return ChatOpenAI(
            model=resolved.model,
            api_key=resolved.api_key,
            base_url=resolved.base_url,
            temperature=resolved.temperature,
            streaming=streaming,
            request_timeout=timeout,
            max_tokens=max_tokens,
        )


llm_service = LLMService()
