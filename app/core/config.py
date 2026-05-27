"""Centralized configuration loaded from environment / .env file.

All other modules MUST import settings from here instead of reading os.environ
directly. This keeps the configuration surface small and testable.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    # ---- App ----
    app_name: str = "PaperMind"
    app_env: Literal["dev", "staging", "prod"] = "dev"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_retention_days: int = 30

    # ---- MinIO ----
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "papermind"
    minio_secret_key: str = "papermind123"
    minio_bucket: str = "papers"
    minio_secure: bool = False

    # ---- Redis ----
    redis_url: str = "redis://localhost:6379/0"
    upload_state_ttl_seconds: int = 24 * 60 * 60

    # ---- Kafka ----
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_parse: str = "papermind.parse"
    kafka_group_id: str = "papermind-worker"
    kafka_client_id: str = "papermind-api"

    # ---- Elasticsearch ----
    es_hosts: str = "http://localhost:9200"
    es_username: Optional[str] = None
    es_password: Optional[str] = None
    es_index_parent: str = "papermind_parents"
    es_index_child: str = "papermind_children"
    es_index_papers: str = "papermind_papers"
    es_vector_dims: int = 1024

    # ---- Embedding ----
    embedding_backend: Literal["dashscope", "local"] = "dashscope"
    embedding_model: str = "text-embedding-v3"
    dashscope_api_key: Optional[str] = None
    local_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # ---- Chunking (sizes/overlaps are token counts; see token_splitting.py) ----
    parent_chunk_size: int = 1024
    parent_chunk_overlap: int = 150
    child_chunk_size: int = 256
    child_chunk_overlap: int = 30

    # ---- Retrieval ----
    # Recall depths for local BM25 + vector fusion. We do RRF in application
    # code so Elasticsearch's license-gated native RRF is not required.
    retrieve_top_k_knn: int = 20
    retrieve_top_k_bm25: int = 20
    retrieve_top_k_children: int = 20  # legacy fallback for older callers
    rrf_k: int = 60
    final_top_k: int = 5
    paper_search_top_k: int = 10
    # When a child hit maps to a parent, also append that parent's `next_parent_id` text.
    retrieve_include_next_parent: bool = True

    # ---- LLM (LCEL QA chain) ----
    llm_model: str = "qwen-plus"
    llm_temperature: float = 0.1
    qa_top_k: int = 5

    # ---- Agent Routing ----
    rule_route_threshold: float = 0.8
    llm_route_temperature: float = 0.0
    llm_route_fallback_route: str = "chunk_qa"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Convenience: ES hosts may be comma-separated
    @property
    def es_hosts_list(self) -> List[str]:
        return [h.strip() for h in self.es_hosts.split(",") if h.strip()]

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, v: str) -> str:
        return v.upper()


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Singleton accessor — read once per process."""
    return AppSettings()


settings = get_settings()
