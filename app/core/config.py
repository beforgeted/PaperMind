"""Centralized configuration loaded from environment / .env file.

All other modules MUST import settings from here instead of reading os.environ
directly. This keeps the configuration surface small and testable.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """PaperMind 统一系统配置。"""

    # ==================== App ====================
    project_name: str = "PaperMind"
    app_name: str = "PaperMind"
    app_env: Literal["dev", "staging", "prod"] = "dev"
    environment: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 2222
    debug: bool = False
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_retention_days: int = 30

    # ==================== Database ====================
    database_url: str = Field(
        default="mysql+pymysql://papermind:papermind@127.0.0.1:3307/papermind?charset=utf8mb4",
        description="数据库连接地址",
    )

    # ==================== MySQL (agent chat) ====================
    mysql_enabled: bool = True
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3307
    mysql_user: Optional[str] = "papermind"
    mysql_password: Optional[str] = "papermind"
    mysql_database: Optional[str] = "papermind"
    mysql_charset: str = "utf8mb4"
    mysql_pool_min_size: int = 1
    mysql_pool_max_size: int = 10
    mysql_pool_recycle_seconds: int = 3600

    # ==================== JWT ====================
    # jwt_secret_key: str = Field(..., description="JWT 密钥，必须在 .env 中设置")
    # jwt_algorithm: str = "HS256"
    # access_token_expire_minutes: int = 60 * 24 * 7  # 7天

    # ==================== Docker / Harbor ====================
    docker_registry: str = "192.168.1.100:8090"
    docker_namespace: str = "papermind"
    docker_insecure: bool = True
    docker_username: Optional[str] = None
    docker_password: Optional[str] = None

    # ==================== MinIO ====================
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "papermind"
    minio_secret_key: str = "papermind123"
    minio_bucket: str = "papers"
    minio_secure: bool = False

    # ==================== Redis ====================
    redis_url: str = "redis://localhost:6379/0"
    upload_state_ttl_seconds: int = 24 * 60 * 60

    # ==================== Kafka ====================
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_parse: str = "papermind.parse"
    kafka_group_id: str = "papermind-worker"
    kafka_client_id: str = "papermind-api"

    # ==================== Elasticsearch ====================
    es_hosts: str = "http://localhost:9201"
    es_username: Optional[str] = None
    es_password: Optional[str] = None
    es_index_parent: str = "papermind_parents"
    es_index_child: str = "papermind_children"
    es_index_papers: str = "papermind_papers"
    es_vector_dims: int = 1024

    # ==================== Embedding ====================
    embedding_backend: Literal["dashscope", "local"] = "dashscope"
    embedding_model: str = "text-embedding-v3"
    dashscope_api_key: Optional[str] = None
    local_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # ==================== Chunking ====================
    parent_chunk_size: int = 1024
    parent_chunk_overlap: int = 150
    child_chunk_size: int = 256
    child_chunk_overlap: int = 30

    # ==================== Retrieval ====================
    retrieve_top_k_knn: int = 20
    retrieve_top_k_bm25: int = 20
    retrieve_top_k_children: int = 20
    rrf_k: int = 60
    final_top_k: int = 5
    paper_search_top_k: int = 10
    retrieve_include_next_parent: bool = True

    # ==================== LLM (OpenAI-compatible) ====================
    llm_model: str = "qwen-plus"
    llm_temperature: float = 0.1
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_model: str = "qwen-plus"
    main_agent_model: Optional[str] = None
    sub_agent_model: Optional[str] = None
    agent_temperature: float = 0.7
    orchestrator_temperature: float = 0.2
    qa_top_k: int = 5

    # ==================== Agent Chat ====================
    chat_max_concurrent_tasks: int = 64
    chat_stream_queue_max_size: int = 256
    chat_stream_queue_put_timeout: float = 1.0

    # ==================== Chat User Validation ====================
    chat_user_validation_enabled: bool = False
    chat_user_table: str = "users"
    chat_user_id_column: str = "id"
    chat_user_active_column: Optional[str] = "is_active"
    chat_user_active_value: str = "1"

    # ==================== Agent Routing ====================
    rule_route_threshold: float = 0.8
    llm_route_temperature: float = 0.0
    llm_route_fallback_route: str = "chunk_qa"

    # ==================== Comparison workflow ====================
    comparison_min_evidence_per_aspect: int = 2
    comparison_max_retry: int = 1
    comparison_max_targets: int = 5
    comparison_paper_resolve_min_score: float = 0.0
    comparison_discovery_top_k: int = 3
    comparison_evidence_top_k: int = 6

    # ==================== Summary (literature review) ====================
    summary_max_papers: int = 8
    summary_max_sections: int = 3
    summary_evidence_top_k: int = 3
    summary_evidence_max_chars: int = 300
    summary_outline_max_chars: int = 8000
    summary_draft_max_chars: int = 6000
    summary_polish_max_chars: int = 10000

    # ==================== Memory system (three-layer) ====================
    memory_es_semantic_index: str = "papermind_memories_semantic"
    memory_es_episodic_index: str = "papermind_memories_episodic"
    memory_working_max_items: int = 50
    memory_working_ttl_seconds: int = 3600
    memory_semantic_max_items: int = 10000
    memory_episodic_max_items: int = 5000
    memory_retrieval_top_k: int = 5
    memory_consolidation_min_importance: float = 0.7
    memory_forget_max_age_days: int = 90
    memory_auto_encode_episodic: bool = True

    # ==================== Upload ====================
    upload_max_size: int = 100 * 1024 * 1024  # 100MB

    # ==================== Model Config ====================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==================== Properties / Validators ====================

    @property
    def es_hosts_list(self) -> List[str]:
        """Convenience: ES hosts may be comma-separated."""
        return [h.strip() for h in self.es_hosts.split(",") if h.strip()]

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, v: str) -> str:
        return v.upper()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor -- read once per process."""
    return Settings()


settings = get_settings()
