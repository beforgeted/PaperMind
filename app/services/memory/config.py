"""Memory system configuration — TTL, capacities, decay, scoring weights."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryConfig:
    """Centralized config for the three-layer memory system."""

    # ── Working memory ──
    working_max_items: int = 50
    working_default_ttl_seconds: int = 3600   # 1 hour
    working_cleanup_interval: int = 300        # 5 minutes between TTL sweeps

    # ── Semantic memory ──
    semantic_es_index: str = "papermind_memories_semantic"
    semantic_max_items: int = 10000
    semantic_embedding_dims: int = 1024
    semantic_retrieval_top_k: int = 5
    semantic_min_score: float = 0.3

    # ── Episodic memory ──
    episodic_es_index: str = "papermind_memories_episodic"
    episodic_max_items: int = 5000
    episodic_retrieval_top_k: int = 10

    # ── Retrieval scoring ──
    score_vector_weight: float = 0.7      # cosine similarity weight
    score_keyword_weight: float = 0.3     # BM25 / keyword weight
    score_time_decay_lambda: float = 0.1  # e^(-λ × days_ago), higher = faster decay
    score_importance_weight: float = 0.5  # (1.0 - w) + w × importance

    # ── Consolidation ──
    consolidation_min_importance: float = 0.7    # working → semantic threshold
    consolidation_min_access_count: int = 3       # must be accessed at least N times

    # ── Forgetting ──
    forget_max_age_days: int = 90                  # auto-delete after 90 days
    forget_low_importance_threshold: float = 0.2   # purge below this if over capacity

    # ── Integration ──
    auto_encode_episodic: bool = True      # auto-record after each agent invocation
    auto_recall_working: bool = True       # auto-inject working memory into queries
    auto_recall_semantic: bool = True      # auto-search semantic memory for relevant context
