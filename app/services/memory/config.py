"""Memory system configuration — TTL, capacities, decay, scoring weights."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemoryConfig:
    """Centralized config for the memory system."""

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
    score_time_decay_lambda: float = 0.1  # e^(-λ × days_ago), higher = faster decay
    score_importance_weight: float = 0.5  # (1.0 - w) + w × importance

    # ── Consolidation ──
    consolidation_min_importance: float = 0.7
    consolidation_min_access_count: int = 3
    consolidation_use_llm: bool = True     # use LLM extractor (fallback to rules)

    # ── Forgetting ──
    forget_max_age_days: int = 90                  # auto-delete after 90 days
    forget_low_importance_threshold: float = 0.2   # purge below this if over capacity

    # ── Integration ──
    auto_encode_episodic: bool = True      # auto-record after each agent invocation
    auto_recall_semantic: bool = True      # auto-search semantic memory for relevant context
