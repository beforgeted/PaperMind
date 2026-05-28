"""PaperMind memory system.

Three-layer: Working (in-memory) → Semantic (ES vector) → Episodic (ES time-series)
Session:   Redis sliding-window dual-key (decouples HTTP/WS from chat state)

Inspired by Hello Agents Ch.8 memory architecture.
"""

from app.services.memory.models import MemoryItem, MemoryScope, MemoryType
from app.services.memory.config import MemoryConfig
from app.services.memory.working import WorkingMemory
from app.services.memory.semantic import SemanticMemory
from app.services.memory.episodic import EpisodicMemory
from app.services.memory.manager import MemoryManager
from app.services.memory.store import JsonMemoryStore, get_memory_store  # backward compat
from app.services.memory.session_store import SessionStore, get_session_store, TurnRecord

__all__ = [
    "MemoryItem",
    "MemoryScope",
    "MemoryType",
    "MemoryConfig",
    "WorkingMemory",
    "SemanticMemory",
    "EpisodicMemory",
    "MemoryManager",
    "JsonMemoryStore",
    "get_memory_store",
    "SessionStore",
    "get_session_store",
    "TurnRecord",
]
