"""PaperMind memory system.

Redis SessionStore keeps the current-session sliding window, ES Episodic archives
raw sessions, and SemanticMemory stores distilled long-term memories.
"""

from app.services.memory.models import MemoryItem, MemoryKind, MemoryScope, MemoryType
from app.services.memory.config import MemoryConfig
from app.services.memory.semantic import SemanticMemory
from app.services.memory.episodic import EpisodicMemory
from app.services.memory.consolidation import MemoryConsolidator
from app.services.memory.manager import MemoryManager, get_memory_manager
from app.services.memory.store import JsonMemoryStore, get_memory_store  # backward compat
from app.services.memory.session_store import SessionStore, get_session_store, TurnRecord

__all__ = [
    "MemoryItem",
    "MemoryKind",
    "MemoryScope",
    "MemoryType",
    "MemoryConfig",
    "SemanticMemory",
    "EpisodicMemory",
    "MemoryConsolidator",
    "MemoryManager",
    "get_memory_manager",
    "JsonMemoryStore",
    "get_memory_store",
    "SessionStore",
    "get_session_store",
    "TurnRecord",
]
