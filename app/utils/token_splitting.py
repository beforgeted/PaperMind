"""Token-length text splitters for parent/child chunking.

Chunk sizes in settings (`parent_chunk_size`, `child_chunk_size`, overlaps)
are interpreted as **token counts**, not characters. Uses `tiktoken` when
available (`cl100k_base`); otherwise a conservative character heuristic.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logging import logger


@lru_cache(maxsize=1)
def _get_tiktoken_encoding():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        logger.warning(
            "tiktoken not installed — chunk sizes use a character heuristic "
            "(~3 chars/token). Install tiktoken for accurate token chunking."
        )
        return None


def count_tokens(text: str) -> int:
    """Return token count for splitter `length_function`."""
    enc = _get_tiktoken_encoding()
    if enc is not None:
        return len(enc.encode(text))
    # Mixed EN/ZH academic PDF text — slightly conservative vs true BPE.
    return max(1, len(text) // 3)


def get_token_length_function() -> Callable[[str], int]:
    """Callable suitable for RecursiveCharacterTextSplitter.length_function."""
    return count_tokens


def parent_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.parent_chunk_size,
        chunk_overlap=settings.parent_chunk_overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", "。", ". ", " ", ""],
    )


def child_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_chunk_overlap,
        length_function=count_tokens,
        separators=["。", "！", "？", ". ", "! ", "? ", "\n", " ", ""],
    )


__all__ = [
    "count_tokens",
    "get_token_length_function",
    "parent_text_splitter",
    "child_text_splitter",
]
