"""LangChain tool decorator shim.

Tests in lightweight environments may not install LangChain. In production,
this module delegates to `langchain_core.tools.tool`; otherwise it leaves the
function callable so pure contract tests can still import tool modules.
"""

from __future__ import annotations

from typing import Any, Callable

try:
    from langchain_core.tools import tool as langchain_tool
except Exception:  # pragma: no cover - exercised only in minimal environments
    langchain_tool = None


def tool(func: Callable | None = None, *args: Any, **kwargs: Any):
    """Decorate a function as a LangChain tool when LangChain is available."""
    if langchain_tool is None:
        if func is None:
            return lambda inner: inner
        return func
    if func is None:
        return langchain_tool(*args, **kwargs)
    return langchain_tool(func, *args, **kwargs)
