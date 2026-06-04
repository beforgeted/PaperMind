"""Shared TypedDict types for agent graph schemas."""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class PlanStep(TypedDict):
    step: int
    action: str
    description: str
    params: dict[str, Any]
