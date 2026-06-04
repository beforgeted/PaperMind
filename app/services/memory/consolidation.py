"""Consolidate ES episodic session archives into long-term semantic memories."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any

import logging; logger = logging.getLogger(__name__)
from app.services.memory.config import MemoryConfig
from app.services.memory.episodic import EpisodicMemory
from app.services.memory.semantic import SemanticMemory

MemoryCandidate = dict[str, Any]
MemoryExtractor = Callable[[dict[str, Any]], Awaitable[list[MemoryCandidate]] | list[MemoryCandidate]]


class MemoryConsolidator:
    """Extract stable preferences and research state from archived sessions."""

    def __init__(
        self,
        episodic: EpisodicMemory | None = None,
        semantic: SemanticMemory | None = None,
        extractor: MemoryExtractor | None = None,
        use_llm: bool = True,
        config: MemoryConfig | None = None,
    ):
        self.config = config or MemoryConfig()
        self.episodic = episodic or EpisodicMemory(self.config)
        self.semantic = semantic or SemanticMemory(self.config)

        if extractor:
            self.extractor: MemoryExtractor = extractor
        elif use_llm and self.config.consolidation_use_llm:
            self.extractor = self._llm_based_extract
        else:
            self.extractor = self._rule_based_extract  # type: ignore[assignment]

    async def consolidate_pending(self, limit: int = 20) -> int:
        """Consolidate sessions whose archive contains unprocessed turns."""
        sessions = self.episodic.list_sessions_needing_consolidation(limit=limit)
        total = 0
        for session in sessions:
            session_id = session.get("session_id", "")
            if not session_id:
                continue
            total += await self.consolidate_session(session_id)
        return total

    async def consolidate_session(self, session_id: str) -> int:
        """Extract long-term memories from one session archive."""
        session_doc = self.episodic.get_session_doc(session_id)
        if not session_doc:
            return 0

        if inspect.iscoroutinefunction(self.extractor):
            candidates = await self.extractor(session_doc)
        else:
            candidates = self.extractor(session_doc)  # type: ignore[assignment]

        if not candidates:
            self.episodic.mark_consolidated(session_id, int(session_doc.get("turn_count") or 0))
            return 0

        user_id = session_doc.get("user_id", "")
        project_id = session_doc.get("project_id", "")
        written = 0
        for candidate in candidates:
            key = str(candidate.get("key") or "").strip()
            content = str(candidate.get("content") or "").strip()
            if not key or not content:
                continue
            await self.semantic.upsert_long_term_memory(
                key=key,
                content=content,
                memory_kind=str(candidate.get("memory_kind") or "fact"),
                user_id=user_id,
                project_id=project_id,
                source_session_ids=[session_id],
                confidence=float(candidate.get("confidence", 0.7)),
                importance=float(candidate.get("importance", 0.7)),
                metadata=dict(candidate.get("metadata") or {}),
            )
            written += 1

        self.episodic.mark_consolidated(session_id, int(session_doc.get("turn_count") or 0))
        logger.info("MemoryConsolidator: session={} wrote {} semantic memories", session_id[:12], written)
        return written

    # ── extractors ─────────────────────────────────────────────────────

    def _rule_based_extract(self, session_doc: dict[str, Any]) -> list[MemoryCandidate]:
        """Extract obvious stable memories without blocking on an LLM."""
        user_text = "\n".join(
            str(turn.get("content") or "")
            for turn in session_doc.get("turns", [])
            if turn.get("role") == "user"
        )
        compact = user_text.lower()
        candidates: list[MemoryCandidate] = []

        if "中文" in user_text and ("回答" in user_text or "回复" in user_text):
            candidates.append(
                {
                    "key": "preferred_language",
                    "content": "用户偏好使用中文回答。",
                    "memory_kind": "preference",
                    "confidence": 0.85,
                    "importance": 0.8,
                    "metadata": {"extractor": "rule_based"},
                }
            )

        if "markdown" in compact or "表格" in user_text:
            candidates.append(
                {
                    "key": "preferred_response_format",
                    "content": "用户偏好结构化回答，必要时使用 Markdown 或表格。",
                    "memory_kind": "preference",
                    "confidence": 0.75,
                    "importance": 0.7,
                    "metadata": {"extractor": "rule_based"},
                }
            )

        if "论文" in user_text and ("研究" in user_text or "综述" in user_text):
            candidates.append(
                {
                    "key": "research_workflow",
                    "content": "用户经常围绕论文检索、综述和研究分析展开长期任务。",
                    "memory_kind": "research_interest",
                    "confidence": 0.65,
                    "importance": 0.65,
                    "metadata": {"extractor": "rule_based"},
                }
            )

        return candidates

    async def _llm_based_extract(self, session_doc: dict[str, Any]) -> list[MemoryCandidate]:
        """Use LLM to extract preferences, research interests, and facts from session turns."""
        turns = session_doc.get("turns", [])
        if not turns:
            return []

        transcript = "\n".join(
            f"[{t.get('role', '')}] {t.get('content', '')[:500]}"
            for t in turns[-20:]
        )

        prompt = (
            "从以下对话中提取长期记忆。返回 JSON 数组，每个元素包含：\n"
            "- key: 简短标签 (如 \"preferred_language\", \"research_topic_水下图像\")\n"
            "- content: 事实/偏好描述 (一句话)\n"
            "- memory_kind: preference | research_interest | project_state | writing_style | fact\n"
            "- confidence: 0.0-1.0\n"
            "- importance: 0.0-1.0\n\n"
            "对话：\n"
            f"{transcript}\n\n"
            "只提取可跨会话复用的稳定信息。只返回 JSON 数组，不要其他文字。"
        )

        try:
            from app.services.llm import get_llm

            llm = get_llm()
            response = await llm.ainvoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            content = content.strip()
            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[-1]
                if content.endswith("```"):
                    content = content[:-3]
            candidates_raw = json.loads(content)
            return [
                {
                    "key": str(c.get("key", "")).strip(),
                    "content": str(c.get("content", "")).strip(),
                    "memory_kind": str(c.get("memory_kind", "fact")),
                    "confidence": float(c.get("confidence", 0.7)),
                    "importance": float(c.get("importance", 0.7)),
                    "metadata": {"extractor": "llm_based"},
                }
                for c in candidates_raw
                if c.get("key") and c.get("content")
            ]
        except Exception as exc:
            logger.warning("MemoryConsolidator: LLM extraction failed, falling back to rules: {}", exc)
            return self._rule_based_extract(session_doc)
