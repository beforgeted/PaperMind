"""Heuristic structure parser for Docling-exported paper Markdown."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


SECTION_TYPES = {
    "abstract",
    "introduction",
    "related_work",
    "method",
    "experiment",
    "ablation",
    "discussion",
    "conclusion",
    "figure_caption",
    "table_caption",
    "reference",
}

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
_FIGURE_RE = re.compile(r"^\s*(fig\.?|figure)\s*\d+", re.IGNORECASE)
_TABLE_RE = re.compile(r"^\s*table\s*\d+", re.IGNORECASE)
_CJK_ASCII_BOUNDARY_RE = re.compile(
    r"(?<=[一-鿿　-〿＀-￯])"
    r"(?=[A-Za-z0-9])"
    r"|"
    r"(?<=[A-Za-z0-9])"
    r"(?=[一-鿿　-〿＀-￯])"
)
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z]*Net|[A-Z]{2,}(?:-[A-Z0-9]+)*|[A-Z][a-z]+[A-Z][A-Za-z0-9]*|[A-Z]+[a-z]*\d+[A-Za-z0-9]*)\b"
)
_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9-]{2,}\b")

_COMMON_KEYWORDS = {
    "PSNR",
    "SSIM",
    "UIQM",
    "UCIQE",
    "LPIPS",
    "NIQE",
    "UIEBD",
    "EUVP",
    "LSUI",
    "DAE",
    "CANet",
    "LCDNet",
    "ablation",
    "underwater",
    "enhancement",
    "restoration",
}


@dataclass
class PaperSection:
    """A contiguous Markdown section with paper-aware metadata."""

    section_title: str
    section_type: str
    subsection: Optional[str]
    content_type: str
    text: str
    page: Optional[int]
    keywords: list[str]
    entities: list[str]


def classify_section(title: str, content: str = "") -> str:
    """Map a section heading or caption line to a normalized section type."""
    normalized = f"{title} {content[:200]}".lower()
    if _TABLE_RE.match(title):
        return "table_caption"
    if _FIGURE_RE.match(title):
        return "figure_caption"
    if "abstract" in normalized:
        return "abstract"
    if "introduction" in normalized:
        return "introduction"
    if "related" in normalized or "background" in normalized:
        return "related_work"
    if "ablation" in normalized:
        return "ablation"
    if any(word in normalized for word in ("experiment", "evaluation", "result", "quantitative", "qualitative", "comparison")):
        return "experiment"
    if any(word in normalized for word in ("method", "approach", "architecture", "framework", "module", "network", "model")):
        return "method"
    if "discussion" in normalized or "limitation" in normalized:
        return "discussion"
    if "conclusion" in normalized or "future work" in normalized:
        return "conclusion"
    if "reference" in normalized or "bibliography" in normalized:
        return "reference"
    return "method" if any(word in normalized for word in ("design", "adaptive", "restoration")) else "introduction"


def extract_entities(text: str) -> list[str]:
    """Extract model names, metrics, datasets and acronym-like entities."""
    # Insert space at CJK-ASCII boundaries so \b word boundaries trigger
    spaced = _CJK_ASCII_BOUNDARY_RE.sub(" ", text) if _CJK_ASCII_BOUNDARY_RE.search(text) else text
    entities = {match.group(0).strip("-") for match in _ENTITY_RE.finditer(spaced)}
    entities.update(keyword for keyword in _COMMON_KEYWORDS if re.search(rf"\b{re.escape(keyword)}\b", spaced, re.IGNORECASE))
    return sorted(entities, key=lambda item: (item.lower(), item))


def extract_keywords(text: str, title: str = "") -> list[str]:
    """Extract lightweight keywords from title plus body text."""
    source = f"{title}\n{text}"
    entities = set(extract_entities(source))
    frequent: dict[str, int] = {}
    for token in _TOKEN_RE.findall(source):
        normalized = token.strip("-")
        if len(normalized) < 3:
            continue
        lower = normalized.lower()
        if lower in {"the", "and", "for", "with", "from", "that", "this", "are", "was", "were"}:
            continue
        frequent[normalized] = frequent.get(normalized, 0) + 1
    ranked = sorted(frequent, key=lambda item: (-frequent[item], item.lower()))
    keywords = list(entities)
    for token in ranked:
        if token not in keywords:
            keywords.append(token)
        if len(keywords) >= 16:
            break
    return keywords


def _content_type(title: str, section_type: str) -> str:
    if section_type in {"figure_caption", "table_caption"}:
        return section_type
    if _TABLE_RE.match(title):
        return "table_caption"
    if _FIGURE_RE.match(title):
        return "figure_caption"
    return "text"


def parse_markdown_sections(markdown: str, title: Optional[str] = None) -> list[PaperSection]:
    """Parse Markdown into section-level blocks without crossing headings."""
    sections: list[PaperSection] = []
    current_title = title or "Document"
    current_subsection: Optional[str] = None
    current_level = 1
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        text = "\n".join(buffer).strip()
        if not text:
            buffer = []
            return
        section_type = classify_section(current_title, text)
        content_type = _content_type(current_title, section_type)
        sections.append(
            PaperSection(
                section_title=current_title,
                section_type=section_type,
                subsection=current_subsection,
                content_type=content_type,
                text=text,
                page=None,
                keywords=extract_keywords(text, current_title),
                entities=extract_entities(f"{current_title}\n{text}"),
            )
        )
        buffer = []

    for line in markdown.splitlines():
        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            current_level = len(heading.group(1))
            current_title = heading.group(2).strip()
            if current_level >= 3:
                current_subsection = current_title
            elif current_level <= 2:
                current_subsection = None
            continue

        if _TABLE_RE.match(line) or _FIGURE_RE.match(line):
            flush()
            caption_title = line.strip()
            section_type = classify_section(caption_title)
            sections.append(
                PaperSection(
                    section_title=current_title,
                    section_type=section_type,
                    subsection=current_subsection,
                    content_type=section_type,
                    text=caption_title,
                    page=None,
                    keywords=extract_keywords(caption_title, current_title),
                    entities=extract_entities(caption_title),
                )
            )
            continue

        buffer.append(line)

    flush()
    return sections
