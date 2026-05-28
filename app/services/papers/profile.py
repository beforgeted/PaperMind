"""Extract paper-level profile from parsed markdown."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Iterable

from langchain_core.prompts import ChatPromptTemplate

from app.core.logging import logger
from app.core.schemas import PaperProfile, ParsedDocument
from app.services.llm import get_llm
from app.shared.paper_utils import parse_markdown_sections

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

_IMAGE_KEYWORDS = (
    "image",
    "images",
    "visual",
    "vision",
    "computer vision",
    "image enhancement",
    "image restoration",
    "image denoising",
    "image dehazing",
    "object detection",
    "segmentation",
    "super resolution",
    "underwater image",
    "图像",
    "视觉",
    "图像增强",
    "图像恢复",
    "目标检测",
    "图像分割",
    "水下图像",
)

_FREQ_KEYWORDS = (
    "fourier",
    "fft",
    "dft",
    "frequency domain",
    "frequency spectrum",
    "spectrum",
    "magnitude spectrum",
    "amplitude spectrum",
    "phase spectrum",
    "wavelet",
    "频域",
    "频谱",
    "傅里叶",
    "快速傅里叶",
    "幅度谱",
    "相位谱",
    "小波",
)

_PROFILE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个论文信息抽取助手。请根据给定论文内容，抽取论文级结构化信息。

要求：
1. 只返回 JSON，不要返回 Markdown，不要返回解释；
2. 不确定的字段可以返回空字符串或空数组；
3. tags 使用英文短语，便于后续检索；
4. summary 使用中文；
5. abstract_summary、introduction_summary、method_summary、contribution_summary、experiment_summary 使用中文；
6. is_image_related 表示论文是否与图像处理、计算机视觉、图像增强、目标检测、图像分割、图像恢复等相关；
7. is_frequency_related 表示论文主体方法是否涉及傅里叶、FFT、DFT、频域、频谱、幅度谱、相位谱、小波变换等频率分析方法；
8. 如果 Fourier / FFT / frequency domain 只出现在 references 或 related work 中，不要判断为 frequency_related；
9. image_confidence、frequency_confidence 取 0 到 1；
10. evidence 数组中的每项必须包含 keyword、field、snippet。

请返回如下 JSON 格式：
{{
  "title": "",
  "abstract": "",
  "clean_abstract": "",
  "abstract_summary": "",
  "introduction_summary": "",
  "method_summary": "",
  "contribution_summary": "",
  "experiment_summary": "",
  "research_problem": "",
  "method_name": "",
  "authors": [],
  "year": "",
  "main_task": "",
  "modality_tags": [],
  "task_tags": [],
  "method_tags": [],
  "domain_tags": [],
  "dataset_tags": [],
  "metric_tags": [],
  "is_image_related": false,
  "is_frequency_related": false,
  "matched_keywords": [],
  "image_confidence": 0.0,
  "frequency_confidence": 0.0,
  "image_evidence": [],
  "frequency_evidence": [],
  "summary": ""
}}""",
        ),
        (
            "human",
            "论文内容如下：\n\n标题：\n{title}\n\n摘要：\n{abstract}\n\n正文片段：\n{content}",
        ),
    ]
)

_FIELD_WEIGHTS = {
    "title": 0.42,
    "abstract": 0.36,
    "method": 0.36,
    "contribution": 0.36,
    "introduction": 0.22,
    "experiment": 0.18,
    "related_work": 0.12,
    "references": 0.0,
}

_SUMMARY_FIELDS = (
    "abstract_summary",
    "introduction_summary",
    "method_summary",
    "contribution_summary",
    "experiment_summary",
)


def _safe_json_load(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        matched = _JSON_BLOCK_RE.search(text)
        if not matched:
            return {}
        try:
            return json.loads(matched.group(0))
        except json.JSONDecodeError:
            return {}


def _pick_abstract(parsed: ParsedDocument) -> str:
    sections = parse_markdown_sections(parsed.markdown, parsed.title)
    for section in sections:
        if section.section_type == "abstract":
            return _clean_text(section.text)[:2000]
    return _clean_text(parsed.markdown)[:2000]


def _pick_content(parsed: ParsedDocument) -> str:
    sections = parse_markdown_sections(parsed.markdown, parsed.title)
    selected = []
    preferred = {"abstract", "introduction", "method", "experiment"}
    for section in sections:
        if section.section_type in preferred:
            selected.append(_clean_text(section.text))
        if len("\n\n".join(selected)) >= 4000:
            break
    if not selected:
        return parsed.markdown[:4000]
    return "\n\n".join(selected)[:4000]


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def _clean_text(text: str) -> str:
    """清理 Markdown 噪声，保留用于画像抽取的正文语义。"""
    cleaned = re.sub(r"!\[[^\]]*]\([^)]*\)", " ", text or "")
    cleaned = re.sub(r"\[[^\]]+]\([^)]*\)", " ", cleaned)
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _as_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _as_float(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        normalized = item.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            out.append(normalized)
    return out


def _append_tag(tags: list[str], value: str) -> None:
    if not value:
        return
    key = value.lower()
    if all(item.lower() != key for item in tags):
        tags.append(value)


def _merge_section_texts(parsed: ParsedDocument) -> dict[str, str]:
    """按论文结构聚合正文，供章节加权兜底使用。"""
    sections = parse_markdown_sections(parsed.markdown, parsed.title)
    grouped: dict[str, list[str]] = {
        "introduction": [],
        "method": [],
        "contribution": [],
        "experiment": [],
        "related_work": [],
        "references": [],
    }
    for section in sections:
        title = (section.section_title or "").lower()
        text = _clean_text(section.text)
        if not text:
            continue
        if "contribution" in title or "贡献" in title:
            grouped["contribution"].append(text)
            continue
        if section.section_type == "reference":
            grouped["references"].append(text)
        elif section.section_type in grouped:
            grouped[section.section_type].append(text)
    return {field: "\n".join(parts)[:3000] for field, parts in grouped.items()}


def _snippet(text: str, keyword: str, *, window: int = 90) -> str:
    lowered = text.lower()
    idx = lowered.find(keyword.lower())
    if idx < 0:
        return _clean_text(text)[: window * 2]
    start = max(0, idx - window)
    end = min(len(text), idx + len(keyword) + window)
    return _clean_text(text[start:end])


def _keyword_in_text(text: str, keyword: str) -> bool:
    if not text or not keyword:
        return False
    if re.search(r"[\u4e00-\u9fff]", keyword):
        return keyword in text
    pattern = rf"(?<![A-Za-z0-9]){re.escape(keyword)}(?![A-Za-z0-9])"
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def _collect_evidence(field_texts: dict[str, str], keywords: tuple[str, ...]) -> list[dict]:
    evidence: list[dict] = []
    for field, text in field_texts.items():
        for keyword in keywords:
            if _keyword_in_text(text, keyword):
                evidence.append(
                    {
                        "keyword": keyword,
                        "field": field,
                        "snippet": _snippet(text, keyword),
                    }
                )
    return evidence


def _confidence_from_evidence(evidence: list[dict]) -> float:
    score = 0.0
    seen_fields: set[str] = set()
    matched_keywords: set[str] = set()
    for item in evidence:
        field = str(item.get("field") or "")
        if field == "references":
            continue
        keyword = str(item.get("keyword") or "")
        matched_keywords.add(keyword.lower())
        if field not in seen_fields:
            seen_fields.add(field)
            score += _FIELD_WEIGHTS.get(field, 0.0)
    if matched_keywords:
        score += min(0.18, max(0, len(matched_keywords) - 1) * 0.04)
    return round(min(score, 1.0), 3)


def _fallback_summary(text: str, default: str = "") -> str:
    text = _clean_text(text)
    if not text:
        return default
    return text[:220]


def _build_paper_search_text(profile: PaperProfile) -> str:
    parts: list[str] = [
        profile.title,
        profile.clean_abstract or profile.abstract,
        profile.abstract_summary,
        profile.introduction_summary,
        profile.method_summary,
        profile.contribution_summary,
        profile.research_problem,
        profile.method_name,
        " ".join(profile.task_tags),
        " ".join(profile.method_tags),
        " ".join(profile.domain_tags),
        " ".join(profile.dataset_tags),
        " ".join(profile.metric_tags),
    ]
    return _clean_text("\n".join(part for part in parts if part))


def _apply_rule_fallback(profile: PaperProfile, parsed: ParsedDocument, content: str) -> PaperProfile:
    section_texts = _merge_section_texts(parsed)
    field_texts = {
        "title": profile.title,
        "abstract": profile.clean_abstract or profile.abstract,
        "introduction": section_texts.get("introduction", ""),
        "method": section_texts.get("method", ""),
        "contribution": section_texts.get("contribution", ""),
        "experiment": section_texts.get("experiment", ""),
        "related_work": section_texts.get("related_work", ""),
        "references": section_texts.get("references", ""),
    }

    image_evidence = _collect_evidence(field_texts, _IMAGE_KEYWORDS)
    freq_evidence = _collect_evidence(field_texts, _FREQ_KEYWORDS)
    image_confidence = max(profile.image_confidence, _confidence_from_evidence(image_evidence))
    frequency_confidence = max(profile.frequency_confidence, _confidence_from_evidence(freq_evidence))

    profile.image_evidence = profile.image_evidence or image_evidence
    profile.frequency_evidence = profile.frequency_evidence or freq_evidence
    profile.image_confidence = round(image_confidence, 3)
    profile.frequency_confidence = round(frequency_confidence, 3)
    profile.is_image_related = profile.image_confidence >= 0.6
    profile.is_frequency_related = profile.frequency_confidence >= 0.6
    profile.matched_keywords = _dedupe(
        [
            *(item.get("keyword", "") for item in profile.image_evidence),
            *(item.get("keyword", "") for item in profile.frequency_evidence),
            *profile.matched_keywords,
        ]
    )

    lowered_main_fields = "\n".join(
        field_texts[field].lower()
        for field in ("title", "abstract", "introduction", "method", "contribution", "experiment")
    )
    if profile.is_image_related:
        _append_tag(profile.modality_tags, "image")
        _append_tag(profile.domain_tags, "computer vision")
    if "enhancement" in lowered_main_fields or "图像增强" in lowered_main_fields:
        _append_tag(profile.task_tags, "image enhancement")
    if "restoration" in lowered_main_fields or "图像恢复" in lowered_main_fields:
        _append_tag(profile.task_tags, "image restoration")
    if "underwater" in lowered_main_fields or "水下" in lowered_main_fields:
        _append_tag(profile.domain_tags, "underwater vision")
    if "detection" in lowered_main_fields or "目标检测" in lowered_main_fields:
        _append_tag(profile.task_tags, "object detection")
    if "segmentation" in lowered_main_fields or "图像分割" in lowered_main_fields:
        _append_tag(profile.task_tags, "segmentation")

    if profile.is_frequency_related:
        _append_tag(profile.domain_tags, "frequency domain")
        if any(term in lowered_main_fields for term in ("fourier", "fft", "dft", "傅里叶", "快速傅里叶")):
            _append_tag(profile.method_tags, "Fourier Transform")
        if "wavelet" in lowered_main_fields or "小波" in lowered_main_fields:
            _append_tag(profile.method_tags, "Wavelet Transform")

    if not profile.year:
        matched = _YEAR_RE.search(content)
        if matched:
            profile.year = matched.group(0)
    if not profile.clean_abstract:
        profile.clean_abstract = _clean_text(profile.abstract)
    if not profile.abstract_summary:
        profile.abstract_summary = _fallback_summary(profile.clean_abstract)
    if not profile.introduction_summary:
        profile.introduction_summary = _fallback_summary(section_texts.get("introduction", ""))
    if not profile.method_summary:
        profile.method_summary = _fallback_summary(section_texts.get("method", ""))
    if not profile.contribution_summary:
        profile.contribution_summary = _fallback_summary(section_texts.get("contribution", ""))
    if not profile.experiment_summary:
        profile.experiment_summary = _fallback_summary(section_texts.get("experiment", ""))
    profile.paper_search_text = _build_paper_search_text(profile)
    return profile


def _profile_from_payload(
    payload: dict[str, Any],
    base: PaperProfile,
    parsed: ParsedDocument,
    now: datetime,
) -> PaperProfile:
    return PaperProfile(
        paper_id=parsed.task_id,
        title=(payload.get("title") or base.title).strip(),
        abstract=(payload.get("abstract") or base.abstract).strip(),
        clean_abstract=(payload.get("clean_abstract") or base.clean_abstract).strip(),
        abstract_summary=(payload.get("abstract_summary") or "").strip(),
        introduction_summary=(payload.get("introduction_summary") or "").strip(),
        method_summary=(payload.get("method_summary") or "").strip(),
        contribution_summary=(payload.get("contribution_summary") or "").strip(),
        experiment_summary=(payload.get("experiment_summary") or "").strip(),
        research_problem=(payload.get("research_problem") or "").strip(),
        method_name=(payload.get("method_name") or "").strip(),
        authors=_as_list(payload.get("authors")),
        year=(payload.get("year") or "").strip(),
        source_file=parsed.original_filename,
        main_task=(payload.get("main_task") or "").strip(),
        modality_tags=_as_list(payload.get("modality_tags")),
        task_tags=_as_list(payload.get("task_tags")),
        method_tags=_as_list(payload.get("method_tags")),
        domain_tags=_as_list(payload.get("domain_tags")),
        dataset_tags=_as_list(payload.get("dataset_tags")),
        metric_tags=_as_list(payload.get("metric_tags")),
        is_image_related=bool(payload.get("is_image_related")),
        is_frequency_related=bool(payload.get("is_frequency_related")),
        matched_keywords=_as_list(payload.get("matched_keywords")),
        image_confidence=_as_float(payload.get("image_confidence")),
        frequency_confidence=_as_float(payload.get("frequency_confidence")),
        image_evidence=payload.get("image_evidence") or [],
        frequency_evidence=payload.get("frequency_evidence") or [],
        summary=(payload.get("summary") or "").strip(),
        created_at=now,
        updated_at=now,
    )


async def extract_paper_profile(parsed: ParsedDocument) -> PaperProfile:
    """提取论文画像：LLM 优先，规则兜底，不抛出阻断异常。"""
    now = datetime.utcnow()
    base = PaperProfile(
        paper_id=parsed.task_id,
        title=parsed.title or "",
        abstract=_pick_abstract(parsed),
        clean_abstract=_pick_abstract(parsed),
        source_file=parsed.original_filename,
        created_at=now,
        updated_at=now,
    )
    content = _pick_content(parsed)
    raw = ""
    try:
        chain = _PROFILE_PROMPT | get_llm()
        response = await chain.ainvoke(
            {
                "title": base.title,
                "abstract": base.abstract,
                "content": content,
            }
        )
        raw = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM paper profile extraction failed (task_id={}): {}", parsed.task_id, exc)

    payload = _safe_json_load(raw)
    if payload:
        try:
            base = _profile_from_payload(payload, base, parsed, now)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Invalid paper profile JSON (task_id={}): {}", parsed.task_id, exc)

    merged_text = f"{base.title}\n{base.abstract}\n{content}"
    base = _apply_rule_fallback(base, parsed, merged_text)
    if not base.main_task:
        if base.is_image_related:
            base.main_task = "image-related research"
        elif base.is_frequency_related:
            base.main_task = "frequency-domain research"
    if not base.summary:
        summary_parts = [getattr(base, field) for field in _SUMMARY_FIELDS if getattr(base, field)]
        base.summary = summary_parts[0] if summary_parts else "该论文已完成自动画像抽取，可进一步查看问答结果获取细节。"
    base.paper_search_text = _build_paper_search_text(base)
    return base

