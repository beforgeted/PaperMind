"""Skill loader -- parses Claude Code SKILL.md files for agent use.

Installed skills live in app/agent/skills/<vendor>/<skill-name>/.

Each skill directory contains:
  - SKILL.md  : YAML frontmatter (name, description, version) + markdown rules
  - references/ : supplementary rule files
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_SKILLS_ROOT = Path(__file__).resolve().parent.parent / "agent" / "skills"

_YAML_BOUNDARY = re.compile(r"^---\s*$", re.MULTILINE)
_SIMPLE_KV = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.*)")
_MAX_RULE_CHARS = 8000


def _parse_simple_frontmatter(text: str) -> dict[str, Any]:
    """Parse simple YAML-like key: value frontmatter without needing pyyaml."""
    result: dict[str, Any] = {}
    for line in text.split("\n"):
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        m = _SIMPLE_KV.match(line)
        if m:
            key = m.group(1).strip()
            value = m.group(2).strip().strip('"').strip("'")
            # handle nested via indentation: skip for simplicity
            if value == "":
                continue
            result[key] = value
    return result


@dataclass(frozen=True)
class SkillMeta:
    name: str
    description: str = ""
    version: str = "0.0.0"
    vendor: str = ""
    path: str = ""  # relative path under skills/
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillContent:
    meta: SkillMeta
    rules: str  # main SKILL.md body (truncated to _MAX_RULE_CHARS)
    references: list[str] = field(default_factory=list)  # available reference file names


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter from markdown body. Returns (meta, body)."""
    parts = _YAML_BOUNDARY.split(text, maxsplit=2)
    if len(parts) >= 3 and parts[0].strip() == "":
        meta = _parse_simple_frontmatter(parts[1])
        body = parts[2].strip()
        return meta, body
    return {}, text.strip()


def _iter_skill_dirs() -> list[Path]:
    """Walk skills/ and return all directories containing SKILL.md."""
    if not _SKILLS_ROOT.exists():
        return []
    skill_dirs: list[Path] = []
    for vendor_dir in _SKILLS_ROOT.iterdir():
        if not vendor_dir.is_dir():
            continue
        for skill_dir in vendor_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            if (skill_dir / "SKILL.md").exists():
                skill_dirs.append(skill_dir)
    return skill_dirs


def _list_references(skill_dir: Path) -> list[str]:
    ref_dir = skill_dir / "references"
    if not ref_dir.is_dir():
        return []
    refs: list[str] = []
    for f in sorted(ref_dir.rglob("*.md")):
        rel = os.path.relpath(f, skill_dir).replace("\\", "/")
        refs.append(rel)
    return refs


def _load_skill(skill_dir: Path) -> Optional[SkillContent]:
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return None
    raw = skill_md.read_text(encoding="utf-8")
    frontmatter, body = _parse_frontmatter(raw)
    rel_path = os.path.relpath(skill_dir, _SKILLS_ROOT).replace("\\", "/")
    vendor = rel_path.split("/")[0] if "/" in rel_path else ""
    meta = SkillMeta(
        name=frontmatter.get("name", skill_dir.name),
        description=frontmatter.get("description", ""),
        version=frontmatter.get("version", frontmatter.get("metadata", {}).get("version", "0.0.0")),
        vendor=vendor,
        path=rel_path,
        metadata={k: v for k, v in frontmatter.items() if k not in ("name", "description", "version")},
    )
    return SkillContent(
        meta=meta,
        rules=body[:_MAX_RULE_CHARS],
        references=_list_references(skill_dir),
    )


class SkillLoader:
    """Loads and caches installed skills."""

    def __init__(self, root: Optional[Path] = None):
        self._root = root or _SKILLS_ROOT
        self._cache: dict[str, SkillContent] = {}

    def list_skills(self) -> list[SkillMeta]:
        """Return metadata for all installed skills."""
        result: list[SkillMeta] = []
        for skill_dir in _iter_skill_dirs():
            content = self._get_content(skill_dir)
            if content:
                result.append(content.meta)
        return result

    def load_skill(self, skill_name: str) -> Optional[SkillContent]:
        """Load a skill by name (e.g. 'nature-polishing' or 'academic-paper')."""
        for skill_dir in _iter_skill_dirs():
            content = self._get_content(skill_dir)
            if content and content.meta.name == skill_name:
                return content
        return None

    def load_reference(self, skill_name: str, ref_path: str) -> Optional[str]:
        """Load a specific reference file from a skill."""
        for skill_dir in _iter_skill_dirs():
            content = self._get_content(skill_dir)
            if content and content.meta.name == skill_name:
                ref_file = skill_dir / ref_path
                if ref_file.exists() and ref_file.is_file():
                    return ref_file.read_text(encoding="utf-8")
        return None

    def _get_content(self, skill_dir: Path) -> Optional[SkillContent]:
        key = str(skill_dir)
        if key not in self._cache:
            self._cache[key] = _load_skill(skill_dir)
        return self._cache[key]


_loader: Optional[SkillLoader] = None


def get_skill_loader() -> SkillLoader:
    global _loader
    if _loader is None:
        _loader = SkillLoader()
    return _loader
