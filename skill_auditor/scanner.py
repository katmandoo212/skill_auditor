# skill_auditor/scanner.py
"""Skill discovery and parsing."""
import yaml
from pathlib import Path
from typing import Optional
from skill_auditor.config import DEFAULT_PLUGIN_CACHE
from skill_auditor.models import Skill


def discover_skills(paths: Optional[list[Path]] = None) -> list[Path]:
    """Find all SKILL.md files in given paths or default plugin cache.

    Args:
        paths: Optional list of paths to scan. Uses DEFAULT_PLUGIN_CACHE if None.

    Returns:
        List of paths to SKILL.md files.
    """
    if paths is None:
        paths = [DEFAULT_PLUGIN_CACHE]

    skill_files = []
    for base_path in paths:
        if not base_path.exists():
            continue
        for skill_file in base_path.rglob("skills/*/SKILL.md"):
            skill_files.append(skill_file)

    return skill_files


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse frontmatter and body from SKILL.md content.

    Args:
        content: Full file content with optional frontmatter.

    Returns:
        Tuple of (frontmatter dict, body string).
    """
    lines = content.strip().split("\n")

    if not lines or lines[0] != "---":
        return {}, content

    # Find closing ---
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i] == "---":
            end_idx = i
            break

    if end_idx is None:
        return {}, content

    frontmatter_str = "\n".join(lines[1:end_idx])
    body = "\n".join(lines[end_idx + 1:])

    try:
        frontmatter = yaml.safe_load(frontmatter_str) or {}
    except yaml.YAMLError:
        frontmatter = {}

    return frontmatter, body.strip()


def parse_skill(skill_path: Path) -> Skill:
    """Parse a SKILL.md file into a Skill dataclass.

    Args:
        skill_path: Path to the SKILL.md file.

    Returns:
        Skill object with parsed metadata.
    """
    content = skill_path.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(content)

    # Extract plugin name from path
    # Path pattern: .../cache/<plugin-name>/.../skills/<skill-name>/SKILL.md
    parts = skill_path.parts
    plugin_name = "unknown"
    for i, part in enumerate(parts):
        if part == "cache" and i + 1 < len(parts):
            plugin_name = parts[i + 1]
            break

    return Skill(
        path=skill_path,
        plugin_name=plugin_name,
        skill_name=skill_path.parent.name,
        display_name=frontmatter.get("name", ""),
        description=frontmatter.get("description", ""),
        triggers=frontmatter.get("trigger", []) or [],
        content=body,
        metadata=frontmatter,
    )