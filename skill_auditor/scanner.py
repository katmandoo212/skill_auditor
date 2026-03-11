# skill_auditor/scanner.py
"""Skill discovery and parsing."""
from pathlib import Path
from typing import Optional
from skill_auditor.config import DEFAULT_PLUGIN_CACHE


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