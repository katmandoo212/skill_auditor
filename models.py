# models.py
"""Data models for skill auditor."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Skill:
    """Represents a Claude Code skill."""
    path: Path
    plugin_name: str
    skill_name: str
    display_name: str
    description: str
    triggers: list[str]
    content: str
    metadata: dict


@dataclass
class DuplicateGroup:
    """Represents a group of similar skills."""
    purpose_tag: str
    skills: list[Skill]
    confidence: str
    notes: str


@dataclass
class SimilarityCandidate:
    """Represents a skill with its similar candidates."""
    skill: Skill
    candidates: list[tuple[Skill, float]]