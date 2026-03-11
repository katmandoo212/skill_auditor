# tests/test_models.py
"""Tests for skill data models."""
import pytest
from pathlib import Path
from skill_auditor.models import Skill, DuplicateGroup, SimilarityCandidate


def test_skill_creation():
    """Test Skill dataclass creation."""
    skill = Skill(
        path=Path("/test/skills/example/SKILL.md"),
        plugin_name="test-plugin",
        skill_name="example",
        display_name="Example Skill",
        description="A test skill",
        triggers=["test trigger"],
        content="Skill content here",
        metadata={"name": "Example Skill"},
    )
    assert skill.skill_name == "example"
    assert skill.plugin_name == "test-plugin"
    assert len(skill.triggers) == 1


def test_similarity_candidate_creation():
    """Test SimilarityCandidate dataclass creation."""
    skill1 = Skill(
        path=Path("/test/skills/a/SKILL.md"),
        plugin_name="plugin1",
        skill_name="a",
        display_name="Skill A",
        description="Description A",
        triggers=[],
        content="Content A",
        metadata={},
    )
    skill2 = Skill(
        path=Path("/test/skills/b/SKILL.md"),
        plugin_name="plugin2",
        skill_name="b",
        display_name="Skill B",
        description="Description B",
        triggers=[],
        content="Content B",
        metadata={},
    )
    candidate = SimilarityCandidate(
        skill=skill1,
        candidates=[(skill2, 0.95)]
    )
    assert candidate.skill.skill_name == "a"
    assert len(candidate.candidates) == 1
    assert candidate.candidates[0][1] == 0.95


def test_duplicate_group_creation():
    """Test DuplicateGroup dataclass creation."""
    skill = Skill(
        path=Path("/test/skills/example/SKILL.md"),
        plugin_name="plugin",
        skill_name="example",
        display_name="Example",
        description="Test",
        triggers=[],
        content="Content",
        metadata={},
    )
    group = DuplicateGroup(
        purpose_tag="code-review",
        skills=[skill],
        confidence="high",
        notes="Both perform code review",
    )
    assert group.purpose_tag == "code-review"
    assert group.confidence == "high"