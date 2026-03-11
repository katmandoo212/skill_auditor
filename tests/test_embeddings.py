# tests/test_embeddings.py
"""Tests for embedding generation."""
import pytest
from pathlib import Path
from skill_auditor.models import Skill
from skill_auditor.embeddings import skill_to_text, find_candidates
import numpy as np


def test_skill_to_text_includes_name():
    """Test skill_to_text includes skill name."""
    skill = Skill(
        path=Path("/test/SKILL.md"),
        plugin_name="test",
        skill_name="test-skill",
        display_name="Test Skill",
        description="A test",
        triggers=[],
        content="Content here",
        metadata={},
    )
    result = skill_to_text(skill)
    assert "Test Skill" in result


def test_skill_to_text_includes_description():
    """Test skill_to_text includes description."""
    skill = Skill(
        path=Path("/test/SKILL.md"),
        plugin_name="test",
        skill_name="test-skill",
        display_name="Test",
        description="This is a description",
        triggers=[],
        content="Content",
        metadata={},
    )
    result = skill_to_text(skill)
    assert "This is a description" in result


def test_skill_to_text_includes_triggers():
    """Test skill_to_text includes triggers when present."""
    skill = Skill(
        path=Path("/test/SKILL.md"),
        plugin_name="test",
        skill_name="test",
        display_name="Test",
        description="Desc",
        triggers=["trigger one", "trigger two"],
        content="Content",
        metadata={},
    )
    result = skill_to_text(skill)
    assert "trigger one" in result
    assert "trigger two" in result


def test_skill_to_text_truncates_long_content():
    """Test skill_to_text truncates content to limit."""
    long_content = "x" * 5000
    skill = Skill(
        path=Path("/test/SKILL.md"),
        plugin_name="test",
        skill_name="test",
        display_name="Test",
        description="Desc",
        triggers=[],
        content=long_content,
        metadata={},
    )
    result = skill_to_text(skill)
    # Content section should be truncated
    assert len(result) < 6000


def test_find_candidates_returns_empty_for_no_matches():
    """Test find_candidates returns empty when no matches above threshold."""
    skill1 = Skill(
        path=Path("/a/SKILL.md"),
        plugin_name="p1",
        skill_name="a",
        display_name="A",
        description="Desc A",
        triggers=[],
        content="Content A",
        metadata={},
    )
    skill2 = Skill(
        path=Path("/b/SKILL.md"),
        plugin_name="p2",
        skill_name="b",
        display_name="B",
        description="Desc B",
        triggers=[],
        content="Content B",
        metadata={},
    )

    # Orthogonal embeddings (no similarity)
    embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    candidates = find_candidates([skill1, skill2], embeddings, threshold=0.9)
    assert len(candidates) == 2
    assert len(candidates[0].candidates) == 0
    assert len(candidates[1].candidates) == 0


def test_find_candidates_finds_similar():
    """Test find_candidates identifies similar skills."""
    skill1 = Skill(
        path=Path("/a/SKILL.md"),
        plugin_name="p1",
        skill_name="a",
        display_name="A",
        description="Desc A",
        triggers=[],
        content="Content A",
        metadata={},
    )
    skill2 = Skill(
        path=Path("/b/SKILL.md"),
        plugin_name="p2",
        skill_name="b",
        display_name="B",
        description="Desc B",
        triggers=[],
        content="Content B",
        metadata={},
    )

    # Identical embeddings (maximum similarity)
    embeddings = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
    ])

    candidates = find_candidates([skill1, skill2], embeddings, threshold=0.5)
    assert len(candidates[0].candidates) == 1
    assert candidates[0].candidates[0][1] == 1.0