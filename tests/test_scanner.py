# tests/test_scanner.py
"""Tests for skill scanner module."""
import pytest
from pathlib import Path
from skill_auditor.scanner import discover_skills
from unittest.mock import patch


def test_discover_skills_empty_path(tmp_path):
    """Test discover_skills returns empty list for non-existent path."""
    result = discover_skills([tmp_path / "nonexistent"])
    assert result == []


def test_discover_skills_finds_skill_md(tmp_path):
    """Test discover_skills finds SKILL.md files."""
    # Create fake skill structure
    skill_dir = tmp_path / "test-plugin" / "skills" / "test-skill"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text("---\nname: Test\n---\nContent")

    result = discover_skills([tmp_path])
    assert len(result) == 1
    assert result[0].name == "SKILL.md"


def test_discover_skills_multiple_plugins(tmp_path):
    """Test discover_skills finds skills across multiple plugins."""
    # Plugin 1
    skill1_dir = tmp_path / "plugin1" / "skills" / "skill-a"
    skill1_dir.mkdir(parents=True)
    (skill1_dir / "SKILL.md").write_text("---\nname: A\n---\nA")

    # Plugin 2
    skill2_dir = tmp_path / "plugin2" / "skills" / "skill-b"
    skill2_dir.mkdir(parents=True)
    (skill2_dir / "SKILL.md").write_text("---\nname: B\n---\nB")

    result = discover_skills([tmp_path])
    assert len(result) == 2