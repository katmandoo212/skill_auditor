# tests/test_scanner.py
"""Tests for skill scanner module."""
import pytest
from pathlib import Path
from skill_auditor.scanner import discover_skills, parse_frontmatter, parse_skill
from skill_auditor.models import Skill


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


def test_parse_frontmatter_basic():
    """Test parsing basic frontmatter."""
    content = """---
name: Test Skill
description: A test skill
---
Skill content here."""

    frontmatter, body = parse_frontmatter(content)
    assert frontmatter["name"] == "Test Skill"
    assert frontmatter["description"] == "A test skill"
    assert "Skill content here" in body


def test_parse_frontmatter_with_triggers():
    """Test parsing frontmatter with trigger list."""
    content = """---
name: Test
trigger:
  - "test trigger"
  - "another trigger"
---
Content."""

    frontmatter, body = parse_frontmatter(content)
    assert len(frontmatter["trigger"]) == 2
    assert "test trigger" in frontmatter["trigger"]


def test_parse_skill(tmp_path):
    """Test parsing a SKILL.md file into a Skill object."""
    # Path pattern: .../cache/<plugin-name>/.../skills/<skill-name>/SKILL.md
    skill_dir = tmp_path / "cache" / "test-plugin" / "skills" / "test-skill"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"

    content = """---
name: Test Skill
description: A test skill for testing
trigger:
  - "test me"
---
This is the skill content.
"""
    skill_file.write_text(content)

    result = parse_skill(skill_file)

    assert result.skill_name == "test-skill"
    assert result.plugin_name == "test-plugin"
    assert result.display_name == "Test Skill"
    assert result.description == "A test skill for testing"
    assert len(result.triggers) == 1
    assert "skill content" in result.content