# tests/test_reporter.py
"""Tests for report generation."""
import pytest
from pathlib import Path
from datetime import datetime
from skill_auditor.models import Skill, DuplicateGroup
from skill_auditor.reporter import generate_report


def test_generate_report_creates_file(tmp_path):
    """Test generate_report creates output file."""
    output = tmp_path / "report.md"

    skill = Skill(
        path=Path("/test/SKILL.md"),
        plugin_name="test",
        skill_name="test-skill",
        display_name="Test Skill",
        description="A test",
        triggers=[],
        content="Content",
        metadata={},
    )

    group = DuplicateGroup(
        purpose_tag="test-purpose",
        skills=[skill],
        confidence="high",
        notes="Test notes",
    )

    stats = {
        "total_skills": 1,
        "duplicated_skills": 1,
        "threshold": 0.8,
        "model": "test-model",
        "all_skills": [skill],
    }

    generate_report([group], stats, output)

    assert output.exists()


def test_generate_report_contains_summary(tmp_path):
    """Test report contains summary section."""
    output = tmp_path / "report.md"

    skill = Skill(
        path=Path("/test/SKILL.md"),
        plugin_name="test",
        skill_name="test-skill",
        display_name="Test Skill",
        description="A test",
        triggers=[],
        content="Content",
        metadata={},
    )

    group = DuplicateGroup(
        purpose_tag="test-purpose",
        skills=[skill],
        confidence="high",
        notes="Test notes",
    )

    stats = {
        "total_skills": 5,
        "duplicated_skills": 2,
        "threshold": 0.8,
        "model": "test-model",
        "all_skills": [skill],
    }

    generate_report([group], stats, output)

    content = output.read_text()
    assert "# Skill Audit Report" in content
    assert "Skills scanned: 5" in content
    assert "Duplicate groups found: 1" in content


def test_generate_report_contains_duplicate_groups(tmp_path):
    """Test report contains duplicate group details."""
    output = tmp_path / "report.md"

    skill = Skill(
        path=Path("/test/SKILL.md"),
        plugin_name="test-plugin",
        skill_name="test-skill",
        display_name="Test Skill",
        description="A test",
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

    stats = {
        "total_skills": 1,
        "duplicated_skills": 1,
        "threshold": 0.8,
        "model": "test-model",
        "all_skills": [skill],
    }

    generate_report([group], stats, output)

    content = output.read_text()
    assert "Group 1: code-review" in content
    assert "Test Skill" in content
    assert "test-plugin" in content
    assert "high" in content


def test_generate_report_contains_all_skills_table(tmp_path):
    """Test report contains table of all scanned skills."""
    output = tmp_path / "report.md"

    skill1 = Skill(
        path=Path("/a/SKILL.md"),
        plugin_name="plugin1",
        skill_name="skill-a",
        display_name="Skill A",
        description="Description for skill A that is longer than fifty characters",
        triggers=[],
        content="Content A",
        metadata={},
    )

    skill2 = Skill(
        path=Path("/b/SKILL.md"),
        plugin_name="plugin2",
        skill_name="skill-b",
        display_name="Skill B",
        description="Description for skill B",
        triggers=[],
        content="Content B",
        metadata={},
    )

    group = DuplicateGroup(
        purpose_tag="test",
        skills=[skill1, skill2],
        confidence="medium",
        notes="Test",
    )

    stats = {
        "total_skills": 2,
        "duplicated_skills": 2,
        "threshold": 0.8,
        "model": "test-model",
        "all_skills": [skill1, skill2],
    }

    generate_report([group], stats, output)

    content = output.read_text()
    assert "All Scanned Skills" in content
    assert "Skill A" in content
    assert "Skill B" in content
    assert "plugin1" in content
    assert "plugin2" in content