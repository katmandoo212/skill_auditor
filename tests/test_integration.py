# tests/test_integration.py
"""Integration tests for skill auditor."""
import pytest
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
import numpy as np


@pytest.fixture
def sample_skills(tmp_path):
    """Create sample skill files for testing."""
    # Plugin 1
    skill1_dir = tmp_path / "plugin1" / "skills" / "code-review"
    skill1_dir.mkdir(parents=True)
    (skill1_dir / "SKILL.md").write_text("""---
name: Code Review
description: Review code for quality issues
trigger:
  - "review code"
  - "check my code"
---
Review code for quality, readability, and best practices.
""")

    # Plugin 2 - similar skill
    skill2_dir = tmp_path / "plugin2" / "skills" / "request-code-review"
    skill2_dir.mkdir(parents=True)
    (skill2_dir / "SKILL.md").write_text("""---
name: Request Code Review
description: Request a code review for your changes
trigger:
  - "code review"
  - "review my code"
---
Request a code review for pull requests or code changes.
""")

    # Plugin 3 - different skill
    skill3_dir = tmp_path / "plugin1" / "skills" / "brainstorm"
    skill3_dir.mkdir(parents=True)
    (skill3_dir / "SKILL.md").write_text("""---
name: Brainstorm
description: Brainstorm ideas for features
trigger:
  - "brainstorm ideas"
---
Generate and explore creative ideas.
""")

    return tmp_path


def test_full_pipeline_discover_parse(sample_skills):
    """Test discovery and parsing pipeline."""
    from skill_auditor.scanner import discover_skills, parse_skill

    # Discover
    skill_files = discover_skills([sample_skills])
    assert len(skill_files) == 3

    # Parse
    skills = [parse_skill(f) for f in skill_files]
    assert len(skills) == 3

    # Check skill names
    skill_names = {s.skill_name for s in skills}
    assert "code-review" in skill_names
    assert "request-code-review" in skill_names
    assert "brainstorm" in skill_names

    # Check skills have expected attributes
    code_review = next(s for s in skills if s.skill_name == "code-review")
    assert code_review.display_name == "Code Review"
    assert "review code" in code_review.triggers
    # Plugin name is 'user-skills' because test paths don't have 'plugins' in them
    # (the parser defaults to 'user-skills' when no plugin is found)
    assert code_review.plugin_name == "user-skills"


def test_full_pipeline_with_mocked_embeddings(sample_skills):
    """Test full pipeline from discovery to find_candidates with mocked embeddings."""
    from skill_auditor.scanner import discover_skills, parse_skill
    from skill_auditor.embeddings import find_candidates
    from skill_auditor.models import SimilarityCandidate

    # Discover
    skill_files = discover_skills([sample_skills])
    assert len(skill_files) == 3

    # Parse
    skills = [parse_skill(f) for f in skill_files]
    assert len(skills) == 3

    # Sort skills by name to ensure consistent ordering
    skills = sorted(skills, key=lambda s: s.skill_name)
    # Order: brainstorm, code-review, request-code-review

    # Create mock embeddings: code-review and request-code-review are similar,
    # brainstorm is different
    # brainstorm: [0.0, 0.0, 1.0] (orthogonal)
    # code-review: [1.0, 0.0, 0.0]
    # request-code-review: [0.95, 0.1, 0.0] (similar to code-review)
    mock_embeddings = np.array([
        [0.0, 0.0, 1.0],      # brainstorm (different)
        [1.0, 0.0, 0.0],      # code-review
        [0.95, 0.1, 0.0],     # request-code-review (similar to code-review)
    ])

    # Find candidates with threshold 0.9
    candidates = find_candidates(skills, mock_embeddings, threshold=0.9)

    assert len(candidates) == 3

    # code-review should have request-code-review as candidate
    code_review_candidates = next(c for c in candidates if c.skill.skill_name == "code-review")
    assert len(code_review_candidates.candidates) == 1
    assert code_review_candidates.candidates[0][0].skill_name == "request-code-review"

    # request-code-review should have code-review as candidate
    request_review_candidates = next(c for c in candidates if c.skill.skill_name == "request-code-review")
    assert len(request_review_candidates.candidates) == 1
    assert request_review_candidates.candidates[0][0].skill_name == "code-review"

    # brainstorm should have no candidates (orthogonal)
    brainstorm_candidates = next(c for c in candidates if c.skill.skill_name == "brainstorm")
    assert len(brainstorm_candidates.candidates) == 0


def test_cli_help():
    """Test CLI shows help."""
    import typer
    from skill_auditor import main

    app = typer.Typer()
    app.command()(main)
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Audit Claude Code skills" in result.output


def test_cli_version():
    """Test CLI version flag works."""
    import typer
    from skill_auditor import main

    app = typer.Typer()
    app.command()(main)
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "0.1.0" in result.output


@patch("skill_auditor.check_ollama_connection")
@patch("skill_auditor.generate_embeddings")
def test_cli_scan_with_path(mock_embeddings, mock_ollama, sample_skills):
    """Test CLI scan command with custom path."""
    import typer
    from skill_auditor import main
    import numpy as np

    # Mock the embeddings to return array of zeros
    mock_embeddings.return_value = np.zeros((3, 384))

    app = typer.Typer()
    app.command()(main)
    runner = CliRunner()
    result = runner.invoke(app, ["-p", str(sample_skills), "-o", str(sample_skills / "report.md")])

    # Should complete without error (though will find no duplicates)
    assert result.exit_code == 0, f"CLI failed with: {result.output}"