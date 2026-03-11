# tests/test_cli.py
"""Tests for CLI."""
import pytest
from typer.testing import CliRunner


def test_cli_help():
    """Test CLI shows help."""
    from skill_auditor import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Audit Claude Code skills" in result.output


def test_cli_version():
    """Test CLI version flag works."""
    from skill_auditor import app

    runner = CliRunner()
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "0.1.0" in result.output