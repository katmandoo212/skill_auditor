# tests/test_cli.py
"""Tests for CLI."""
import pytest
from typer.testing import CliRunner
import typer
from skill_auditor import main


def test_cli_help():
    """Test CLI shows help."""
    app = typer.Typer()
    app.command()(main)
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Audit Claude Code skills" in result.output


def test_cli_version():
    """Test CLI version flag works."""
    app = typer.Typer()
    app.command()(main)
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "0.1.0" in result.output