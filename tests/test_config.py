# tests/test_config.py
"""Tests for configuration module."""
import pytest
from pathlib import Path
from skill_auditor.config import (
    DEFAULT_PLUGIN_CACHE,
    DEFAULT_OUTPUT,
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    DEFAULT_MAX_CANDIDATES,
    EMBEDDING_MODEL,
)


def test_default_plugin_cache_path():
    """Test default plugin cache path resolves correctly."""
    assert DEFAULT_PLUGIN_CACHE.name == "cache"
    assert ".claude" in str(DEFAULT_PLUGIN_CACHE)


def test_default_output_path():
    """Test default output filename."""
    assert DEFAULT_OUTPUT.name == "skill_audit_report.md"


def test_default_model():
    """Test default Ollama model."""
    assert DEFAULT_MODEL == "glm5:cloud"


def test_default_threshold():
    """Test default similarity threshold."""
    assert DEFAULT_THRESHOLD == 0.8


def test_embedding_model_name():
    """Test embedding model name."""
    assert EMBEDDING_MODEL == "all-MiniLM-L6-v2"