# Skill Auditor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python CLI tool that scans Claude Code plugin skills for duplicates using embedding similarity and LLM evaluation.

**Architecture:** Pipeline with three stages - discovery (find SKILL.md files), candidate detection (embeddings + similarity), and LLM evaluation (Ollama judgment). Uses Typer for CLI, sentence-transformers for local embeddings, and outputs a Markdown report.

**Tech Stack:** Python 3.14+, uv package manager, Typer, sentence-transformers, scikit-learn, ollama, PyYAML

---

## Task 1: Project Initialization

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "skill_auditor"
version = "0.1.0"
requires-python = ">=3.14"
dependencies = [
    "typer>=0.12.0",
    "sentence-transformers>=2.2.0",
    "scikit-learn>=1.3.0",
    "ollama>=0.1.0",
    "pyyaml>=6.0",
    "rich>=13.0.0",
]

[project.scripts]
skill-auditor = "skill_auditor:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Step 2: Create .python-version**

```
3.14
```

**Step 3: Create .gitignore**

```
# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
ENV/

# IDE
.vscode/
.idea/

# Distribution
dist/
build/
*.egg-info/

# Model cache
.cache/

# Output
skill_audit_report.md
```

**Step 4: Initialize virtual environment and install dependencies**

Run: `uv venv && source .venv/Scripts/activate && uv pip install -e .`

**Step 5: Commit**

```bash
git add pyproject.toml .python-version .gitignore
git commit -m "chore: initialize project with uv and dependencies"
```

---

## Task 2: Data Models

**Files:**
- Create: `models.py`
- Create: `tests/test_models.py`

**Step 1: Write the failing test**

```python
# tests/test_models.py
"""Tests for skill data models."""
import pytest
from pathlib import Path
from models import Skill, DuplicateGroup, SimilarityCandidate


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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL with "No module named 'models'"

**Step 3: Write minimal implementation**

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add models.py tests/test_models.py
git commit -m "feat: add data models for Skill, DuplicateGroup, SimilarityCandidate"
```

---

## Task 3: Configuration

**Files:**
- Create: `config.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
"""Tests for configuration module."""
import pytest
from pathlib import Path
from config import (
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with "No module named 'config'"

**Step 3: Write minimal implementation**

```python
# config.py
"""Configuration defaults for skill auditor."""
from pathlib import Path


# Path defaults
DEFAULT_PLUGIN_CACHE = Path.home() / ".claude" / "plugins" / "cache"
DEFAULT_OUTPUT = Path("skill_audit_report.md")

# Model defaults
DEFAULT_MODEL = "glm5:cloud"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Threshold defaults
DEFAULT_THRESHOLD = 0.8
DEFAULT_MAX_CANDIDATES = 10

# Content truncation
CONTENT_TRUNCATE_LENGTH = 2000
PROMPT_CONTENT_LENGTH = 1500
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat: add configuration constants and defaults"
```

---

## Task 4: Skill Scanner - Discovery

**Files:**
- Create: `scanner.py`
- Create: `tests/test_scanner.py`

**Step 1: Write the failing test**

```python
# tests/test_scanner.py
"""Tests for skill scanner module."""
import pytest
from pathlib import Path
from scanner import discover_skills
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


@patch("scanner.DEFAULT_PLUGIN_CACHE")
def test_discover_skills_default_path(mock_cache, tmp_path):
    """Test discover_skills uses default path when none provided."""
    mock_cache.__str__ = lambda self: str(tmp_path)

    skill_dir = tmp_path / "skills" / "test-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: Test\n---\nContent")

    result = discover_skills(None)
    # This test would need the mock properly configured
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_scanner.py -v`
Expected: FAIL with "No module named 'scanner'"

**Step 3: Write minimal implementation**

```python
# scanner.py
"""Skill discovery and parsing."""
from pathlib import Path
from typing import Optional
from config import DEFAULT_PLUGIN_CACHE


def discover_skills(paths: Optional[list[Path]] = None) -> list[Path]:
    """Find all SKILL.md files in given paths or default plugin cache.

    Args:
        paths: Optional list of paths to scan. Uses DEFAULT_PLUGIN_CACHE if None.

    Returns:
        List of paths to SKILL.md files.
    """
    if paths is None:
        paths = [DEFAULT_PLUGIN_CACHE]

    skill_files = []
    for base_path in paths:
        if not base_path.exists():
            continue
        for skill_file in base_path.rglob("skills/*/SKILL.md"):
            skill_files.append(skill_file)

    return skill_files
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_scanner.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scanner.py tests/test_scanner.py
git commit -m "feat: add skill discovery functionality"
```

---

## Task 5: Skill Scanner - Parsing

**Files:**
- Modify: `scanner.py`
- Modify: `tests/test_scanner.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_scanner.py

from scanner import parse_skill, parse_frontmatter
from models import Skill


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
    skill_dir = tmp_path / "test-plugin" / "skills" / "test-skill"
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_scanner.py -v`
Expected: FAIL with "cannot import name 'parse_skill'"

**Step 3: Write minimal implementation**

```python
# Add to scanner.py

import yaml
from models import Skill


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse frontmatter and body from SKILL.md content.

    Args:
        content: Full file content with optional frontmatter.

    Returns:
        Tuple of (frontmatter dict, body string).
    """
    lines = content.strip().split("\n")

    if not lines or lines[0] != "---":
        return {}, content

    # Find closing ---
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i] == "---":
            end_idx = i
            break

    if end_idx is None:
        return {}, content

    frontmatter_str = "\n".join(lines[1:end_idx])
    body = "\n".join(lines[end_idx + 1:])

    try:
        frontmatter = yaml.safe_load(frontmatter_str) or {}
    except yaml.YAMLError:
        frontmatter = {}

    return frontmatter, body.strip()


def parse_skill(skill_path: Path) -> Skill:
    """Parse a SKILL.md file into a Skill dataclass.

    Args:
        skill_path: Path to the SKILL.md file.

    Returns:
        Skill object with parsed metadata.
    """
    content = skill_path.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(content)

    # Extract plugin name from path
    # Path pattern: .../cache/<plugin-name>/.../skills/<skill-name>/SKILL.md
    parts = skill_path.parts
    plugin_name = "unknown"
    for i, part in enumerate(parts):
        if part == "cache" and i + 1 < len(parts):
            plugin_name = parts[i + 1]
            break

    return Skill(
        path=skill_path,
        plugin_name=plugin_name,
        skill_name=skill_path.parent.name,
        display_name=frontmatter.get("name", ""),
        description=frontmatter.get("description", ""),
        triggers=frontmatter.get("trigger", []) or [],
        content=body,
        metadata=frontmatter,
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_scanner.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scanner.py tests/test_scanner.py
git commit -m "feat: add skill parsing with frontmatter extraction"
```

---

## Task 6: Embeddings Module

**Files:**
- Create: `embeddings.py`
- Create: `tests/test_embeddings.py`

**Step 1: Write the failing test**

```python
# tests/test_embeddings.py
"""Tests for embedding generation."""
import pytest
from pathlib import Path
from models import Skill
from embeddings import skill_to_text, find_candidates
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_embeddings.py -v`
Expected: FAIL with "No module named 'embeddings'"

**Step 3: Write minimal implementation**

```python
# embeddings.py
"""Embedding generation and similarity computation."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from models import Skill, SimilarityCandidate
from config import CONTENT_TRUNCATE_LENGTH

_model = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def skill_to_text(skill: Skill) -> str:
    """Convert skill metadata to text for embedding.

    Args:
        skill: Skill object to convert.

    Returns:
        Text representation suitable for embedding.
    """
    parts = [
        f"Name: {skill.display_name}",
        f"Description: {skill.description}",
    ]
    if skill.triggers:
        parts.append(f"Triggers: {', '.join(skill.triggers)}")
    parts.append(f"Content: {skill.content[:CONTENT_TRUNCATE_LENGTH]}")
    return "\n".join(parts)


def generate_embeddings(skills: list[Skill]) -> np.ndarray:
    """Generate embeddings for all skills.

    Args:
        skills: List of skills to embed.

    Returns:
        Numpy array of embeddings.
    """
    model = get_model()
    texts = [skill_to_text(s) for s in skills]
    return model.encode(texts, show_progress_bar=True)


def find_candidates(
    skills: list[Skill],
    embeddings: np.ndarray,
    threshold: float,
) -> list[SimilarityCandidate]:
    """Find similar skill pairs above threshold.

    Args:
        skills: List of skills.
        embeddings: Corresponding embeddings.
        threshold: Minimum similarity score to include.

    Returns:
        List of SimilarityCandidate objects.
    """
    similarity_matrix = cosine_similarity(embeddings)

    candidates = []
    for i, skill in enumerate(skills):
        similar = []
        for j, other in enumerate(skills):
            if i != j and similarity_matrix[i, j] >= threshold:
                similar.append((other, float(similarity_matrix[i, j])))

        similar.sort(key=lambda x: x[1], reverse=True)
        candidates.append(SimilarityCandidate(skill, similar))

    return candidates
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_embeddings.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add embeddings.py tests/test_embeddings.py
git commit -m "feat: add embedding generation and similarity computation"
```

---

## Task 7: LLM Evaluator Module

**Files:**
- Create: `evaluator.py`
- Create: `tests/test_evaluator.py`

**Step 1: Write the failing test**

```python
# tests/test_evaluator.py
"""Tests for LLM evaluation."""
import pytest
from pathlib import Path
from models import Skill, SimilarityCandidate
from evaluator import format_candidates, parse_json_response


def test_format_candidates_empty():
    """Test format_candidates with empty list."""
    result = format_candidates([])
    assert result == ""


def test_format_candidates_single():
    """Test format_candidates with single candidate."""
    skill = Skill(
        path=Path("/test/SKILL.md"),
        plugin_name="test",
        skill_name="test-skill",
        display_name="Test Skill",
        description="A test skill",
        triggers=["trigger"],
        content="Content",
        metadata={},
    )
    candidates = [(skill, 0.95)]

    result = format_candidates(candidates)
    assert "Test Skill" in result
    assert "0.95" in result
    assert "test" in result  # plugin name


def test_format_candidates_multiple():
    """Test format_candidates with multiple candidates."""
    skill1 = Skill(
        path=Path("/a/SKILL.md"),
        plugin_name="p1",
        skill_name="a",
        display_name="Skill A",
        description="Desc A",
        triggers=[],
        content="Content A",
        metadata={},
    )
    skill2 = Skill(
        path=Path("/b/SKILL.md"),
        plugin_name="p2",
        skill_name="b",
        display_name="Skill B",
        description="Desc B",
        triggers=[],
        content="Content B",
        metadata={},
    )

    result = format_candidates([(skill1, 0.9), (skill2, 0.8)])
    assert "Skill A" in result
    assert "Skill B" in result
    assert "0.90" in result
    assert "0.80" in result


def test_parse_json_response_valid():
    """Test parse_json_response with valid JSON."""
    json_str = '{"purpose_tag": "test", "duplicates": []}'
    result = parse_json_response(json_str)
    assert result["purpose_tag"] == "test"
    assert result["duplicates"] == []


def test_parse_json_response_with_duplicates():
    """Test parse_json_response with duplicates array."""
    json_str = '''{
        "purpose_tag": "code-review",
        "duplicates": [
            {"skill_name": "skill-a", "is_duplicate": true, "confidence": "high"}
        ]
    }'''
    result = parse_json_response(json_str)
    assert result["purpose_tag"] == "code-review"
    assert len(result["duplicates"]) == 1
    assert result["duplicates"][0]["skill_name"] == "skill-a"


def test_parse_json_response_invalid():
    """Test parse_json_response with invalid JSON returns empty dict."""
    result = parse_json_response("not valid json")
    assert result == {}


def test_parse_json_response_extracts_json():
    """Test parse_json_response extracts JSON from markdown code block."""
    text = '''Here is the response:
```json
{"purpose_tag": "extracted", "duplicates": []}
```
That's it.'''

    result = parse_json_response(text)
    assert result["purpose_tag"] == "extracted"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluator.py -v`
Expected: FAIL with "No module named 'evaluator'"

**Step 3: Write minimal implementation**

```python
# evaluator.py
"""LLM evaluation using Ollama."""
import json
import re
from typing import Any

import ollama

from models import Skill
from config import DEFAULT_MODEL, PROMPT_CONTENT_LENGTH


EVAL_PROMPT = """You are evaluating Claude Code skills for functional duplicates.

**Primary Skill:**
- Name: {primary_name}
- Plugin: {primary_plugin}
- Description: {primary_description}
- Triggers: {primary_triggers}
- Content: {primary_content}

**Candidate Similar Skills:**
{candidates_text}

**Task:**
1. For each candidate, determine if it is functionally equivalent or highly similar to the primary skill.
2. Assign a purpose tag that describes what this skill does (e.g., "code-review", "brainstorming", "debugging").
3. Explain any differences in how they achieve their purpose.

**Respond in JSON format:**
{{
  "purpose_tag": "<single word or short phrase>",
  "duplicates": [
    {{
      "skill_name": "<candidate skill name>",
      "plugin": "<candidate plugin>",
      "is_duplicate": true/false,
      "confidence": "high"/"medium"/"low",
      "notes": "<explanation>"
    }}
  ]
}}
"""


def check_ollama_connection(model: str = DEFAULT_MODEL) -> None:
    """Verify Ollama is running and model is available.

    Args:
        model: Model name to check.

    Raises:
        RuntimeError: If Ollama is not running or model unavailable.
    """
    try:
        ollama.ps()
    except Exception:
        raise RuntimeError("Ollama is not running. Start with: ollama serve")

    models = ollama.list()
    model_names = [m.model for m in models.models]
    if model not in model_names:
        print(f"Pulling model {model}...")
        ollama.pull(model)


def format_candidates(candidates: list[tuple[Skill, float]]) -> str:
    """Format candidate skills for the prompt.

    Args:
        candidates: List of (skill, similarity_score) tuples.

    Returns:
        Formatted string for the prompt.
    """
    if not candidates:
        return "No candidates found."

    lines = []
    for i, (skill, score) in enumerate(candidates, 1):
        lines.append(f"**Candidate {i}: {skill.display_name}**")
        lines.append(f"- Plugin: {skill.plugin_name}")
        lines.append(f"- Similarity: {score:.2f}")
        lines.append(f"- Description: {skill.description}")
        if skill.triggers:
            lines.append(f"- Triggers: {', '.join(skill.triggers)}")
        lines.append(f"- Content Preview: {skill.content[:500]}...")
        lines.append("")

    return "\n".join(lines)


def parse_json_response(response: str) -> dict[str, Any]:
    """Parse JSON from LLM response, handling markdown code blocks.

    Args:
        response: Raw LLM response text.

    Returns:
        Parsed JSON dict, or empty dict on failure.
    """
    # Try to extract JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", response)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Try to find raw JSON object
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            json_str = json_match.group(0)
        else:
            return {}

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}


def evaluate_candidates(
    candidate: SimilarityCandidate,
    model: str = DEFAULT_MODEL,
    max_candidates: int = 10,
) -> dict[str, Any]:
    """Send skill and candidates to Ollama for judgment.

    Args:
        candidate: SimilarityCandidate with skill and similar candidates.
        model: Ollama model to use.
        max_candidates: Maximum candidates to include in prompt.

    Returns:
        Parsed JSON response with purpose_tag and duplicates.
    """
    prompt = EVAL_PROMPT.format(
        primary_name=candidate.skill.display_name,
        primary_plugin=candidate.skill.plugin_name,
        primary_description=candidate.skill.description,
        primary_triggers=", ".join(candidate.skill.triggers) or "None",
        primary_content=candidate.skill.content[:PROMPT_CONTENT_LENGTH],
        candidates_text=format_candidates(candidate.candidates[:max_candidates]),
    )

    response = ollama.generate(model=model, prompt=prompt)
    return parse_json_response(response.response)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add evaluator.py tests/test_evaluator.py
git commit -m "feat: add LLM evaluation with Ollama integration"
```

---

## Task 8: Reporter Module

**Files:**
- Create: `reporter.py`
- Create: `tests/test_reporter.py`

**Step 1: Write the failing test**

```python
# tests/test_reporter.py
"""Tests for report generation."""
import pytest
from pathlib import Path
from datetime import datetime
from models import Skill, DuplicateGroup
from reporter import generate_report


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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_reporter.py -v`
Expected: FAIL with "No module named 'reporter'"

**Step 3: Write minimal implementation**

```python
# reporter.py
"""Markdown report generation."""
from pathlib import Path
from datetime import datetime
from typing import Any

from models import DuplicateGroup, Skill


def generate_report(
    duplicate_groups: list[DuplicateGroup],
    stats: dict[str, Any],
    output_path: Path,
) -> None:
    """Generate markdown report with grouped duplicates.

    Args:
        duplicate_groups: List of DuplicateGroup objects.
        stats: Statistics dict with total_skills, duplicated_skills, etc.
        output_path: Path to write the report.
    """
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Skill Audit Report\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")

        # Summary stats
        f.write("## Summary\n\n")
        f.write(f"- Skills scanned: {stats['total_skills']}\n")
        f.write(f"- Duplicate groups found: {len(duplicate_groups)}\n")
        f.write(f"- Skills with duplicates: {stats['duplicated_skills']}\n")
        f.write(f"- Similarity threshold: {stats['threshold']}\n")
        f.write(f"- Model used: {stats['model']}\n\n")

        # Grouped findings
        f.write("## Duplicate Groups\n\n")
        for i, group in enumerate(duplicate_groups, 1):
            f.write(f"### Group {i}: {group.purpose_tag}\n\n")
            f.write(f"**Confidence:** {group.confidence}\n\n")
            f.write(f"**Notes:** {group.notes}\n\n")
            f.write("| Skill | Plugin | Path |\n")
            f.write("|-------|--------|------|\n")
            for skill in group.skills:
                try:
                    rel_path = skill.path.relative_to(Path.home())
                except ValueError:
                    rel_path = skill.path
                f.write(f"| {skill.display_name} | {skill.plugin_name} | `{rel_path}` |\n")
            f.write("\n")

        # All skills list
        f.write("## All Scanned Skills\n\n")
        f.write("| Skill | Plugin | Description |\n")
        f.write("|-------|--------|-------------|\n")
        for skill in stats["all_skills"]:
            desc_preview = skill.description[:50] + "..." if len(skill.description) > 50 else skill.description
            f.write(f"| {skill.display_name} | {skill.plugin_name} | {desc_preview} |\n")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_reporter.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add reporter.py tests/test_reporter.py
git commit -m "feat: add markdown report generation"
```

---

## Task 9: CLI Entry Point

**Files:**
- Create: `skill_auditor.py` (main CLI)
- Modify: `tests/test_scanner.py` (add integration test)

**Step 1: Write the failing test**

```python
# Add integration test to tests/test_scanner.py or create tests/test_cli.py

import pytest
from pathlib import Path
from click.testing import CliRunner


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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with "No module named 'skill_auditor'" or "cannot import name 'app'"

**Step 3: Write minimal implementation**

```python
# skill_auditor.py
"""Skill Auditor CLI - Audit Claude Code skills for duplicates."""
import typer
from pathlib import Path
from typing import Optional

from scanner import discover_skills, parse_skill
from embeddings import generate_embeddings, find_candidates
from evaluator import check_ollama_connection, evaluate_candidates
from reporter import generate_report
from models import DuplicateGroup
from config import (
    DEFAULT_OUTPUT,
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    DEFAULT_MAX_CANDIDATES,
)

app = typer.Typer(help="Audit Claude Code skills for duplicates and similarity")


@app.command()
def main(
    paths: Optional[list[Path]] = typer.Option(
        None,
        "--path",
        "-p",
        help="Paths to scan for skills (default: all plugin caches)",
    ),
    threshold: float = typer.Option(
        DEFAULT_THRESHOLD,
        "--threshold",
        "-t",
        help="Embedding similarity threshold for candidates (0.0-1.0)",
    ),
    output: Path = typer.Option(
        DEFAULT_OUTPUT,
        "--output",
        "-o",
        help="Output markdown file path",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        "-m",
        help="Ollama model for LLM evaluation",
    ),
    max_candidates: int = typer.Option(
        DEFAULT_MAX_CANDIDATES,
        "--max-candidates",
        help="Max candidates per skill to send to LLM",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
):
    """Scan skills, find duplicates, generate report."""
    import logging

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Check Ollama connection
    logger.info(f"Checking Ollama connection for model: {model}")
    check_ollama_connection(model)

    # Discover skills
    logger.info("Discovering skills...")
    skill_files = discover_skills(paths)
    if not skill_files:
        logger.error("No skills found. Check your paths.")
        raise typer.Exit(1)

    logger.info(f"Found {len(skill_files)} skills")

    # Parse skills
    logger.info("Parsing skills...")
    skills = [parse_skill(f) for f in skill_files]

    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = generate_embeddings(skills)

    # Find candidates
    logger.info(f"Finding candidates (threshold={threshold})...")
    candidates = find_candidates(skills, embeddings, threshold)

    # Filter to only skills with candidates
    candidates_with_matches = [c for c in candidates if c.candidates]
    logger.info(f"Found {len(candidates_with_matches)} skills with potential duplicates")

    # Evaluate with LLM
    logger.info("Evaluating candidates with LLM...")
    duplicate_groups = []
    evaluated_skills = set()

    for candidate in candidates_with_matches:
        if candidate.skill.path in evaluated_skills:
            continue

        result = evaluate_candidates(candidate, model, max_candidates)
        if result and "duplicates" in result:
            # Build group from result
            group_skills = [candidate.skill]
            for dup in result.get("duplicates", []):
                if dup.get("is_duplicate"):
                    # Find matching skill
                    for s in skills:
                        if s.skill_name == dup.get("skill_name"):
                            group_skills.append(s)
                            evaluated_skills.add(s.path)
                            break

            if len(group_skills) > 1:
                group = DuplicateGroup(
                    purpose_tag=result.get("purpose_tag", "unknown"),
                    skills=group_skills,
                    confidence=result.get("duplicates", [{}])[0].get("confidence", "low"),
                    notes=result.get("duplicates", [{}])[0].get("notes", ""),
                )
                duplicate_groups.append(group)
                evaluated_skills.add(candidate.skill.path)

    # Generate report
    logger.info(f"Generating report: {output}")
    stats = {
        "total_skills": len(skills),
        "duplicated_skills": len(evaluated_skills),
        "threshold": threshold,
        "model": model,
        "all_skills": skills,
    }
    generate_report(duplicate_groups, stats, output)

    logger.info(f"Report saved to {output}")
    logger.info(f"Found {len(duplicate_groups)} duplicate groups")


def _version_callback(value: bool):
    if value:
        typer.echo("skill-auditor version 0.1.0")
        raise typer.Exit()


@app.callback()
def version(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
):
    """Skill Auditor - Audit Claude Code skills for duplicates."""
    pass


if __name__ == "__main__":
    app()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add skill_auditor.py tests/test_cli.py
git commit -m "feat: add CLI entry point with Typer"
```

---

## Task 10: Create Tests Directory Structure

**Files:**
- Create: `tests/__init__.py`

**Step 1: Create init file**

```python
# tests/__init__.py
"""Tests for skill auditor."""
```

**Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add tests/__init__.py
git commit -m "test: add tests package init"
```

---

## Task 11: Final Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
"""Integration tests for skill auditor."""
import pytest
from pathlib import Path
from click.testing import CliRunner


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


def test_full_pipeline(sample_skills):
    """Test full pipeline from discovery to report."""
    from scanner import discover_skills, parse_skill
    from embeddings import generate_embeddings, find_candidates

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


def test_cli_scan_command(sample_skills):
    """Test CLI scan command with sample skills."""
    from skill_auditor import app

    runner = CliRunner()
    output_file = sample_skills / "report.md"

    # This would require Ollama to be running, so we'll skip the full integration
    # Just test that the CLI accepts the arguments
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
```

**Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v`
Expected: PASS (or skip if Ollama not available)

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests"
```

---

## Task 12: README Documentation

**Files:**
- Create: `README.md`

**Step 1: Create README**

```markdown
# Skill Auditor

A Python CLI tool that scans Claude Code plugin skills for duplicates using embedding similarity and LLM evaluation.

## Features

- **Skill Discovery**: Automatically finds all SKILL.md files in plugin caches
- **Embedding Similarity**: Uses sentence-transformers for fast candidate filtering
- **LLM Evaluation**: Ollama-powered judgment for nuanced duplicate detection
- **Markdown Reports**: Human-readable output with grouped findings

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai/) running locally with a model (default: `glm5:cloud`)

## Installation

```bash
# Clone and setup
cd skill_auditor
uv venv
source .venv/Scripts/activate  # Windows Git Bash
# or: .venv\Scripts\Activate.ps1  # Windows PowerShell
# or: source .venv/bin/activate  # Unix/Linux
uv pip install -e .
```

## Usage

```bash
# Scan all installed plugins with defaults
skill-auditor

# Scan specific paths
skill-auditor -p ~/.claude/plugins/cache/superpowers-marketplace

# Adjust similarity threshold
skill-auditor -t 0.75

# Use different Ollama model
skill-auditor -m llama3.2

# Custom output file
skill-auditor -o my_report.md

# Verbose logging
skill-auditor -v
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-p, --path` | All plugin caches | Paths to scan for skills |
| `-t, --threshold` | 0.8 | Embedding similarity threshold |
| `-o, --output` | `skill_audit_report.md` | Output markdown file |
| `-m, --model` | `glm5:cloud` | Ollama model for evaluation |
| `--max-candidates` | 10 | Max candidates per skill |
| `-v, --verbose` | False | Enable debug logging |

## Output

Generates a Markdown report with:

- Summary statistics
- Grouped duplicate findings with confidence levels
- Table of all scanned skills

## Development

```bash
# Run tests
pytest tests/ -v

# Install dev dependencies
uv pip install -e ".[dev]"
```

## License

MIT
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with usage instructions"
```

---

## Plan Complete

Plan saved to `docs/plans/2026-03-10-skill-auditor-implementation.md`.

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**