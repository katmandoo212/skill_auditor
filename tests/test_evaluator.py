# tests/test_evaluator.py
"""Tests for LLM evaluation."""
import pytest
from pathlib import Path
from skill_auditor.models import Skill, SimilarityCandidate
from skill_auditor.evaluator import format_candidates, parse_json_response


def test_format_candidates_empty():
    """Test format_candidates with empty list."""
    result = format_candidates([])
    assert result == "No candidates found."


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