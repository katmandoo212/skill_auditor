# skill_auditor/evaluator.py
"""LLM evaluation using Ollama."""
import json
import re
from typing import Any

import ollama

from skill_auditor.models import Skill, SimilarityCandidate
from skill_auditor.config import DEFAULT_MODEL, PROMPT_CONTENT_LENGTH


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