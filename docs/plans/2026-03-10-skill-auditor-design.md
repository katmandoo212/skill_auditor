# Skill Auditor Design Document

**Created:** 2026-03-10

## Overview

A Python CLI tool that scans all installed Claude Code plugin skills, identifies functionally similar or duplicate skills using a hybrid embedding + LLM approach, and generates a Markdown report.

## Architecture

**Pipeline Pattern with Three Stages:**
1. **Discovery** - Find and parse all skill files from configurable paths
2. **Candidate Detection** - Generate embeddings, compute similarity matrix, filter candidates above threshold
3. **LLM Evaluation** - For each skill, send all its candidates to Ollama for one-vs-many judgment with purpose tagging

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Python version | 3.14+ | Modern features, user requirement |
| Package manager | uv | Fast dependency resolution, modern Python packaging |
| Virtual environment | `.venv` via uv | Isolated dependencies, managed by `uv venv` |
| Skill scope | All installed plugins | Comprehensive audit |
| Output format | Markdown report | Human-readable, version controllable |
| Similarity approach | Hybrid: embeddings + LLM | Fast candidate filtering, nuanced judgment |
| Evaluation method | One-vs-many with tagging | Efficient, provides readability via tags |
| Data included | Full skill metadata | Rich context for LLM judgment |
| Embeddings | Local sentence-transformers | Offline, no API dependency |
| Skill discovery | Configurable paths | Flexibility for edge cases |
| Threshold | CLI flag with 0.8 default | Tunable for precision/recall |
| CLI framework | Typer | Modern, type hints, auto-help |
| Ollama failures | Fail fast | Essential dependency, clean user feedback |

## Project Setup with uv

**Package Manager:** uv - Fast, modern Python package manager

**Virtual Environment:**
```bash
# Create virtual environment
uv venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate (Windows Git Bash)
source .venv/Scripts/activate

# Activate (Unix/Linux)
source .venv/bin/activate
```

**Dependency Management:**
```bash
# Install dependencies
uv pip install -e .

# Add new dependency
uv pip install <package>

# Sync dependencies from pyproject.toml
uv pip sync
```

## File Structure

```
skill_auditor/
├── .venv/                # Virtual environment (created by uv venv)
├── skill_auditor.py      # Main CLI entry point
├── scanner.py            # Skill discovery and parsing
├── embeddings.py         # Embedding generation and similarity
├── evaluator.py          # LLM evaluation logic
├── reporter.py           # Markdown report generation
├── models.py             # Pydantic models for skill data
├── config.py             # Default paths and settings
├── pyproject.toml        # Dependencies and project config
└── .python-version       # Python version file (3.14)
```

## Data Models

```python
@dataclass
class Skill:
    path: Path                    # Full path to SKILL.md
    plugin_name: str              # e.g., "superpowers", "compound-engineering"
    skill_name: str               # Directory name, e.g., "brainstorming"
    display_name: str             # From frontmatter 'name' field
    description: str              # From frontmatter 'description' field
    triggers: list[str]           # From frontmatter 'trigger' field (if present)
    content: str                  # Full SKILL.md content after frontmatter
    metadata: dict                # Raw frontmatter as parsed dict

@dataclass
class DuplicateGroup:
    purpose_tag: str              # LLM-assigned purpose category
    skills: list[Skill]           # All skills in this group
    confidence: str               # "high", "medium", or "low"
    notes: str                    # LLM's explanation of similarity

@dataclass
class SimilarityCandidate:
    skill: Skill                   # The skill being compared
    candidates: list[tuple[Skill, float]]  # (similar_skill, similarity_score)
```

## CLI Interface

```python
@app.command()
def main(
    paths: list[Path] = typer.Option(None, "--path", "-p"),
    threshold: float = typer.Option(0.8, "--threshold", "-t"),
    output: Path = typer.Option(Path("skill_audit_report.md"), "--output", "-o"),
    model: str = typer.Option("glm5:cloud", "--model", "-m"),
    max_candidates: int = typer.Option(50, "--max-candidates"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
```

**Example Usage:**
```bash
# Run with uv (recommended - uses project's virtual environment)
uv run skill-auditor

# Or activate venv first, then run directly
uv venv && source .venv/bin/activate
skill-auditor

# Scan specific paths with custom threshold
uv run skill-auditor -p ~/.claude/plugins/cache/superpowers-marketplace -t 0.75

# Use different model and output
uv run skill-auditor -m llama3.2 -o my_report.md
```

## Discovery and Parsing

```python
def discover_skills(paths: list[Path] | None = None) -> list[Path]:
    """Find all SKILL.md files in given paths or default plugin cache."""
    if not paths:
        plugin_cache = Path.home() / ".claude" / "plugins" / "cache"
        paths = [plugin_cache]

    skill_files = []
    for base_path in paths:
        for skill_file in base_path.rglob("skills/*/SKILL.md"):
            skill_files.append(skill_file)

    return skill_files

def parse_skill(skill_path: Path) -> Skill:
    """Parse a SKILL.md file into a Skill dataclass."""
    content = skill_path.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(content)

    # Extract plugin name from path
    parts = skill_path.parts
    cache_index = parts.index("cache")
    plugin_name = parts[cache_index + 1]

    return Skill(
        path=skill_path,
        plugin_name=plugin_name,
        skill_name=skill_path.parent.name,
        display_name=frontmatter.get("name", ""),
        description=frontmatter.get("description", ""),
        triggers=frontmatter.get("trigger", []),
        content=body.strip(),
        metadata=frontmatter,
    )
```

## Embedding Generation

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_model = None

def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def skill_to_text(skill: Skill) -> str:
    """Convert skill metadata to text for embedding."""
    parts = [
        f"Name: {skill.display_name}",
        f"Description: {skill.description}",
    ]
    if skill.triggers:
        parts.append(f"Triggers: {', '.join(skill.triggers)}")
    parts.append(f"Content: {skill.content[:2000]}")
    return "\n".join(parts)

def find_candidates(skills: list[Skill], embeddings: np.ndarray, threshold: float) -> list[SimilarityCandidate]:
    """Find similar skill pairs above threshold."""
    similarity_matrix = cosine_similarity(embeddings)

    candidates = []
    for i, skill in enumerate(skills):
        similar = []
        for j, other in enumerate(skills):
            if i != j and similarity_matrix[i, j] >= threshold:
                similar.append((other, similarity_matrix[i, j]))
        similar.sort(key=lambda x: x[1], reverse=True)
        candidates.append(SimilarityCandidate(skill, similar))

    return candidates
```

## LLM Evaluation

```python
import ollama

def check_ollama_connection(model: str) -> None:
    """Verify Ollama is running and model is available."""
    try:
        ollama.ps()
    except Exception:
        raise RuntimeError("Ollama is not running. Start with: ollama serve")

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
2. Assign a purpose tag that describes what this skill does.
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

def evaluate_candidates(candidate: SimilarityCandidate, model: str, max_candidates: int = 10) -> dict:
    """Send skill and candidates to Ollama for judgment."""
    prompt = EVAL_PROMPT.format(
        primary_name=candidate.skill.display_name,
        primary_plugin=candidate.skill.plugin_name,
        primary_description=candidate.skill.description,
        primary_triggers=", ".join(candidate.skill.triggers),
        primary_content=candidate.skill.content[:1500],
        candidates_text=format_candidates(candidate.candidates[:max_candidates]),
    )

    response = ollama.generate(model=model, prompt=prompt)
    return parse_json_response(response.response)
```

## Report Generation

```python
def generate_report(duplicate_groups: list[DuplicateGroup], stats: dict, output_path: Path) -> None:
    """Generate markdown report with grouped duplicates."""
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Skill Audit Report\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")

        # Summary stats
        f.write("## Summary\n\n")
        f.write(f"- Skills scanned: {stats['total_skills']}\n")
        f.write(f"- Duplicate groups found: {len(duplicate_groups)}\n")
        f.write(f"- Skills with duplicates: {stats['duplicated_skills']}\n")

        # Grouped findings
        f.write("## Duplicate Groups\n\n")
        for i, group in enumerate(duplicate_groups, 1):
            f.write(f"### Group {i}: {group.purpose_tag}\n\n")
            f.write(f"**Confidence:** {group.confidence}\n")
            f.write(f"**Notes:** {group.notes}\n\n")
            f.write("| Skill | Plugin | Path |\n|-------|--------|------|\n")
            for skill in group.skills:
                rel_path = skill.path.relative_to(Path.home())
                f.write(f"| {skill.display_name} | {skill.plugin_name} | `{rel_path}` |\n")
            f.write("\n")

        # All skills list
        f.write("## All Scanned Skills\n\n")
        f.write("| Skill | Plugin | Description |\n|-------|--------|-------------|\n")
        for skill in stats['all_skills']:
            f.write(f"| {skill.display_name} | {skill.plugin_name} | {skill.description[:50]}... |\n")
```

## Dependencies

**pyproject.toml:**
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
```

**.python-version:**
```
3.14
```

**Setup Commands:**
```bash
# Initialize project with uv
uv venv
source .venv/Scripts/activate  # Windows Git Bash
# or: .venv\Scripts\Activate.ps1  # Windows PowerShell
uv pip install -e .
```

## Configuration Defaults

```python
DEFAULT_PLUGIN_CACHE = Path.home() / ".claude" / "plugins" / "cache"
DEFAULT_OUTPUT = Path("skill_audit_report.md")
DEFAULT_MODEL = "glm5:cloud"
DEFAULT_THRESHOLD = 0.8
DEFAULT_MAX_CANDIDATES = 10
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CONTENT_TRUNCATE_LENGTH = 2000
PROMPT_CONTENT_LENGTH = 1500
```