# Skill Auditor

Audit Claude Code skills for duplicates and similarity.

## Features

- **Skill Discovery**: Automatically finds all `SKILL.md` files in Claude Code plugin cache directories
- **Metadata Parsing**: Extracts frontmatter metadata (name, description, triggers) from skill files
- **Embedding Generation**: Uses sentence-transformers to create semantic embeddings of skill content
- **Similarity Detection**: Finds potentially duplicate skills using cosine similarity
- **LLM Evaluation**: Uses Ollama to evaluate candidate duplicates for functional equivalence
- **Markdown Reports**: Generates human-readable reports grouped by purpose

## Requirements

- Python 3.14+
- [Ollama](https://ollama.ai/) running locally with a model available

## Installation

```bash
# Clone the repository
git clone https://github.com/katmandoo212/skill_auditor.git
cd skill_auditor

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

## Usage

```bash
# Scan default plugin cache
skill-auditor

# Scan custom paths
skill-auditor -p /path/to/plugins -p /another/path

# Specify output file
skill-auditor -o my_report.md

# Adjust similarity threshold (default: 0.8)
skill-auditor -t 0.75

# Use a different Ollama model
skill-auditor -m llama3.2

# Verbose output
skill-auditor -v
```

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--path` | `-p` | `~/.claude/plugins/cache` | Paths to scan for skills (can be specified multiple times) |
| `--output` | `-o` | `skill_audit_report.md` | Output markdown file path |
| `--threshold` | `-t` | `0.8` | Embedding similarity threshold (0.0-1.0) |
| `--model` | `-m` | `glm-5:cloud` | Ollama model for LLM evaluation |
| `--max-candidates` | | `10` | Maximum candidates per skill to send to LLM |
| `--verbose` | `-v` | `False` | Enable verbose logging |
| `--version` | | | Show version and exit |

## Output

The tool generates a markdown report with:

### Summary

- Total skills scanned
- Number of duplicate groups found
- Skills with potential duplicates
- Similarity threshold and model used

### Duplicate Groups

Each group contains:

- Purpose tag (e.g., "code-review", "brainstorming")
- Confidence level (high/medium/low)
- Notes explaining the similarity
- Table of duplicate skills with plugin and path

### All Scanned Skills

A complete table of all discovered skills with their plugins and descriptions.

## How It Works

1. **Discovery**: Scans directories for `skills/*/SKILL.md` pattern
2. **Parsing**: Extracts YAML frontmatter and content from each skill file
3. **Embedding**: Generates semantic embeddings using `all-MiniLM-L6-v2`
4. **Similarity**: Computes cosine similarity between all skill pairs
5. **Candidate Selection**: Skills with similarity >= threshold become candidates
6. **LLM Evaluation**: Sends skill + candidates to Ollama for functional analysis
7. **Report Generation**: Creates markdown report grouped by purpose

## Development

```bash
# Install dev dependencies
uv sync

# Run tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_integration.py
```

### Project Structure

```
skill_auditor/
├── __init__.py      # CLI entry point
├── config.py       # Configuration defaults
├── models.py       # Data models (Skill, DuplicateGroup, SimilarityCandidate)
├── scanner.py      # Skill discovery and parsing
├── embeddings.py   # Embedding generation and similarity
├── evaluator.py    # LLM evaluation via Ollama
└── reporter.py     # Markdown report generation

tests/
├── test_cli.py
├── test_config.py
├── test_embeddings.py
├── test_evaluator.py
├── test_integration.py
├── test_models.py
├── test_reporter.py
└── test_scanner.py
```

## License

MIT License