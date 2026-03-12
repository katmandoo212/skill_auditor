# config.py
"""Configuration defaults for skill auditor."""
from pathlib import Path


# Path defaults
DEFAULT_SKILL_PATHS = [
    Path.home() / ".claude" / "plugins" / "marketplaces",
    Path.home() / ".claude" / "skills",
]
DEFAULT_OUTPUT = Path("skill_audit_report.md")

# Model defaults
DEFAULT_MODEL = "glm-5:cloud"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Threshold defaults
DEFAULT_THRESHOLD = 0.8
DEFAULT_MAX_CANDIDATES = 10

# Content truncation
CONTENT_TRUNCATE_LENGTH = 2000
PROMPT_CONTENT_LENGTH = 1500