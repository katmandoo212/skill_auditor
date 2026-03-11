# skill_auditor/reporter.py
"""Markdown report generation."""
from pathlib import Path
from datetime import datetime
from typing import Any

from skill_auditor.models import DuplicateGroup, Skill


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