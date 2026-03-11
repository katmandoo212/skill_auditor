"""Skill Auditor CLI - Audit Claude Code skills for duplicates."""
import typer
from pathlib import Path
from typing import Optional
import logging

from skill_auditor.scanner import discover_skills, parse_skill
from skill_auditor.embeddings import generate_embeddings, find_candidates
from skill_auditor.evaluator import check_ollama_connection, evaluate_candidates
from skill_auditor.reporter import generate_report
from skill_auditor.models import DuplicateGroup
from skill_auditor.config import (
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