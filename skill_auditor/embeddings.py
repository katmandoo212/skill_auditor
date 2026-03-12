# skill_auditor/embeddings.py
"""Embedding generation and similarity computation."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from skill_auditor.models import Skill, SimilarityCandidate
from skill_auditor.config import CONTENT_TRUNCATE_LENGTH, EMBEDDING_MODEL

_model = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
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