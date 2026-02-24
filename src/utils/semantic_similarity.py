"""
Semantic Similarity Module - SpaCy-based text similarity.

This module provides functions to calculate semantic similarity between
archaeological site descriptions using SpaCy word embeddings.

The cosine similarity between document vectors provides a more robust
measure of textual similarity compared to keyword-based Jaccard similarity.
"""

import logging
from functools import lru_cache
from typing import Any

import spacy
from spacy.language import Language

logger = logging.getLogger(__name__)

# Global SpaCy model instance (lazy loaded)
_nlp: Language | None = None


def get_spacy_model() -> Language:
    """
    Get the SpaCy model instance (lazy loading with caching).

    Returns:
        SpaCy Language model with Italian word vectors
    """
    global _nlp
    if _nlp is None:
        logger.info("Loading SpaCy model 'it_core_news_lg'...")
        _nlp = spacy.load("it_core_news_lg")
        logger.info("SpaCy model loaded (vectors: %s)", _nlp.vocab.vectors.shape)
    return _nlp


def compute_text_vector(text: str) -> Any:
    """
    Compute the document vector for a text using SpaCy.

    Args:
        text: Input text (description, historical info, etc.)

    Returns:
        NumPy array representing the document vector (mean of word vectors)
    """
    nlp = get_spacy_model()
    doc = nlp(text)
    return doc.vector


def cosine_similarity_texts(text1: str, text2: str) -> float:
    """
    Calculate cosine similarity between two texts using SpaCy embeddings.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    if not text1 or not text2:
        return 0.0

    nlp = get_spacy_model()
    doc1 = nlp(text1)
    doc2 = nlp(text2)

    # SpaCy's similarity() method uses cosine similarity on document vectors
    similarity = doc1.similarity(doc2)

    # Ensure result is in [0, 1] range (can be negative for very dissimilar texts)
    return max(0.0, min(1.0, similarity))


def cosine_similarity_vectors(vec1: Any, vec2: Any) -> float:
    """
    Calculate cosine similarity between two pre-computed vectors.

    Args:
        vec1: First vector (NumPy array)
        vec2: Second vector (NumPy array)

    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    import numpy as np

    # Handle zero vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    return max(0.0, min(1.0, float(similarity)))


def build_description_vectors(entities: list) -> dict[str, Any]:
    """
    Pre-compute document vectors for all entity descriptions.

    This is more efficient than computing vectors on-demand for each pair.

    Args:
        entities: List of DolmenEntity objects

    Returns:
        Dict mapping URI to document vector
    """
    vectors = {}
    nlp = get_spacy_model()

    for entity in entities:
        # Combine all textual information
        text_parts = []
        if entity.description:
            text_parts.append(entity.description)
        if entity.historical_info:
            text_parts.append(entity.historical_info)
        if entity.category:
            text_parts.append(entity.category)

        combined_text = " ".join(text_parts)

        if combined_text.strip():
            doc = nlp(combined_text)
            vectors[entity.uri] = doc.vector
        else:
            vectors[entity.uri] = None

    logger.debug("Built description vectors for %d entities", len(vectors))
    return vectors


def validate_similarity_with_embeddings(
    source_uri: str,
    target_uri: str,
    description_vectors: dict[str, Any],
    min_similarity: float = 0.5,
    llm_confidence: float = 0.0,
    llm_weight: float = 0.7,
) -> tuple[bool, float, dict]:
    """
    Validate a similarity pair using cosine similarity on embeddings.

    Uses a hybrid approach combining:
    - LLM confidence (subjective, semantic understanding)
    - Cosine similarity (objective, vector-based)

    Args:
        source_uri: URI of source entity
        target_uri: URI of target entity
        description_vectors: Pre-computed vectors for all entities
        min_similarity: Minimum hybrid similarity to accept (default 0.5)
        llm_confidence: LLM's confidence in the similarity (0.0-1.0)
        llm_weight: Weight for LLM confidence in hybrid score (default 0.7)

    Returns:
        Tuple of (is_valid, hybrid_score, details_dict)
    """
    source_vec = description_vectors.get(source_uri)
    target_vec = description_vectors.get(target_uri)

    details = {
        "source_uri": source_uri,
        "target_uri": target_uri,
        "llm_confidence": llm_confidence,
    }

    # Check if vectors exist
    if source_vec is None:
        return False, 0.0, {**details, "reason": f"No text vector for source: {source_uri}"}
    if target_vec is None:
        return False, 0.0, {**details, "reason": f"No text vector for target: {target_uri}"}

    # Calculate cosine similarity
    cosine_sim = cosine_similarity_vectors(source_vec, target_vec)
    details["cosine_similarity"] = round(cosine_sim, 4)

    # Calculate hybrid score
    # LLM captures semantic/domain understanding, cosine captures textual similarity
    algorithmic_weight = 1.0 - llm_weight
    hybrid_score = (llm_confidence * llm_weight) + (cosine_sim * algorithmic_weight)
    details["hybrid_score"] = round(hybrid_score, 4)
    details["llm_weight"] = llm_weight
    details["algorithmic_weight"] = algorithmic_weight

    # Validation decision
    is_valid = hybrid_score >= min_similarity

    if not is_valid:
        details["reason"] = f"Hybrid score {hybrid_score:.3f} below threshold {min_similarity}"

    return is_valid, hybrid_score, details


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    "get_spacy_model",
    "compute_text_vector",
    "cosine_similarity_texts",
    "cosine_similarity_vectors",
    "build_description_vectors",
    "validate_similarity_with_embeddings",
]
