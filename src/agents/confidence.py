"""
Confidence Calculator - Hybrid confidence scoring for LLM-validated assertions.

This module provides functions to calculate combined confidence scores
from LLM proposals (70%) and algorithmic validation (30%).

The hybrid approach ensures that:
1. High LLM confidence without validation = medium overall score
2. Low LLM confidence with high validation = medium overall score
3. High LLM confidence + high validation = high overall score
4. Low LLM confidence + low validation = low overall score
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class ConfidenceWeights:
    """Weights for hybrid confidence calculation."""

    llm_weight: float = 0.7
    validation_weight: float = 0.3

    def __post_init__(self):
        """Ensure weights sum to 1.0."""
        total = self.llm_weight + self.validation_weight
        if abs(total - 1.0) > 0.01:
            logger.warning("Confidence weights sum to %.2f, normalizing to 1.0", total)
            self.llm_weight /= total
            self.validation_weight /= total


# Default weights (70/30 as specified)
DEFAULT_WEIGHTS = ConfidenceWeights(llm_weight=0.7, validation_weight=0.3)


# =============================================================================
# CONFIDENCE CALCULATION
# =============================================================================


def calculate_hybrid_confidence(
    llm_confidence: float,
    validation_score: float,
    weights: ConfidenceWeights | None = None,
    apply_penalty: bool = True,
) -> float:
    """
    Calculate hybrid confidence from LLM and validation scores.

    Uses weighted average with optional penalty for large discrepancies.

    Args:
        llm_confidence: LLM's self-reported confidence (0.0 - 1.0)
        validation_score: Algorithmic validation score (0.0 - 1.0)
        weights: Custom weights (defaults to 70/30)
        apply_penalty: Apply penalty for large LLM-validation discrepancy

    Returns:
        Combined confidence score (0.0 - 1.0)
    """
    weights = weights or DEFAULT_WEIGHTS

    # Clamp inputs
    llm_confidence = max(0.0, min(1.0, llm_confidence))
    validation_score = max(0.0, min(1.0, validation_score))

    # Weighted average
    base_score = weights.llm_weight * llm_confidence + weights.validation_weight * validation_score

    if not apply_penalty:
        return base_score

    # Apply penalty for large discrepancies
    # If LLM is very confident but validation fails (or vice versa),
    # reduce the score to flag uncertainty
    discrepancy = abs(llm_confidence - validation_score)
    if discrepancy > 0.4:
        # Significant disagreement - reduce score proportionally
        penalty = (discrepancy - 0.4) * 0.3  # Max penalty of ~0.18
        base_score = max(0.0, base_score - penalty)
        logger.debug(
            "Applied discrepancy penalty: LLM=%.2f, Val=%.2f, Penalty=%.2f",
            llm_confidence,
            validation_score,
            penalty,
        )

    return base_score


def calculate_cluster_confidence(
    llm_confidence: float,
    validation_score: float,
    member_count: int,
    min_members: int = 2,
    bonus_threshold: int = 5,
    weights: ConfidenceWeights | None = None,
) -> float:
    """
    Calculate confidence for a cluster assertion.

    Applies bonus for larger clusters (more evidence).

    Args:
        llm_confidence: LLM's confidence
        validation_score: Algorithmic validation score
        member_count: Number of members in cluster
        min_members: Minimum members (below this, reduce score)
        bonus_threshold: Above this, apply bonus
        weights: Custom weights

    Returns:
        Adjusted confidence score
    """
    base_score = calculate_hybrid_confidence(llm_confidence, validation_score, weights)

    if member_count < min_members:
        # Penalty for tiny clusters
        penalty = 0.2 * (min_members - member_count) / min_members
        return max(0.0, base_score - penalty)

    if member_count >= bonus_threshold:
        # Bonus for larger clusters (more evidence)
        bonus = 0.1 * min(1.0, (member_count - bonus_threshold) / 5)
        return min(1.0, base_score + bonus)

    return base_score


def calculate_relation_confidence(
    llm_confidence: float,
    validation_score: float,
    relation_strength: float | None = None,
    weights: ConfidenceWeights | None = None,
) -> float:
    """
    Calculate confidence for a relation assertion.

    Optionally factors in relation strength (e.g., proximity, similarity).

    Args:
        llm_confidence: LLM's confidence
        validation_score: Algorithmic validation score
        relation_strength: Optional strength metric (0-1)
        weights: Custom weights

    Returns:
        Adjusted confidence score
    """
    base_score = calculate_hybrid_confidence(llm_confidence, validation_score, weights)

    if relation_strength is not None:
        # Blend with relation strength
        relation_strength = max(0.0, min(1.0, relation_strength))
        # Give relation strength 20% influence
        adjusted = 0.8 * base_score + 0.2 * relation_strength
        return adjusted

    return base_score


def calculate_path_confidence(
    llm_confidence: float,
    validation_score: float,
    stop_count: int,
    narrative_quality_score: float | None = None,
    weights: ConfidenceWeights | None = None,
) -> float:
    """
    Calculate confidence for a thematic path.

    Considers path coherence and narrative quality.

    Args:
        llm_confidence: LLM's confidence in path design
        validation_score: Geographic feasibility score
        stop_count: Number of stops in path
        narrative_quality_score: Optional narrative quality (0-1)
        weights: Custom weights

    Returns:
        Adjusted confidence score
    """
    base_score = calculate_hybrid_confidence(llm_confidence, validation_score, weights)

    # Paths with more stops need higher coherence
    if stop_count >= 5:
        # Slightly reduce confidence for longer paths (harder to maintain quality)
        complexity_factor = 1.0 - 0.05 * (stop_count - 5)
        base_score *= max(0.8, complexity_factor)

    if narrative_quality_score is not None:
        # Blend with narrative quality
        narrative_quality_score = max(0.0, min(1.0, narrative_quality_score))
        # Give narrative 15% influence
        adjusted = 0.85 * base_score + 0.15 * narrative_quality_score
        return adjusted

    return base_score


def calculate_feature_confidence(
    entity_count: int,
    total_entities: int,
    category: str,
) -> float:
    """
    Calculate confidence for an extracted feature assertion.

    Since the LLM does not provide per-feature confidence, this uses
    algorithmic signals: how many entities share the feature (prevalence)
    and how objectively verifiable the feature category is.

    Args:
        entity_count: Number of entities that share this feature
        total_entities: Total entities analysed (for prevalence ratio)
        category: Feature category (architectural, functional, contextual, material)

    Returns:
        Confidence score (0.30 - 0.95)
    """
    # Base score: no LLM confidence available, use a neutral baseline
    base_score = 0.70

    # Prevalence bonus: features shared by more entities are more reliable
    if total_entities > 0:
        prevalence = entity_count / total_entities
        prevalence_bonus = min(0.15, prevalence * 0.30)  # Max +0.15
        base_score += prevalence_bonus

    # Category weight: architectural/material features are more objective
    category_bonuses = {
        "architectural": 0.05,
        "material": 0.05,
        "functional": 0.0,
        "contextual": 0.0,
    }
    base_score += category_bonuses.get(category, 0.0)

    # Clamp to realistic range
    return max(0.30, min(0.95, round(base_score, 2)))


# =============================================================================
# CONFIDENCE THRESHOLDS
# =============================================================================


@dataclass
class ConfidenceThresholds:
    """Thresholds for filtering assertions by confidence."""

    high: float = 0.8  # High confidence - include in primary output
    medium: float = 0.5  # Medium confidence - include with warning
    low: float = 0.3  # Low confidence - flag for review

    def classify(self, score: float) -> str:
        """Classify a confidence score."""
        if score >= self.high:
            return "high"
        elif score >= self.medium:
            return "medium"
        elif score >= self.low:
            return "low"
        else:
            return "very_low"


DEFAULT_THRESHOLDS = ConfidenceThresholds()


def filter_by_confidence(
    assertions: list[Any],
    min_confidence: float = 0.3,
    confidence_attr: str = "confidence_score",
) -> list[Any]:
    """
    Filter assertions by minimum confidence threshold.

    Args:
        assertions: List of assertion objects
        min_confidence: Minimum confidence to include
        confidence_attr: Attribute name for confidence score

    Returns:
        Filtered list of assertions
    """
    filtered = []
    for assertion in assertions:
        score = getattr(assertion, confidence_attr, 0.0)
        if isinstance(assertion, dict):
            score = assertion.get(confidence_attr, 0.0)
        if score >= min_confidence:
            filtered.append(assertion)
        else:
            logger.debug(
                "Filtered out assertion with confidence %.2f < %.2f",
                score,
                min_confidence,
            )
    return filtered


def sort_by_confidence(
    assertions: list[Any],
    confidence_attr: str = "confidence_score",
    reverse: bool = True,
) -> list[Any]:
    """
    Sort assertions by confidence score.

    Args:
        assertions: List of assertion objects
        confidence_attr: Attribute name for confidence score
        reverse: If True, highest confidence first

    Returns:
        Sorted list of assertions
    """

    def get_score(a):
        if isinstance(a, dict):
            return a.get(confidence_attr, 0.0)
        return getattr(a, confidence_attr, 0.0)

    return sorted(assertions, key=get_score, reverse=reverse)


# =============================================================================
# CONFIDENCE REPORT
# =============================================================================


def generate_confidence_report(
    assertions: list[Any],
    confidence_attr: str = "confidence_score",
) -> dict[str, Any]:
    """
    Generate a summary report of confidence scores.

    Args:
        assertions: List of assertion objects
        confidence_attr: Attribute name for confidence score

    Returns:
        Report dictionary with statistics
    """

    def get_score(a):
        if isinstance(a, dict):
            return a.get(confidence_attr, 0.0)
        return getattr(a, confidence_attr, 0.0)

    if not assertions:
        return {
            "count": 0,
            "avg_confidence": 0.0,
            "min_confidence": 0.0,
            "max_confidence": 0.0,
            "distribution": {"high": 0, "medium": 0, "low": 0, "very_low": 0},
        }

    scores = [get_score(a) for a in assertions]
    thresholds = DEFAULT_THRESHOLDS

    distribution = {"high": 0, "medium": 0, "low": 0, "very_low": 0}
    for score in scores:
        category = thresholds.classify(score)
        distribution[category] += 1

    return {
        "count": len(assertions),
        "avg_confidence": sum(scores) / len(scores),
        "min_confidence": min(scores),
        "max_confidence": max(scores),
        "distribution": distribution,
    }
