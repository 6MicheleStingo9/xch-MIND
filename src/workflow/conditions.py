"""
LangGraph Routing Conditions - Controls workflow transitions.

Defines conditional functions for routing between nodes.
"""

import logging
from typing import Literal

from src.agents import AgentType

logger = logging.getLogger(__name__)


# =============================================================================
# ROUTING FROM ORCHESTRATOR
# =============================================================================


def route_from_orchestrator(
    state: dict,
) -> Literal["geo_analyzer", "temporal_analyzer", "type_analyzer", "path_generator", "end"]:
    """
    Route from orchestrator to the next worker agent.

    Based on the orchestrator's decision stored in state['next_agent'].
    """
    next_agent = state.get("next_agent", "end")

    # Map AgentType to node names
    routing_map = {
        AgentType.GEO_ANALYZER: "geo_analyzer",
        AgentType.GEO_ANALYZER.value: "geo_analyzer",
        "geo_analyzer": "geo_analyzer",
        "geospatial": "geo_analyzer",
        AgentType.TEMPORAL_ANALYZER: "temporal_analyzer",
        AgentType.TEMPORAL_ANALYZER.value: "temporal_analyzer",
        "temporal_analyzer": "temporal_analyzer",
        "chronological": "temporal_analyzer",
        AgentType.TYPE_ANALYZER: "type_analyzer",
        AgentType.TYPE_ANALYZER.value: "type_analyzer",
        "type_analyzer": "type_analyzer",
        "typological": "type_analyzer",
        AgentType.PATH_GENERATOR: "path_generator",
        AgentType.PATH_GENERATOR.value: "path_generator",
        "path_generator": "path_generator",
        "narrative": "path_generator",
        "end": "end",
        "END": "end",
    }

    result = routing_map.get(next_agent, "end")
    logger.info("Routing from orchestrator to: %s (based on: %s)", result, next_agent)

    return result


# =============================================================================
# CONTINUATION CONDITIONS
# =============================================================================


def should_continue(
    state: dict,
) -> Literal["orchestrator", "end"]:
    """
    Determine if the workflow should continue or end.

    Behavior depends on workflow mode:
    - "comprehensive": Run ALL analyzers + PathGenerator before ending
    - "autonomous": LLM-guided agent selection, stop when coverage + diversity thresholds are met

    Always ends when:
    - Max iterations reached
    """
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 20)
    assertions = state.get("assertions", [])
    messages = state.get("messages", [])
    workflow_mode = state.get("mode", "comprehensive")

    # Always check max iterations (safety limit)
    if iteration >= max_iterations:
        logger.info("Ending: max iterations reached (%d/%d)", iteration, max_iterations)
        return "end"

    # Get completed analysis types
    completed_types = _get_completed_analysis_types(messages)

    # Calculate coverage stats for logging
    entity_count = state.get("entity_count", 0)
    covered_count = _calculate_coverage(assertions) if entity_count > 0 else 0
    coverage_ratio = covered_count / entity_count if entity_count > 0 else 0

    # === COMPREHENSIVE MODE ===
    # Must run ALL 4 analyzer types before ending
    if workflow_mode == "comprehensive":
        if len(completed_types) >= 4:  # geo, temporal, type, path
            logger.info(
                "Ending (comprehensive): all 4 analysis types completed (%s)",
                ", ".join(sorted(completed_types)),
            )
            return "end"

        logger.info(
            "Continuing (comprehensive): %d/4 analyzers done (%s), %d assertions, coverage %.1f%%",
            len(completed_types),
            ", ".join(sorted(completed_types)) if completed_types else "none",
            len(assertions),
            coverage_ratio * 100,
        )
        return "orchestrator"

    # === AUTONOMOUS MODE ===
    # LLM-guided agent selection, stop when coverage + diversity thresholds are met
    min_coverage = state.get("min_entity_coverage", 0.8)
    min_diversity = state.get("min_diversity", 2)
    diversity_count = _count_assertion_types(assertions)

    # End if coverage AND diversity thresholds are met
    if coverage_ratio >= min_coverage and diversity_count >= min_diversity:
        logger.info(
            "Ending (autonomous): target coverage (%.1f%% >= %.1f%%) AND diversity (%d >= %d) reached",
            coverage_ratio * 100,
            min_coverage * 100,
            diversity_count,
            min_diversity,
        )
        return "end"

    # Also end if all analyzers completed (even in autonomous mode)
    if len(completed_types) >= 4:
        logger.info("Ending (autonomous): all analysis types completed")
        return "end"

    logger.info(
        "Continuing (autonomous): iteration %d, %d assertions, coverage %.1f%%, diversity %d/%d",
        iteration,
        len(assertions),
        coverage_ratio * 100,
        diversity_count,
        min_diversity,
    )
    return "orchestrator"


def should_generate_paths(
    state: dict,
) -> Literal["path_generator", "orchestrator"]:
    """
    Determine if we should generate paths or continue analysis.

    Generate paths if:
    1. We have enough clusters/relations to work with
    2. We've completed at least one analysis pass
    """
    assertions = state.get("assertions", [])

    # Count clusters and relations
    clusters = sum(1 for a in assertions if _is_cluster(a))
    relations = sum(1 for a in assertions if _is_relation(a))

    if clusters >= 2 or relations >= 3:
        logger.info(
            "Ready for path generation: %d clusters, %d relations",
            clusters,
            relations,
        )
        return "path_generator"

    return "orchestrator"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _calculate_coverage(assertions: list) -> int:
    """Calculate number of unique entities covered by assertions."""
    covered_uris = set()
    for a in assertions:
        # Handle dict or object access
        if isinstance(a, dict):
            subjects = a.get("subject_uris", [])
            objects = a.get("object_uris", [])
        else:
            subjects = getattr(a, "subject_uris", [])
            objects = getattr(a, "object_uris", [])

        covered_uris.update(subjects)
        covered_uris.update(objects)

    return len(covered_uris)


def _has_sufficient_diversity(assertions: list) -> bool:
    """Check if assertions have sufficient type diversity (legacy function)."""
    return _count_assertion_types(assertions) >= 2


def _count_assertion_types(assertions: list) -> int:
    """Count how many different assertion types are present."""
    types_found = set()

    for a in assertions:
        if isinstance(a, dict):
            assertion_id = a.get("assertion_id", "")
        else:
            assertion_id = getattr(a, "assertion_id", "")

        if "geo" in assertion_id or "near" in assertion_id:
            types_found.add("geo")
        elif "chrono" in assertion_id or "period" in assertion_id or "contemporary" in assertion_id:
            types_found.add("chrono")
        elif "type" in assertion_id or "similar" in assertion_id:
            types_found.add("type")
        elif "path" in assertion_id:
            types_found.add("path")

    return len(types_found)


def _get_completed_analysis_types(messages: list) -> set:
    """Extract which analysis types have been completed from messages."""
    completed = set()

    for msg in messages:
        if isinstance(msg, str):
            if "GeoAnalyzer" in msg:
                completed.add("geo")
            elif "TemporalAnalyzer" in msg:
                completed.add("temporal")
            elif "TypeAnalyzer" in msg:
                completed.add("type")
            elif "PathGenerator" in msg:
                completed.add("path")

    return completed


def _is_cluster(assertion) -> bool:
    """Check if assertion is a cluster."""
    if isinstance(assertion, dict):
        aid = assertion.get("assertion_id", "")
    else:
        aid = getattr(assertion, "assertion_id", "")
    return "cluster" in aid.lower()


def _is_relation(assertion) -> bool:
    """Check if assertion is a relation."""
    if isinstance(assertion, dict):
        aid = assertion.get("assertion_id", "")
    else:
        aid = getattr(assertion, "assertion_id", "")
    return any(kw in aid.lower() for kw in ["near", "similar", "contemporary"])
