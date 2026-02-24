"""
LangGraph Workflow Graph - Main workflow definition.

Defines the StateGraph that orchestrates all agents.
"""

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.loaders import DolmenEntity
from src.utils.rate_limiter import RateLimiter

from .conditions import route_from_orchestrator, should_continue
from .nodes import (
    NodeContext,
    geo_analyzer_node,
    orchestrator_node,
    path_generator_node,
    temporal_analyzer_node,
    type_analyzer_node,
)

logger = logging.getLogger(__name__)


# =============================================================================
# STATE DEFINITION
# =============================================================================


class WorkflowState(TypedDict, total=False):
    """
    State schema for the LangGraph workflow.

    Uses TypedDict for LangGraph compatibility while maintaining
    compatibility with our Pydantic AgentState.
    """

    # Iteration control
    iteration: int
    max_iterations: int

    # Entity tracking
    entity_count: int
    entities_processed: list[str]
    filtered_entities: dict[str, list[str]]  # reason -> [entity_names]

    # Assertions (stored as dicts for serialization)
    assertions: list[dict[str, Any]]

    # Workflow control
    next_agent: str
    current_task: dict[str, Any] | None

    # Logging
    messages: list[str]

    # Termination
    min_assertions: int

    # Run identification
    run_id: str

    # Analysis flags
    geo_analysis_done: bool
    temporal_analysis_done: bool
    type_analysis_done: bool
    path_generation_done: bool

    # Novelty filtering statistics (populated by worker nodes)
    novelty_stats: dict[str, dict[str, int]]


# =============================================================================
# GRAPH BUILDER
# =============================================================================


def build_workflow(
    config: dict[str, Any],
    entities: list[DolmenEntity],
    rate_limiter: RateLimiter | None = None,
    output_dir: str | None = None,
) -> StateGraph:
    """
    Build the LangGraph workflow for dolmen analysis.

    Args:
        config: Full configuration dictionary
        entities: List of DolmenEntity objects to analyze
        rate_limiter: Optional rate limiter for API throttling
        output_dir: Optional output directory for memory persistence

    Returns:
        Compiled LangGraph StateGraph
    """
    logger.info("Building LangGraph workflow for %d entities", len(entities))

    # Initialize shared context with memory support
    NodeContext.initialize(config, entities, rate_limiter, output_dir)

    # Create the graph
    graph = StateGraph(WorkflowState)

    # ===================
    # ADD NODES
    # ===================

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("geo_analyzer", geo_analyzer_node)
    graph.add_node("temporal_analyzer", temporal_analyzer_node)
    graph.add_node("type_analyzer", type_analyzer_node)
    graph.add_node("path_generator", path_generator_node)

    # ===================
    # ADD EDGES
    # ===================

    # Entry point: start with orchestrator
    graph.set_entry_point("orchestrator")

    # Orchestrator routes to workers based on decision
    graph.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "geo_analyzer": "geo_analyzer",
            "temporal_analyzer": "temporal_analyzer",
            "type_analyzer": "type_analyzer",
            "path_generator": "path_generator",
            "end": END,
        },
    )

    # Workers return to check if we should continue
    for worker in ["geo_analyzer", "temporal_analyzer", "type_analyzer"]:
        graph.add_conditional_edges(
            worker,
            should_continue,
            {
                "orchestrator": "orchestrator",
                "end": END,
            },
        )

    # Path generator also checks continuation
    graph.add_conditional_edges(
        "path_generator",
        should_continue,
        {
            "orchestrator": "orchestrator",
            "end": END,
        },
    )

    logger.info("Workflow graph built successfully")

    return graph


def create_initial_state(
    entities: list[DolmenEntity],
    max_iterations: int = 10,
    min_assertions: int = 5,
    run_id: str = "unknown_run",
) -> WorkflowState:
    """
    Create the initial state for workflow execution.

    Args:
        entities: List of entities to process
        max_iterations: Maximum workflow iterations
        min_assertions: Minimum assertions before termination
        run_id: External run identifier (e.g., 'xch_run_20260219_105059')

    Returns:
        Initial WorkflowState
    """
    return WorkflowState(
        iteration=0,
        max_iterations=max_iterations,
        entity_count=len(entities),
        entities_processed=[],
        assertions=[],
        next_agent="",
        current_task=None,
        messages=[f"Workflow initialized with {len(entities)} entities"],
        min_assertions=min_assertions,
        run_id=run_id,
        geo_analysis_done=False,
        temporal_analysis_done=False,
        type_analysis_done=False,
        path_generation_done=False,
    )


# =============================================================================
# WORKFLOW RUNNER
# =============================================================================


class DolmenWorkflow:
    """
    High-level interface for running the dolmen analysis workflow.

    Provides a simple API for executing the complete analysis pipeline.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the workflow.

        Args:
            config: Configuration dictionary (from config.yaml)
        """
        self.config = config
        self._graph = None
        self._compiled = None

    def prepare(
        self,
        entities: list[DolmenEntity],
        rate_limiter: RateLimiter | None = None,
        output_dir: str | None = None,
    ) -> None:
        """
        Prepare the workflow for execution.

        Args:
            entities: List of DolmenEntity objects to analyze
            rate_limiter: Optional RateLimiter for tracking and throttling
            output_dir: Optional output directory for memory persistence
        """
        self._graph = build_workflow(self.config, entities, rate_limiter, output_dir)
        self._compiled = self._graph.compile()
        self._entities = entities
        self._output_dir = output_dir
        logger.info("Workflow prepared for %d entities", len(entities))

    def run(
        self,
        max_iterations: int = 10,
        min_assertions: int = 5,
        run_id: str = "unknown_run",
    ) -> dict[str, Any]:
        """
        Execute the workflow.

        Args:
            max_iterations: Maximum number of workflow iterations
            min_assertions: Minimum assertions before early termination
            run_id: External run identifier (e.g., 'xch_run_20260219_105059')

        Returns:
            Final workflow state with all generated assertions
        """
        if self._compiled is None:
            raise RuntimeError("Workflow not prepared. Call prepare() first.")

        # Create initial state
        initial_state = create_initial_state(
            entities=self._entities,
            max_iterations=max_iterations,
            min_assertions=min_assertions,
            run_id=run_id,
        )

        logger.info("Starting workflow execution")
        logger.info("  Max iterations: %d", max_iterations)
        logger.info("  Min assertions: %d", min_assertions)

        # Run the workflow
        final_state = self._compiled.invoke(initial_state)

        logger.info("Workflow completed")
        logger.info("  Total iterations: %d", final_state.get("iteration", 0))
        logger.info("  Total assertions: %d", len(final_state.get("assertions", [])))

        # Finalize memory history
        self._finalize_memory(final_state)

        return final_state

    def _finalize_memory(self, final_state: dict[str, Any]) -> None:
        """Finalize memory store after workflow completion."""
        ctx = NodeContext.get()
        if ctx.has_memory():
            run_id = final_state.get("run_id", "unknown_run")
            ctx.memory_store.finalize_run(run_id)
            logger.info(
                "Memory finalized: %d total clusters, %d total relations, %d total paths",
                ctx.memory_store.history.statistics.total_clusters,
                ctx.memory_store.history.statistics.total_relations,
                ctx.memory_store.history.statistics.total_paths,
            )

    def stream(
        self,
        max_iterations: int = 10,
        min_assertions: int = 5,
        run_id: str = "unknown_run",
    ):
        """
        Execute the workflow with streaming updates.

        Yields intermediate states for progress monitoring.

        Args:
            max_iterations: Maximum number of workflow iterations
            min_assertions: Minimum assertions before early termination
            run_id: External run identifier (e.g., 'xch_run_20260219_105059')

        Yields:
            Intermediate workflow states
        """
        if self._compiled is None:
            raise RuntimeError("Workflow not prepared. Call prepare() first.")

        initial_state = create_initial_state(
            entities=self._entities,
            max_iterations=max_iterations,
            min_assertions=min_assertions,
            run_id=run_id,
        )

        logger.info("Starting streaming workflow execution")

        for state in self._compiled.stream(initial_state):
            yield state

        logger.info("Streaming workflow completed")

    def get_assertions(self, final_state: dict[str, Any]) -> list[dict]:
        """
        Extract assertions from the final state.

        Args:
            final_state: Final workflow state

        Returns:
            List of assertion dictionaries
        """
        return final_state.get("assertions", [])

    def get_summary(self, final_state: dict[str, Any]) -> dict[str, Any]:
        """
        Generate a summary of the workflow execution.

        Args:
            final_state: Final workflow state

        Returns:
            Summary dictionary with statistics
        """
        assertions = final_state.get("assertions", [])

        # Count by type
        type_counts = {}
        for a in assertions:
            aid = a.get("assertion_id", "") if isinstance(a, dict) else a.assertion_id
            if "geo_cluster" in aid:
                type_counts["geographic_clusters"] = type_counts.get("geographic_clusters", 0) + 1
            elif "chrono" in aid or "period" in aid:
                type_counts["chronological_clusters"] = (
                    type_counts.get("chronological_clusters", 0) + 1
                )
            elif "type_cluster" in aid:
                type_counts["typological_clusters"] = type_counts.get("typological_clusters", 0) + 1
            elif "near" in aid:
                type_counts["nearTo_relations"] = type_counts.get("nearTo_relations", 0) + 1
            elif "similar" in aid:
                type_counts["similarTo_relations"] = type_counts.get("similarTo_relations", 0) + 1
            elif "contemporary" in aid:
                type_counts["contemporaryWith_relations"] = (
                    type_counts.get("contemporaryWith_relations", 0) + 1
                )
            elif "path" in aid.lower() and "stop" not in aid.lower():
                type_counts["thematic_paths"] = type_counts.get("thematic_paths", 0) + 1
            elif "stop" in aid.lower():
                type_counts["path_stops"] = type_counts.get("path_stops", 0) + 1

        return {
            "total_iterations": final_state.get("iteration", 0),
            "total_assertions": len(assertions),
            "entities_processed": len(final_state.get("entities_processed", [])),
            "assertion_types": type_counts,
            "messages": final_state.get("messages", []),
        }
