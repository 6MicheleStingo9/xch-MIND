"""
LangGraph Node Functions - Wrappers for agent execution.

Each node function:
1. Receives the current AgentState
2. Executes the corresponding agent
3. Updates and returns the modified state
"""

import logging
from pathlib import Path
from typing import Any

from src.agents import (
    AgentState,
    AgentTask,
    AgentType,
    GeoAnalyzerAgent,
    OrchestratorAgent,
    PathGeneratorAgent,
    TemporalAnalyzerAgent,
    TypeAnalyzerAgent,
)
from src.loaders import DolmenEntity
from src.memory.context_builder import ContextBuilder
from src.memory.history_store import KnowledgeHistoryStore
from src.memory.novelty_detector import NoveltyDetector
from src.utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


# =============================================================================
# NODE CONTEXT - Shared across nodes
# =============================================================================


class NodeContext:
    """
    Shared context for all nodes in the workflow.

    Holds configuration, entities, cached agent instances, and memory components.
    """

    _instance = None

    def __init__(
        self,
        config: dict[str, Any],
        entities: list[DolmenEntity],
        rate_limiter: RateLimiter | None = None,
        output_dir: Path | str | None = None,
    ):
        self.config = config
        self.entities = entities
        self.rate_limiter = rate_limiter
        self._agents: dict[AgentType, Any] = {}

        # Initialize memory components
        if output_dir:
            self.memory_store = KnowledgeHistoryStore(output_dir)
            min_confidence = config.get("memory", {}).get("min_confidence", 0.70)
            self.novelty_detector = NoveltyDetector(
                history_store=self.memory_store,
                cluster_overlap_threshold=config.get("memory", {}).get(
                    "cluster_overlap_threshold", 0.8
                ),
                path_overlap_threshold=config.get("memory", {}).get("path_overlap_threshold", 0.7),
                min_confidence=min_confidence,
            )
            self.context_builder = ContextBuilder(
                self.memory_store,
                min_confidence=min_confidence,
            )
            logger.info(
                "Memory initialized: %d clusters, %d relations, %d paths from previous runs",
                len(self.memory_store.history.clusters),
                len(self.memory_store.history.relations),
                len(self.memory_store.history.paths),
            )
        else:
            self.memory_store = None
            self.novelty_detector = None
            self.context_builder = None
            logger.info("Memory disabled (no output_dir provided)")

    @classmethod
    def initialize(
        cls,
        config: dict[str, Any],
        entities: list[DolmenEntity],
        rate_limiter: RateLimiter | None = None,
        output_dir: Path | str | None = None,
    ) -> "NodeContext":
        """Initialize the singleton context."""
        cls._instance = cls(config, entities, rate_limiter, output_dir)
        return cls._instance

    @classmethod
    def get(cls) -> "NodeContext":
        """Get the singleton context."""
        if cls._instance is None:
            raise RuntimeError("NodeContext not initialized. Call initialize() first.")
        return cls._instance

    def has_memory(self) -> bool:
        """Check if memory components are available."""
        return self.memory_store is not None

    def get_agent(self, agent_type: AgentType):
        """Get or create an agent instance."""
        if agent_type not in self._agents:
            self._agents[agent_type] = self._create_agent(agent_type)
        return self._agents[agent_type]

    def _create_agent(self, agent_type: AgentType):
        """Create an agent instance based on type."""
        agent_classes = {
            AgentType.ORCHESTRATOR: OrchestratorAgent,
            AgentType.GEO_ANALYZER: GeoAnalyzerAgent,
            AgentType.TEMPORAL_ANALYZER: TemporalAnalyzerAgent,
            AgentType.TYPE_ANALYZER: TypeAnalyzerAgent,
            AgentType.PATH_GENERATOR: PathGeneratorAgent,
        }
        agent_class = agent_classes.get(agent_type)
        if agent_class is None:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return agent_class(config=self.config, rate_limiter=self.rate_limiter)


# =============================================================================
# ORCHESTRATOR NODE
# =============================================================================


def orchestrator_node(state: dict) -> dict:
    """
    Orchestrator node - decides which agent to call next.

    Updates state with:
    - next_agent: The agent to route to
    - current_task: The task for the next agent
    - iteration: Incremented
    """
    logger.info("=" * 50)
    logger.info("ORCHESTRATOR NODE - Iteration %d", state.get("iteration", 0) + 1)
    logger.info("=" * 50)

    ctx = NodeContext.get()
    orchestrator = ctx.get_agent(AgentType.ORCHESTRATOR)

    # Build AgentState from dict
    agent_state = AgentState(**state)
    agent_state.entity_count = len(ctx.entities)

    # Create entities summary
    entities_summary = f"Total entities: {len(ctx.entities)}"

    # Execute orchestrator - returns AgentTask or None
    next_task = orchestrator.decide_next_agent(agent_state, entities_summary)

    # Update state
    new_state = state.copy()
    new_state["iteration"] = state.get("iteration", 0) + 1

    if next_task is None:
        new_state["next_agent"] = "end"
        new_state["current_task"] = None
        new_state["messages"] = state.get("messages", []) + [
            "Orchestrator decided: workflow complete"
        ]
    else:
        # Map AgentType to node name
        agent_to_node = {
            AgentType.GEO_ANALYZER: "geo_analyzer",
            AgentType.TEMPORAL_ANALYZER: "temporal_analyzer",
            AgentType.TYPE_ANALYZER: "type_analyzer",
            AgentType.PATH_GENERATOR: "path_generator",
        }
        new_state["next_agent"] = agent_to_node.get(next_task.agent, "end")
        new_state["current_task"] = {
            "task_id": next_task.task_id,
            "description": next_task.description,
            "params": next_task.params,
        }
        new_state["messages"] = state.get("messages", []) + [
            f"Orchestrator decided: {new_state['next_agent']}"
        ]

    logger.info("Orchestrator decision: %s", new_state["next_agent"])

    return new_state


# =============================================================================
# WORKER NODES
# =============================================================================


def geo_analyzer_node(state: dict) -> dict:
    """
    Geographic analyzer node - identifies spatial relationships.

    Updates state with geographic assertions.
    Integrates with memory system for novelty detection.
    """
    logger.info("-" * 40)
    logger.info("GEO ANALYZER NODE")
    logger.info("-" * 40)

    ctx = NodeContext.get()
    agent = ctx.get_agent(AgentType.GEO_ANALYZER)

    # Build AgentState
    agent_state = AgentState(**state)

    # Prepare task params with historical context
    task_params = state.get("current_task", {}).get("params", {}).copy()
    if ctx.has_memory():
        task_params["historical_context"] = ctx.context_builder.build_geo_context()
        logger.info("GeoAnalyzer: Injected historical context into LLM prompt")

    # Create task
    task = AgentTask(
        agent=AgentType.GEO_ANALYZER,
        task_id=f"geo_{agent_state.iteration}",
        description="Analyze geographic proximity and clusters",
        params=task_params,
    )

    # Execute analysis
    result = agent.analyze(agent_state, task, ctx.entities)

    # Filter novel assertions and update history
    novel_assertions = []
    for assertion in result.assertions:
        is_novel = True

        if ctx.has_memory():
            if assertion.assertion_type.value == "geographic_cluster":
                is_novel, details = ctx.novelty_detector.is_novel_cluster(
                    assertion.subject_uris, "geographic"
                )
                if is_novel:
                    # Add to history
                    ctx.memory_store.add_cluster(
                        cluster_id=assertion.assertion_id,
                        cluster_type="geographic",
                        label=assertion.label,
                        member_uris=assertion.subject_uris,
                        confidence=assertion.confidence_score,
                        run_id=state.get("run_id", "unknown_run"),
                        metadata={"region": getattr(assertion, "region", None)},
                    )
                else:
                    logger.info(
                        "GeoAnalyzer: Filtered duplicate cluster '%s' (%.2f overlap)",
                        assertion.label,
                        details.get("max_overlap", 0),
                    )
            elif assertion.assertion_type.value == "near_to":
                is_novel, _ = ctx.novelty_detector.is_novel_relation(
                    assertion.source_uri, assertion.target_uri, "nearTo"
                )
                if is_novel:
                    ctx.memory_store.add_relation(
                        relation_id=assertion.assertion_id,
                        relation_type="nearTo",
                        source_uri=assertion.source_uri,
                        target_uri=assertion.target_uri,
                        confidence=assertion.confidence_score,
                        run_id=state.get("run_id", "unknown_run"),
                        value=assertion.relation_value,
                    )

        if is_novel:
            novel_assertions.append(assertion)

    # Invalidate novelty cache after updates
    if ctx.has_memory():
        ctx.novelty_detector.invalidate_caches()

    # Update state
    new_state = state.copy()
    existing_assertions = state.get("assertions", [])
    new_assertions = [a.model_dump() for a in novel_assertions]
    new_state["assertions"] = existing_assertions + new_assertions
    new_state["entities_processed"] = list(
        set(state.get("entities_processed", []) + [e.uri for e in ctx.entities if e.latitude])
    )
    new_state["messages"] = state.get("messages", []) + [
        f"GeoAnalyzer generated {len(novel_assertions)} novel assertions (filtered {len(result.assertions) - len(novel_assertions)} duplicates)"
    ]

    # Track novelty statistics
    novelty_stats = state.get("novelty_stats", {})
    novelty_stats["geo_analyzer"] = {
        "proposed": len(result.assertions),
        "novel": len(novel_assertions),
        "filtered_as_duplicates": len(result.assertions) - len(novel_assertions),
    }
    new_state["novelty_stats"] = novelty_stats

    new_state["geo_analysis_done"] = True

    logger.info(
        "GeoAnalyzer: %d novel assertions (%d filtered as duplicates)",
        len(novel_assertions),
        len(result.assertions) - len(novel_assertions),
    )

    return new_state


def temporal_analyzer_node(state: dict) -> dict:
    """
    Temporal analyzer node - identifies chronological relationships.

    Updates state with temporal assertions.
    Integrates with memory system for novelty detection.
    """
    logger.info("-" * 40)
    logger.info("TEMPORAL ANALYZER NODE")
    logger.info("-" * 40)

    ctx = NodeContext.get()
    agent = ctx.get_agent(AgentType.TEMPORAL_ANALYZER)

    # Build AgentState
    agent_state = AgentState(**state)

    # Prepare task params with historical context
    task_params = state.get("current_task", {}).get("params", {}).copy()
    if ctx.has_memory():
        task_params["historical_context"] = ctx.context_builder.build_temporal_context()
        logger.info("TemporalAnalyzer: Injected historical context into LLM prompt")

    # Create task
    task = AgentTask(
        agent=AgentType.TEMPORAL_ANALYZER,
        task_id=f"temporal_{agent_state.iteration}",
        description="Analyze temporal periods and contemporaneity",
        params=task_params,
    )

    # Execute analysis
    result = agent.analyze(agent_state, task, ctx.entities)

    # Filter novel assertions and update history
    novel_assertions = []
    for assertion in result.assertions:
        is_novel = True

        if ctx.has_memory():
            if assertion.assertion_type.value == "chronological_cluster":
                is_novel, details = ctx.novelty_detector.is_novel_cluster(
                    assertion.subject_uris, "chronological"
                )
                if is_novel:
                    ctx.memory_store.add_cluster(
                        cluster_id=assertion.assertion_id,
                        cluster_type="chronological",
                        label=assertion.label,
                        member_uris=assertion.subject_uris,
                        confidence=assertion.confidence_score,
                        run_id=state.get("run_id", "unknown_run"),
                        metadata={
                            "period_label": getattr(assertion, "period_label", None),
                            "start_year": getattr(assertion, "start_year", None),
                            "end_year": getattr(assertion, "end_year", None),
                        },
                    )
                else:
                    logger.info(
                        "TemporalAnalyzer: Filtered duplicate cluster '%s' (%.2f overlap)",
                        assertion.label,
                        details.get("max_overlap", 0),
                    )
            elif assertion.assertion_type.value == "contemporary_with":
                is_novel, _ = ctx.novelty_detector.is_novel_relation(
                    assertion.source_uri, assertion.target_uri, "contemporaryWith"
                )
                if is_novel:
                    ctx.memory_store.add_relation(
                        relation_id=assertion.assertion_id,
                        relation_type="contemporaryWith",
                        source_uri=assertion.source_uri,
                        target_uri=assertion.target_uri,
                        confidence=assertion.confidence_score,
                        run_id=state.get("run_id", "unknown_run"),
                        value=assertion.relation_value,
                    )

        if is_novel:
            novel_assertions.append(assertion)

    # Invalidate novelty cache after updates
    if ctx.has_memory():
        ctx.novelty_detector.invalidate_caches()

    # Update state
    new_state = state.copy()
    existing_assertions = state.get("assertions", [])
    new_assertions = [a.model_dump() for a in novel_assertions]
    new_state["assertions"] = existing_assertions + new_assertions
    new_state["messages"] = state.get("messages", []) + [
        f"TemporalAnalyzer generated {len(novel_assertions)} novel assertions (filtered {len(result.assertions) - len(novel_assertions)} duplicates)"
    ]

    # Track novelty statistics
    novelty_stats = state.get("novelty_stats", {})
    novelty_stats["temporal_analyzer"] = {
        "proposed": len(result.assertions),
        "novel": len(novel_assertions),
        "filtered_as_duplicates": len(result.assertions) - len(novel_assertions),
    }
    new_state["novelty_stats"] = novelty_stats

    new_state["temporal_analysis_done"] = True

    logger.info(
        "TemporalAnalyzer: %d novel assertions (%d filtered as duplicates)",
        len(novel_assertions),
        len(result.assertions) - len(novel_assertions),
    )

    return new_state


def type_analyzer_node(state: dict) -> dict:
    """
    Type analyzer node - identifies typological similarities.

    Updates state with typological assertions.
    Integrates with memory system for novelty detection.
    """
    logger.info("-" * 40)
    logger.info("TYPE ANALYZER NODE")
    logger.info("-" * 40)

    ctx = NodeContext.get()
    agent = ctx.get_agent(AgentType.TYPE_ANALYZER)

    # Build AgentState
    agent_state = AgentState(**state)

    # Prepare task params with historical context
    task_params = state.get("current_task", {}).get("params", {}).copy()
    if ctx.has_memory():
        task_params["historical_context"] = ctx.context_builder.build_type_context()
        logger.info("TypeAnalyzer: Injected historical context into LLM prompt")

    # Create task
    task = AgentTask(
        agent=AgentType.TYPE_ANALYZER,
        task_id=f"type_{agent_state.iteration}",
        description="Analyze typological similarities",
        params=task_params,
    )

    # Execute analysis
    result = agent.analyze(agent_state, task, ctx.entities)

    # Filter novel assertions and update history
    novel_assertions = []
    for assertion in result.assertions:
        is_novel = True

        if ctx.has_memory():
            if assertion.assertion_type.value == "typological_cluster":
                is_novel, details = ctx.novelty_detector.is_novel_cluster(
                    assertion.subject_uris, "typological"
                )
                if is_novel:
                    ctx.memory_store.add_cluster(
                        cluster_id=assertion.assertion_id,
                        cluster_type="typological",
                        label=assertion.label,
                        member_uris=assertion.subject_uris,
                        confidence=assertion.confidence_score,
                        run_id=state.get("run_id", "unknown_run"),
                        metadata={
                            "typology": getattr(assertion, "typology", None),
                            "shared_features": getattr(assertion, "shared_features", []),
                        },
                    )
                else:
                    logger.info(
                        "TypeAnalyzer: Filtered duplicate cluster '%s' (%.2f overlap)",
                        assertion.label,
                        details.get("max_overlap", 0),
                    )
            elif assertion.assertion_type.value == "similar_to":
                is_novel, _ = ctx.novelty_detector.is_novel_relation(
                    assertion.source_uri, assertion.target_uri, "similarTo"
                )
                if is_novel:
                    ctx.memory_store.add_relation(
                        relation_id=assertion.assertion_id,
                        relation_type="similarTo",
                        source_uri=assertion.source_uri,
                        target_uri=assertion.target_uri,
                        confidence=assertion.confidence_score,
                        run_id=state.get("run_id", "unknown_run"),
                        value=assertion.relation_value,
                    )

        if is_novel:
            novel_assertions.append(assertion)

    # Invalidate novelty cache after updates
    if ctx.has_memory():
        ctx.novelty_detector.invalidate_caches()

    # Update state
    new_state = state.copy()
    existing_assertions = state.get("assertions", [])
    new_assertions = [a.model_dump() for a in novel_assertions]
    new_state["assertions"] = existing_assertions + new_assertions
    new_state["messages"] = state.get("messages", []) + [
        f"TypeAnalyzer generated {len(novel_assertions)} novel assertions (filtered {len(result.assertions) - len(novel_assertions)} duplicates)"
    ]

    # Track novelty statistics
    novelty_stats = state.get("novelty_stats", {})
    novelty_stats["type_analyzer"] = {
        "proposed": len(result.assertions),
        "novel": len(novel_assertions),
        "filtered_as_duplicates": len(result.assertions) - len(novel_assertions),
    }
    new_state["novelty_stats"] = novelty_stats

    new_state["type_analysis_done"] = True

    logger.info(
        "TypeAnalyzer: %d novel assertions (%d filtered as duplicates)",
        len(novel_assertions),
        len(result.assertions) - len(novel_assertions),
    )

    return new_state


def path_generator_node(state: dict) -> dict:
    """
    Path generator node - creates thematic visit paths.

    Updates state with path assertions.
    Integrates with memory system for novelty detection.
    """
    logger.info("-" * 40)
    logger.info("PATH GENERATOR NODE")
    logger.info("-" * 40)

    ctx = NodeContext.get()
    agent = ctx.get_agent(AgentType.PATH_GENERATOR)

    # Build AgentState with assertions reconstructed
    agent_state = _reconstruct_state(state)

    # Prepare task params with historical context
    task_params = state.get("current_task", {}).get("params", {}).copy()
    if ctx.has_memory():
        task_params["historical_context"] = ctx.context_builder.build_path_context()
        logger.info("PathGenerator: Injected historical context into LLM prompt")

    # Create task
    task = AgentTask(
        agent=AgentType.PATH_GENERATOR,
        task_id=f"path_{agent_state.iteration}",
        description="Generate thematic visit paths",
        params=task_params,
    )

    # Execute analysis
    result = agent.analyze(agent_state, task, ctx.entities)

    # Filter novel assertions and update history
    novel_assertions = []
    for assertion in result.assertions:
        is_novel = True

        if ctx.has_memory() and assertion.assertion_type.value == "thematic_path":
            # Get stop URIs from the path
            stop_uris = [stop.site_uri for stop in getattr(assertion, "stops", [])]
            is_novel, details = ctx.novelty_detector.is_novel_path(
                stop_uris, theme=getattr(assertion, "theme", None)
            )
            if is_novel:
                ctx.memory_store.add_path(
                    path_id=assertion.assertion_id,
                    theme=getattr(assertion, "theme", "unknown"),
                    path_type=getattr(assertion, "path_type", "mixed"),
                    stop_uris=stop_uris,
                    confidence=assertion.confidence_score,
                    run_id=state.get("run_id", "unknown_run"),
                    metadata={
                        "narrative": getattr(assertion, "narrative", None),
                        "estimated_duration": getattr(assertion, "estimated_duration", None),
                    },
                )
            else:
                logger.info(
                    "PathGenerator: Filtered duplicate path '%s' (%.2f overlap)",
                    getattr(assertion, "theme", "unknown"),
                    details.get("max_overlap", 0),
                )

        if is_novel:
            novel_assertions.append(assertion)

    # Invalidate novelty cache after updates
    if ctx.has_memory():
        ctx.novelty_detector.invalidate_caches()

    # Update state
    new_state = state.copy()
    existing_assertions = state.get("assertions", [])
    new_assertions = [a.model_dump() for a in novel_assertions]
    new_state["assertions"] = existing_assertions + new_assertions
    new_state["messages"] = state.get("messages", []) + [
        f"PathGenerator generated {len(novel_assertions)} novel assertions (filtered {len(result.assertions) - len(novel_assertions)} duplicates)"
    ]

    # Track novelty statistics
    novelty_stats = state.get("novelty_stats", {})
    novelty_stats["path_generator"] = {
        "proposed": len(result.assertions),
        "novel": len(novel_assertions),
        "filtered_as_duplicates": len(result.assertions) - len(novel_assertions),
    }
    new_state["novelty_stats"] = novelty_stats

    new_state["path_generation_done"] = True

    logger.info(
        "PathGenerator: %d novel assertions (%d filtered as duplicates)",
        len(novel_assertions),
        len(result.assertions) - len(novel_assertions),
    )

    return new_state


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _reconstruct_state(state: dict) -> AgentState:
    """
    Reconstruct AgentState with proper assertion objects.

    LangGraph stores state as dicts, so we need to reconstruct
    the Pydantic models for agents that need typed assertions.
    """
    from src.agents.models import (
        ChronologicalCluster,
        ExtractedFeatureAssertion,
        GeographicCluster,
        InterpretiveAssertion,
        PathStop,
        SiteRelation,
        ThematicPath,
        TypologicalCluster,
    )

    assertions = []
    for a in state.get("assertions", []):
        if isinstance(a, dict):
            # Reconstruct based on type hints in the dict
            assertion_id = a.get("assertion_id", "")
            assertion_type = a.get("assertion_type", "")
            if assertion_type == "extracted_feature" or "feature" in assertion_id:
                assertions.append(ExtractedFeatureAssertion(**a))
            elif "geo_cluster" in assertion_id:
                assertions.append(GeographicCluster(**a))
            elif "chrono_cluster" in assertion_id or "period" in assertion_id:
                assertions.append(ChronologicalCluster(**a))
            elif "type_cluster" in assertion_id:
                assertions.append(TypologicalCluster(**a))
            elif "path" in assertion_id.lower() and "stop" not in assertion_id.lower():
                assertions.append(ThematicPath(**a))
            elif "stop" in assertion_id.lower():
                assertions.append(PathStop(**a))
            elif (
                "near" in assertion_id
                or "similar" in assertion_id
                or "contemporary" in assertion_id
            ):
                assertions.append(SiteRelation(**a))
            else:
                assertions.append(InterpretiveAssertion(**a))
        else:
            assertions.append(a)

    return AgentState(
        iteration=state.get("iteration", 0),
        max_iterations=state.get("max_iterations", 20),
        entity_count=state.get("entity_count", 0),
        entities_processed=state.get("entities_processed", []),
        assertions=assertions,
        messages=state.get("messages", []),
    )
