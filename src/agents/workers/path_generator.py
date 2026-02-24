"""
Path Generator Agent - LLM-First Thematic Path Generation.

Uses LLM as primary engine to create narrative-driven visit itineraries.
Synthesizes results from Geo, Chrono, and Type agents to propose coherent paths.

Flow:
1. Prepare Context (20%): Entities + Existing Clusters (Geo, Chrono, Type)
2. LLM Proposal (60%): Generate thematic paths with specific stops and narratives
3. Validation (10%): Verify geographic feasibility and logical connectivity
4. Hybrid Confidence (10%): 70% LLM, 30% Validation
"""

import json
import logging
import uuid
from typing import Any

from src.llm import PromptLibrary

from ..base import BaseAgent
from ..confidence import calculate_path_confidence
from ..models import (
    AgentResult,
    AgentState,
    AgentTask,
    AgentType,
    AssertionType,
    ChronologicalCluster,
    GeographicCluster,
    InterpretiveAssertion,
    NarrativeAnalysisResponse,
    PathStop,
    SiteRelation,
    ThematicPath,
    TypologicalCluster,
)
from ..validation import validate_thematic_path

logger = logging.getLogger(__name__)


class PathGeneratorAgent(BaseAgent):
    """
    Path generation agent using LLM-first approach.

    The LLM analyzes entities and existing clusters to propose:
    1. Thematic paths (Geographic, Chronological, Typological, Mixed)
    2. Ordered stops with visitation logic
    3. Narrative descriptions connecting the stops

    Proposals are validated for geographic feasibility (travel distances).
    """

    agent_type = AgentType.PATH_GENERATOR

    def analyze(self, state: AgentState, task: AgentTask, entities: list) -> AgentResult:
        """
        Perform LLM-first path generation.

        Args:
            state: Current system state with previous assertions
            task: Task with parameters (max_stops, theme)
            entities: List of DolmenEntity objects

        Returns:
            AgentResult with validated thematic paths and stops
        """
        logger.info("PathGenerator: Starting LLM-first path generation")

        # Get parameters
        max_stops = task.params.get("max_stops", self.agent_config.get("max_path_length", 10))
        min_stops = task.params.get("min_stops", self.agent_config.get("min_path_length", 3))
        # theme parameter is passed to LLM but it can generate mixed themes
        requested_theme = task.params.get("theme", "mixed")

        # Optional limits for LLM proposals (fallback to agent_config)
        max_paths = task.params.get("max_paths", self.agent_config.get("max_paths"))
        min_stops_per_path = task.params.get("min_stops_per_path", min_stops)
        max_stops_per_path = task.params.get("max_stops_per_path", max_stops)

        if len(entities) < 2:
            return self._create_result(
                task=task,
                assertions=[],
                metadata={"message": "Not enough entities to generate paths"},
            )

        # Build URI -> entity mapping
        entities_by_uri = {e.uri: e for e in entities}

        # =================================================================
        # PHASE 1: Prepare LLM Context (20%)
        # =================================================================
        # Extract clusters from previous agents
        clusters = self._extract_clusters(state)
        llm_context = self._prepare_llm_context(entities, clusters, requested_theme)
        logger.debug(
            "PathGenerator: Prepared context with %d clusters and %d entities",
            len(clusters),
            len(entities),
        )

        # =================================================================
        # PHASE 2: LLM Proposals (60%)
        # =================================================================
        try:
            llm_response = self._get_llm_proposals(
                llm_context,
                max_paths=max_paths,
                min_stops=min_stops_per_path,
                max_stops=max_stops_per_path,
            )
            logger.info("PathGenerator: LLM proposed %d paths", len(llm_response.paths))
        except Exception as e:
            # Propagate daily quota exhaustion - don't fallback
            from src.llm.provider import DailyQuotaExhaustedError

            if isinstance(e, DailyQuotaExhaustedError):
                raise
            logger.error("PathGenerator: LLM proposal failed: %s", e)
            llm_response = NarrativeAnalysisResponse(
                paths=[],
                overall_narrative="LLM analysis failed",
            )

        assertions = []

        # =================================================================
        # PHASE 3: Validate and Convert Proposals (10%)
        # =================================================================
        for proposal in llm_response.paths:
            logger.info(
                "PathGenerator: [LLM PROPOSAL] Path '%s' (%s) with %d stops",
                proposal.path_title,
                getattr(proposal, "path_type", "mixed"),
                len(proposal.stops),
            )
            # Log each stop proposed by LLM
            for i, stop in enumerate(proposal.stops, 1):
                entity = entities_by_uri.get(stop.site_uri)
                resolved_name = entity.display_name if entity else "[NOT FOUND]"
                logger.info(
                    "PathGenerator: [LLM PROPOSAL]   Stop %d: URI=%s | LLM_name='%s' | Resolved='%s'",
                    i,
                    stop.site_uri.split("/")[-1],
                    stop.site_name,
                    resolved_name,
                )
                if not entity:
                    logger.warning(
                        "PathGenerator: [WARNING] Stop %d URI not found in entities: %s",
                        i,
                        stop.site_uri,
                    )

            # Check stop count limits
            if len(proposal.stops) > max_stops * 1.5:  # Allow slight flexibility
                proposal.stops = proposal.stops[:max_stops]

            # Validate path feasibility
            validation = validate_thematic_path(
                stops=[{"site_uri": s.site_uri, "order": i} for i, s in enumerate(proposal.stops)],
                entities_by_uri=entities_by_uri,
                path_type=getattr(proposal, "path_type", "mixed"),
            )

            # Note: We are lenient with validation issues (warnings instead of rejections)
            # unless the path is physically impossible (e.g. unknown sites)
            if not validation.is_valid:
                logger.warning(
                    "PathGenerator: [REJECTED] Path '%s' - Reason: %s",
                    proposal.path_title,
                    validation.details.get("reason"),
                )
                # Continue anyway, but score will be lower

            # Calculate hybrid confidence
            confidence = calculate_path_confidence(
                llm_confidence=proposal.llm_confidence,
                validation_score=validation.validation_score,
                stop_count=len(proposal.stops),
            )

            path_id = f"path_{uuid.uuid4().hex[:8]}"

            # Create ThematicPath assertion
            path = ThematicPath(
                assertion_id=path_id,
                label=proposal.path_title,
                description=proposal.overall_narrative,
                narrative=proposal.overall_narrative,
                subject_uris=[s.site_uri for s in proposal.stops],
                generated_by=self.agent_type,
                confidence_score=round(confidence, 2),
                reasoning=f"LLM proposed path. Validation score: {validation.validation_score:.2f}",
                theme=proposal.theme,
                path_type=getattr(proposal, "path_type", "mixed"),
                estimated_duration=proposal.estimated_duration_hours,
                difficulty=proposal.difficulty,
            )
            assertions.append(path)

            # Create individual PathStop assertions
            path_stops = []
            for i, stop_proposal in enumerate(proposal.stops, 1):
                entity = entities_by_uri.get(stop_proposal.site_uri)

                # Use entity display_name if found, otherwise use LLM's site_name with warning
                if entity:
                    stop_label = entity.display_name
                else:
                    # Fallback to LLM name but mark it as unverified
                    stop_label = stop_proposal.site_name or "Unknown Site"
                    logger.warning(
                        "PathGenerator: Stop %d uses unverified name '%s' (URI not found: %s)",
                        i,
                        stop_label,
                        stop_proposal.site_uri,
                    )

                stop = PathStop(
                    assertion_id=f"stop_{path_id}_{i}",
                    assertion_type=AssertionType.PATH_STOP,
                    label=f"Stop {i}: {stop_label}",
                    description=stop_proposal.narrative,
                    subject_uris=[stop_proposal.site_uri],
                    generated_by=self.agent_type,
                    confidence_score=path.confidence_score,
                    reasoning=f"Stop {i} in path {path_id}",
                    path_id=path_id,
                    site_uri=stop_proposal.site_uri,
                    order=i,
                    justification=stop_proposal.narrative,
                )
                assertions.append(stop)
                path_stops.append(stop)

            path.stops = path_stops
            logger.info(
                "PathGenerator: Accepted path '%s' (%s) with %d stops",
                proposal.path_title,
                proposal.theme,
                len(path_stops),
            )
            logger.debug(
                "PathGenerator: Accepted path '%s' (confidence: %.2f)",
                proposal.path_title,
                confidence,
            )

        # =================================================================
        # FALLBACK: Algorithmic Fallback (if no LLM paths)
        # =================================================================
        if not any(isinstance(a, ThematicPath) for a in assertions):
            logger.warning("PathGenerator: No LLM paths accepted, using algorithmic fallback")
            fallback_assertions = self._algorithmic_fallback(state, entities, max_stops)
            assertions.extend(fallback_assertions)

        logger.info("PathGenerator: Generated %d total assertions", len(assertions))

        return self._create_result(
            task=task,
            assertions=assertions,
            metadata={
                "entities_available": len(entities),
                "llm_paths_proposed": len(llm_response.paths),
                "assertions_generated": len(assertions),
                "overall_narrative": llm_response.overall_narrative,
                "fallback_used": len(assertions) > 0 and len(llm_response.paths) == 0,
            },
        )

    def _extract_clusters(self, state: AgentState) -> list[InterpretiveAssertion]:
        """Extract cluster assertions from state."""
        clusters = []
        for assertion in state.assertions:
            if isinstance(assertion, (GeographicCluster, ChronologicalCluster, TypologicalCluster)):
                clusters.append(assertion)
        return clusters

    def _prepare_llm_context(
        self, entities: list, clusters: list, requested_theme: str
    ) -> dict[str, Any]:
        """Prepare context for LLM: sites and existing clusters."""

        # specific instructions based on theme
        theme_instructions = f"Focus on {requested_theme} connections."

        # Serialize sites (lite version)
        sites_data = []
        for e in entities:
            sites_data.append(
                {
                    "uri": e.uri,
                    "name": e.display_name,
                    "period": e.period_label,
                    "category": e.category,
                    "municipality": e.municipality or e.coverage,
                    "description_excerpt": e.description[:100] if e.description else "",
                }
            )

        # Serialize clusters
        clusters_data = []
        for c in clusters:
            clusters_data.append(
                {
                    "type": c.__class__.__name__,
                    "label": c.label,
                    "description": c.description,
                    "members": c.subject_uris,
                }
            )

        return {
            "total_entities": len(entities),
            "theme_instructions": theme_instructions,
            "sites": sites_data,
            "clusters": clusters_data,
        }

    def _get_llm_proposals(
        self,
        context: dict,
        max_paths: int | None = None,
        min_stops: int | None = None,
        max_stops: int | None = None,
    ) -> NarrativeAnalysisResponse:
        """Call LLM to propose thematic paths.

        Args:
            context: LLM context with site and cluster data
            max_paths: Optional maximum number of paths (None = no limit)
            min_stops: Optional minimum stops per path (None = no limit)
            max_stops: Optional maximum stops per path (None = no limit)
        """
        sites_json = json.dumps(context["sites"], indent=2, ensure_ascii=False)
        clusters_json = json.dumps(context["clusters"], indent=2, ensure_ascii=False)

        system_prompt = PromptLibrary.PATH_GENERATOR_SYSTEM_V2
        human_prompt = PromptLibrary.PATH_GENERATOR_TASK_V2

        response = self.provider.invoke_structured(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            output_schema=NarrativeAnalysisResponse,
            variables={
                "theme_instructions": context["theme_instructions"],
                "sites_json": sites_json,
                "clusters_json": clusters_json,
                "max_paths_hint": PromptLibrary.get_max_paths_hint(max_paths),
                "stops_range_hint": PromptLibrary.get_stops_range_hint(min_stops, max_stops),
            },
        )

        return response

    def _algorithmic_fallback(
        self, state: AgentState, entities: list, max_stops: int
    ) -> list[InterpretiveAssertion]:
        """Fallback path generation using simple cluster-to-path logic."""
        assertions = []
        clusters = self._extract_clusters(state)
        entity_map = {e.uri: e for e in entities}

        # Create one path per cluster (up to 3 paths to avoid spam)
        for cluster in clusters[:3]:
            members = [uri for uri in cluster.subject_uris if uri in entity_map]
            if len(members) < 2:
                continue

            path_members = members[:max_stops]

            path_id = f"path_algo_{uuid.uuid4().hex[:8]}"
            path = ThematicPath(
                assertion_id=path_id,
                label=f"Visit: {cluster.label}",
                description=f"Path based on {cluster.label}",
                subject_uris=path_members,
                generated_by=self.agent_type,
                confidence_score=0.5,
                reasoning="Algorithmic fallback based on cluster",
                theme="mixed",
                estimated_duration=len(path_members) * 0.5,
                difficulty="medium",
            )
            assertions.append(path)

            for i, uri in enumerate(path_members, 1):
                stop = PathStop(
                    assertion_id=f"stop_{path_id}_{i}",
                    assertion_type=AssertionType.PATH_STOP,
                    label=f"Stop {i}",
                    description="Visit site",
                    subject_uris=[uri],
                    generated_by=self.agent_type,
                    confidence_score=0.5,
                    reasoning="Algorithmic stop",
                    path_id=path_id,
                    site_uri=uri,
                    order=i,
                    justification="Cluster member",
                )
                assertions.append(stop)

        return assertions
