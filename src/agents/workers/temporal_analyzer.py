"""
Temporal Analyzer Agent - LLM-First Chronological Analysis.

Uses LLM as primary analysis engine for:
1. Normalizing heterogeneous period labels
2. Proposing meaningful chronological clusters
3. Identifying significant contemporary relationships

Flow:
1. Prepare entity context for LLM (10%)
2. LLM proposes period normalizations, clusters, relations (70%)
3. Algorithmic validation using period overlap (20%)
4. Hybrid confidence scoring (70% LLM, 30% validation)
"""

import json
import logging
import re
import uuid
from typing import Any

from src.llm import PromptLibrary

from ..base import BaseAgent
from ..confidence import calculate_cluster_confidence, calculate_relation_confidence
from ..models import (
    AgentResult,
    AgentState,
    AgentTask,
    AgentType,
    AssertionType,
    ChronologicalAnalysisResponse,
    ChronologicalCluster,
    SiteRelation,
)
from ..validation import validate_chronological_cluster, validate_contemporary_relation

logger = logging.getLogger(__name__)


# Period normalization mapping (used for validation fallback)
PERIOD_MAP = {
    "bronzo antico": (-2300, -1700),
    "bronzo medio": (-1700, -1350),
    "bronzo recente": (-1350, -1150),
    "bronzo finale": (-1150, -900),
    "età del bronzo": (-2300, -900),
    "eneolitico": (-3000, -2300),
    "calcolitico": (-3000, -2300),
    "età del rame": (-3000, -2300),
    "neolitico": (-5000, -3000),
    "età del ferro": (-900, -27),
    "bronzo": (-2300, -900),
    "neolithic": (-5000, -3000),
    "eneolithic": (-3000, -2300),
    "chalcolithic": (-3000, -2300),
    "copper age": (-3000, -2300),
    "bronze age": (-2300, -900),
    "iron age": (-900, -27),
}


def normalize_period_fallback(period_label: str | None) -> tuple[str, int, int] | None:
    """
    Fallback period normalization using keyword matching.
    Used when LLM normalization fails or is unavailable.
    """
    if not period_label:
        return None

    period_lower = period_label.lower().strip()

    # Try specific matches first
    for period_name, (start, end) in PERIOD_MAP.items():
        if period_name in period_lower:
            return (period_name.title(), start, end)

    # Try millennium patterns
    if "millennio" in period_lower or "millennium" in period_lower:
        millennium_match = re.search(r"(\d+|[ivx]+)", period_lower)
        if millennium_match:
            return ("Millennium-based", -5000, -2000)

    return None


class TemporalAnalyzerAgent(BaseAgent):
    """
    Temporal analysis agent using LLM-first approach.

    The LLM analyzes period labels, historical information, and descriptions to:
    1. Normalize heterogeneous dating labels to standard periods
    2. Propose meaningful chronological clusters
    3. Identify significant contemporary relationships

    Proposals are validated algorithmically and scored with hybrid confidence.
    """

    agent_type = AgentType.TEMPORAL_ANALYZER

    def analyze(self, state: AgentState, task: AgentTask, entities: list) -> AgentResult:
        """
        Perform LLM-first temporal analysis.

        Args:
            state: Current system state
            task: Task with parameters
            entities: List of DolmenEntity objects

        Returns:
            AgentResult with validated chronological assertions
        """
        logger.info("TemporalAnalyzer: Starting LLM-first analysis of %d entities", len(entities))

        # Get parameters
        period_tolerance = task.params.get(
            "period_tolerance_years", self.agent_config.get("period_tolerance_years", 500)
        )

        # Optional limits for LLM proposals (fallback to agent_config)
        max_clusters = (
            task.params.get("max_clusters") or self.agent_config.get("max_clusters")
        ) or 10
        max_pairs = (task.params.get("max_pairs") or self.agent_config.get("max_pairs")) or 20

        # Cluster size constraints (from config)
        min_cluster_size = task.params.get(
            "min_cluster_size", self.agent_config.get("min_cluster_size", None)
        )
        max_cluster_size = task.params.get(
            "max_cluster_size", self.agent_config.get("max_cluster_size", None)
        )

        # Filter entities with any temporal information
        entities_with_temporal = [e for e in entities if e.period_label or e.historical_info]
        logger.info(
            "TemporalAnalyzer: %d entities have temporal information", len(entities_with_temporal)
        )

        # Log warnings for filtered entities
        filtered_entities = [e for e in entities if not (e.period_label or e.historical_info)]
        for entity in filtered_entities:
            logger.warning(
                "TemporalAnalyzer: Entity filtered - '%s' - No period information",
                entity.display_name,
            )

        if len(entities_with_temporal) < 2:
            return self._create_result(
                task=task,
                assertions=[],
                metadata={"message": "Not enough entities with temporal information"},
            )

        # Build URI -> entity mapping
        entities_by_uri = {e.uri: e for e in entities_with_temporal}

        # =================================================================
        # PHASE 1: Prepare LLM Context (10%)
        # =================================================================
        llm_context = self._prepare_llm_context(entities_with_temporal)
        logger.debug("TemporalAnalyzer: Prepared LLM context")

        # =================================================================
        # PHASE 2: LLM Proposals (70%)
        # =================================================================
        try:
            llm_response = self._get_llm_proposals(
                llm_context,
                max_clusters=max_clusters,
                max_pairs=max_pairs,
                min_cluster_size=min_cluster_size,
                max_cluster_size=max_cluster_size,
            )
            logger.info(
                "TemporalAnalyzer: LLM proposed %d normalizations, %d clusters, %d pairs",
                len(llm_response.normalizations),
                len(llm_response.clusters),
                len(llm_response.contemporary_pairs),
            )
        except Exception as e:
            # Propagate daily quota exhaustion - don't fallback
            from src.llm.provider import DailyQuotaExhaustedError

            if isinstance(e, DailyQuotaExhaustedError):
                raise
            logger.error("TemporalAnalyzer: LLM proposal failed: %s", e)
            llm_response = ChronologicalAnalysisResponse(
                normalizations=[],
                clusters=[],
                contemporary_pairs=[],
                chronological_narrative="LLM analysis failed",
            )

        assertions = []

        # Store LLM normalizations for validation
        llm_normalizations = {
            n.original_label: (n.normalized_period, n.start_year, n.end_year)
            for n in llm_response.normalizations
            if n.start_year and n.end_year
        }

        # =================================================================
        # PHASE 3: Validate and Convert Cluster Proposals
        # =================================================================
        logger.info(
            "TemporalAnalyzer: LLM proposed %d clusters, validating...", len(llm_response.clusters)
        )
        for proposal in llm_response.clusters:
            logger.info(
                "TemporalAnalyzer: [LLM PROPOSAL] Cluster '%s' (%d-%d) with %d members",
                proposal.period_label,
                proposal.start_year or 0,
                proposal.end_year or 0,
                len(proposal.member_uris),
            )

            validation = validate_chronological_cluster(
                member_uris=proposal.member_uris,
                entities_by_uri=entities_by_uri,
                period_tolerance_years=period_tolerance,
                min_cluster_size=min_cluster_size or 2,
                max_cluster_size=max_cluster_size,
            )

            if not validation.is_valid:
                logger.warning(
                    "TemporalAnalyzer: [REJECTED] Cluster '%s' - Reason: %s",
                    proposal.period_label,
                    validation.details.get("reason", "validation failed"),
                )
                continue

            # Calculate hybrid confidence
            confidence = calculate_cluster_confidence(
                llm_confidence=proposal.llm_confidence,
                validation_score=validation.validation_score,
                member_count=len(proposal.member_uris),
            )

            cluster = ChronologicalCluster(
                assertion_id=f"chrono_cluster_{uuid.uuid4().hex[:8]}",
                label=f"Chronological Cluster - {proposal.period_label}",
                description=proposal.temporal_narrative,
                subject_uris=proposal.member_uris,
                generated_by=self.agent_type,
                confidence_score=round(confidence, 2),
                reasoning=proposal.reasoning,
                period_label=proposal.period_label,
                start_year=proposal.start_year,
                end_year=proposal.end_year,
            )
            assertions.append(cluster)
            logger.info(
                "TemporalAnalyzer: Accepted cluster '%s' with %d sites (Range: %d to %d)",
                proposal.period_label,
                len(proposal.member_uris),
                proposal.start_year,
                proposal.end_year,
            )

        # =================================================================
        # PHASE 3b: Validate and Convert Contemporary Relations
        # =================================================================
        logger.info(
            "TemporalAnalyzer: LLM proposed %d contemporary pairs, validating...",
            len(llm_response.contemporary_pairs),
        )
        for proposal in llm_response.contemporary_pairs:
            logger.info(
                "TemporalAnalyzer: [LLM PROPOSAL] Contemporary: %s <-> %s | Period: %s",
                proposal.source_uri.split("/")[-1],
                proposal.target_uri.split("/")[-1],
                proposal.shared_period,
            )

            validation = validate_contemporary_relation(
                source_uri=proposal.source_uri,
                target_uri=proposal.target_uri,
                entities_by_uri=entities_by_uri,
            )

            if not validation.is_valid:
                logger.warning(
                    "TemporalAnalyzer: [REJECTED] Contemporary %s <-> %s - Reason: %s",
                    proposal.source_uri.split("/")[-1],
                    proposal.target_uri.split("/")[-1],
                    validation.details.get("reason", "validation failed"),
                )
                continue

            # Calculate hybrid confidence
            confidence = calculate_relation_confidence(
                llm_confidence=proposal.llm_confidence,
                validation_score=validation.validation_score,
            )

            # Get entity names
            source_name = getattr(
                entities_by_uri.get(proposal.source_uri), "display_name", "Unknown"
            )
            target_name = getattr(
                entities_by_uri.get(proposal.target_uri), "display_name", "Unknown"
            )

            relation = SiteRelation(
                assertion_id=f"contemp_{uuid.uuid4().hex[:8]}",
                assertion_type=AssertionType.CONTEMPORARY_WITH,
                label=f"{source_name} contemporary with {target_name}",
                description=f"Both sites date to {proposal.shared_period}",
                source_uri=proposal.source_uri,
                target_uri=proposal.target_uri,
                subject_uris=[proposal.source_uri],
                object_uris=[proposal.target_uri],
                generated_by=self.agent_type,
                confidence_score=round(confidence, 2),
                reasoning=proposal.reasoning,
            )
            assertions.append(relation)
            logger.info(
                "TemporalAnalyzer: Accepted contemporary relationship: %s <-> %s",
                source_name,
                target_name,
            )

        # =================================================================
        # FALLBACK: If LLM produced nothing, use algorithmic approach
        # =================================================================
        if not assertions and len(entities_with_temporal) >= 2:
            logger.warning(
                "TemporalAnalyzer: No LLM proposals accepted, using algorithmic fallback"
            )
            fallback_assertions = self._algorithmic_fallback(entities_with_temporal)
            assertions.extend(fallback_assertions)

        logger.info("TemporalAnalyzer: Generated %d total assertions", len(assertions))

        return self._create_result(
            task=task,
            assertions=assertions,
            metadata={
                "entities_analyzed": len(entities_with_temporal),
                "llm_clusters_proposed": len(llm_response.clusters),
                "llm_relations_proposed": len(llm_response.contemporary_pairs),
                "assertions_accepted": len(assertions),
                "chronological_narrative": llm_response.chronological_narrative,
                "fallback_used": len(assertions) > 0
                and len(llm_response.clusters) + len(llm_response.contemporary_pairs) == 0,
            },
        )

    def _prepare_llm_context(self, entities: list) -> dict[str, Any]:
        """Prepare entity data for LLM consumption."""
        sites = []
        for e in entities:
            site = {
                "uri": e.uri,
                "name": e.display_name,
                "period_label": e.period_label or "Unknown",
            }
            if e.historical_info:
                site["historical_info"] = (
                    e.historical_info[:300] + "..."
                    if len(e.historical_info) > 300
                    else e.historical_info
                )
            if e.description:
                # Look for temporal keywords in description
                desc_lower = e.description.lower()
                if any(
                    kw in desc_lower for kw in ["millennio", "secolo", "bronze", "neolitic", "età"]
                ):
                    site["description_temporal_hints"] = e.description[:200]
            sites.append(site)

        return {
            "total_entities": len(entities),
            "sites": sites,
        }

    def _get_llm_proposals(
        self,
        context: dict,
        max_clusters: int | None = None,
        max_pairs: int | None = None,
        min_cluster_size: int | None = None,
        max_cluster_size: int | None = None,
    ) -> ChronologicalAnalysisResponse:
        """Call LLM to propose chronological analysis.

        Args:
            context: LLM context with site data
            max_clusters: Optional maximum number of clusters (None = no limit)
            max_pairs: Optional maximum number of contemporary pairs (None = no limit)
            min_cluster_size: Optional minimum members per cluster (None = no limit)
            max_cluster_size: Optional maximum members per cluster (None = no limit)
        """
        sites_json = json.dumps(context["sites"], indent=2, ensure_ascii=False)

        system_prompt = PromptLibrary.TEMPORAL_ANALYZER_SYSTEM_V2
        human_prompt = PromptLibrary.TEMPORAL_ANALYZER_TASK_V2

        response = self.provider.invoke_structured(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            output_schema=ChronologicalAnalysisResponse,
            variables={
                "total_entities": context["total_entities"],
                "sites_json": sites_json,
                "max_clusters": max_clusters or 10,
                "max_pairs": max_pairs or 15,
                "max_clusters_hint": PromptLibrary.get_max_clusters_hint(max_clusters),
                "max_pairs_hint": PromptLibrary.get_max_pairs_hint(max_pairs),
                "cluster_size_hint": PromptLibrary.get_cluster_size_hint(
                    min_cluster_size, max_cluster_size
                ),
            },
        )

        return response

    def _algorithmic_fallback(self, entities: list) -> list:
        """Fallback to algorithmic period grouping."""
        assertions = []
        period_groups: dict[str, list] = {}

        for entity in entities:
            period_info = normalize_period_fallback(entity.period_label)
            if period_info:
                period_name = period_info[0]
                if period_name not in period_groups:
                    period_groups[period_name] = []
                period_groups[period_name].append(
                    {
                        "entity": entity,
                        "period_name": period_name,
                        "start_year": period_info[1],
                        "end_year": period_info[2],
                    }
                )

        for period_name, members in period_groups.items():
            if len(members) >= 2:
                start_year = min(m["start_year"] for m in members)
                end_year = max(m["end_year"] for m in members)

                cluster = ChronologicalCluster(
                    assertion_id=f"chrono_cluster_{uuid.uuid4().hex[:8]}",
                    label=f"Chronological Cluster - {period_name}",
                    description=f"Algorithmic cluster of {len(members)} sites",
                    subject_uris=[m["entity"].uri for m in members],
                    generated_by=self.agent_type,
                    confidence_score=0.5,
                    reasoning="Algorithmic fallback clustering",
                    period_label=period_name,
                    start_year=start_year,
                    end_year=end_year,
                )
                assertions.append(cluster)

        return assertions
