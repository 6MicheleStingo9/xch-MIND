"""
Geographic Analyzer Agent - LLM-First Spatial Analysis.

Uses LLM as primary analysis engine with algorithmic validation.
The LLM proposes meaningful geographic clusters and spatial relationships,
which are then validated using Haversine distance calculations.

Flow:
1. Prepare entity context for LLM (10%)
2. LLM proposes clusters and relations (70%)
3. Algorithmic validation (20%)
4. Hybrid confidence scoring (70% LLM, 30% validation)
"""

import json
import logging
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
    GeoAnalysisResponse,
    GeographicCluster,
    SiteRelation,
)
from ..validation import (
    ValidationResult,
    haversine_distance,
    validate_coordinates,
    validate_geographic_cluster,
    validate_near_relation,
)

logger = logging.getLogger(__name__)


class GeoAnalyzerAgent(BaseAgent):
    """
    Geographic analysis agent using LLM-first approach.

    The LLM analyzes entity descriptions and spatial context to propose:
    1. Meaningful geographic clusters (based on territorial context, not just distance)
    2. Significant nearTo relationships (with archaeological/cultural significance)

    Proposals are validated algorithmically and scored with hybrid confidence.
    """

    agent_type = AgentType.GEO_ANALYZER

    def analyze(self, state: AgentState, task: AgentTask, entities: list) -> AgentResult:
        """
        Perform LLM-first geographic analysis.

        Args:
            state: Current system state
            task: Task with parameters
            entities: List of DolmenEntity objects

        Returns:
            AgentResult with validated geographic assertions
        """
        logger.info("GeoAnalyzer: Starting LLM-first analysis of %d entities", len(entities))

        # Get validation thresholds from config
        max_cluster_radius = task.params.get(
            "max_cluster_radius_km", self.agent_config.get("clustering_threshold", 50)
        )
        max_near_distance = task.params.get(
            "max_distance_km", self.agent_config.get("max_distance_km", 30)
        )

        # Optional limits for LLM proposals (fallback to agent_config)
        max_clusters = (
            task.params.get("max_clusters") or self.agent_config.get("max_clusters")
        ) or 15
        max_pairs = (task.params.get("max_pairs") or self.agent_config.get("max_pairs")) or 30

        # Cluster size constraints (from config)
        min_cluster_size = task.params.get(
            "min_cluster_size", self.agent_config.get("min_cluster_size", None)
        )
        max_cluster_size = task.params.get(
            "max_cluster_size", self.agent_config.get("max_cluster_size", None)
        )

        # Filter entities with valid coordinates
        entities_with_coords = [
            e
            for e in entities
            if e.has_coordinates and validate_coordinates(e.latitude, e.longitude)
        ]
        logger.info("GeoAnalyzer: %d entities have valid coordinates", len(entities_with_coords))

        # Log warnings for filtered entities
        filtered_entities = [
            e
            for e in entities
            if not (e.has_coordinates and validate_coordinates(e.latitude, e.longitude))
        ]
        for entity in filtered_entities:
            reason = (
                "Missing coordinates" if not entity.has_coordinates else "Invalid coordinate values"
            )
            logger.warning(
                "GeoAnalyzer: Entity filtered - '%s' (%s) - lat=%s, lon=%s",
                entity.display_name,
                reason,
                getattr(entity, "latitude", None),
                getattr(entity, "longitude", None),
            )

        if len(entities_with_coords) < 2:
            return self._create_result(
                task=task,
                assertions=[],
                metadata={"message": "Not enough entities with coordinates"},
            )

        # Build URI -> entity mapping for validation
        entities_by_uri = {e.uri: e for e in entities_with_coords}

        # =================================================================
        # PHASE 1: Prepare LLM Context (10%)
        # =================================================================
        llm_context = self._prepare_llm_context(entities_with_coords)
        logger.debug(
            "GeoAnalyzer: Prepared LLM context with %d entities", len(entities_with_coords)
        )

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
                "GeoAnalyzer: LLM proposed %d clusters, %d relations",
                len(llm_response.clusters),
                len(llm_response.relations),
            )
        except Exception as e:
            # Propagate daily quota exhaustion - don't fallback
            from src.llm.provider import DailyQuotaExhaustedError

            if isinstance(e, DailyQuotaExhaustedError):
                raise
            logger.error("GeoAnalyzer: LLM proposal failed: %s", e)
            # Fallback to empty proposals - will use algorithmic fallback
            llm_response = GeoAnalysisResponse(
                clusters=[],
                relations=[],
                territorial_observations="LLM analysis failed",
            )

        assertions = []

        # =================================================================
        # PHASE 3: Validate and Convert Cluster Proposals (20%)
        # =================================================================
        logger.info(
            "GeoAnalyzer: LLM proposed %d clusters, validating...", len(llm_response.clusters)
        )
        for proposal in llm_response.clusters:
            logger.info(
                "GeoAnalyzer: [LLM PROPOSAL] Cluster '%s' with %d members in region '%s'",
                proposal.cluster_label,
                len(proposal.member_uris),
                proposal.region or "Unknown",
            )

            validation = validate_geographic_cluster(
                member_uris=proposal.member_uris,
                entities_by_uri=entities_by_uri,
                max_radius_km=max_cluster_radius,
                min_cluster_size=min_cluster_size or 2,
                max_cluster_size=max_cluster_size,
            )

            if not validation.is_valid:
                logger.warning(
                    "GeoAnalyzer: [REJECTED] Cluster '%s' - Reason: %s | Radius: %.1fkm (max: %.1fkm)",
                    proposal.cluster_label,
                    validation.details.get("reason", "validation failed"),
                    validation.details.get("radius_km", 0),
                    max_cluster_radius,
                )
                continue

            # Calculate hybrid confidence
            confidence = calculate_cluster_confidence(
                llm_confidence=proposal.llm_confidence,
                validation_score=validation.validation_score,
                member_count=len(proposal.member_uris),
            )

            # Convert to GeographicCluster assertion
            cluster = GeographicCluster(
                assertion_id=f"geo_cluster_{uuid.uuid4().hex[:8]}",
                label=proposal.cluster_label,
                description=proposal.reasoning,
                subject_uris=proposal.member_uris,
                generated_by=self.agent_type,
                confidence_score=round(confidence, 2),
                reasoning=f"LLM proposal validated. Radius: {validation.details.get('radius_km', 0):.1f}km",
                centroid_lat=validation.details.get("centroid", {}).get("lat"),
                centroid_lon=validation.details.get("centroid", {}).get("lon"),
                radius_km=round(validation.details.get("radius_km", 0), 2),
                site_count=validation.details.get("member_count", len(proposal.member_uris)),
                region=proposal.region,
            )
            assertions.append(cluster)
            logger.info(
                "GeoAnalyzer: Accepted cluster '%s' with %d sites, Radius: %.1fkm",
                proposal.cluster_label,
                len(proposal.member_uris),
                cluster.radius_km,
            )

        # =================================================================
        # PHASE 3b: Validate and Convert Relation Proposals
        # =================================================================
        logger.info(
            "GeoAnalyzer: LLM proposed %d nearTo relations, validating...",
            len(llm_response.relations),
        )
        for proposal in llm_response.relations:
            logger.info(
                "GeoAnalyzer: [LLM PROPOSAL] NearTo: %s -> %s | Context: %s",
                proposal.source_uri.split("/")[-1],
                proposal.target_uri.split("/")[-1],
                proposal.spatial_context[:50] if proposal.spatial_context else "N/A",
            )

            validation = validate_near_relation(
                source_uri=proposal.source_uri,
                target_uri=proposal.target_uri,
                entities_by_uri=entities_by_uri,
                max_distance_km=max_near_distance,
            )

            if not validation.is_valid:
                logger.warning(
                    "GeoAnalyzer: [REJECTED] NearTo %s -> %s - Reason: %s",
                    proposal.source_uri.split("/")[-1],
                    proposal.target_uri.split("/")[-1],
                    validation.details.get("reason", "validation failed"),
                )
                continue

            # Calculate hybrid confidence
            distance = validation.details.get("distance_km", 0)
            confidence = calculate_relation_confidence(
                llm_confidence=proposal.llm_confidence,
                validation_score=validation.validation_score,
                relation_strength=(
                    1.0 - (distance / max_near_distance) if distance < max_near_distance else 0
                ),
            )

            # Get entity names for label
            source_name = entities_by_uri.get(proposal.source_uri, {})
            target_name = entities_by_uri.get(proposal.target_uri, {})
            source_display = getattr(source_name, "display_name", proposal.source_uri[-20:])
            target_display = getattr(target_name, "display_name", proposal.target_uri[-20:])

            relation = SiteRelation(
                assertion_id=f"near_{uuid.uuid4().hex[:8]}",
                assertion_type=AssertionType.NEAR_TO,
                label=f"{source_display} near to {target_display}",
                description=proposal.spatial_context,
                source_uri=proposal.source_uri,
                target_uri=proposal.target_uri,
                subject_uris=[proposal.source_uri],
                object_uris=[proposal.target_uri],
                relation_value=distance,
                generated_by=self.agent_type,
                confidence_score=round(confidence, 2),
                reasoning=f"{proposal.reasoning}. Distance: {distance:.1f}km",
            )
            assertions.append(relation)
            logger.info(
                "GeoAnalyzer: Accepted relation: %s -> %s (Dist: %.1fkm)",
                source_display,
                target_display,
                distance,
            )

        # =================================================================
        # FALLBACK: If LLM produced nothing, use algorithmic approach
        # =================================================================
        if not assertions and len(entities_with_coords) >= 2:
            logger.warning("GeoAnalyzer: No LLM proposals accepted, using algorithmic fallback")
            fallback_assertions = self._algorithmic_fallback(
                entities_with_coords, max_cluster_radius, max_near_distance
            )
            assertions.extend(fallback_assertions)

        logger.info("GeoAnalyzer: Generated %d total assertions", len(assertions))

        return self._create_result(
            task=task,
            assertions=assertions,
            metadata={
                "entities_analyzed": len(entities_with_coords),
                "llm_clusters_proposed": len(llm_response.clusters),
                "llm_relations_proposed": len(llm_response.relations),
                "assertions_accepted": len(assertions),
                "territorial_observations": llm_response.territorial_observations,
                "fallback_used": len(assertions) > 0
                and len(llm_response.clusters) + len(llm_response.relations) == 0,
            },
        )

    def _prepare_llm_context(self, entities: list) -> dict[str, Any]:
        """
        Prepare entity data for LLM consumption.

        Creates a structured summary of each entity with relevant spatial context.
        """
        sites = []
        for e in entities:
            site = {
                "uri": e.uri,
                "name": e.display_name,
                "coordinates": {"lat": e.latitude, "lon": e.longitude},
                "municipality": e.municipality or e.coverage or "Unknown",
                "region": e.region_name or e.region_id or "Unknown",
            }
            # Include description excerpt for context
            if e.description:
                site["description_excerpt"] = (
                    e.description[:200] + "..." if len(e.description) > 200 else e.description
                )
            if e.historical_info:
                site["historical_context"] = (
                    e.historical_info[:150] + "..."
                    if len(e.historical_info) > 150
                    else e.historical_info
                )
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
    ) -> GeoAnalysisResponse:
        """
        Call LLM to propose geographic clusters and relations.

        Uses structured output for reliable parsing.

        Args:
            context: LLM context with site data
            max_clusters: Optional maximum number of clusters (None = no limit)
            max_pairs: Optional maximum number of near-to pairs (None = no limit)
            min_cluster_size: Optional minimum members per cluster (None = no limit)
            max_cluster_size: Optional maximum members per cluster (None = no limit)
        """
        sites_json = json.dumps(context["sites"], indent=2, ensure_ascii=False)

        system_prompt = PromptLibrary.GEO_ANALYZER_SYSTEM_V2
        human_prompt = PromptLibrary.GEO_ANALYZER_TASK_V2

        response = self.provider.invoke_structured(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            output_schema=GeoAnalysisResponse,
            variables={
                "total_entities": context["total_entities"],
                "sites_json": sites_json,
                "max_clusters": max_clusters or 15,
                "max_pairs": max_pairs or 20,
                "max_clusters_hint": PromptLibrary.get_max_clusters_hint(max_clusters),
                "max_pairs_hint": PromptLibrary.get_max_pairs_hint(max_pairs),
                "cluster_size_hint": PromptLibrary.get_cluster_size_hint(
                    min_cluster_size, max_cluster_size
                ),
            },
        )

        return response

    def _algorithmic_fallback(self, entities: list, max_radius: float, max_distance: float) -> list:
        """
        Fallback to algorithmic clustering when LLM fails or produces nothing.

        Uses the original distance-based approach as a safety net.
        """
        assertions = []

        # Simple distance-based clustering
        assigned = set()
        for entity in entities:
            if entity.uri in assigned:
                continue

            cluster_members = [entity]
            assigned.add(entity.uri)

            for other in entities:
                if other.uri in assigned:
                    continue
                for member in cluster_members:
                    distance = haversine_distance(
                        member.latitude, member.longitude, other.latitude, other.longitude
                    )
                    if distance <= max_radius / 2:  # Use half radius for tighter clusters
                        cluster_members.append(other)
                        assigned.add(other.uri)
                        break

            if len(cluster_members) >= 2:
                # Calculate centroid
                avg_lat = sum(e.latitude for e in cluster_members) / len(cluster_members)
                avg_lon = sum(e.longitude for e in cluster_members) / len(cluster_members)

                # Determine region
                regions = {}
                for e in cluster_members:
                    region = e.region_name or e.region_id or "Unknown"
                    regions[region] = regions.get(region, 0) + 1
                dominant_region = max(regions, key=regions.get)

                cluster = GeographicCluster(
                    assertion_id=f"geo_cluster_{uuid.uuid4().hex[:8]}",
                    label=f"Geographic Cluster - {dominant_region}",
                    description=f"Algorithmic cluster of {len(cluster_members)} sites",
                    subject_uris=[m.uri for m in cluster_members],
                    generated_by=self.agent_type,
                    confidence_score=0.5,  # Lower confidence for fallback
                    reasoning="Algorithmic fallback clustering",
                    centroid_lat=avg_lat,
                    centroid_lon=avg_lon,
                    radius_km=max_radius / 2,
                    site_count=len(cluster_members),
                    region=dominant_region,
                )
                assertions.append(cluster)

        return assertions
