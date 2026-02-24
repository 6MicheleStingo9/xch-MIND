"""
Type Analyzer Agent - LLM-First Typological Analysis.

Uses LLM as primary analysis engine for:
1. Extracting architectural and functional features from unstructured text
2. Proposing meaningful typological clusters
3. Identifying similar sites based on extracted features

Flow:
1. Prepare entity context for LLM (10%)
2. LLM performs feature extraction and proposes clusters/pairs (70%)
3. Algorithmic validation using cosine similarity on SpaCy embeddings (20%)
4. Hybrid confidence scoring (70% LLM, 30% validation)
"""

import json
import logging
import uuid
from typing import Any

from src.llm import PromptLibrary
from src.utils.feature_normalizer import (
    deduplicate_feature_labels,
    normalize_feature_label,
)
from src.utils.semantic_similarity import (
    build_description_vectors,
    validate_similarity_with_embeddings,
)

from ..base import BaseAgent
from ..confidence import (
    calculate_cluster_confidence,
    calculate_feature_confidence,
    calculate_relation_confidence,
)
from ..models import (
    AgentResult,
    AgentState,
    AgentTask,
    AgentType,
    AssertionType,
    ExtractedFeatureAssertion,
    SiteRelation,
    TypologicalAnalysisResponse,
    TypologicalCluster,
)
from ..validation import validate_typological_cluster

logger = logging.getLogger(__name__)


# Feature extraction keywords for algorithmic fallback
ARCHITECTURAL_FEATURES_FALLBACK = [
    "lastra",
    "lastrone",
    "lastre",
    "copertura",
    "ortostati",
    "camera",
    "corridoio",
    "dromos",
    "ingresso",
    "tumulo",
    "cairn",
    "megalite",
    "monolite",
    "menhir",
    "pietra",
    "blocco",
    "allée couverte",
    "tholos",
    "cista",
    "galleria",
    "portale",
    "slab",
    "cover",
    "chamber",
    "corridor",
    "entrance",
    "capstone",
    "portal",
    "falsa cupola",
]

FUNCTIONAL_CATEGORIES_FALLBACK = {
    "funerary": [
        "funerario",
        "funeraria",
        "sepoltura",
        "tomba",
        "funerary",
        "burial",
        "tomb",
        "necropoli",
    ],
    "cultic": ["cultuale", "sacro", "rituale", "cultic", "sacred", "ritual"],
    "domestic": ["abitativo", "insediamento", "domestic", "settlement"],
}


class TypeAnalyzerAgent(BaseAgent):
    """
    Typological analysis agent using LLM-first approach.

    The LLM analyzes site descriptions to:
    1. Extract structural and functional features (which are often implicit)
    2. Propose typological clusters based on shared characteristics
    3. Identify similar sites

    Proposals are validated algorithmically using Jaccard similarity on the extracted features.
    """

    agent_type = AgentType.TYPE_ANALYZER

    def analyze(self, state: AgentState, task: AgentTask, entities: list) -> AgentResult:
        """
        Perform LLM-first typological analysis.

        Args:
            state: Current system state
            task: Task with parameters
            entities: List of DolmenEntity objects

        Returns:
            AgentResult with validated typological assertions
        """
        logger.info("TypeAnalyzer: Starting LLM-first analysis of %d entities", len(entities))

        # Optional limits for LLM proposals (fallback to agent_config)
        max_clusters = task.params.get("max_clusters", self.agent_config.get("max_clusters")) or 10
        max_pairs = task.params.get("max_pairs", self.agent_config.get("max_pairs")) or 20

        # Cluster size constraints (from config)
        min_cluster_size = (
            task.params.get("min_cluster_size") or self.agent_config.get("min_cluster_size") or 2
        )
        max_cluster_size = (
            task.params.get("max_cluster_size") or self.agent_config.get("max_cluster_size") or 6
        )

        # Majority threshold for cluster validation (from config)
        majority_threshold = task.params.get(
            "majority_threshold", self.agent_config.get("majority_threshold", 0.6)
        )

        entities_with_text = [
            e for e in entities if e.description or e.historical_info or e.category
        ]
        logger.info("TypeAnalyzer: %d entities have textual descriptions", len(entities_with_text))

        # Log warnings for filtered entities
        filtered_entities = [
            e for e in entities if not (e.description or e.historical_info or e.category)
        ]
        for entity in filtered_entities:
            logger.warning(
                "TypeAnalyzer: Entity filtered - '%s' - No description/features found",
                entity.display_name,
            )

        if len(entities_with_text) < 2:
            return self._create_result(
                task=task,
                assertions=[],
                metadata={"message": "Not enough entities with text for analysis"},
            )

        # Build URI -> entity mapping
        entities_by_uri = {e.uri: e for e in entities_with_text}

        # =================================================================
        # PHASE 1: Prepare LLM Context (10%)
        # =================================================================
        llm_context = self._prepare_llm_context(entities_with_text)
        logger.debug("TypeAnalyzer: Prepared LLM context")

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
                "TypeAnalyzer: LLM extracted features for %d sites, proposed %d clusters, %d pairs",
                len(llm_response.feature_extractions),
                len(llm_response.clusters),
                len(llm_response.similarity_pairs),
            )
        except Exception as e:
            # Propagate daily quota exhaustion - don't fallback
            from src.llm.provider import DailyQuotaExhaustedError

            if isinstance(e, DailyQuotaExhaustedError):
                raise
            logger.error("TypeAnalyzer: LLM proposal failed: %s", e)
            llm_response = TypologicalAnalysisResponse(
                feature_extractions=[],
                clusters=[],
                similarity_pairs=[],
                typological_observations="LLM analysis failed",
            )

        assertions = []

        # Map URI to feature set for validation
        feature_map: dict[str, set[str]] = {}
        for extraction in llm_response.feature_extractions:
            features = set()
            features.update(extraction.architectural_features)
            features.update(extraction.functional_features)
            features.update(extraction.contextual_features)
            feature_map[extraction.entity_uri] = features

        # Also fallback extract features for entities missed by LLM
        for uri, entity in entities_by_uri.items():
            if uri not in feature_map:
                feature_map[uri] = self._extract_features_fallback(entity)

        # =================================================================
        # PHASE 2b: Build Description Vectors for Semantic Similarity
        # =================================================================
        logger.info(
            "TypeAnalyzer: Building SpaCy embedding vectors for %d entities",
            len(entities_with_text),
        )
        description_vectors = build_description_vectors(entities_with_text)

        # Get similarity threshold for embeddings (can be different from Jaccard)
        embedding_similarity_threshold = task.params.get(
            "embedding_similarity_threshold",
            self.agent_config.get("embedding_similarity_threshold", 0.5),
        )

        # =================================================================
        # PHASE 3: Validate and Convert Cluster Proposals
        # =================================================================
        logger.info(
            "TypeAnalyzer: LLM proposed %d clusters, validating...", len(llm_response.clusters)
        )
        for proposal in llm_response.clusters:
            logger.info(
                "TypeAnalyzer: [LLM PROPOSAL] Cluster '%s' with %d members: %s",
                proposal.typology_label,
                len(proposal.member_uris),
                [uri.split("/")[-1] for uri in proposal.member_uris[:5]],  # Show first 5 URIs
            )
            logger.info(
                "TypeAnalyzer: [LLM PROPOSAL] Defining features: %s, LLM confidence: %.2f",
                proposal.defining_features[:5] if proposal.defining_features else [],
                proposal.llm_confidence,
            )

            validation = validate_typological_cluster(
                member_uris=proposal.member_uris,
                feature_extractions=feature_map,
                min_shared_features=1,
                majority_threshold=majority_threshold,
                min_cluster_size=min_cluster_size,
                max_cluster_size=max_cluster_size,
            )

            if not validation.is_valid:
                logger.warning(
                    "TypeAnalyzer: [REJECTED] Cluster '%s' - Reason: %s | Shared features: %s",
                    proposal.typology_label,
                    validation.details.get("reason", "validation failed"),
                    validation.details.get("shared_features", [])[:5],
                )
                continue

            # Calculate hybrid confidence
            confidence = calculate_cluster_confidence(
                llm_confidence=proposal.llm_confidence,
                validation_score=validation.validation_score,
                member_count=len(proposal.member_uris),
            )

            cluster = TypologicalCluster(
                assertion_id=f"type_cluster_{uuid.uuid4().hex[:8]}",
                label=f"Typological Cluster - {proposal.typology_label}",
                description=proposal.reasoning,
                subject_uris=proposal.member_uris,
                generated_by=self.agent_type,
                confidence_score=round(confidence, 2),
                reasoning=f"Shared features: {validation.details.get('shared_features', [])[:5]}",
                typology=proposal.typology_label,
                shared_features=validation.details.get("shared_features", []),
            )
            assertions.append(cluster)
            logger.info(
                "TypeAnalyzer: Accepted typological cluster '%s' with %d sites, Defining features: %s",
                proposal.typology_label,
                len(proposal.member_uris),
                list(cluster.shared_features)[:3],
            )

        # =================================================================
        # PHASE 3b: Validate and Convert Similarity Relations (using embeddings)
        # =================================================================
        logger.info(
            "TypeAnalyzer: LLM proposed %d similarity pairs, validating with SpaCy embeddings...",
            len(llm_response.similarity_pairs),
        )
        for proposal in llm_response.similarity_pairs:
            logger.info(
                "TypeAnalyzer: [LLM PROPOSAL] Similarity: %s <-> %s | Shared: %s | LLM Conf: %.2f",
                proposal.source_uri.split("/")[-1],
                proposal.target_uri.split("/")[-1],
                proposal.shared_features[:3] if proposal.shared_features else [],
                proposal.llm_confidence,
            )

            # Use new embedding-based validation with hybrid scoring
            is_valid, hybrid_score, details = validate_similarity_with_embeddings(
                source_uri=proposal.source_uri,
                target_uri=proposal.target_uri,
                description_vectors=description_vectors,
                min_similarity=embedding_similarity_threshold,
                llm_confidence=proposal.llm_confidence,
                llm_weight=0.7,  # 70% LLM, 30% cosine similarity
            )

            if not is_valid:
                logger.warning(
                    "TypeAnalyzer: [REJECTED] Similarity %s <-> %s - Cosine: %.3f, Hybrid: %.3f - %s",
                    proposal.source_uri.split("/")[-1],
                    proposal.target_uri.split("/")[-1],
                    details.get("cosine_similarity", 0),
                    details.get("hybrid_score", 0),
                    details.get("reason", "validation failed"),
                )
                continue

            source_name = getattr(
                entities_by_uri.get(proposal.source_uri), "display_name", "Unknown"
            )
            target_name = getattr(
                entities_by_uri.get(proposal.target_uri), "display_name", "Unknown"
            )

            relation = SiteRelation(
                assertion_id=f"similar_{uuid.uuid4().hex[:8]}",
                assertion_type=AssertionType.SIMILAR_TO,
                label=f"{source_name} similar to {target_name}",
                description=f"Cosine similarity: {details.get('cosine_similarity', 0):.3f}, Hybrid: {hybrid_score:.3f}",
                source_uri=proposal.source_uri,
                target_uri=proposal.target_uri,
                subject_uris=[proposal.source_uri],
                object_uris=[proposal.target_uri],
                relation_value=details.get("cosine_similarity", 0),
                generated_by=self.agent_type,
                confidence_score=round(hybrid_score, 2),
                reasoning=proposal.reasoning,
            )
            assertions.append(relation)
            logger.info(
                "TypeAnalyzer: [ACCEPTED] Similarity: %s <-> %s | Cosine: %.3f | Hybrid: %.3f",
                source_name,
                target_name,
                details.get("cosine_similarity", 0),
                hybrid_score,
            )

        logger.info("TypeAnalyzer: Generated %d total assertions from LLM", len(assertions))

        # =================================================================
        # PHASE 4: Materialise ExtractedFeature assertions
        # =================================================================
        # Only materialise features that appear in ≥2 entities (cluster-relevant)
        # Labels are normalised (lowercase + lemma) and deduplicated by similarity
        feature_entities: dict[tuple[str, str], list[tuple[str, str]]] = (
            {}
        )  # (category, normalised_label) -> [(uri, name)]
        for extraction in llm_response.feature_extractions:
            uri = extraction.entity_uri
            name = getattr(entities_by_uri.get(uri), "display_name", "")
            for feat in extraction.architectural_features:
                key = ("architectural", normalize_feature_label(feat))
                feature_entities.setdefault(key, []).append((uri, name))
            for feat in extraction.functional_features:
                key = ("functional", normalize_feature_label(feat))
                feature_entities.setdefault(key, []).append((uri, name))
            for feat in extraction.contextual_features:
                key = ("contextual", normalize_feature_label(feat))
                feature_entities.setdefault(key, []).append((uri, name))
            for feat in extraction.material_features:
                key = ("material", normalize_feature_label(feat))
                feature_entities.setdefault(key, []).append((uri, name))

        # Similarity-based deduplication within each category
        feature_entities = deduplicate_feature_labels(feature_entities)

        total_entities_analysed = len(entities_by_uri)
        feature_count = 0
        for (category, label), entity_list in feature_entities.items():
            if len(entity_list) < 2:
                continue  # Skip features unique to a single entity
            confidence = calculate_feature_confidence(
                entity_count=len(entity_list),
                total_entities=total_entities_analysed,
                category=category,
            )
            for entity_uri, entity_name in entity_list:
                feat_assertion = ExtractedFeatureAssertion(
                    assertion_id=f"feature_{uuid.uuid4().hex[:8]}",
                    label=f"{label} ({category})",
                    description=f"Extracted {category} feature from {entity_name or entity_uri}",
                    subject_uris=[entity_uri],
                    generated_by=self.agent_type,
                    confidence_score=confidence,
                    reasoning=f"LLM-extracted {category} feature shared by {len(entity_list)} entities",
                    feature_label=label,
                    feature_category=category,
                    entity_uri=entity_uri,
                    entity_name=entity_name,
                )
                assertions.append(feat_assertion)
                feature_count += 1

        logger.info(
            "TypeAnalyzer: Materialised %d ExtractedFeature assertions (from %d shared features)",
            feature_count,
            sum(1 for v in feature_entities.values() if len(v) >= 2),
        )

        # ─────────────────────────────────────────────────────────────
        # FALLBACK: If LLM produced nothing valid, use algorithmic approach
        # ─────────────────────────────────────────────────────────────
        if len(assertions) == 0 and len(entities_with_text) >= 2:
            logger.warning(
                "TypeAnalyzer: No LLM proposals accepted, activating algorithmic fallback"
            )

            # Fallback B: Category-based clusters
            fallback_clusters = self._algorithmic_fallback(
                entities=entities_with_text,
                feature_map=feature_map,
                entities_by_uri=entities_by_uri,
                min_cluster_size=2,
                max_cluster_size=6,
            )
            assertions.extend(fallback_clusters)
            logger.info(
                "TypeAnalyzer: [FALLBACK] Created %d category-based clusters",
                len(fallback_clusters),
            )

            # Fallback C: Jaccard similarity pairs
            fallback_pairs = self._generate_similarity_fallback(
                feature_map=feature_map,
                entities_by_uri=entities_by_uri,
                min_jaccard=0.3,
                max_pairs=20,
            )
            assertions.extend(fallback_pairs)
            logger.info(
                "TypeAnalyzer: [FALLBACK] Created %d Jaccard similarity pairs",
                len(fallback_pairs),
            )

            logger.info(
                "TypeAnalyzer: Fallback produced %d total assertions",
                len(assertions),
            )

        return self._create_result(
            task=task,
            assertions=assertions,
            metadata={
                "entities_analyzed": len(entities_with_text),
                "features_extracted": len(feature_map),
                "llm_clusters_proposed": len(llm_response.clusters),
                "llm_pairs_proposed": len(llm_response.similarity_pairs),
                "assertions_accepted": len(assertions),
                "typological_observations": llm_response.typological_observations,
                "fallback_used": len(assertions) > 0
                and len(llm_response.clusters) + len(llm_response.similarity_pairs) == 0,
            },
        )

    def _prepare_llm_context(self, entities: list) -> dict[str, Any]:
        """Prepare entity data for LLM consumption."""
        sites = []
        for e in entities:
            site = {
                "uri": e.uri,
                "name": e.display_name,
                "category": e.category or "Unknown",
                "context": e.context_type or "Unknown",
            }
            if e.description:
                site["description"] = e.description[:500]  # First 500 chars
            if e.historical_info:
                site["historical_info"] = e.historical_info[:300]
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
    ) -> TypologicalAnalysisResponse:
        """Call LLM to perform typological analysis.

        Args:
            context: LLM context with site data
            max_clusters: Optional maximum number of clusters (None = no limit)
            max_pairs: Optional maximum number of similarity pairs (None = no limit)
            min_cluster_size: Optional minimum members per cluster (None = no limit)
            max_cluster_size: Optional maximum members per cluster (None = no limit)
        """
        sites_json = json.dumps(context["sites"], indent=2, ensure_ascii=False)

        system_prompt = PromptLibrary.TYPE_ANALYZER_SYSTEM_V2
        human_prompt = PromptLibrary.TYPE_ANALYZER_TASK_V2

        response = self.provider.invoke_structured(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            output_schema=TypologicalAnalysisResponse,
            variables={
                "total_entities": context["total_entities"],
                "sites_json": sites_json,
                "max_clusters": max_clusters or 10,
                "max_pairs": max_pairs or 20,
                "max_clusters_hint": PromptLibrary.get_max_clusters_hint(max_clusters),
                "max_pairs_hint": PromptLibrary.get_max_pairs_hint(max_pairs),
                "cluster_size_hint": PromptLibrary.get_cluster_size_hint(
                    min_cluster_size, max_cluster_size
                ),
            },
        )

        return response

    def _extract_features_fallback(self, entity) -> set[str]:
        """Extract features using keyword matching (fallback)."""
        text = (entity.description or "") + " " + (entity.historical_info or "")
        text_lower = text.lower()
        features = set()

        for kw in ARCHITECTURAL_FEATURES_FALLBACK:
            if kw in text_lower:
                features.add(kw)

        for cat, kws in FUNCTIONAL_CATEGORIES_FALLBACK.items():
            if any(k in text_lower for k in kws):
                features.add(f"cat:{cat}")

        return features

    def _algorithmic_fallback(
        self,
        entities: list,
        feature_map: dict[str, set[str]],
        entities_by_uri: dict,
        min_cluster_size: int = 2,
        max_cluster_size: int = 6,
    ) -> list:
        """
        Fallback: create typological clusters based on functional categories.

        Groups sites by their functional category (funerary, cultic, domestic).
        """
        assertions = []

        # Group by functional category
        category_to_uris: dict[str, list[str]] = {}
        for uri, features in feature_map.items():
            for feat in features:
                if feat.startswith("cat:"):
                    cat = feat.replace("cat:", "")
                    if cat not in category_to_uris:
                        category_to_uris[cat] = []
                    category_to_uris[cat].append(uri)

        # Create cluster for each category with enough members
        for category, member_uris in category_to_uris.items():
            if len(member_uris) < min_cluster_size:
                continue

            # Limit to max_cluster_size
            cluster_uris = member_uris[:max_cluster_size]

            cluster = TypologicalCluster(
                assertion_id=f"type_cluster_{uuid.uuid4().hex[:8]}",
                label=f"{category.title()} Sites",
                description=f"Algorithmic cluster of {len(cluster_uris)} sites with {category} function",
                subject_uris=cluster_uris,
                generated_by=self.agent_type,
                confidence_score=0.5,  # Fixed for fallback
                reasoning=f"Algorithmic fallback: sites identified as {category} based on keyword matching",
                defining_features=[f"function:{category}"],
                site_count=len(cluster_uris),
            )
            assertions.append(cluster)
            logger.info(
                "TypeAnalyzer: [FALLBACK] Created cluster '%s' with %d sites",
                cluster.label,
                len(cluster_uris),
            )

        return assertions

    def _generate_similarity_fallback(
        self,
        feature_map: dict[str, set[str]],
        entities_by_uri: dict,
        min_jaccard: float = 0.3,
        max_pairs: int = 20,
    ) -> list:
        """
        Fallback: create similarity pairs based on Jaccard similarity of features.

        Computes Jaccard index = |intersection| / |union| between feature sets.
        """
        assertions = []
        uris = [uri for uri, feats in feature_map.items() if len(feats) >= 2]

        pairs_created = 0
        for i, uri1 in enumerate(uris):
            if pairs_created >= max_pairs:
                break
            for uri2 in uris[i + 1 :]:
                if pairs_created >= max_pairs:
                    break

                f1, f2 = feature_map[uri1], feature_map[uri2]

                # Jaccard similarity
                intersection = len(f1 & f2)
                union = len(f1 | f2)
                jaccard = intersection / union if union > 0 else 0

                if jaccard >= min_jaccard:
                    shared = list(f1 & f2)
                    source_name = getattr(entities_by_uri.get(uri1), "display_name", "Unknown")
                    target_name = getattr(entities_by_uri.get(uri2), "display_name", "Unknown")

                    relation = SiteRelation(
                        assertion_id=f"similar_{uuid.uuid4().hex[:8]}",
                        assertion_type=AssertionType.SIMILAR_TO,
                        label=f"{source_name} similar to {target_name}",
                        description=f"Jaccard similarity: {jaccard:.3f}, shared features: {shared}",
                        source_uri=uri1,
                        target_uri=uri2,
                        subject_uris=[uri1],
                        object_uris=[uri2],
                        relation_value=jaccard,
                        generated_by=self.agent_type,
                        confidence_score=0.5,  # Fixed for fallback
                        reasoning=f"Algorithmic fallback: Jaccard={jaccard:.2f}, shared: {shared}",
                    )
                    assertions.append(relation)
                    pairs_created += 1
                    logger.info(
                        "TypeAnalyzer: [FALLBACK] Similarity: %s <-> %s | Jaccard: %.3f | Shared: %s",
                        source_name,
                        target_name,
                        jaccard,
                        shared[:3],
                    )

        return assertions
