"""
XCH Pipeline - Complete integration pipeline for xch-CORE.

Orchestrates the entire flow: entity loading → multi-agent workflow →
triple generation → validation → serialization.
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rdflib import Graph

from src.agents.models import InterpretiveAssertion
from src.config.settings import Settings, get_settings
from src.loaders import DolmenEntity, load_all_entities
from src.triples import TripleGenerator, TripleSerializer, TripleValidator
from src.utils.logging import add_file_handler, remove_file_handler
from src.utils.rate_limiter import RateLimiter
from src.workflow import DolmenWorkflow, create_initial_state

logger = logging.getLogger(__name__)


# =============================================================================
# PIPELINE RESULT
# =============================================================================


@dataclass
class PipelineResult:
    """Result from a complete pipeline execution."""

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    duration_seconds: float = 0.0

    # Entities
    entities_loaded: int = 0
    entities_processed: int = 0
    entities_failed: list[str] = field(default_factory=list)
    entities_filtered: dict[str, list[str]] = field(
        default_factory=dict
    )  # reason -> [entity_names]

    # Assertions
    assertions_generated: int = 0
    assertions_by_type: dict[str, int] = field(default_factory=dict)

    # Triples
    triples_generated: int = 0

    # Validation
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)

    # Output
    output_files: list[str] = field(default_factory=list)

    # Novelty filtering stats
    novelty_stats: dict[str, dict[str, int]] = field(default_factory=dict)

    # Stats
    llm_calls: int = 0
    rate_limit_waits: int = 0

    def finalize(self) -> None:
        """Mark pipeline as complete and calculate duration."""
        self.completed_at = datetime.now()
        self.duration_seconds = (self.completed_at - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timing": {
                "started_at": self.started_at.isoformat(),
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "duration_seconds": round(self.duration_seconds, 2),
            },
            "entities": {
                "loaded": self.entities_loaded,
                "processed": self.entities_processed,
                "failed": self.entities_failed,
                "filtered": self.entities_filtered,
            },
            "assertions": {
                "total": self.assertions_generated,
                "by_type": self.assertions_by_type,
            },
            "triples": {
                "total": self.triples_generated,
            },
            "validation": {
                "errors": len(self.validation_errors),
                "warnings": len(self.validation_warnings),
                "error_details": self.validation_errors[:10],  # First 10
                "warning_details": self.validation_warnings[:10],
            },
            "output": {
                "files": self.output_files,
            },
            "novelty_filtering": self.novelty_stats,
            "stats": {
                "llm_calls": self.llm_calls,
                "rate_limit_waits": self.rate_limit_waits,
            },
        }

    def print_summary(self) -> None:
        """Print a summary of the pipeline execution."""
        print("\n" + "=" * 60)
        print("📊 PIPELINE EXECUTION SUMMARY")
        print("=" * 60)

        print(f"\n⏱️  Duration: {self.duration_seconds:.2f}s")
        print(f"📁 Entities: {self.entities_processed}/{self.entities_loaded} processed")

        if self.entities_failed:
            print(f"   ⚠️  Failed: {len(self.entities_failed)}")

        if self.entities_filtered:
            total_filtered = sum(len(v) for v in self.entities_filtered.values())
            print(f"   ⏭️  Filtered (analysis stage): {total_filtered}")
            for reason, entities in self.entities_filtered.items():
                print(f"      • {reason}: {len(entities)}")

        print(f"\n📝 Assertions: {self.assertions_generated} generated")
        if self.assertions_by_type:
            for atype, count in sorted(self.assertions_by_type.items()):
                print(f"   • {atype}: {count}")

        print(f"\n🔗 Triples: {self.triples_generated} generated")

        if self.validation_errors:
            print(f"\n❌ Validation Errors: {len(self.validation_errors)}")
        if self.validation_warnings:
            print(f"⚠️  Validation Warnings: {len(self.validation_warnings)}")

        print(f"\n📤 Output Files: {len(self.output_files)}")
        for f in self.output_files:
            print(f"   • {f}")

        print(f"\n🤖 LLM Calls: {self.llm_calls}")
        if self.rate_limit_waits:
            print(f"   Rate limit waits: {self.rate_limit_waits}")

        print("=" * 60)


# =============================================================================
# PIPELINE
# =============================================================================


class Pipeline:
    """
    xch-MIND Pipeline

    Orchestrates:
    1. Loading dolmen entities from XML
    2. Running multi-agent LangGraph workflow
    3. Generating RDF triples from assertions
    4. Validating output
    5. Serializing to Turtle/JSON-LD

    Usage:
        pipeline = XCHPipeline()
        result = pipeline.execute(limit=5, dry_run=True)
        result.print_summary()
    """

    def __init__(
        self,
        settings: Settings | None = None,
        entities_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
    ):
        """
        Initialize the pipeline.

        Args:
            settings: Configuration settings (uses default if None)
            entities_dir: Override directory for entity XML files
            output_dir: Override directory for output files
        """
        self.settings = settings or get_settings()

        # Directories
        self.entities_dir = Path(entities_dir) if entities_dir else self.settings.paths.entities_dir
        self.output_dir = Path(output_dir) if output_dir else self.settings.paths.output_dir

        # Rate limiter based on configuration
        # - Enabled for Gemini with rate_limiting.enabled=True (Free Tier)
        # - Disabled for Ollama (local, no limits) or paid tiers
        rate_cfg = self.settings.llm.rate_limiting
        if self.settings.llm.provider == "gemini" and rate_cfg.enabled:
            self.rate_limiter = RateLimiter(
                requests_per_minute=rate_cfg.requests_per_minute,
                requests_per_day=rate_cfg.requests_per_day,
                max_retries=rate_cfg.max_retries,
                base_delay=rate_cfg.base_delay,
                state_path=str(self.output_dir / ".rate_limit.json"),
            )
            logger.info(
                "Rate limiter enabled: %d RPM, %d RPD",
                rate_cfg.requests_per_minute,
                rate_cfg.requests_per_day,
            )
        else:
            self.rate_limiter = None
            if self.settings.llm.provider == "gemini":
                logger.info("Rate limiter disabled (paid tier or manual override)")
            else:
                logger.info("Rate limiter disabled (local Ollama provider)")

        # Components (lazy initialization)
        self._workflow: DolmenWorkflow | None = None
        self._triple_generator: TripleGenerator | None = None
        self._serializer: TripleSerializer | None = None
        self._validator: TripleValidator | None = None

        # State
        self._entities: list[DolmenEntity] = []
        self._assertions: list[InterpretiveAssertion] = []
        self._raw_assertion_dicts: list[dict] = []  # Raw dicts preserving subclass fields
        self._graph: Graph | None = None

        logger.info("Pipeline initialized")
        logger.info("  Entities dir: %s", self.entities_dir)
        logger.info("  Output dir: %s", self.output_dir)

    # =========================================================================
    # COMPONENT INITIALIZATION
    # =========================================================================

    def _get_config_dict(self) -> dict[str, Any]:
        """Convert settings to config dict for workflow."""
        return {
            "llm": {
                "provider": self.settings.llm.provider,
                "gemini": {
                    "model": self.settings.llm.gemini.model,
                    "temperature": self.settings.llm.gemini.temperature,
                    "max_tokens": self.settings.llm.gemini.max_tokens,
                },
                "ollama": {
                    "model": self.settings.llm.ollama.model,
                    "base_url": self.settings.llm.ollama.base_url,
                    "temperature": self.settings.llm.ollama.temperature,
                    "max_tokens": self.settings.llm.ollama.max_tokens,
                },
            },
            "agents": {
                "orchestrator": {
                    "temperature": self.settings.agents.orchestrator.temperature,
                },
                "geospatial": {
                    "enabled": self.settings.agents.geospatial.enabled,
                    "temperature": self.settings.agents.geospatial.temperature,
                    "max_distance_km": self.settings.agents.geospatial.max_distance_km,
                    "clustering_threshold": self.settings.agents.geospatial.clustering_threshold,
                    "min_cluster_size": self.settings.agents.geospatial.min_cluster_size or 2,
                    "max_cluster_size": self.settings.agents.geospatial.max_cluster_size or 6,
                    "max_clusters": self.settings.agents.geospatial.max_clusters or 15,
                    "max_pairs": self.settings.agents.geospatial.max_pairs or 20,
                },
                "chronological": {
                    "enabled": self.settings.agents.chronological.enabled,
                    "temperature": self.settings.agents.chronological.temperature,
                    "period_tolerance_years": self.settings.agents.chronological.period_tolerance_years,
                    "min_cluster_size": self.settings.agents.chronological.min_cluster_size or 2,
                    "max_cluster_size": self.settings.agents.chronological.max_cluster_size or 6,
                    "max_clusters": self.settings.agents.chronological.max_clusters or 10,
                    "max_pairs": self.settings.agents.chronological.max_pairs or 15,
                },
                "typological": {
                    "enabled": self.settings.agents.typological.enabled,
                    "temperature": self.settings.agents.typological.temperature,
                    "similarity_threshold": self.settings.agents.typological.similarity_threshold,
                    "min_cluster_size": self.settings.agents.typological.min_cluster_size or 2,
                    "max_cluster_size": self.settings.agents.typological.max_cluster_size or 6,
                    "max_clusters": self.settings.agents.typological.max_clusters or 10,
                    "max_pairs": self.settings.agents.typological.max_pairs or 20,
                    "embedding_similarity_threshold": self.settings.agents.typological.embedding_similarity_threshold
                    or 0.7,
                    "majority_threshold": self.settings.agents.typological.majority_threshold
                    or 0.6,
                },
                "narrative": {
                    "enabled": self.settings.agents.narrative.enabled,
                    "temperature": self.settings.agents.narrative.temperature,
                    "max_path_length": self.settings.agents.narrative.max_path_length,
                    "min_path_length": self.settings.agents.narrative.min_path_length,
                    "max_paths": self.settings.agents.narrative.max_paths or 5,
                    "min_stops_per_path": self.settings.agents.narrative.min_stops_per_path or 3,
                    "max_stops_per_path": self.settings.agents.narrative.max_stops_per_path or 8,
                },
            },
            "workflow": {
                "max_iterations": self.settings.workflow.max_iterations,
                "mode": self.settings.workflow.mode,
                "min_entity_coverage": self.settings.workflow.min_entity_coverage,
                "min_diversity": self.settings.workflow.min_diversity,
            },
            "output": {
                "format": self.settings.output.format,
                "include_provenance": self.settings.output.include_provenance,
                "include_confidence_scores": self.settings.output.include_confidence_scores,
            },
        }

    @property
    def triple_generator(self) -> TripleGenerator:
        """Get or create triple generator."""
        if self._triple_generator is None:
            self._triple_generator = TripleGenerator()
        return self._triple_generator

    def get_quota_info(self) -> dict[str, Any] | None:
        """Get current API quota usage information.

        Returns None if rate limiting is not enabled (e.g., Ollama provider).
        """
        if self.rate_limiter is None:
            return None
        return self.rate_limiter.get_quota_info()

    @property
    def serializer(self) -> TripleSerializer:
        """Get or create serializer."""
        if self._serializer is None:
            self._serializer = TripleSerializer()
        return self._serializer

    @property
    def validator(self) -> TripleValidator:
        """Get or create validator."""
        if self._validator is None:
            shapes_path = Path(self.settings.paths.xch_ontology).parent / "xch-shapes.ttl"
            ontology_path = Path(self.settings.paths.xch_ontology)
            self._validator = TripleValidator(
                shapes_path=shapes_path if shapes_path.exists() else None,
                ontology_path=ontology_path if ontology_path.exists() else None,
            )
        return self._validator

    # =========================================================================
    # PIPELINE STAGES
    # =========================================================================

    def load_entities(self, limit: int | None = None, progress_callback=None) -> list[DolmenEntity]:
        """
        Load dolmen entities from XML files.

        Args:
            limit: Maximum number of entities to load (None = all)
            progress_callback: Optional callback for progress updates

        Returns:
            List of loaded DolmenEntity objects
        """
        logger.info("Loading entities from %s", self.entities_dir)

        # Load entities with limit optimization
        self._entities = load_all_entities(
            str(self.entities_dir), progress_callback=progress_callback, limit=limit
        )

        if limit is not None and limit > 0:
            # Re-slice just in case, though load_all_entities handles it now
            self._entities = self._entities[:limit]
            logger.info("Limited to %d entities", limit)

        logger.info("Loaded %d entities", len(self._entities))

        return self._entities

    def run_workflow(
        self,
        entities: list[DolmenEntity] | None = None,
        max_iterations: int | None = None,
        dry_run: bool = False,
        progress_callback=None,
    ) -> list[InterpretiveAssertion]:
        """
        Run the multi-agent workflow on entities.

        Args:
            entities: Entities to process (uses loaded entities if None)
            max_iterations: Override max iterations
            dry_run: If True, simulate without LLM calls
            progress_callback: Optional callback for progress updates

        Returns:
            List of generated InterpretiveAssertions
        """
        entities = entities or self._entities
        if not entities:
            raise ValueError("No entities to process. Call load_entities() first.")

        max_iter = max_iterations or self.settings.workflow.max_iterations

        logger.info(
            "Running workflow on %d entities (max_iter=%d, dry_run=%s)",
            len(entities),
            max_iter,
            dry_run,
        )

        if dry_run:
            # Simulate workflow without LLM calls
            self._assertions = self._simulate_workflow(entities)
        else:
            # Real workflow execution
            self._assertions = self._execute_workflow(entities, max_iter, progress_callback)

        logger.info("Generated %d assertions", len(self._assertions))

        return self._assertions

    def _execute_workflow(
        self,
        entities: list[DolmenEntity],
        max_iterations: int,
        progress_callback=None,
    ) -> list[InterpretiveAssertion]:
        """Execute real workflow with LLM calls."""
        config = self._get_config_dict()

        # Initialize workflow with memory support
        workflow = DolmenWorkflow(config)
        workflow.prepare(
            entities,
            rate_limiter=self.rate_limiter,
            output_dir=str(self.output_dir),  # Enable cross-run memory
        )

        # Run with rate limiting consideration
        final_state = None

        # Get external run_id from pipeline (set in execute())
        run_id = getattr(self, "_current_run_id", "unknown_run")

        if progress_callback:
            # Use streaming for progress updates
            for state in workflow.stream(
                max_iterations=max_iterations,
                min_assertions=5,
                run_id=run_id,
            ):
                final_state = state
                progress_callback(state)
        else:
            final_state = workflow.run(
                max_iterations=max_iterations,
                min_assertions=5,
                run_id=run_id,
            )

        # Create assertions if None (should not happen if stream worked)
        if final_state is None:
            # Just in case stream yielded nothing
            logger.warning("Workflow stream yielded no state, falling back to run()")
            final_state = workflow.run(
                max_iterations=max_iterations, min_assertions=5, run_id=run_id
            )

        # Extract novelty filtering stats from state
        self._novelty_stats = final_state.get("novelty_stats", {})

        # Extract assertions from state
        # Store raw dicts to preserve subclass-specific fields (path_type, theme, stops, etc.)
        raw_dicts = final_state.get("assertions", [])
        self._raw_assertion_dicts = raw_dicts

        assertions = []
        for a_dict in raw_dicts:
            try:
                assertion = InterpretiveAssertion(**a_dict)
                assertions.append(assertion)
            except Exception as e:
                logger.warning("Failed to parse assertion: %s", e)

        return assertions

    def _simulate_workflow(
        self,
        entities: list[DolmenEntity],
    ) -> list[InterpretiveAssertion]:
        """Simulate workflow for dry-run mode."""
        from collections import defaultdict

        from src.agents.models import AgentType, AssertionType
        from src.loaders import get_region_name

        logger.info("DRY RUN: Simulating workflow...")

        assertions = []

        # Group entities by region for realistic geographic clustering
        by_region: dict[str, list[DolmenEntity]] = defaultdict(list)
        by_period: dict[str, list[DolmenEntity]] = defaultdict(list)

        for entity in entities:
            region_id = getattr(entity, "region_id", None)
            region = get_region_name(region_id) if region_id else "Unknown"
            by_region[region or "Unknown"].append(entity)

            period = getattr(entity, "period_label", None) or "Unknown Period"
            by_period[period].append(entity)

        # Create one GeographicCluster per region (with at least 2 members)
        for region, region_entities in by_region.items():
            if len(region_entities) >= 2:  # Only clusters with 2+ members
                cluster_uris = [e.uri for e in region_entities]
                region_safe = (region or "unknown").lower().replace(" ", "_")

                assertions.append(
                    InterpretiveAssertion(
                        assertion_id=f"sim_geo_cluster_{region_safe}",
                        assertion_type=AssertionType.GEOGRAPHIC_CLUSTER,
                        label=f"Cluster geografico: Dolmen della {region}",
                        description=f"[DRY RUN] Raggruppamento di {len(region_entities)} dolmen nella regione {region}",
                        subject_uris=cluster_uris,
                        object_uris=[],
                        confidence_score=0.85,
                        reasoning=f"Dolmen raggruppati per prossimità geografica nella regione {region}",
                        generated_by=AgentType.GEO_ANALYZER,
                        created_at=datetime.now(),
                    )
                )

        # Create near_to relations between consecutive entities in the same region
        for region, region_entities in by_region.items():
            if len(region_entities) < 2:
                continue
            region_safe = (region or "unknown").lower().replace(" ", "_")
            for i, entity in enumerate(region_entities[:-1]):
                other = region_entities[i + 1]
                assertions.append(
                    InterpretiveAssertion(
                        assertion_id=f"sim_near_{region_safe}_{i}",
                        assertion_type=AssertionType.NEAR_TO,
                        label=f"{entity.label[:30]} vicino a {other.label[:30]}",
                        description=f"[DRY RUN] Relazione di prossimità nella regione {region}",
                        subject_uris=[entity.uri],
                        object_uris=[other.uri],
                        confidence_score=0.75,
                        reasoning=f"Entrambi i dolmen si trovano nella regione {region}",
                        generated_by=AgentType.GEO_ANALYZER,
                        created_at=datetime.now(),
                    )
                )

        # Create chronological clusters per period (with at least 2 members)
        for period, period_entities in by_period.items():
            if len(period_entities) >= 2:
                cluster_uris = [e.uri for e in period_entities]
                period_safe = (period or "unknown").lower().replace(" ", "_")[:30]

                assertions.append(
                    InterpretiveAssertion(
                        assertion_id=f"sim_chrono_cluster_{period_safe}",
                        assertion_type=AssertionType.CHRONOLOGICAL_CLUSTER,
                        label=f"Cluster cronologico: {period}",
                        description=f"[DRY RUN] Raggruppamento di {len(period_entities)} dolmen nel periodo {period}",
                        subject_uris=cluster_uris,
                        object_uris=[],
                        confidence_score=0.80,
                        reasoning=f"Dolmen datati allo stesso periodo storico: {period}",
                        generated_by=AgentType.TEMPORAL_ANALYZER,
                        created_at=datetime.now(),
                    )
                )

        # Create contemporary_with relations between entities in same period
        for period, period_entities in by_period.items():
            if len(period_entities) < 2:
                continue
            period_safe = (period or "unknown").lower().replace(" ", "_")[:30]
            for i, entity in enumerate(period_entities[:-1]):
                other = period_entities[i + 1]
                assertions.append(
                    InterpretiveAssertion(
                        assertion_id=f"sim_contemporary_{period_safe}_{i}",
                        assertion_type=AssertionType.CONTEMPORARY_WITH,
                        label=f"{entity.label[:20]} contemporaneo a {other.label[:20]}",
                        description=f"[DRY RUN] Relazione cronologica nel periodo {period}",
                        subject_uris=[entity.uri],
                        object_uris=[other.uri],
                        confidence_score=0.72,
                        reasoning=f"Entrambi i dolmen datati al periodo {period}",
                        generated_by=AgentType.TEMPORAL_ANALYZER,
                        created_at=datetime.now(),
                    )
                )

        logger.info("DRY RUN: Generated %d simulated assertions", len(assertions))

        return assertions

    def generate_triples(
        self,
        assertions: list[InterpretiveAssertion] | list[dict] | None = None,
    ) -> Graph:
        """
        Generate RDF triples from assertions.

        Args:
            assertions: Assertions to convert (uses workflow output if None)

        Returns:
            RDFLib Graph with generated triples
        """
        # Prefer raw dicts which preserve subclass-specific fields (path_type, theme, etc.)
        # that are lost when reconstructing as base InterpretiveAssertion
        assertions = assertions or self._raw_assertion_dicts or self._assertions
        if not assertions:
            raise ValueError("No assertions to convert. Run workflow first.")

        logger.info("Generating triples from %d assertions", len(assertions))

        self._graph = self.triple_generator.generate(assertions)

        triple_count = len(self._graph)
        logger.info("Generated %d triples", triple_count)

        return self._graph

    def validate(
        self,
        graph: Graph | None = None,
    ) -> tuple[list[str], list[str]]:
        """
        Validate generated triples.

        Args:
            graph: Graph to validate (uses generated graph if None)

        Returns:
            Tuple of (errors, warnings)
        """
        graph = graph or self._graph
        if graph is None:
            raise ValueError("No graph to validate. Generate triples first.")

        logger.info("Validating graph with %d triples", len(graph))

        result = self.validator.validate(graph)

        logger.info(
            "Validation complete: %d errors, %d warnings", len(result.errors), len(result.warnings)
        )

        return result.errors, result.warnings

    def serialize(
        self,
        graph: Graph | None = None,
        output_format: str | None = None,
        output_path: str | Path | None = None,
    ) -> str:
        """
        Serialize graph to file.

        Args:
            graph: Graph to serialize (uses generated graph if None)
            output_format: Output format (turtle, json-ld, xml)
            output_path: Override output file path

        Returns:
            Path to output file
        """
        graph = graph or self._graph
        if graph is None:
            raise ValueError("No graph to serialize. Generate triples first.")

        fmt = output_format or self.settings.output.format

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = {"turtle": ".ttl", "json-ld": ".jsonld", "xml": ".rdf"}.get(fmt, ".ttl")

        if output_path:
            filepath = Path(output_path)
        else:
            filepath = self.output_dir / f"xch_output_{timestamp}{ext}"

        logger.info("Serializing to %s (format: %s)", filepath, fmt)

        self.serializer.to_file(graph, str(filepath), format=fmt)

        return str(filepath)

    def _generate_review_queue(self, run_dir: Path) -> str | None:
        """
        Extract assertions with PendingReview status into a JSON file for manual review.

        Queries the generated RDF graph for resources with
        xch:hasValidationStatus xch:PendingReview and exports their
        metadata to a structured JSON file.

        Args:
            run_dir: Directory for this pipeline run

        Returns:
            Path to review_queue.json, or None if no items to review
        """
        query = """
        PREFIX xch: <https://w3id.org/xch-mind/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dc: <http://purl.org/dc/elements/1.1/>

        SELECT ?assertion ?label ?description ?confidence ?agent ?type
        WHERE {
            ?assertion xch:hasValidationStatus xch:PendingReview .
            OPTIONAL { ?assertion rdfs:label ?label }
            OPTIONAL { ?assertion dc:description ?description }
            OPTIONAL { ?assertion xch:confidenceScore ?confidence }
            OPTIONAL { ?assertion xch:generatedBy ?agent }
            OPTIONAL { ?assertion a ?type . FILTER(?type != rdfs:Resource) }
        }
        ORDER BY ?confidence
        """

        results = list(self._graph.query(query))

        if not results:
            logger.info("No assertions with PendingReview status — skipping review queue")
            return None

        review_items = []
        for row in results:
            item = {
                "assertion_uri": str(row.assertion),
                "label": str(row.label) if row.label else None,
                "description": str(row.description) if row.description else None,
                "confidence_score": float(row.confidence) if row.confidence else None,
                "generated_by": str(row.agent).split("/")[-1] if row.agent else None,
                "rdf_type": str(row.type).split("/")[-1] if row.type else None,
                "status": "pending_review",
            }
            review_items.append(item)

        # Deduplicate by assertion URI (SPARQL may return multiple rows per assertion
        # if it has multiple rdf:type values)
        seen = set()
        unique_items = []
        for item in review_items:
            uri = item["assertion_uri"]
            if uri not in seen:
                seen.add(uri)
                unique_items.append(item)

        review_queue = {
            "generated_at": datetime.now().isoformat(),
            "total_items": len(unique_items),
            "confidence_range": {
                "min": min(
                    (i["confidence_score"] for i in unique_items if i["confidence_score"]),
                    default=None,
                ),
                "max": max(
                    (i["confidence_score"] for i in unique_items if i["confidence_score"]),
                    default=None,
                ),
            },
            "items": unique_items,
        }

        review_path = run_dir / "review_queue.json"
        with open(review_path, "w", encoding="utf-8") as f:
            json.dump(review_queue, f, indent=2, ensure_ascii=False)

        logger.info(
            "Generated review queue: %d assertions pending review → %s",
            len(unique_items),
            review_path,
        )
        return str(review_path)

    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================

    def _create_run_directory(self) -> Path:
        """
        Create a unique directory for this pipeline run.

        Returns:
            Path to the run directory (e.g., output/xch_run_20260203_120000/)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"xch_run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _save_metadata(self, run_dir: Path, result: "PipelineResult") -> str:
        """
        Save execution metadata to JSON file.

        Args:
            run_dir: Directory for this run
            result: PipelineResult with execution stats

        Returns:
            Path to metadata.json
        """
        metadata_path = run_dir / "metadata.json"
        metadata = result.to_dict()

        # Add additional context
        metadata["run_info"] = {
            "run_directory": str(run_dir),
            "provider": self.settings.llm.provider,
            "model": (
                self.settings.llm.gemini.model
                if self.settings.llm.provider == "gemini"
                else self.settings.llm.ollama.model
            ),
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        logger.info("Saved metadata to %s", metadata_path)
        return str(metadata_path)

    def execute(
        self,
        limit: int | None = None,
        dry_run: bool = False,
        output_format: str | None = None,
        skip_validation: bool = False,
        progress_callback: Any | None = None,
        enable_file_logging: bool = True,
    ) -> PipelineResult:
        """
        Execute the complete pipeline.

        Args:
            limit: Maximum entities to process (None = all)
            dry_run: Simulate without LLM calls
            output_format: Override output format
            skip_validation: Skip validation step
            progress_callback: Optional callback for progress updates
            enable_file_logging: Save execution log to file (default: True)

        Returns:
            PipelineResult with execution summary
        """
        result = PipelineResult()
        run_dir = None

        try:
            # Create run directory for this execution
            run_dir = self._create_run_directory()
            self._current_run_id = run_dir.name  # e.g., 'xch_run_20260219_105059'
            logger.info("Run directory: %s", run_dir)

            # Setup file logging
            if enable_file_logging:
                log_file = run_dir / "execution.log"
                add_file_handler(log_file)

            # Stage 1: Load entities
            logger.info("--- STAGE 1: Loading entities ---")
            entities = self.load_entities(limit=limit)
            result.entities_loaded = len(entities)

            if progress_callback:
                progress_callback("loaded", len(entities))

            # Stage 2: Run workflow
            logger.info("--- STAGE 2: Running multi-agent workflow ---")
            assertions = self.run_workflow(dry_run=dry_run)
            result.assertions_generated = len(assertions)
            result.entities_processed = len(entities)
            result.novelty_stats = getattr(self, "_novelty_stats", {})

            # Count by type
            for a in assertions:
                atype = a.assertion_type.value
                result.assertions_by_type[atype] = result.assertions_by_type.get(atype, 0) + 1

            if progress_callback:
                progress_callback("workflow", len(assertions))

            # Stage 3: Generate triples
            logger.info("--- STAGE 3: Generating RDF Knowledge Graph ---")
            graph = self.generate_triples()
            result.triples_generated = len(graph)

            if progress_callback:
                progress_callback("triples", len(graph))

            # Stage 4: Validate
            if not skip_validation:
                logger.info("--- STAGE 4: Validating Knowledge Graph ---")
                errors, warnings = self.validate()
                result.validation_errors = errors
                result.validation_warnings = warnings

            # Stage 5: Serialize to run directory
            logger.info("--- STAGE 5: Serializing output files ---")

            # Determine output format from config or override
            fmt = output_format or self.settings.output.format
            ext_map = {"turtle": ".ttl", "json-ld": ".jsonld", "xml": ".rdf"}
            ext = ext_map.get(fmt, ".ttl")
            run_suffix = run_dir.name.replace("xch_run_", "")

            out_filename = f"xch_output_{run_suffix}{ext}"
            out_path = run_dir / out_filename
            self.serializer.to_file(self._graph, str(out_path), format=fmt)
            result.output_files.append(str(out_path))
            logger.info("Saved %s: %s", fmt, out_path)

            # Generate manual review queue if enabled
            if self.settings.output.manual_review_queue and self._graph:
                review_path = self._generate_review_queue(run_dir)
                if review_path:
                    result.output_files.append(review_path)

            # Get rate limiter stats (only for Gemini)
            if self.rate_limiter:
                stats = self.rate_limiter.get_stats()
                result.llm_calls = stats.get("session_requests", 0)
                result.rate_limit_waits = stats.get("total_waits", 0)

        except Exception as e:
            logger.exception("Pipeline execution failed: %s", e)
            raise

        finally:
            result.finalize()

            # Save metadata
            if run_dir:
                metadata_path = self._save_metadata(run_dir, result)
                result.output_files.append(metadata_path)

            # Cleanup file handler
            if enable_file_logging:
                remove_file_handler()
                # Add log file to output list
                log_path = run_dir / "execution.log" if run_dir else None
                if log_path and log_path.exists():
                    result.output_files.append(str(log_path))

        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def run_pipeline(
    limit: int | None = None,
    dry_run: bool = False,
    output_format: str = "turtle",
    verbose: bool = True,
) -> PipelineResult:
    """
    Convenience function to run the complete pipeline.

    Args:
        limit: Maximum entities to process
        dry_run: Simulate without LLM calls
        output_format: Output format (turtle, json-ld, xml)
        verbose: Print progress to stdout

    Returns:
        PipelineResult with execution summary
    """
    # Setup logging
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    # Run pipeline
    pipeline = XCHPipeline()
    result = pipeline.execute(
        limit=limit,
        dry_run=dry_run,
        output_format=output_format,
    )

    if verbose:
        result.print_summary()

    return result
