"""
Triple Validator - Validates generated triples against the xch ontology.

Performs syntactic and SHACL-based semantic validation of RDF graphs.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyshacl
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS

logger = logging.getLogger(__name__)

# xch-MIND namespace
XCH = Namespace("https://w3id.org/xch-mind/ontology/")


# =============================================================================
# VALIDATION RESULT
# =============================================================================


@dataclass
class ValidationResult:
    """Result of triple validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def summary(self) -> str:
        """Get a summary of the validation result."""
        status = "VALID" if self.is_valid else "INVALID"
        return (
            f"Validation {status}: " f"{len(self.errors)} errors, " f"{len(self.warnings)} warnings"
        )


# =============================================================================
# TRIPLE VALIDATOR
# =============================================================================


class TripleValidator:
    """
    Validates RDF triples against the xch-MIND ontology.

    Performs:
    - Syntactic validation (well-formed RDF)
    - Namespace validation (correct prefixes)
    - Type validation (correct rdf:type usage)
    - Property validation (properties exist in ontology)
    """

    # Expected xch: classes
    XCH_CLASSES = {
        XCH.InterpretiveAssertion,
        XCH.GeographicCluster,
        XCH.ChronologicalCluster,
        XCH.TypologicalCluster,
        XCH.ThematicPath,
        XCH.GeographicPath,
        XCH.ChronologicalPath,
        XCH.TypologicalPath,
        XCH.NarrativePath,
        XCH.PathStop,
        XCH.SpatialRelation,
        XCH.TypologicalRelation,
        XCH.TemporalRelation,
        XCH.ExtractedFeature,
        XCH.ArchitecturalFeature,
        XCH.FunctionalFeature,
    }

    # Expected xch: properties (aligned with xch-core.ttl ontology)
    XCH_PROPERTIES = {
        # Core relations
        XCH.includesSite,
        XCH.nearTo,
        XCH.similarTo,
        XCH.contemporaryWith,
        XCH.relatesTo,
        XCH.groundedOn,
        XCH.derivedFrom,
        # Path structure
        XCH.hasTheme,
        XCH.hasStop,
        XCH.stopSite,
        XCH.nextStop,
        XCH.stopOrder,
        XCH.belongsToPath,
        XCH.pathLength,
        XCH.narrativeSnippet,
        XCH.connectionReason,
        XCH.estimatedDuration,
        XCH.difficulty,
        # Provenance
        XCH.generatedBy,
        XCH.generatedAt,
        XCH.validatedBy,
        XCH.validatedAt,
        XCH.hasValidationStatus,
        XCH.confidenceScore,
        XCH.reasoningTrace,
        XCH.validationNotes,
        # Spatial
        XCH.distanceKm,
        XCH.regionName,
        # Temporal
        XCH.periodLabel,
        XCH.temporalExtent,
        XCH.startYear,
        XCH.endYear,
        # Typological
        XCH.typologyLabel,
        XCH.definingFeature,
        XCH.similarityScore,
        # Features
        XCH.hasFeature,
        XCH.featureOf,
        XCH.featureLabel,
        # Relation reification
        XCH.source,
        XCH.target,
        XCH.about,
        # Additional path
        XCH.narrative,
    }

    def __init__(
        self, ontology_path: Path | str | None = None, shapes_path: Path | str | None = None
    ):
        """
        Initialize the validator.

        Args:
            ontology_path: Optional path to xch ontology file
            shapes_path: Optional path to SHACL shapes file (.ttl)
        """
        self.ontology = None
        self.shacl_graph = None
        if ontology_path:
            self._load_ontology(ontology_path)
        if shapes_path:
            self._load_shacl_shapes(shapes_path)

    def _load_ontology(self, path: Path | str) -> None:
        """Load the ontology file."""
        path = Path(path)
        if path.exists():
            self.ontology = Graph()
            self.ontology.parse(path, format="turtle")
            logger.info("Loaded ontology from %s (%d triples)", path, len(self.ontology))

    def _load_shacl_shapes(self, path: Path | str) -> None:
        """Load SHACL shapes from a Turtle file."""
        path = Path(path)
        if path.exists():
            self.shacl_graph = Graph()
            self.shacl_graph.parse(path, format="turtle")
            logger.info("Loaded SHACL shapes from %s (%d triples)", path, len(self.shacl_graph))
        else:
            logger.warning("SHACL shapes file not found: %s", path)

    def validate(self, graph: Graph) -> ValidationResult:
        """
        Validate an RDF graph.

        Args:
            graph: RDF graph to validate

        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult(is_valid=True)

        # Basic checks
        logger.info("TripleValidator: Checking graph size...")
        self._check_not_empty(graph, result)

        logger.info("TripleValidator: Validating namespace bindings...")
        self._check_namespaces(graph, result)

        logger.info("TripleValidator: Validating class types...")
        self._check_types(graph, result)

        logger.info("TripleValidator: Validating properties and external terms...")
        self._check_properties(graph, result)

        logger.info("TripleValidator: Validating literal well-formedness...")
        self._check_literals(graph, result)

        # SHACL validation (if shapes are loaded)
        if self.shacl_graph is not None:
            logger.info("TripleValidator: Running SHACL validation...")
            self._check_shacl(graph, result)

        # Gather info
        result.info["triple_count"] = len(graph)
        result.info["subject_count"] = len(set(s for s, _, _ in graph))

        logger.info("Validation complete: %s", result.summary())
        return result

    def _check_not_empty(self, graph: Graph, result: ValidationResult) -> None:
        """Check that graph is not empty."""
        if len(graph) == 0:
            result.add_error("Graph is empty")

    def _check_namespaces(self, graph: Graph, result: ValidationResult) -> None:
        """Check that required namespaces are bound."""
        bound_prefixes = {prefix for prefix, _ in graph.namespaces()}

        # Check for xch namespace
        if "xch" not in bound_prefixes:
            result.add_warning("xch namespace not bound")

    def _check_types(self, graph: Graph, result: ValidationResult) -> None:
        """Check that rdf:type statements use valid classes."""
        for s, p, o in graph.triples((None, RDF.type, None)):
            if isinstance(o, URIRef):
                # Check if it's an xch class
                if str(o).startswith(str(XCH)):
                    if o not in self.XCH_CLASSES:
                        result.add_warning(f"Unknown xch class: {o}")

    def _check_properties(self, graph: Graph, result: ValidationResult) -> None:
        """Check that properties are valid."""
        # Allowed external namespaces (simplified check)
        EXTERNAL_PROPERTIES = {
            # PROV-O
            URIRef("http://www.w3.org/ns/prov#wasGeneratedBy"),
            URIRef("http://www.w3.org/ns/prov#generatedAtTime"),
            URIRef("http://www.w3.org/ns/prov#used"),
            # GeoSPARQL
            URIRef("http://www.opengis.net/ont/geosparql#hasCentroid"),
            # WGS84
            URIRef("http://www.w3.org/2003/01/geo/wgs84_pos#lat"),
            URIRef("http://www.w3.org/2003/01/geo/wgs84_pos#long"),
            # RDF/RDFS/DC
            RDF.type,
            RDFS.label,
            URIRef("http://purl.org/dc/elements/1.1/description"),
        }

        for s, p, o in graph:
            if isinstance(p, URIRef):
                # Check xch properties
                if str(p).startswith(str(XCH)):
                    if p not in self.XCH_PROPERTIES:
                        result.add_warning(f"Unknown xch property: {p}")
                # Check known external properties
                elif any(
                    str(p).startswith(base)
                    for base in [
                        "http://www.w3.org/ns/prov#",
                        "http://www.opengis.net/ont/geosparql#",
                        "http://www.w3.org/2003/01/geo/wgs84_pos#",
                    ]
                ):
                    if p not in EXTERNAL_PROPERTIES:
                        result.add_warning(f"Unknown external property: {p}")

    def _check_literals(self, graph: Graph, result: ValidationResult) -> None:
        """Check that literals are well-formed."""
        from rdflib import Literal

        for s, p, o in graph:
            if isinstance(o, Literal):
                # Check for empty strings
                if str(o) == "":
                    result.add_warning(f"Empty literal for {p} on {s}")

    def _check_shacl(self, graph: Graph, result: ValidationResult) -> None:
        """Run SHACL validation using pyshacl."""
        try:
            conforms, results_graph, results_text = pyshacl.validate(
                data_graph=graph,
                shacl_graph=self.shacl_graph,
                ont_graph=self.ontology,
                inference="none",
                abort_on_first=False,
                allow_infos=True,
                allow_warnings=True,
            )

            # Parse SHACL results
            SH = Namespace("http://www.w3.org/ns/shacl#")
            violations = 0
            warnings_count = 0

            for report in results_graph.subjects(RDF.type, SH.ValidationResult):
                severity = results_graph.value(report, SH.resultSeverity)
                message = results_graph.value(report, SH.resultMessage)
                focus = results_graph.value(report, SH.focusNode)
                path = results_graph.value(report, SH.resultPath)

                detail = f"SHACL: {message}"
                if focus:
                    detail += f" (node: {focus})"
                if path:
                    detail += f" (path: {path})"

                if severity == SH.Violation:
                    result.add_error(detail)
                    violations += 1
                elif severity == SH.Warning:
                    result.add_warning(detail)
                    warnings_count += 1
                # Info severity is logged but not added to results

            result.info["shacl_conforms"] = conforms
            result.info["shacl_violations"] = violations
            result.info["shacl_warnings"] = warnings_count

            if conforms:
                logger.info("SHACL validation: CONFORMS")
            else:
                logger.warning(
                    "SHACL validation: %d violations, %d warnings",
                    violations,
                    warnings_count,
                )

        except Exception as e:
            logger.error("SHACL validation failed with exception: %s", e)
            result.add_warning(f"SHACL validation could not be completed: {e}")

    def check_consistency(self, graph: Graph) -> list[str]:
        """
        Check logical consistency of the graph.

        Returns:
            List of inconsistency messages
        """
        issues = []

        # Check that paths have stops
        for path in graph.subjects(RDF.type, XCH.ThematicPath):
            stops = list(graph.objects(path, XCH.hasStop))
            if not stops:
                issues.append(f"ThematicPath {path} has no stops")

        # Check that stops reference sites
        for stop in graph.subjects(RDF.type, XCH.PathStop):
            sites = list(graph.objects(stop, XCH.stopSite))
            if not sites:
                issues.append(f"PathStop {stop} has no site reference")

        # Check that clusters have members
        for cluster_type in [
            XCH.GeographicCluster,
            XCH.ChronologicalCluster,
            XCH.TypologicalCluster,
        ]:
            for cluster in graph.subjects(RDF.type, cluster_type):
                members = list(graph.objects(cluster, XCH.includesSite))
                if not members:
                    issues.append(f"Cluster {cluster} has no members")

        # Check that relations have source and target
        for rel_type in [XCH.SpatialRelation, XCH.TypologicalRelation, XCH.TemporalRelation]:
            for rel in graph.subjects(RDF.type, rel_type):
                sources = list(graph.objects(rel, XCH.source))
                targets = list(graph.objects(rel, XCH.target))
                if not sources:
                    issues.append(f"Relation {rel} has no source")
                if not targets:
                    issues.append(f"Relation {rel} has no target")

        return issues


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def validate_graph(graph: Graph) -> ValidationResult:
    """Quick function to validate a graph."""
    return TripleValidator().validate(graph)


def check_consistency(graph: Graph) -> list[str]:
    """Quick function to check graph consistency."""
    return TripleValidator().check_consistency(graph)
