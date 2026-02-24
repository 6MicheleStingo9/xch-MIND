"""
Triple Generator - Converts InterpretiveAssertions to RDF triples.

Generates valid RDF triples according to the xch: (xch-MIND) ontology.
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DC, DCTERMS, OWL, RDF, RDFS, SKOS, XSD

logger = logging.getLogger(__name__)


# =============================================================================
# NAMESPACE DEFINITIONS
# =============================================================================

# xch-MIND ontology namespace
XCH = Namespace("https://w3id.org/xch-mind/ontology/")

# ArCo resources namespace
ARCO_RES = Namespace("https://w3id.org/arco/resource/ArchaeologicalProperty/")
ARCO_ONT = Namespace("https://w3id.org/arco/ontology/arco/")

# GeoSPARQL namespace
GEO = Namespace("http://www.opengis.net/ont/geosparql#")

# PROV-O namespace
PROV = Namespace("http://www.w3.org/ns/prov#")

# WGS84 namespace
WGS84 = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")

# xch-MIND resource namespace
XCH_RES = Namespace("https://w3id.org/xch-mind/resource/")


# =============================================================================
# TRIPLE GENERATOR
# =============================================================================


class TripleGenerator:
    """
    Generates RDF triples from InterpretiveAssertion objects.

    Maps assertion types to appropriate RDF structures according
    to the xch-MIND ontology.
    """

    def __init__(self, base_uri: str = "https://w3id.org/xch-mind/data/"):
        """
        Initialize the triple generator.

        Args:
            base_uri: Base URI for generated resources
        """
        self.base_uri = base_uri
        self.data_ns = Namespace(base_uri)
        self.graph = Graph()  # Empty graph for initialization

    def generate(self, assertions: list[dict | Any]) -> Graph:
        """
        Generate RDF graph from a list of assertions.

        Args:
            assertions: List of assertion dicts or Pydantic models

        Returns:
            rdflib Graph containing all generated triples
        """
        graph = Graph()

        # Bind namespaces for clean serialization
        graph.bind("xch", XCH)
        graph.bind("arco", ARCO_RES)
        graph.bind("arco-ont", ARCO_ONT)
        graph.bind("geo", GEO)
        graph.bind("prov", PROV)
        graph.bind("wgs84", WGS84)
        graph.bind("dc", DC)
        graph.bind("dcterms", DCTERMS)
        graph.bind("data", self.data_ns)
        graph.bind("xch-res", XCH_RES)
        graph.bind("skos", SKOS)

        logger.info("Generating triples for %d assertions", len(assertions))

        for assertion in assertions:
            # Convert to dict if Pydantic model
            if hasattr(assertion, "model_dump"):
                assertion = assertion.model_dump()

            try:
                self._add_assertion(graph, assertion)
            except Exception as e:
                logger.warning(
                    "Failed to generate triples for assertion %s: %s",
                    assertion.get("assertion_id", "unknown"),
                    e,
                )

        logger.info("Generated %d triples", len(graph))
        return graph

    def _add_assertion(self, graph: Graph, assertion: dict) -> None:
        """Route assertion to appropriate handler based on type."""
        assertion_type = assertion.get("assertion_type", "")

        # Handle enum values
        if hasattr(assertion_type, "value"):
            assertion_type = assertion_type.value

        # Route based on assertion type
        assertion_label = assertion.get("label") or assertion_type
        logger.info("TripleGenerator: Mapping %s '%s'", assertion_type, assertion_label)

        if assertion_type == "geographic_cluster":
            self._add_geographic_cluster(graph, assertion)
        elif assertion_type == "chronological_cluster":
            self._add_chronological_cluster(graph, assertion)
        elif assertion_type == "typological_cluster":
            self._add_typological_cluster(graph, assertion)
        elif assertion_type == "near_to":
            self._add_near_relation(graph, assertion)
        elif assertion_type == "similar_to":
            self._add_similar_relation(graph, assertion)
        elif assertion_type == "contemporary_with":
            self._add_contemporary_relation(graph, assertion)
        elif assertion_type == "thematic_path":
            self._add_thematic_path(graph, assertion)
        elif assertion_type == "path_stop":
            self._add_path_stop(graph, assertion)
        elif assertion_type == "extracted_feature":
            self._add_extracted_feature(graph, assertion)
        else:
            self._add_generic_assertion(graph, assertion)

    # =========================================================================
    # CLUSTER GENERATORS
    # =========================================================================

    def _add_geographic_cluster(self, graph: Graph, assertion: dict) -> None:
        """Generate triples for a GeographicCluster."""
        cluster_uri = self.data_ns[assertion["assertion_id"]]

        # Type and label
        graph.add((cluster_uri, RDF.type, XCH.GeographicCluster))
        graph.add((cluster_uri, RDFS.label, Literal(assertion.get("label", ""))))

        if assertion.get("description"):
            graph.add((cluster_uri, DC.description, Literal(assertion["description"])))

        # Members
        for member_uri in assertion.get("subject_uris", []):
            member_ref = self._uri_to_ref(member_uri)
            graph.add((cluster_uri, XCH.includesSite, member_ref))

        # Region
        if assertion.get("region"):
            graph.add((cluster_uri, XCH.regionName, Literal(assertion["region"])))

        # Centroid (if available)
        if assertion.get("centroid_lat") and assertion.get("centroid_lon"):
            centroid = BNode()
            graph.add((cluster_uri, GEO.hasCentroid, centroid))
            graph.add((centroid, RDF.type, GEO.Point))
            graph.add(
                (
                    centroid,
                    WGS84.lat,
                    Literal(Decimal(str(assertion["centroid_lat"])), datatype=XSD.decimal),
                )
            )
            graph.add(
                (
                    centroid,
                    WGS84.long,
                    Literal(Decimal(str(assertion["centroid_lon"])), datatype=XSD.decimal),
                )
            )

        # Provenance
        self._add_provenance(graph, cluster_uri, assertion)

    def _add_chronological_cluster(self, graph: Graph, assertion: dict) -> None:
        """Generate triples for a ChronologicalCluster."""
        cluster_uri = self.data_ns[assertion["assertion_id"]]

        graph.add((cluster_uri, RDF.type, XCH.ChronologicalCluster))
        graph.add((cluster_uri, RDFS.label, Literal(assertion.get("label", ""))))

        if assertion.get("description"):
            graph.add((cluster_uri, DC.description, Literal(assertion["description"])))

        # Members
        for member_uri in assertion.get("subject_uris", []):
            member_ref = self._uri_to_ref(member_uri)
            graph.add((cluster_uri, XCH.includesSite, member_ref))

        # Period information - support both 'period' and 'period_label'
        period = assertion.get("period") or assertion.get("period_label")
        if period:
            graph.add((cluster_uri, XCH.periodLabel, Literal(period)))

        # Temporal extent (startYear - endYear)
        if assertion.get("start_year") or assertion.get("end_year"):
            temporal_extent = []
            if assertion.get("start_year"):
                temporal_extent.append(str(assertion["start_year"]))
                graph.add(
                    (
                        cluster_uri,
                        XCH.startYear,
                        Literal(assertion["start_year"], datatype=XSD.integer),
                    )
                )
            if assertion.get("end_year"):
                temporal_extent.append(str(assertion["end_year"]))
                graph.add(
                    (cluster_uri, XCH.endYear, Literal(assertion["end_year"], datatype=XSD.integer))
                )
            if temporal_extent:
                graph.add((cluster_uri, XCH.temporalExtent, Literal(" to ".join(temporal_extent))))

        self._add_provenance(graph, cluster_uri, assertion)

    def _add_typological_cluster(self, graph: Graph, assertion: dict) -> None:
        """Generate triples for a TypologicalCluster."""
        cluster_uri = self.data_ns[assertion["assertion_id"]]

        graph.add((cluster_uri, RDF.type, XCH.TypologicalCluster))
        graph.add((cluster_uri, RDFS.label, Literal(assertion.get("label", ""))))

        if assertion.get("description"):
            graph.add((cluster_uri, DC.description, Literal(assertion["description"])))

        # Members
        for member_uri in assertion.get("subject_uris", []):
            member_ref = self._uri_to_ref(member_uri)
            graph.add((cluster_uri, XCH.includesSite, member_ref))

        # Typology
        if assertion.get("typology"):
            graph.add((cluster_uri, XCH.typologyLabel, Literal(assertion["typology"])))

        # Shared/defining features
        for feature in assertion.get("shared_features", []):
            graph.add((cluster_uri, XCH.definingFeature, Literal(feature)))

        self._add_provenance(graph, cluster_uri, assertion)

    # =========================================================================
    # RELATION GENERATORS
    # =========================================================================

    def _add_near_relation(self, graph: Graph, assertion: dict) -> None:
        """Generate triples for a nearTo relation."""
        # Support both source_uri/target_uri and subject_uris/object_uris
        source_uri = assertion.get("source_uri") or (
            assertion.get("subject_uris", [None])[0] if assertion.get("subject_uris") else None
        )
        target_uri = assertion.get("target_uri") or (
            assertion.get("object_uris", [None])[0] if assertion.get("object_uris") else None
        )

        source_ref = self._uri_to_ref(source_uri) if source_uri else None
        target_ref = self._uri_to_ref(target_uri) if target_uri else None

        if not source_ref or not target_ref:
            return

        # Main relation + parent property for non-reasoning SPARQL interop
        graph.add((source_ref, XCH.nearTo, target_ref))
        graph.add((source_ref, XCH.relatesTo, target_ref))

        # Reified relation for additional properties
        relation_uri = self.data_ns[assertion["assertion_id"]]
        graph.add((relation_uri, RDF.type, XCH.SpatialRelation))
        graph.add((relation_uri, XCH.source, source_ref))
        graph.add((relation_uri, XCH.target, target_ref))

        # Label and description
        if assertion.get("label"):
            graph.add((relation_uri, RDFS.label, Literal(assertion["label"])))
        if assertion.get("description"):
            graph.add((relation_uri, DC.description, Literal(assertion["description"])))

        # Distance
        if assertion.get("relation_value") is not None:
            graph.add(
                (
                    relation_uri,
                    XCH.distanceKm,
                    Literal(Decimal(str(assertion["relation_value"])), datatype=XSD.decimal),
                )
            )

        self._add_provenance(graph, relation_uri, assertion)

    def _add_similar_relation(self, graph: Graph, assertion: dict) -> None:
        """Generate triples for a similarTo relation."""
        # Support both source_uri/target_uri and subject_uris/object_uris
        source_uri = assertion.get("source_uri") or (
            assertion.get("subject_uris", [None])[0] if assertion.get("subject_uris") else None
        )
        target_uri = assertion.get("target_uri") or (
            assertion.get("object_uris", [None])[0] if assertion.get("object_uris") else None
        )

        source_ref = self._uri_to_ref(source_uri) if source_uri else None
        target_ref = self._uri_to_ref(target_uri) if target_uri else None

        if not source_ref or not target_ref:
            return

        # Main relation + parent property for non-reasoning SPARQL interop
        graph.add((source_ref, XCH.similarTo, target_ref))
        graph.add((source_ref, XCH.relatesTo, target_ref))

        # Reified relation
        relation_uri = self.data_ns[assertion["assertion_id"]]
        graph.add((relation_uri, RDF.type, XCH.TypologicalRelation))
        graph.add((relation_uri, XCH.source, source_ref))
        graph.add((relation_uri, XCH.target, target_ref))

        # Label and description
        if assertion.get("label"):
            graph.add((relation_uri, RDFS.label, Literal(assertion["label"])))
        if assertion.get("description"):
            graph.add((relation_uri, DC.description, Literal(assertion["description"])))

        # Similarity score
        if assertion.get("relation_value") is not None:
            graph.add(
                (
                    relation_uri,
                    XCH.similarityScore,
                    Literal(Decimal(str(assertion["relation_value"])), datatype=XSD.decimal),
                )
            )

        self._add_provenance(graph, relation_uri, assertion)

    def _add_contemporary_relation(self, graph: Graph, assertion: dict) -> None:
        """Generate triples for a contemporaryWith relation."""
        # Support both source_uri/target_uri and subject_uris/object_uris
        source_uri = assertion.get("source_uri") or (
            assertion.get("subject_uris", [None])[0] if assertion.get("subject_uris") else None
        )
        target_uri = assertion.get("target_uri") or (
            assertion.get("object_uris", [None])[0] if assertion.get("object_uris") else None
        )

        source_ref = self._uri_to_ref(source_uri) if source_uri else None
        target_ref = self._uri_to_ref(target_uri) if target_uri else None

        if not source_ref or not target_ref:
            return

        # Main relation (symmetric) + parent property for non-reasoning SPARQL interop
        graph.add((source_ref, XCH.contemporaryWith, target_ref))
        graph.add((target_ref, XCH.contemporaryWith, source_ref))
        graph.add((source_ref, XCH.relatesTo, target_ref))
        graph.add((target_ref, XCH.relatesTo, source_ref))

        # Reified relation
        relation_uri = self.data_ns[assertion["assertion_id"]]
        graph.add((relation_uri, RDF.type, XCH.TemporalRelation))
        graph.add((relation_uri, XCH.source, source_ref))
        graph.add((relation_uri, XCH.target, target_ref))

        # Label and description
        if assertion.get("label"):
            graph.add((relation_uri, RDFS.label, Literal(assertion["label"])))
        if assertion.get("description"):
            graph.add((relation_uri, DC.description, Literal(assertion["description"])))

        self._add_provenance(graph, relation_uri, assertion)

    # =========================================================================
    # PATH GENERATORS
    # =========================================================================

    def _add_thematic_path(self, graph: Graph, assertion: dict) -> None:
        """Generate triples for a ThematicPath."""
        path_uri = self.data_ns[assertion["assertion_id"]]

        # Always assert ThematicPath superclass
        graph.add((path_uri, RDF.type, XCH.ThematicPath))

        # Map path_type to ontology subclass
        path_type = (assertion.get("path_type") or "mixed").lower()
        path_subclass_map = {
            "geographic": XCH.GeographicPath,
            "chronological": XCH.ChronologicalPath,
            "typological": XCH.TypologicalPath,
            "narrative": XCH.NarrativePath,
            "mixed": XCH.NarrativePath,
        }
        if subclass := path_subclass_map.get(path_type):
            graph.add((path_uri, RDF.type, subclass))

        graph.add((path_uri, RDFS.label, Literal(assertion.get("label", ""))))

        if assertion.get("description"):
            graph.add((path_uri, DC.description, Literal(assertion["description"])))

        # Theme as SKOS Concept
        if assertion.get("theme"):
            theme_str = assertion["theme"]
            # Create a SKOS Concept URI from the theme label
            theme_slug = theme_str.lower().replace(" ", "_").replace("-", "_")
            theme_uri = self.data_ns[f"theme_{theme_slug}"]
            graph.add((theme_uri, RDF.type, SKOS.Concept))
            graph.add((theme_uri, SKOS.prefLabel, Literal(theme_str)))
            graph.add((path_uri, XCH.hasTheme, theme_uri))

        # Stops (embedded as list of dicts or objects)
        stops = assertion.get("stops", [])
        stop_uris = []  # Track for nextStop linking
        for i, stop in enumerate(stops, 1):
            # Convert Pydantic model to dict if needed
            if hasattr(stop, "model_dump"):
                stop = stop.model_dump()

            # Create stop URI
            stop_id = (
                stop.get("assertion_id")
                or stop.get("stop_id")
                or f"{assertion['assertion_id']}-stop-{i}"
            )
            stop_uri = self.data_ns[stop_id]

            # Link path to stop
            graph.add((path_uri, XCH.hasStop, stop_uri))

            # Add stop details
            graph.add((stop_uri, RDF.type, XCH.PathStop))
            graph.add((stop_uri, RDFS.label, Literal(stop.get("label", f"Stop {i}"))))
            graph.add(
                (stop_uri, XCH.stopOrder, Literal(stop.get("order", i), datatype=XSD.integer))
            )
            graph.add((stop_uri, XCH.belongsToPath, path_uri))

            # Site reference
            site_uri = stop.get("site_uri")
            if site_uri:
                site_ref = self._uri_to_ref(site_uri)
                graph.add((stop_uri, XCH.stopSite, site_ref))

            # Narrative snippet
            if stop.get("narrative"):
                graph.add((stop_uri, XCH.narrativeSnippet, Literal(stop["narrative"])))

            # Connection reason (why this stop connects to the next)
            if stop.get("connection_reason"):
                graph.add((stop_uri, XCH.connectionReason, Literal(stop["connection_reason"])))

            stop_uris.append(stop_uri)

        # Link consecutive stops with xch:nextStop
        for j in range(len(stop_uris) - 1):
            graph.add((stop_uris[j], XCH.nextStop, stop_uris[j + 1]))

        # Path length
        if stops:
            graph.add((path_uri, XCH.pathLength, Literal(len(stops), datatype=XSD.integer)))

        # Duration (convert float hours to xsd:duration ISO 8601)
        if assertion.get("estimated_duration"):
            duration_iso = self._hours_to_xsd_duration(assertion["estimated_duration"])
            graph.add(
                (path_uri, XCH.estimatedDuration, Literal(duration_iso, datatype=XSD.duration))
            )

        # Difficulty
        if assertion.get("difficulty"):
            graph.add((path_uri, XCH.difficulty, Literal(assertion["difficulty"])))

        # Narrative
        if assertion.get("narrative"):
            graph.add((path_uri, XCH.narrative, Literal(assertion["narrative"])))

        self._add_provenance(graph, path_uri, assertion)

    def _add_path_stop(self, graph: Graph, assertion: dict) -> None:
        """Generate triples for a PathStop."""
        stop_uri = self.data_ns[assertion["assertion_id"]]

        graph.add((stop_uri, RDF.type, XCH.PathStop))
        graph.add((stop_uri, RDFS.label, Literal(assertion.get("label", ""))))

        # Site reference
        if assertion.get("site_uri"):
            site_ref = self._uri_to_ref(assertion["site_uri"])
            graph.add((stop_uri, XCH.stopSite, site_ref))

        # Order in path
        if assertion.get("order"):
            graph.add((stop_uri, XCH.stopOrder, Literal(assertion["order"], datatype=XSD.integer)))

        # Parent path
        if assertion.get("path_id"):
            path_ref = self.data_ns[assertion["path_id"]]
            graph.add((stop_uri, XCH.belongsToPath, path_ref))

        # Justification
        if assertion.get("justification"):
            graph.add((stop_uri, XCH.connectionReason, Literal(assertion["justification"])))

        # Narrative snippet
        if assertion.get("narrative"):
            graph.add((stop_uri, XCH.narrativeSnippet, Literal(assertion["narrative"])))

        self._add_provenance(graph, stop_uri, assertion)

    # =========================================================================
    # FEATURE GENERATORS
    # =========================================================================

    def _add_extracted_feature(self, graph: Graph, assertion: dict) -> None:
        """Generate triples for an ExtractedFeature."""
        feature_uri = self.data_ns[assertion["assertion_id"]]

        # Map feature category to ontology subclass
        category = (assertion.get("feature_category") or "").lower()
        category_class_map = {
            "architectural": XCH.ArchitecturalFeature,
            "functional": XCH.FunctionalFeature,
        }
        feature_class = category_class_map.get(category, XCH.ExtractedFeature)

        graph.add((feature_uri, RDF.type, feature_class))
        # Always assert superclass too
        if feature_class != XCH.ExtractedFeature:
            graph.add((feature_uri, RDF.type, XCH.ExtractedFeature))

        graph.add((feature_uri, RDFS.label, Literal(assertion.get("label", ""))))

        # Feature label
        if assertion.get("feature_label"):
            graph.add((feature_uri, XCH.featureLabel, Literal(assertion["feature_label"])))

        if assertion.get("description"):
            graph.add((feature_uri, DC.description, Literal(assertion["description"])))

        # Link feature to entity (featureOf) and entity to feature (hasFeature)
        entity_uri = assertion.get("entity_uri")
        if entity_uri:
            entity_ref = self._uri_to_ref(entity_uri)
            if entity_ref:
                graph.add((feature_uri, XCH.featureOf, entity_ref))
                graph.add((entity_ref, XCH.hasFeature, feature_uri))

        self._add_provenance(graph, feature_uri, assertion)

    # =========================================================================
    # GENERIC AND HELPERS
    # =========================================================================

    def _add_generic_assertion(self, graph: Graph, assertion: dict) -> None:
        """Generate triples for a generic InterpretiveAssertion."""
        assertion_uri = self.data_ns[assertion["assertion_id"]]

        graph.add((assertion_uri, RDF.type, XCH.InterpretiveAssertion))
        graph.add((assertion_uri, RDFS.label, Literal(assertion.get("label", ""))))

        if assertion.get("description"):
            graph.add((assertion_uri, DC.description, Literal(assertion["description"])))

        # Subject URIs
        for uri in assertion.get("subject_uris", []):
            ref = self._uri_to_ref(uri)
            graph.add((assertion_uri, XCH.about, ref))

        self._add_provenance(graph, assertion_uri, assertion)

    def _add_provenance(self, graph: Graph, subject: URIRef, assertion: dict) -> None:
        """Add provenance and validation metadata to a resource."""
        # Generated by agent — use xch:generatedBy (subPropertyOf prov:wasGeneratedBy)
        if assertion.get("generated_by"):
            agent_map = {
                "orchestrator": XCH_RES.OrchestratorAgent,
                "geo_analyzer": XCH_RES.GeoSpatialAgent,
                "geospatial": XCH_RES.GeoSpatialAgent,
                "temporal_analyzer": XCH_RES.ChronologicalAgent,
                "chronological": XCH_RES.ChronologicalAgent,
                "type_analyzer": XCH_RES.TypologicalAgent,
                "typological": XCH_RES.TypologicalAgent,
                "path_generator": XCH_RES.NarrativeAgent,
                "narrative": XCH_RES.NarrativeAgent,
                "validator": XCH_RES.ValidationAgent,
            }
            agent_val = assertion["generated_by"]
            if hasattr(agent_val, "value"):
                agent_val = agent_val.value

            # Use URI if mapped, otherwise fall back to literal
            if agent_uri := agent_map.get(str(agent_val)):
                graph.add((subject, XCH.generatedBy, agent_uri))
            else:
                graph.add((subject, XCH.generatedBy, Literal(str(agent_val))))

        # Confidence score
        if assertion.get("confidence_score") is not None:
            graph.add(
                (
                    subject,
                    XCH.confidenceScore,
                    Literal(Decimal(str(assertion["confidence_score"])), datatype=XSD.decimal),
                )
            )

        # Reasoning trace
        if assertion.get("reasoning"):
            graph.add((subject, XCH.reasoningTrace, Literal(assertion["reasoning"])))

        # Generation timestamp
        timestamp = assertion.get("created_at")
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        elif hasattr(timestamp, "isoformat"):
            timestamp = timestamp.isoformat()
        graph.add(
            (
                subject,
                XCH.generatedAt,
                Literal(timestamp, datatype=XSD.dateTime),
            )
        )

        # --- Validation metadata ---
        # Three-tier classification: Validated (≥0.70), PendingReview (0.60-0.70), Rejected (<0.60)
        # All assertions reaching the generator have already passed inline agent validation;
        # classification is based on confidence score alone.
        confidence = assertion.get("confidence_score", 0.0)
        if confidence >= 0.70:
            graph.add((subject, XCH.hasValidationStatus, XCH.Validated))
        elif confidence >= 0.60:
            graph.add((subject, XCH.hasValidationStatus, XCH.PendingReview))
        else:
            graph.add((subject, XCH.hasValidationStatus, XCH.Rejected))

        # Validated by (same agent that generated, since validation is inline)
        if assertion.get("generated_by"):
            agent_val = assertion["generated_by"]
            if hasattr(agent_val, "value"):
                agent_val = agent_val.value
            agent_map_val = {
                "orchestrator": XCH_RES.OrchestratorAgent,
                "geo_analyzer": XCH_RES.GeoSpatialAgent,
                "geospatial": XCH_RES.GeoSpatialAgent,
                "temporal_analyzer": XCH_RES.ChronologicalAgent,
                "chronological": XCH_RES.ChronologicalAgent,
                "type_analyzer": XCH_RES.TypologicalAgent,
                "typological": XCH_RES.TypologicalAgent,
                "path_generator": XCH_RES.NarrativeAgent,
                "narrative": XCH_RES.NarrativeAgent,
                "validator": XCH_RES.ValidationAgent,
            }
            if val_agent := agent_map_val.get(str(agent_val)):
                graph.add((subject, XCH.validatedBy, val_agent))

        # Validated at timestamp
        graph.add(
            (
                subject,
                XCH.validatedAt,
                Literal(timestamp, datatype=XSD.dateTime),
            )
        )

        # Validation notes
        if assertion.get("validation_notes"):
            graph.add((subject, XCH.validationNotes, Literal(assertion["validation_notes"])))

        # --- Grounding ---
        # groundedOn: link assertion to source ArCo entities
        # Also emit prov:used for PROV-O interoperability
        for uri in assertion.get("subject_uris", []):
            ref = self._uri_to_ref(uri)
            if ref:
                graph.add((subject, XCH.groundedOn, ref))
                graph.add((subject, PROV.used, ref))
        for uri in assertion.get("object_uris", []):
            ref = self._uri_to_ref(uri)
            if ref:
                graph.add((subject, XCH.groundedOn, ref))
                graph.add((subject, PROV.used, ref))

        # derivedFrom: link to source assertions (e.g., paths derived from clusters)
        for source_id in assertion.get("derived_from", []):
            graph.add((subject, XCH.derivedFrom, self.data_ns[source_id]))

    def _uri_to_ref(self, uri: str) -> URIRef | None:
        """Convert a URI string to rdflib URIRef."""
        if not uri:
            logger.debug("Empty URI provided to _uri_to_ref, returning None")
            return None

        # Handle ArCo URIs
        if "dati.beniculturali.it" in uri or "arco" in uri.lower():
            # Extract identifier and use ARCO namespace
            parts = uri.rstrip("/").split("/")
            identifier = parts[-1]
            return ARCO_RES[identifier]

        # Handle full URIs
        if uri.startswith("http://") or uri.startswith("https://"):
            return URIRef(uri)

        # Handle local references
        return self.data_ns[uri]

    @staticmethod
    def _hours_to_xsd_duration(hours_value) -> str:
        """Convert a numeric hours value to ISO 8601 duration string (e.g. 'PT2H30M').

        Args:
            hours_value: float or int representing hours (e.g. 2.5 → 'PT2H30M')

        Returns:
            ISO 8601 duration string suitable for xsd:duration
        """
        try:
            h = float(hours_value)
        except (TypeError, ValueError):
            # If it's already a string like "PT2H", return as-is
            return str(hours_value)

        whole_hours = int(h)
        remaining_minutes = int(round((h - whole_hours) * 60))

        parts = ["PT"]
        if whole_hours > 0:
            parts.append(f"{whole_hours}H")
        if remaining_minutes > 0:
            parts.append(f"{remaining_minutes}M")
        if whole_hours == 0 and remaining_minutes == 0:
            parts.append("0H")

        return "".join(parts)
