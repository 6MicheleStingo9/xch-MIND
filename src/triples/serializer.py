"""
Triple Serializer - Serializes RDF graphs to various formats.

Supports Turtle, JSON-LD, N-Triples, and RDF/XML output.
"""

import json
import logging
from pathlib import Path
from typing import Any

from rdflib import Graph

logger = logging.getLogger(__name__)


# =============================================================================
# SUPPORTED FORMATS
# =============================================================================

FORMATS = {
    "turtle": {"extension": ".ttl", "mime": "text/turtle"},
    "ttl": {"extension": ".ttl", "mime": "text/turtle"},
    "json-ld": {"extension": ".jsonld", "mime": "application/ld+json"},
    "jsonld": {"extension": ".jsonld", "mime": "application/ld+json"},
    "n3": {"extension": ".n3", "mime": "text/n3"},
    "nt": {"extension": ".nt", "mime": "application/n-triples"},
    "ntriples": {"extension": ".nt", "mime": "application/n-triples"},
    "xml": {"extension": ".rdf", "mime": "application/rdf+xml"},
    "rdf": {"extension": ".rdf", "mime": "application/rdf+xml"},
}


# =============================================================================
# TRIPLE SERIALIZER
# =============================================================================


class TripleSerializer:
    """
    Serializes RDF graphs to various output formats.

    Provides clean, formatted output suitable for publication
    and further processing.
    """

    def __init__(self):
        """Initialize the serializer."""
        pass

    def to_turtle(self, graph: Graph) -> str:
        """
        Serialize graph to Turtle format.

        Args:
            graph: RDF graph to serialize

        Returns:
            Turtle string representation
        """
        return graph.serialize(format="turtle")

    def to_jsonld(self, graph: Graph, context: dict | None = None) -> dict:
        """
        Serialize graph to JSON-LD format.

        Args:
            graph: RDF graph to serialize
            context: Optional JSON-LD context

        Returns:
            JSON-LD dictionary
        """
        # Serialize to JSON-LD string
        jsonld_str = graph.serialize(format="json-ld")

        # Parse to dict
        result = json.loads(jsonld_str)

        # Add custom context if provided
        if context:
            if isinstance(result, dict):
                result["@context"] = context
            elif isinstance(result, list) and result:
                result = {"@context": context, "@graph": result}

        return result

    def to_ntriples(self, graph: Graph) -> str:
        """
        Serialize graph to N-Triples format.

        Args:
            graph: RDF graph to serialize

        Returns:
            N-Triples string representation
        """
        return graph.serialize(format="nt")

    def to_rdfxml(self, graph: Graph) -> str:
        """
        Serialize graph to RDF/XML format.

        Args:
            graph: RDF graph to serialize

        Returns:
            RDF/XML string representation
        """
        return graph.serialize(format="xml")

    def to_file(
        self,
        graph: Graph,
        path: Path | str,
        format: str = "turtle",
    ) -> None:
        """
        Serialize graph to a file.

        Args:
            graph: RDF graph to serialize
            path: Output file path
            format: Output format (turtle, json-ld, nt, xml)
        """
        path = Path(path)
        format_lower = format.lower()

        if format_lower not in FORMATS:
            raise ValueError(f"Unsupported format: {format}. Supported: {list(FORMATS.keys())}")

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize based on format
        if format_lower in ("turtle", "ttl"):
            content = self.to_turtle(graph)
        elif format_lower in ("json-ld", "jsonld"):
            content = json.dumps(self.to_jsonld(graph), indent=2, ensure_ascii=False)
        elif format_lower in ("nt", "ntriples"):
            content = self.to_ntriples(graph)
        elif format_lower in ("xml", "rdf"):
            content = self.to_rdfxml(graph)
        else:
            content = graph.serialize(format=format_lower)

        # Write to file
        path.write_text(content, encoding="utf-8")
        logger.info("Serialized %d triples to %s (%s)", len(graph), path, format)

    def get_statistics(self, graph: Graph) -> dict[str, Any]:
        """
        Get statistics about the graph.

        Args:
            graph: RDF graph to analyze

        Returns:
            Dictionary with graph statistics
        """
        from collections import Counter

        # Count triples
        total_triples = len(graph)

        # Count by predicate
        predicates = Counter()
        subjects = set()
        objects_uris = set()

        for s, p, o in graph:
            predicates[str(p)] += 1
            subjects.add(str(s))
            if hasattr(o, "startswith"):  # URI or string
                if str(o).startswith("http"):
                    objects_uris.add(str(o))

        # Get namespaces
        namespaces = {prefix: str(uri) for prefix, uri in graph.namespaces()}

        return {
            "total_triples": total_triples,
            "unique_subjects": len(subjects),
            "unique_predicates": len(predicates),
            "unique_object_uris": len(objects_uris),
            "predicates": dict(predicates.most_common(20)),
            "namespaces": namespaces,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def serialize_to_turtle(graph: Graph) -> str:
    """Quick function to serialize a graph to Turtle."""
    return TripleSerializer().to_turtle(graph)


def serialize_to_file(graph: Graph, path: Path | str, format: str = "turtle") -> None:
    """Quick function to serialize a graph to a file."""
    TripleSerializer().to_file(graph, path, format)
