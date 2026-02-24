"""
Triples Module - RDF triple generation and serialization.

This module provides functionality for converting InterpretiveAssertions
to valid RDF triples according to the xch-MIND ontology.

Components:
- generator.py: Converts assertions to RDF triples
- serializer.py: Serializes graphs to Turtle/JSON-LD
- validator.py: Validates triples against the ontology
"""

from .generator import TripleGenerator, XCH, ARCO_RES, GEO, PROV
from .serializer import TripleSerializer, serialize_to_turtle, serialize_to_file
from .validator import TripleValidator, ValidationResult, validate_graph, check_consistency

__all__ = [
    # Generator
    "TripleGenerator",
    # Namespaces
    "XCH",
    "ARCO_RES",
    "GEO",
    "PROV",
    # Serializer
    "TripleSerializer",
    "serialize_to_turtle",
    "serialize_to_file",
    # Validator
    "TripleValidator",
    "ValidationResult",
    "validate_graph",
    "check_consistency",
]
