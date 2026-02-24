"""Loaders package for ArCo data extraction."""

from .arco_loader import (
    DolmenEntity,
    SPARQLResolver,
    load_all_entities,
    parse_entity_file,
    get_region_name,
    NAMESPACES,
    REGION_ID_TO_NAME,
)

__all__ = [
    "DolmenEntity",
    "SPARQLResolver",
    "load_all_entities",
    "parse_entity_file",
    "get_region_name",
    "NAMESPACES",
    "REGION_ID_TO_NAME",
]
