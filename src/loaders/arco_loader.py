"""
ArCo Entity Loader - Extracts dolmen data from XML files and resolves URIs via SPARQL.

This module provides:
- XML parsing for ArCo RDF/XML entity files
- SPARQL resolution for linked resources (Geometry, Address, Dating)
- Local caching to avoid repeated SPARQL queries
- Graceful fallback when SPARQL is unavailable
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from lxml import etree
from pydantic import BaseModel, Field, computed_field

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# NAMESPACE DEFINITIONS
# =============================================================================

NAMESPACES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "arco": "https://w3id.org/arco/ontology/arco/",
    "a-cd": "https://w3id.org/arco/ontology/context-description/",
    "a-loc": "https://w3id.org/arco/ontology/location/",
    "a-dd": "https://w3id.org/arco/ontology/denotative-description/",
    "a-cat": "https://w3id.org/arco/ontology/catalogue/",
    "a-con": "https://w3id.org/arco/ontology/construction-description/",
    "core": "https://w3id.org/arco/ontology/core/",
    "clv": "https://w3id.org/italia/onto/CLV/",
    "l0": "https://w3id.org/italia/onto/l0/",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "pico": "http://data.cochrane.org/ontologies/pico/",
    "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
}

# Reverse mapping for URI to prefix
URI_TO_PREFIX = {v: k for k, v in NAMESPACES.items()}


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class DolmenEntity(BaseModel):
    """
    Represents a dolmen entity extracted from ArCo data.

    Combines data directly available in XML files with data resolved
    via SPARQL queries to the ArCo endpoint.
    """

    # === Identification ===
    uri: str = Field(description="Full ArCo URI of the entity")
    identifier: str = Field(description="Numeric identifier (e.g., '1600389226')")
    label: str = Field(description="Full label from rdfs:label")
    name: str = Field(description="Short name extracted from label")

    # === Descriptions (from XML) ===
    description: str | None = Field(default=None, description="dc:description or core:description")
    historical_info: str | None = Field(default=None, description="a-cd:historicalInformation")

    # === Classification (from XML) ===
    property_type: str = Field(default="dolmen", description="dc:type")
    category: str | None = Field(default=None, description="Cultural property category")
    category_uri: str | None = Field(default=None, description="URI of category")
    context_type: str | None = Field(default=None, description="Urban/rural context")
    context_type_uri: str | None = Field(default=None, description="URI of context type")

    # === Location - Textual (from XML) ===
    region_id: str | None = Field(
        default=None, description="Region identifier (e.g., '20' for Sardegna)"
    )
    coverage: str | None = Field(default=None, description="dc:coverage (e.g., 'Gesturi (VS)')")

    # === Location - Resolved (from SPARQL) ===
    address_full: str | None = Field(default=None, description="Full address from SPARQL")
    municipality: str | None = Field(default=None, description="Municipality name")
    province: str | None = Field(default=None, description="Province code or name")
    region_name: str | None = Field(default=None, description="Region name")
    latitude: float | None = Field(default=None, description="Latitude from geometry")
    longitude: float | None = Field(default=None, description="Longitude from geometry")

    # === Temporal (from XML label + SPARQL) ===
    dating_label: str | None = Field(default=None, description="Dating label from SPARQL")
    period_label: str | None = Field(default=None, description="Period extracted from label")

    # === URIs of linked resources (for SPARQL resolution) ===
    geometry_uri: str | None = Field(default=None, description="URI of Geometry resource")
    dating_uri: str | None = Field(default=None, description="URI of Dating resource")
    address_uri: str | None = Field(default=None, description="URI of Address resource")

    # === Media (from XML) ===
    image_urls: list[str] = Field(default_factory=list, description="Image URLs")

    # === Metadata ===
    source_file: str = Field(description="Original XML filename")
    sparql_resolved: bool = Field(default=False, description="Whether SPARQL resolution succeeded")

    @computed_field
    @property
    def has_coordinates(self) -> bool:
        """Check if entity has valid coordinates."""
        return self.latitude is not None and self.longitude is not None

    @computed_field
    @property
    def display_name(self) -> str:
        """Return a clean display name for the entity."""
        return self.name or self.label.split(",")[0].strip()

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a summary dictionary for LLM consumption."""
        return {
            "uri": self.uri,
            "name": self.display_name,
            "description": self.description,
            "historical_info": self.historical_info,
            "location": {
                "municipality": self.municipality,
                "province": self.province,
                "region": self.region_name or self.region_id,
                "coordinates": (
                    {"lat": self.latitude, "lon": self.longitude} if self.has_coordinates else None
                ),
            },
            "period": self.period_label,
            "category": self.category,
            "context": self.context_type,
        }


# =============================================================================
# XML EXTRACTION
# =============================================================================


def _get_text(element: etree._Element, xpath: str, namespaces: dict) -> str | None:
    """Extract text content from an element using XPath."""
    results = element.xpath(xpath, namespaces=namespaces)
    if results:
        if isinstance(results[0], str):
            return results[0].strip() if results[0].strip() else None
        elif hasattr(results[0], "text") and results[0].text:
            return results[0].text.strip()
    return None


def _get_resource(element: etree._Element, xpath: str, namespaces: dict) -> str | None:
    """Extract rdf:resource attribute from an element using XPath."""
    results = element.xpath(xpath, namespaces=namespaces)
    if results:
        return results[0].get(f"{{{NAMESPACES['rdf']}}}resource")
    return None


def _get_all_resources(element: etree._Element, xpath: str, namespaces: dict) -> list[str]:
    """Extract all rdf:resource attributes from elements matching XPath."""
    results = element.xpath(xpath, namespaces=namespaces)
    resources = []
    for r in results:
        uri = r.get(f"{{{NAMESPACES['rdf']}}}resource")
        if uri:
            resources.append(uri)
    return resources


def _extract_name_from_label(label: str) -> str:
    """
    Extract clean name from ArCo label.

    Examples:
        "dolmen, Dolmen dell'Accettula (PERIODIZZAZIONI/ PROTOSTORIA/ Età del Bronzo)"
        -> "Dolmen dell'Accettula"

        "Dolmen 1 di Matta Larentu (dolmen) - Suni (OR)  (V-III millennio BC cal)"
        -> "Dolmen 1 di Matta Larentu"
    """
    # Remove leading "dolmen, " if present
    if label.lower().startswith("dolmen,"):
        label = label[7:].strip()

    # Try to extract name before parentheses
    match = re.match(r"^([^(]+)", label)
    if match:
        name = match.group(1).strip()
        # Remove trailing " - Location" if present
        if " - " in name:
            name = name.split(" - ")[0].strip()
        # Remove trailing "(dolmen)" if present
        name = re.sub(r"\s*\(dolmen\)\s*$", "", name, flags=re.IGNORECASE)
        return name.strip()

    return label.split(",")[0].strip()


def _extract_period_from_label(label: str) -> str | None:
    """
    Extract period information from ArCo label.

    Examples:
        "... (PERIODIZZAZIONI/ PROTOSTORIA/ Età del Bronzo)" -> "Età del Bronzo"
        "... (V-III millennio BC cal)" -> "V-III millennio BC cal"
        "... (Neolitico)" -> "Neolitico"
    """
    # Look for period in parentheses at the end
    matches = re.findall(r"\(([^)]+)\)\s*$", label)
    if matches:
        period_raw = matches[-1]
        # If it contains PERIODIZZAZIONI, extract the last part
        if "PERIODIZZAZIONI" in period_raw.upper():
            parts = period_raw.split("/")
            if parts:
                return parts[-1].strip()
        # Otherwise return as-is if it looks like a period
        period_keywords = [
            "bronzo",
            "neolitico",
            "millennio",
            "eneolitico",
            "calcolitico",
            "bc",
            "a.c.",
        ]
        if any(kw in period_raw.lower() for kw in period_keywords):
            return period_raw.strip()
    return None


def _extract_location_from_coverage(coverage: str) -> tuple[str | None, str | None]:
    """
    Extract municipality and province from dc:coverage.

    Examples:
        "Gesturi (VS)" -> ("Gesturi", "VS")
        "Suni (OR)" -> ("Suni", "OR")
        "Macomer (NU)" -> ("Macomer", "NU")
    """
    if not coverage:
        return None, None

    match = re.match(r"([^(]+)\s*\(([^)]+)\)", coverage)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    return coverage.strip(), None


def _extract_category_from_uri(uri: str | None) -> str | None:
    """Extract category name from ArCo category URI."""
    if not uri:
        return None
    # URI like: https://w3id.org/arco/resource/CulturalPropertyCategory/area-ad-uso-funerario
    parts = uri.rstrip("/").split("/")
    if parts:
        return parts[-1].replace("-", " ")
    return None


def _extract_context_from_uri(uri: str | None) -> str | None:
    """Extract context type from ArCo context URI."""
    if not uri:
        return None
    # URI like: https://w3id.org/arco/resource/TypeOfContext/contesto-rurale
    parts = uri.rstrip("/").split("/")
    if parts:
        return parts[-1].replace("-", " ").replace("contesto ", "")
    return None


def parse_entity_file(filepath: Path) -> DolmenEntity | None:
    """
    Parse a single ArCo XML entity file and extract relevant data.

    Args:
        filepath: Path to the XML file

    Returns:
        DolmenEntity object or None if parsing fails
    """
    try:
        tree = etree.parse(str(filepath))
        root = tree.getroot()

        # Find the main rdf:Description element
        descriptions = root.xpath("//rdf:Description", namespaces=NAMESPACES)
        if not descriptions:
            logger.warning(f"No rdf:Description found in {filepath.name}")
            return None

        desc = descriptions[0]

        # Get URI
        uri = desc.get(f"{{{NAMESPACES['rdf']}}}about")
        if not uri:
            logger.warning(f"No rdf:about found in {filepath.name}")
            return None

        # Extract identifier from URI
        identifier = uri.rstrip("/").split("/")[-1]

        # Get label (try both languages, prefer Italian)
        label = _get_text(desc, "rdfs:label[@xml:lang='it']/text()", NAMESPACES)
        if not label:
            label = _get_text(desc, "rdfs:label/text()", NAMESPACES)
        if not label:
            label = _get_text(desc, "dc:title/text()", NAMESPACES) or f"Entity {identifier}"

        # Extract name and period from label
        name = _extract_name_from_label(label)
        period_label = _extract_period_from_label(label)

        # Get descriptions
        description = _get_text(desc, "dc:description/text()", NAMESPACES)
        if not description:
            description = _get_text(desc, "core:description/text()", NAMESPACES)

        historical_info = _get_text(desc, "a-cd:historicalInformation/text()", NAMESPACES)

        # Get type
        property_type = _get_text(desc, "dc:type/text()", NAMESPACES) or "dolmen"

        # Get category
        category_uri = _get_resource(desc, "arco:hasCulturalPropertyCategory", NAMESPACES)
        category = _extract_category_from_uri(category_uri)

        # Get context type
        context_uri = _get_resource(desc, "a-loc:hasTypeOfContext", NAMESPACES)
        context_type = _extract_context_from_uri(context_uri)

        # Get region identifier
        region_id = _get_text(desc, "arco:regionIdentifier/text()", NAMESPACES)

        # Get coverage
        coverage = _get_text(desc, "dc:coverage/text()", NAMESPACES)
        municipality, province = _extract_location_from_coverage(coverage)

        # Get linked resource URIs
        geometry_uri = _get_resource(desc, "clv:hasGeometry", NAMESPACES)
        dating_uri = _get_resource(desc, "a-cd:hasDating", NAMESPACES)
        address_uri = _get_resource(desc, "a-loc:hasCulturalPropertyAddress", NAMESPACES)
        if not address_uri:
            address_uri = _get_resource(desc, "dcterms:spatial", NAMESPACES)

        # Get image URLs
        image_urls = []
        for prop in ["foaf:depiction", "pico:preview"]:
            urls = _get_all_resources(desc, prop, NAMESPACES)
            image_urls.extend(urls)
        # Remove duplicates while preserving order
        image_urls = list(dict.fromkeys(image_urls))

        return DolmenEntity(
            uri=uri,
            identifier=identifier,
            label=label,
            name=name,
            description=description,
            historical_info=historical_info,
            property_type=property_type,
            category=category,
            category_uri=category_uri,
            context_type=context_type,
            context_type_uri=context_uri,
            region_id=region_id,
            coverage=coverage,
            municipality=municipality,
            province=province,
            period_label=period_label,
            geometry_uri=geometry_uri,
            dating_uri=dating_uri,
            address_uri=address_uri,
            image_urls=image_urls,
            source_file=filepath.name,
        )

    except etree.XMLSyntaxError as e:
        logger.error(f"XML syntax error in {filepath.name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error parsing {filepath.name}: {e}")
        return None


# =============================================================================
# SPARQL RESOLUTION
# =============================================================================


class SPARQLResolver:
    """
    Resolves linked resource URIs via SPARQL queries to ArCo endpoint.

    Features:
    - Batch queries for efficiency
    - Local caching to avoid repeated queries
    - Graceful fallback on errors
    """

    ENDPOINT = "https://dati.beniculturali.it/sparql"
    CACHE_FILE = "sparql_cache.json"

    def __init__(self, cache_dir: Path | None = None):
        """
        Initialize the resolver.

        Args:
            cache_dir: Directory for cache file. If None, uses current directory.
        """
        self.cache_dir = cache_dir or Path(".")
        self.cache_path = self.cache_dir / self.CACHE_FILE
        self._cache: dict[str, Any] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from disk if available."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r") as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded SPARQL cache with {len(self._cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to load SPARQL cache: {e}")
                self._cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f, indent=2)
            logger.debug(f"Saved SPARQL cache with {len(self._cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to save SPARQL cache: {e}")

    def _cache_key(self, uris: list[str]) -> str:
        """Generate cache key from list of URIs."""
        sorted_uris = sorted(set(uris))
        return hashlib.md5("".join(sorted_uris).encode()).hexdigest()

    def _parse_wkt_point(self, wkt: str) -> tuple[float, float] | None:
        """
        Parse WKT POINT to (latitude, longitude).

        Args:
            wkt: WKT string like "POINT(17.208769 40.562705)"

        Returns:
            Tuple of (latitude, longitude) or None if parsing fails
        """
        match = re.match(r"POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)", wkt, re.IGNORECASE)
        if match:
            lon = float(match.group(1))
            lat = float(match.group(2))
            return (lat, lon)
        return None

    def resolve_entities(self, entities: list[DolmenEntity]) -> list[DolmenEntity]:
        """
        Resolve linked resources for a list of entities via SPARQL.

        Args:
            entities: List of DolmenEntity objects to enrich

        Returns:
            Same list with SPARQL data filled in where available
        """
        try:
            from SPARQLWrapper import JSON, SPARQLWrapper
        except ImportError:
            logger.error("SPARQLWrapper not installed. Run: pip install sparqlwrapper")
            return entities

        # Collect all URIs that need resolution
        geometry_uris = [e.geometry_uri for e in entities if e.geometry_uri]
        address_uris = [e.address_uri for e in entities if e.address_uri]
        dating_uris = [e.dating_uri for e in entities if e.dating_uri]

        all_uris = geometry_uris + address_uris + dating_uris
        if not all_uris:
            logger.info("No URIs to resolve via SPARQL")
            return entities

        # Check cache
        cache_key = self._cache_key(all_uris)
        if cache_key in self._cache:
            logger.info("Using cached SPARQL results")
            return self._apply_cached_results(entities, self._cache[cache_key])

        # Build and execute batch queries
        results = {}

        # Query geometries
        if geometry_uris:
            geo_results = self._query_geometries(geometry_uris)
            results["geometries"] = geo_results

        # Query addresses
        if address_uris:
            addr_results = self._query_addresses(address_uris)
            results["addresses"] = addr_results

        # Query datings
        if dating_uris:
            dating_results = self._query_datings(dating_uris)
            results["datings"] = dating_results

        # Fallback: Query Wikidata for entities without geometry but with municipality
        # This handles cases where ArCo doesn't have direct coordinates
        entities_needing_fallback = [e for e in entities if not e.geometry_uri and e.municipality]
        if entities_needing_fallback:
            municipalities = list(
                set(e.municipality for e in entities_needing_fallback if e.municipality)
            )
            wikidata_results = self._query_wikidata_coordinates(municipalities)
            results["wikidata_coordinates"] = wikidata_results

        # Cache results
        self._cache[cache_key] = results
        self._save_cache()

        # Apply results to entities
        return self._apply_results(entities, results)

    def _query_geometries(self, uris: list[str]) -> dict[str, dict]:
        """
        Query geometry data for multiple URIs.

        Handles two ArCo geometry formats:
        1. geometry-point: Uses clv:serialization with WKT POINT
        2. geometry-1: Uses a-loc:hasCoordinates -> lat/long (WGS84)
        """
        try:
            from SPARQLWrapper import JSON, SPARQLWrapper
        except ImportError:
            return {}

        # Build VALUES clause
        values = " ".join(f"<{uri}>" for uri in set(uris))

        # Unified query that handles both geometry formats:
        # - clv:serialization for geometry-point (WKT format)
        # - a-loc:hasCoordinates -> lat/long for geometry-1 (WGS84)
        query = f"""
        PREFIX clv: <https://w3id.org/italia/onto/CLV/>
        PREFIX a-loc: <https://w3id.org/arco/ontology/location/>
        
        SELECT ?geo ?wkt ?lat ?lon WHERE {{
            VALUES ?geo {{ {values} }}
            OPTIONAL {{
                ?geo clv:serialization ?wkt .
            }}
            OPTIONAL {{
                ?geo a-loc:hasCoordinates ?coords .
                ?coords a-loc:lat ?lat .
                ?coords a-loc:long ?lon .
            }}
        }}
        """

        try:
            sparql = SPARQLWrapper(self.ENDPOINT)
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            sparql.setTimeout(60)

            response = sparql.query().convert()

            results = {}
            for binding in response.get("results", {}).get("bindings", []):
                geo_uri = binding.get("geo", {}).get("value")
                if not geo_uri:
                    continue

                # Try hasCoordinates first (more precise, already separate lat/lon)
                lat_str = binding.get("lat", {}).get("value")
                lon_str = binding.get("lon", {}).get("value")

                if lat_str and lon_str:
                    try:
                        results[geo_uri] = {
                            "latitude": float(lat_str),
                            "longitude": float(lon_str),
                        }
                        continue
                    except ValueError:
                        pass

                # Fallback to WKT serialization
                wkt = binding.get("wkt", {}).get("value")
                if wkt:
                    coords = self._parse_wkt_point(wkt)
                    if coords:
                        results[geo_uri] = {"latitude": coords[0], "longitude": coords[1]}

            logger.info(f"Resolved {len(results)} geometry URIs via SPARQL")
            return results

        except Exception as e:
            logger.error(f"SPARQL geometry query failed: {e}")
            return {}

    def _query_addresses(self, uris: list[str]) -> dict[str, dict]:
        """Query address data for multiple URIs."""
        try:
            from SPARQLWrapper import JSON, SPARQLWrapper
        except ImportError:
            return {}

        values = " ".join(f"<{uri}>" for uri in set(uris))

        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX clv: <https://w3id.org/italia/onto/CLV/>
        
        SELECT ?addr ?label ?cityLabel WHERE {{
            VALUES ?addr {{ {values} }}
            ?addr rdfs:label ?label .
            OPTIONAL {{
                ?addr clv:hasCity ?city .
                ?city rdfs:label ?cityLabel .
            }}
        }}
        """

        try:
            sparql = SPARQLWrapper(self.ENDPOINT)
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            sparql.setTimeout(60)

            response = sparql.query().convert()

            results = {}
            for binding in response.get("results", {}).get("bindings", []):
                addr_uri = binding.get("addr", {}).get("value")
                label = binding.get("label", {}).get("value")
                city = binding.get("cityLabel", {}).get("value")

                if addr_uri:
                    results[addr_uri] = {
                        "address_full": label,
                        "municipality": city,
                    }
                    # Try to extract region and province from full address
                    if label:
                        self._parse_address_components(label, results[addr_uri])

            logger.info(f"Resolved {len(results)} address URIs via SPARQL")
            return results

        except Exception as e:
            logger.error(f"SPARQL address query failed: {e}")
            return {}

    def _parse_address_components(self, address: str, result: dict) -> None:
        """
        Parse address components from full address string.

        Example: "ITALIA, Puglia, TA, Statte, strada Vicinale Accetta Piccola, 74010 Statte TA"
        """
        parts = [p.strip() for p in address.split(",")]
        if len(parts) >= 2:
            # Second part is usually region
            result["region_name"] = parts[1] if parts[1] != "ITALIA" else None
        if len(parts) >= 3:
            # Third part is usually province code
            province = parts[2]
            if len(province) <= 3:  # Province codes are 2-3 chars
                result["province"] = province

    def _query_datings(self, uris: list[str]) -> dict[str, dict]:
        """Query dating data for multiple URIs."""
        try:
            from SPARQLWrapper import JSON, SPARQLWrapper
        except ImportError:
            return {}

        values = " ".join(f"<{uri}>" for uri in set(uris))

        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?dating ?label WHERE {{
            VALUES ?dating {{ {values} }}
            ?dating rdfs:label ?label .
        }}
        """

        try:
            sparql = SPARQLWrapper(self.ENDPOINT)
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            sparql.setTimeout(60)

            response = sparql.query().convert()

            results = {}
            for binding in response.get("results", {}).get("bindings", []):
                dating_uri = binding.get("dating", {}).get("value")
                label = binding.get("label", {}).get("value")
                if dating_uri and label:
                    results[dating_uri] = {"dating_label": label}

            logger.info(f"Resolved {len(results)} dating URIs via SPARQL")
            return results

        except Exception as e:
            logger.error(f"SPARQL dating query failed: {e}")
            return {}

    def _query_wikidata_coordinates(self, municipalities: list[str]) -> dict[str, dict]:
        """
        Query Wikidata for coordinates of Italian municipalities.

        This is a fallback for entities that don't have geometry in ArCo
        but have a municipality name. Uses Wikidata to find the comune
        and its coordinates.

        Args:
            municipalities: List of municipality names (in Italian)

        Returns:
            Dictionary mapping lowercase municipality name to coordinates
        """
        if not municipalities:
            return {}

        try:
            from SPARQLWrapper import JSON, SPARQLWrapper
        except ImportError:
            return {}

        # Build VALUES clause with Italian labels
        values = " ".join(f'"{m}"@it' for m in municipalities)

        # Query Wikidata for Italian municipalities (comuni)
        # wdt:P31/wdt:P279* wd:Q747074 = instance of (or subclass of) "comune of Italy"
        # wdt:P625 = coordinate location
        query = f"""
        SELECT ?city ?cityLabel ?lat ?lon WHERE {{
            VALUES ?name {{ {values} }}
            ?city wdt:P31/wdt:P279* wd:Q747074 .
            ?city rdfs:label ?name .
            ?city wdt:P625 ?coords .
            BIND(geof:latitude(?coords) AS ?lat)
            BIND(geof:longitude(?coords) AS ?lon)
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "it,en" }}
        }}
        """

        try:
            sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            sparql.setTimeout(60)
            # Wikidata requires a User-Agent header
            sparql.addCustomHttpHeader(
                "User-Agent",
                "xch-MIND/1.0 (https://github.com/6MicheleStingo9/xch-MIND; research project)",
            )

            response = sparql.query().convert()

            results = {}
            for binding in response.get("results", {}).get("bindings", []):
                city_label = binding.get("cityLabel", {}).get("value", "")
                lat_str = binding.get("lat", {}).get("value")
                lon_str = binding.get("lon", {}).get("value")

                if city_label and lat_str and lon_str:
                    try:
                        # Store with lowercase key for case-insensitive matching
                        results[city_label.lower()] = {
                            "latitude": float(lat_str),
                            "longitude": float(lon_str),
                        }
                    except ValueError:
                        pass

            logger.info(
                f"Resolved {len(results)}/{len(municipalities)} municipality coordinates "
                f"via Wikidata fallback"
            )
            return results

        except Exception as e:
            logger.warning(f"Wikidata coordinates query failed: {e}")
            return {}

    def _apply_results(self, entities: list[DolmenEntity], results: dict) -> list[DolmenEntity]:
        """Apply SPARQL results to entities."""
        geometries = results.get("geometries", {})
        addresses = results.get("addresses", {})
        datings = results.get("datings", {})
        wikidata_coords = results.get("wikidata_coordinates", {})

        for entity in entities:
            resolved = False

            # Apply geometry
            if entity.geometry_uri and entity.geometry_uri in geometries:
                geo = geometries[entity.geometry_uri]
                entity.latitude = geo.get("latitude")
                entity.longitude = geo.get("longitude")
                resolved = True

            # Apply address
            if entity.address_uri and entity.address_uri in addresses:
                addr = addresses[entity.address_uri]
                entity.address_full = addr.get("address_full")
                if addr.get("municipality"):
                    entity.municipality = addr.get("municipality")
                if addr.get("province"):
                    entity.province = addr.get("province")
                if addr.get("region_name"):
                    entity.region_name = addr.get("region_name")
                resolved = True

            # Apply dating
            if entity.dating_uri and entity.dating_uri in datings:
                dating = datings[entity.dating_uri]
                entity.dating_label = dating.get("dating_label")
                resolved = True

            # Fallback: if no coordinates yet, try Wikidata by municipality name
            if not entity.has_coordinates and entity.municipality:
                mun_key = entity.municipality.lower()
                if mun_key in wikidata_coords:
                    coords = wikidata_coords[mun_key]
                    entity.latitude = coords.get("latitude")
                    entity.longitude = coords.get("longitude")
                    resolved = True
                    logger.debug(
                        f"Applied Wikidata fallback coordinates for {entity.display_name} "
                        f"({entity.municipality})"
                    )

            entity.sparql_resolved = resolved

        return entities

    def _apply_cached_results(
        self, entities: list[DolmenEntity], cached: dict
    ) -> list[DolmenEntity]:
        """Apply cached SPARQL results to entities."""
        return self._apply_results(entities, cached)


# =============================================================================
# MAIN LOADER FUNCTION
# =============================================================================


def load_all_entities(
    entities_dir: Path | str,
    resolve_sparql: bool = True,
    cache_dir: Path | str | None = None,
    progress_callback=None,
    limit: int | None = None,
) -> list[DolmenEntity]:
    """
    Load all dolmen entities from a directory of XML files.

    Args:
        entities_dir: Directory containing ArCo XML entity files
        resolve_sparql: Whether to resolve linked resources via SPARQL
        cache_dir: Directory for SPARQL cache. Defaults to entities_dir.
        progress_callback: Optional function called for each file processed
        limit: Optional maximum number of entities to load

    Returns:
        List of DolmenEntity objects
    """
    entities_dir = Path(entities_dir)
    if not entities_dir.exists():
        raise FileNotFoundError(f"Entities directory not found: {entities_dir}")

    # Find all XML files
    xml_files = sorted(entities_dir.glob("*.xml"))
    if not xml_files:
        logger.warning(f"No XML files found in {entities_dir}")
        return []

    logger.info(f"Found {len(xml_files)} XML files in {entities_dir}")

    # Parse all files
    entities: list[DolmenEntity] = []

    # Initialize progress callback if provided
    if progress_callback:
        # Notify total count if callback supports it (e.g. wrapper)
        # But we just call it per item
        pass

    for filepath in xml_files:
        # Check limit early
        if limit is not None and len(entities) >= limit:
            break

        entity = parse_entity_file(filepath)
        if entity:
            entities.append(entity)
        else:
            logger.warning(f"Skipped {filepath.name} (parsing failed)")

        if progress_callback:
            progress_callback()

    logger.info(f"Successfully parsed {len(entities)} entities")

    # Resolve SPARQL if requested
    if resolve_sparql and entities:
        cache_path = Path(cache_dir) if cache_dir else entities_dir
        resolver = SPARQLResolver(cache_dir=cache_path)
        entities = resolver.resolve_entities(entities)

        # Log resolution stats
        resolved_count = sum(1 for e in entities if e.sparql_resolved)
        coords_count = sum(1 for e in entities if e.has_coordinates)
        logger.info(
            f"SPARQL resolution: {resolved_count}/{len(entities)} entities resolved, "
            f"{coords_count} with coordinates"
        )

    return entities


# =============================================================================
# REGION MAPPING
# =============================================================================

REGION_ID_TO_NAME = {
    "01": "Piemonte",
    "02": "Valle d'Aosta",
    "03": "Lombardia",
    "04": "Trentino-Alto Adige",
    "05": "Veneto",
    "06": "Friuli-Venezia Giulia",
    "07": "Liguria",
    "08": "Emilia-Romagna",
    "09": "Toscana",
    "10": "Umbria",
    "11": "Marche",
    "12": "Lazio",
    "13": "Abruzzo",
    "14": "Molise",
    "15": "Campania",
    "16": "Puglia",
    "17": "Basilicata",
    "18": "Calabria",
    "19": "Sicilia",
    "20": "Sardegna",
}


def get_region_name(region_id: str | None) -> str | None:
    """Get region name from ArCo region identifier."""
    if not region_id:
        return None
    # Pad to 2 digits if needed
    region_id = region_id.zfill(2)
    return REGION_ID_TO_NAME.get(region_id)
