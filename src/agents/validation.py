"""
Validation Layer - Algorithmic validation of LLM proposals.

This module provides functions to validate LLM-generated proposals
using deterministic algorithms (Haversine distance, period overlap,
Jaccard similarity, etc.).

The validation layer contributes 30% to the final confidence score,
ensuring that LLM proposals are grounded in verifiable facts.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION RESULTS
# =============================================================================


@dataclass
class ValidationResult:
    """Result of validating an LLM proposal."""

    is_valid: bool
    validation_score: float  # 0.0 - 1.0
    details: dict[str, Any]
    warnings: list[str]


# =============================================================================
# GEOGRAPHIC VALIDATION
# =============================================================================


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points in kilometers.

    Uses the Haversine formula for accuracy.

    Args:
        lat1, lon1: Coordinates of first point
        lat2, lon2: Coordinates of second point

    Returns:
        Distance in kilometers
    """
    R = 6371.0  # Earth's radius in km

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def validate_coordinates(lat: float | None, lon: float | None) -> bool:
    """Validate that coordinates are valid and not null island."""
    if lat is None or lon is None:
        return False
    if lat == 0.0 and lon == 0.0:
        return False  # Null island
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return False
    return True


def validate_geographic_cluster(
    member_uris: list[str],
    entities_by_uri: dict[str, Any],
    max_radius_km: float = 10.0,
    min_cluster_size: int = 2,
    max_cluster_size: int | None = None,
) -> ValidationResult:
    """
    Validate a proposed geographic cluster.

    Checks:
    1. All member URIs exist in entities
    2. All members have valid coordinates
    3. Cluster size is within min/max bounds
    4. Cluster radius is within threshold
    5. Members are actually proximate to each other

    Args:
        member_uris: URIs of proposed cluster members
        entities_by_uri: Dict mapping URI to entity object
        max_radius_km: Maximum allowed cluster radius
        min_cluster_size: Minimum number of members (default 2)
        max_cluster_size: Maximum number of members (None = no limit)

    Returns:
        ValidationResult with score based on cluster tightness
    """
    warnings = []
    details = {}

    # Check URIs exist
    valid_members = []
    for uri in member_uris:
        if uri not in entities_by_uri:
            warnings.append(f"URI not found: {uri}")
            continue
        entity = entities_by_uri[uri]
        if not validate_coordinates(entity.latitude, entity.longitude):
            warnings.append(f"Invalid coordinates for {uri}")
            continue
        valid_members.append(entity)

    # Check minimum cluster size
    if len(valid_members) < min_cluster_size:
        return ValidationResult(
            is_valid=False,
            validation_score=0.0,
            details={"reason": f"Less than {min_cluster_size} valid members with coordinates"},
            warnings=warnings,
        )

    # Check maximum cluster size
    if max_cluster_size is not None and len(valid_members) > max_cluster_size:
        return ValidationResult(
            is_valid=False,
            validation_score=0.0,
            details={
                "reason": f"Cluster has {len(valid_members)} members, exceeds max {max_cluster_size}"
            },
            warnings=warnings,
        )

    # Calculate centroid
    avg_lat = sum(e.latitude for e in valid_members) / len(valid_members)
    avg_lon = sum(e.longitude for e in valid_members) / len(valid_members)
    details["centroid"] = {"lat": avg_lat, "lon": avg_lon}

    # Calculate max distance from centroid
    max_dist = 0.0
    for entity in valid_members:
        dist = haversine_distance(avg_lat, avg_lon, entity.latitude, entity.longitude)
        max_dist = max(max_dist, dist)

    details["radius_km"] = max_dist
    details["member_count"] = len(valid_members)

    # Validate radius
    if max_dist > max_radius_km:
        return ValidationResult(
            is_valid=False,
            validation_score=0.0,
            details={**details, "reason": f"Radius {max_dist:.1f}km exceeds max {max_radius_km}km"},
            warnings=warnings,
        )

    # Calculate score based on cluster tightness
    # Tighter clusters = higher score
    # Score of 1.0 for radius < 5km, decreasing to 0.5 at max_radius_km
    if max_dist < 5.0:
        score = 1.0
    else:
        score = 1.0 - (0.5 * (max_dist - 5.0) / (max_radius_km - 5.0))

    return ValidationResult(
        is_valid=True,
        validation_score=score,
        details=details,
        warnings=warnings,
    )


def validate_near_relation(
    source_uri: str,
    target_uri: str,
    entities_by_uri: dict[str, Any],
    max_distance_km: float = 30.0,
) -> ValidationResult:
    """
    Validate a proposed nearTo relation.

    Checks that both sites exist and are within distance threshold.

    Args:
        source_uri: URI of source site
        target_uri: URI of target site
        entities_by_uri: Dict mapping URI to entity
        max_distance_km: Maximum distance to be considered "near"

    Returns:
        ValidationResult with score based on proximity
    """
    warnings = []

    # Get entities
    source = entities_by_uri.get(source_uri)
    target = entities_by_uri.get(target_uri)

    if not source:
        return ValidationResult(
            False, 0.0, {"reason": f"Source URI not found: {source_uri}"}, warnings
        )
    if not target:
        return ValidationResult(
            False, 0.0, {"reason": f"Target URI not found: {target_uri}"}, warnings
        )

    # Validate coordinates
    if not validate_coordinates(source.latitude, source.longitude):
        return ValidationResult(False, 0.0, {"reason": "Source has invalid coordinates"}, warnings)
    if not validate_coordinates(target.latitude, target.longitude):
        return ValidationResult(False, 0.0, {"reason": "Target has invalid coordinates"}, warnings)

    # Calculate distance
    distance = haversine_distance(
        source.latitude, source.longitude, target.latitude, target.longitude
    )

    details = {"distance_km": distance}

    if distance > max_distance_km:
        return ValidationResult(
            False,
            0.0,
            {**details, "reason": f"Distance {distance:.1f}km exceeds max {max_distance_km}km"},
            warnings,
        )

    # Score: 1.0 for very close (<5km), decreasing to 0.5 at max_distance
    if distance < 5.0:
        score = 1.0
    else:
        score = 1.0 - (0.5 * (distance - 5.0) / (max_distance_km - 5.0))

    return ValidationResult(True, score, details, warnings)


# =============================================================================
# CHRONOLOGICAL VALIDATION
# =============================================================================


# Standard period definitions with approximate date ranges
PERIOD_RANGES = {
    "paleolithic": (-2500000, -10000),
    "mesolithic": (-10000, -6000),
    "neolithic": (-6000, -3000),
    "eneolithic": (-3000, -2300),
    "chalcolithic": (-3000, -2300),
    "copper age": (-3000, -2300),
    "bronze age": (-2300, -900),
    "early bronze age": (-2300, -1700),
    "middle bronze age": (-1700, -1350),
    "late bronze age": (-1350, -900),
    "iron age": (-900, -27),
    "protohistory": (-3000, -27),
}


def periods_overlap(
    start1: int | None,
    end1: int | None,
    start2: int | None,
    end2: int | None,
    tolerance_years: int = 100,
) -> tuple[bool, float]:
    """
    Check if two date ranges overlap.

    Args:
        start1, end1: First period range (negative = BC)
        start2, end2: Second period range
        tolerance_years: Buffer for overlap detection (default 100yr for prehistoric chronologies)

    Returns:
        Tuple of (overlaps: bool, overlap_score: float)
    """
    if any(v is None for v in [start1, end1, start2, end2]):
        return False, 0.0

    # Add tolerance
    s1, e1 = start1 - tolerance_years, end1 + tolerance_years
    s2, e2 = start2 - tolerance_years, end2 + tolerance_years

    # Check overlap
    overlaps = s1 <= e2 and s2 <= e1

    if not overlaps:
        return False, 0.0

    # Calculate overlap score based on how much they overlap
    overlap_start = max(s1, s2)
    overlap_end = min(e1, e2)
    overlap_duration = overlap_end - overlap_start

    # Normalize by the smaller period duration
    period1_duration = e1 - s1
    period2_duration = e2 - s2
    min_duration = min(period1_duration, period2_duration)

    if min_duration <= 0:
        return True, 0.5  # Default score for degenerate cases

    score = min(1.0, overlap_duration / min_duration)
    return True, score


def validate_chronological_cluster(
    member_uris: list[str],
    entities_by_uri: dict[str, Any],
    period_tolerance_years: int = 100,
    min_cluster_size: int = 2,
    max_cluster_size: int | None = None,
) -> ValidationResult:
    """
    Validate a proposed chronological cluster.

    Checks that members have overlapping periods.

    Args:
        member_uris: URIs of proposed cluster members
        entities_by_uri: Dict mapping URI to entity
        period_tolerance_years: Tolerance for period overlap
        min_cluster_size: Minimum number of members (default 2)
        max_cluster_size: Maximum number of members (None = no limit)

    Returns:
        ValidationResult with score based on temporal coherence
    """
    warnings = []
    details = {}

    # Get valid members with period info
    valid_members = []
    for uri in member_uris:
        entity = entities_by_uri.get(uri)
        if not entity:
            warnings.append(f"URI not found: {uri}")
            continue
        if not entity.period_label:
            warnings.append(f"No period info for {uri}")
            continue
        valid_members.append(entity)

    # Check minimum cluster size
    if len(valid_members) < min_cluster_size:
        return ValidationResult(
            False,
            0.0,
            {"reason": f"Less than {min_cluster_size} members with period info"},
            warnings,
        )

    # Check maximum cluster size
    if max_cluster_size is not None and len(valid_members) > max_cluster_size:
        return ValidationResult(
            False,
            0.0,
            {"reason": f"Cluster has {len(valid_members)} members, exceeds max {max_cluster_size}"},
            warnings,
        )

    details["member_count"] = len(valid_members)

    # For now, use a simpler validation: check if period labels match
    # (Full implementation would parse dates from labels)
    period_labels = [e.period_label.lower() for e in valid_members if e.period_label]
    unique_periods = set(period_labels)

    details["unique_periods"] = list(unique_periods)

    # Score based on period consistency
    if len(unique_periods) == 1:
        score = 1.0  # All same period
    elif len(unique_periods) == 2:
        score = 0.7  # Two related periods
    elif len(unique_periods) <= len(valid_members) / 2:
        score = 0.5  # Some consistency
    else:
        score = 0.3  # Low consistency

    return ValidationResult(True, score, details, warnings)


def validate_contemporary_relation(
    source_uri: str,
    target_uri: str,
    entities_by_uri: dict[str, Any],
) -> ValidationResult:
    """Validate a proposed contemporaryWith relation."""
    warnings = []

    source = entities_by_uri.get(source_uri)
    target = entities_by_uri.get(target_uri)

    if not source:
        return ValidationResult(False, 0.0, {"reason": f"Source not found: {source_uri}"}, warnings)
    if not target:
        return ValidationResult(False, 0.0, {"reason": f"Target not found: {target_uri}"}, warnings)

    source_period = source.period_label.lower() if source.period_label else ""
    target_period = target.period_label.lower() if target.period_label else ""

    if not source_period or not target_period:
        return ValidationResult(False, 0.0, {"reason": "Missing period info"}, warnings)

    # Simple period matching
    if source_period == target_period:
        return ValidationResult(True, 1.0, {"shared_period": source_period}, warnings)

    # Check for partial match
    for period in PERIOD_RANGES:
        if period in source_period and period in target_period:
            return ValidationResult(True, 0.8, {"matched_period": period}, warnings)

    return ValidationResult(
        False, 0.3, {"source_period": source_period, "target_period": target_period}, warnings
    )


# =============================================================================
# TYPOLOGICAL VALIDATION
# =============================================================================


def jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def validate_typological_cluster(
    member_uris: list[str],
    feature_extractions: dict[str, set[str]],
    min_shared_features: int = 1,
    majority_threshold: float = 0.6,
    min_cluster_size: int = 2,
    max_cluster_size: int | None = None,
) -> ValidationResult:
    """
    Validate a proposed typological cluster using MAJORITY logic.

    Instead of requiring ALL members to share features, we use a majority approach:
    a feature is considered "shared" if at least majority_threshold (60%) of members have it.

    Args:
        member_uris: URIs of proposed cluster members
        feature_extractions: Dict mapping URI to set of features
        min_shared_features: Minimum features shared by majority of members
        majority_threshold: Fraction of members that must share a feature (default 0.6 = 60%)
        min_cluster_size: Minimum number of members (default 2)
        max_cluster_size: Maximum number of members (None = no limit)

    Returns:
        ValidationResult with score based on feature overlap
    """
    warnings = []
    details = {}

    # Get features for all members
    member_features = []
    member_uris_with_features = []
    for uri in member_uris:
        features = feature_extractions.get(uri)
        if not features:
            warnings.append(f"No features extracted for {uri}")
            continue
        member_features.append(features)
        member_uris_with_features.append(uri)

    # Check minimum cluster size
    if len(member_features) < min_cluster_size:
        return ValidationResult(
            False, 0.0, {"reason": f"Less than {min_cluster_size} members with features"}, warnings
        )

    # Check maximum cluster size
    if max_cluster_size is not None and len(member_features) > max_cluster_size:
        return ValidationResult(
            False,
            0.0,
            {
                "reason": f"Cluster has {len(member_features)} members, exceeds max {max_cluster_size}"
            },
            warnings,
        )

    # MAJORITY LOGIC: Count how many members have each feature
    all_features = set()
    for features in member_features:
        all_features.update(features)

    feature_counts = {}
    for feature in all_features:
        count = sum(1 for mf in member_features if feature in mf)
        feature_counts[feature] = count

    # A feature is "shared" if at least majority_threshold of members have it
    min_members_for_majority = int(len(member_features) * majority_threshold)
    min_members_for_majority = max(2, min_members_for_majority)  # At least 2 members

    shared_by_majority = [
        f for f, count in feature_counts.items() if count >= min_members_for_majority
    ]

    details["shared_features"] = shared_by_majority
    details["member_count"] = len(member_features)
    details["majority_threshold"] = majority_threshold
    details["min_members_for_majority"] = min_members_for_majority
    details["feature_counts"] = {
        f: c for f, c in feature_counts.items() if c >= 2
    }  # Features shared by at least 2

    if len(shared_by_majority) < min_shared_features:
        return ValidationResult(
            False,
            0.3,
            {
                **details,
                "reason": f"Only {len(shared_by_majority)} features shared by majority ({min_members_for_majority}+ members)",
            },
            warnings,
        )

    # Score based on number of shared features and coverage
    feature_score = min(1.0, len(shared_by_majority) / 5)  # Max score at 5+ shared features

    # Bonus for high coverage (many members sharing features)
    avg_coverage = (
        sum(feature_counts[f] for f in shared_by_majority)
        / (len(shared_by_majority) * len(member_features))
        if shared_by_majority
        else 0
    )
    coverage_bonus = avg_coverage * 0.2  # Up to 0.2 bonus

    score = min(1.0, feature_score + coverage_bonus)

    return ValidationResult(True, round(score, 2), details, warnings)


# =============================================================================
# PATH VALIDATION
# =============================================================================


def validate_thematic_path(
    stops: list[dict],
    entities_by_uri: dict[str, Any],
    path_type: str = "mixed",
    max_total_distance_km: float = 200.0,
) -> ValidationResult:
    """
    Validate a proposed thematic path.

    Validation strategy depends on path_type:
    - 'geographic': Strict distance validation (max 200km total)
    - 'chronological': Flexible - validates entity existence only, no distance limit
    - 'typological': Flexible - validates entity existence only, no distance limit
    - 'mixed': Flexible - validates entity existence only, no distance limit

    Checks for all types:
    1. All stop URIs exist in entities
    2. Valid coordinates for distance calculation (if applicable)

    Additional checks for 'geographic' paths:
    3. Total path distance within max_total_distance_km
    4. Path has reasonable leg consistency

    Args:
        stops: List of stop dicts with 'site_uri' and 'order'
        entities_by_uri: Dict mapping URI to entity
        path_type: Type of path - determines validation strictness
        max_total_distance_km: Maximum allowed total distance (only for geographic paths)
        stops: List of stop dicts with 'site_uri' and 'order'
        entities_by_uri: Dict mapping URI to entity
        path_type: Type of path - determines validation strictness
        max_total_distance_km: Maximum allowed total distance (only for geographic paths)

    Returns:
        ValidationResult with score based on path quality
    """
    warnings = []
    details = {"path_type": path_type}

    if len(stops) < 2:
        return ValidationResult(False, 0.0, {"reason": "Path needs at least 2 stops"}, warnings)

    # Sort stops by order
    sorted_stops = sorted(stops, key=lambda s: s.get("order", 0))

    # Validate stop entities exist
    valid_stops = []
    for stop in sorted_stops:
        uri = stop.get("site_uri")
        entity = entities_by_uri.get(uri)
        if not entity:
            warnings.append(f"Stop URI not found: {uri}")
            continue
        valid_stops.append(entity)

    if len(valid_stops) < 2:
        return ValidationResult(False, 0.0, {"reason": "Less than 2 valid stops"}, warnings)

    details["stop_count"] = len(valid_stops)

    # For non-geographic paths, we don't enforce distance limits
    # Chronological/Typological paths can span entire regions
    is_geographic_path = path_type.lower() == "geographic"

    # Calculate distances for informational purposes (all path types)
    total_distance = 0.0
    leg_distances = []
    coords_valid = True

    for entity in valid_stops:
        if not validate_coordinates(entity.latitude, entity.longitude):
            coords_valid = False
            break

    if coords_valid:
        for i in range(len(valid_stops) - 1):
            dist = haversine_distance(
                valid_stops[i].latitude,
                valid_stops[i].longitude,
                valid_stops[i + 1].latitude,
                valid_stops[i + 1].longitude,
            )
            leg_distances.append(dist)
            total_distance += dist

        details["total_distance_km"] = round(total_distance, 2)
        details["leg_distances_km"] = [round(d, 2) for d in leg_distances]
    else:
        warnings.append("Some stops have invalid coordinates - distance not calculated")
        details["total_distance_km"] = None

    # === GEOGRAPHIC PATH: Strict distance validation ===
    if is_geographic_path:
        if not coords_valid:
            return ValidationResult(
                False,
                0.0,
                {**details, "reason": "Geographic path requires valid coordinates"},
                warnings,
            )

        if total_distance > max_total_distance_km:
            return ValidationResult(
                False,
                0.0,
                {
                    **details,
                    "reason": f"Total distance {total_distance:.1f}km exceeds max {max_total_distance_km}km",
                },
                warnings,
            )

        # Score based on path efficiency and reasonable leg lengths
        if leg_distances:
            avg_leg = total_distance / len(leg_distances)
            variance = sum((d - avg_leg) ** 2 for d in leg_distances) / len(leg_distances)
            cv = (variance**0.5) / avg_leg if avg_leg > 0 else 0

            # Lower coefficient of variation = more consistent = better score
            consistency_score = max(0.5, 1.0 - cv / 2)
        else:
            consistency_score = 0.5

        # Also factor in reasonable total distance
        distance_score = 1.0 - (total_distance / max_total_distance_km) * 0.5

        score = (consistency_score + distance_score) / 2
        return ValidationResult(True, score, details, warnings)

    # === CHRONOLOGICAL/TYPOLOGICAL/MIXED PATH: Flexible validation ===
    # These paths prioritize thematic coherence over geographic proximity
    # They can span large distances as they connect sites by period or type

    # Base score: entity existence validation passed
    base_score = 0.8

    # Bonus for having more valid stops
    stop_bonus = min(0.1, len(valid_stops) * 0.02)

    # Small penalty if coordinates are missing (informational warning only)
    coord_penalty = 0.0 if coords_valid else 0.05

    score = base_score + stop_bonus - coord_penalty

    details["validation_mode"] = "flexible (non-geographic)"

    return ValidationResult(True, round(score, 2), details, warnings)
