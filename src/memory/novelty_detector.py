"""
Novelty Detector - Identifies duplicate or overlapping proposals.

This module compares new proposals against the knowledge history to filter
out duplicates and ensure only novel assertions are generated.
"""

import logging
from typing import Any

from src.memory.history_store import KnowledgeHistoryStore, StoredCluster, StoredRelation

logger = logging.getLogger(__name__)


class NoveltyDetector:
    """
    Detects whether proposed assertions are novel or duplicates.

    Uses various similarity metrics to determine if a proposed cluster,
    relation, or path already exists in the knowledge history.
    """

    def __init__(
        self,
        history_store: KnowledgeHistoryStore,
        cluster_overlap_threshold: float = 0.8,
        relation_duplicate_check: bool = True,
        path_overlap_threshold: float = 0.7,
        min_confidence: float = 0.70,
    ):
        """
        Initialize the novelty detector.

        Args:
            history_store: The knowledge history store to check against
            cluster_overlap_threshold: Jaccard similarity threshold for cluster duplicates (0.0-1.0)
            relation_duplicate_check: Whether to check for exact relation duplicates
            path_overlap_threshold: Stop overlap threshold for path duplicates (0.0-1.0)
            min_confidence: Minimum confidence for assertions to be considered in
                novelty detection. Assertions below this threshold are ignored,
                allowing the LLM to re-propose improved versions of the same
                knowledge in future runs.
        """
        self.history = history_store
        self.cluster_overlap_threshold = cluster_overlap_threshold
        self.relation_duplicate_check = relation_duplicate_check
        self.path_overlap_threshold = path_overlap_threshold
        self.min_confidence = min_confidence

        # Cache for faster lookups
        self._cluster_sets: dict[str, set[str]] | None = None
        self._relation_pairs: set[tuple[str, str, str]] | None = None
        self._path_sets: dict[str, set[str]] | None = None

    def _build_caches(self) -> None:
        """Build lookup caches from history, filtering by min_confidence."""
        if self._cluster_sets is None:
            self._cluster_sets = {}
            for cluster in self.history.history.clusters:
                if cluster.confidence >= self.min_confidence:
                    self._cluster_sets[cluster.cluster_id] = set(cluster.member_uris)

        if self._relation_pairs is None:
            self._relation_pairs = set()
            for rel in self.history.history.relations:
                if rel.confidence >= self.min_confidence:
                    # Store both directions for symmetric relations
                    self._relation_pairs.add((rel.relation_type, rel.source_uri, rel.target_uri))
                    if rel.relation_type in ("nearTo", "contemporaryWith", "similarTo"):
                        self._relation_pairs.add(
                            (rel.relation_type, rel.target_uri, rel.source_uri)
                        )

        if self._path_sets is None:
            self._path_sets = {}
            for path in self.history.history.paths:
                if path.confidence >= self.min_confidence:
                    self._path_sets[path.path_id] = set(path.stop_uris)

    def invalidate_caches(self) -> None:
        """Invalidate caches after history updates."""
        self._cluster_sets = None
        self._relation_pairs = None
        self._path_sets = None

    # =========================================================================
    # CLUSTER NOVELTY
    # =========================================================================

    def is_novel_cluster(
        self,
        member_uris: list[str],
        cluster_type: str,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Check if a proposed cluster is novel.

        A cluster is considered novel if its Jaccard similarity with all
        existing clusters of the same type is below the threshold.

        Args:
            member_uris: URIs of the proposed cluster members
            cluster_type: Type of cluster (geographic, chronological, typological)

        Returns:
            Tuple of (is_novel, details) where details contains similarity info
        """
        self._build_caches()

        proposed_set = set(member_uris)
        if not proposed_set:
            return False, {"reason": "empty_cluster"}

        # Get existing clusters of same type
        existing_clusters = self.history.get_clusters_by_type(cluster_type)

        max_overlap = 0.0
        most_similar_cluster: StoredCluster | None = None

        for existing in existing_clusters:
            existing_set = self._cluster_sets.get(existing.cluster_id, set())
            if not existing_set:
                continue

            # Calculate Jaccard similarity
            intersection = len(proposed_set & existing_set)
            union = len(proposed_set | existing_set)
            jaccard = intersection / union if union > 0 else 0.0

            if jaccard > max_overlap:
                max_overlap = jaccard
                most_similar_cluster = existing

        is_novel = max_overlap < self.cluster_overlap_threshold

        details = {
            "max_overlap": max_overlap,
            "threshold": self.cluster_overlap_threshold,
            "is_novel": is_novel,
        }

        if most_similar_cluster:
            details["most_similar"] = {
                "cluster_id": most_similar_cluster.cluster_id,
                "label": most_similar_cluster.label,
                "jaccard": max_overlap,
            }

        if not is_novel:
            logger.debug(
                "Cluster rejected as duplicate (%.2f overlap with '%s')",
                max_overlap,
                most_similar_cluster.label if most_similar_cluster else "unknown",
            )

        return is_novel, details

    # =========================================================================
    # RELATION NOVELTY
    # =========================================================================

    def is_novel_relation(
        self,
        source_uri: str,
        target_uri: str,
        relation_type: str,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Check if a proposed relation is novel.

        A relation is considered duplicate if the same pair (in either direction)
        already exists for the same relation type.

        Args:
            source_uri: URI of the source entity
            target_uri: URI of the target entity
            relation_type: Type of relation (nearTo, contemporaryWith, similarTo)

        Returns:
            Tuple of (is_novel, details)
        """
        if not self.relation_duplicate_check:
            return True, {"reason": "duplicate_check_disabled"}

        self._build_caches()

        # Check if this exact relation exists (in either direction)
        forward = (relation_type, source_uri, target_uri)
        backward = (relation_type, target_uri, source_uri)

        exists = forward in self._relation_pairs or backward in self._relation_pairs

        details = {
            "relation_type": relation_type,
            "exists_in_history": exists,
            "is_novel": not exists,
        }

        if exists:
            logger.debug(
                "Relation rejected as duplicate: %s %s %s",
                source_uri.split("/")[-1],
                relation_type,
                target_uri.split("/")[-1],
            )

        return not exists, details

    # =========================================================================
    # PATH NOVELTY
    # =========================================================================

    def is_novel_path(
        self,
        stop_uris: list[str],
        theme: str | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Check if a proposed path is novel.

        A path is considered duplicate if it has high stop overlap with an
        existing path, optionally considering the theme as well.

        Args:
            stop_uris: URIs of the path stops
            theme: Optional theme to consider in duplicate detection

        Returns:
            Tuple of (is_novel, details)
        """
        self._build_caches()

        proposed_set = set(stop_uris)
        if not proposed_set:
            return False, {"reason": "empty_path"}

        max_overlap = 0.0
        most_similar_path_id: str | None = None

        for path in self.history.history.paths:
            existing_set = self._path_sets.get(path.path_id, set())
            if not existing_set:
                continue

            # Calculate stop overlap
            intersection = len(proposed_set & existing_set)
            union = len(proposed_set | existing_set)
            overlap = intersection / union if union > 0 else 0.0

            # Boost overlap score if themes match
            if theme and path.theme.lower() == theme.lower():
                overlap = min(overlap * 1.2, 1.0)

            if overlap > max_overlap:
                max_overlap = overlap
                most_similar_path_id = path.path_id

        is_novel = max_overlap < self.path_overlap_threshold

        details = {
            "max_overlap": max_overlap,
            "threshold": self.path_overlap_threshold,
            "is_novel": is_novel,
        }

        if most_similar_path_id:
            details["most_similar_path_id"] = most_similar_path_id

        if not is_novel:
            logger.debug("Path rejected as duplicate (%.2f overlap)", max_overlap)

        return is_novel, details

    # =========================================================================
    # BATCH FILTERING
    # =========================================================================

    def filter_novel_clusters(
        self,
        proposals: list[dict[str, Any]],
        cluster_type: str,
    ) -> list[dict[str, Any]]:
        """
        Filter a list of cluster proposals, keeping only novel ones.

        Args:
            proposals: List of cluster proposals with 'member_uris' key
            cluster_type: Type of clusters

        Returns:
            List of novel proposals
        """
        novel = []
        for proposal in proposals:
            member_uris = proposal.get("member_uris", [])
            is_novel, details = self.is_novel_cluster(member_uris, cluster_type)
            if is_novel:
                proposal["_novelty_details"] = details
                novel.append(proposal)
            else:
                logger.info(
                    "Filtered duplicate cluster: %s (%.2f overlap)",
                    proposal.get("label", "unknown"),
                    details.get("max_overlap", 0),
                )

        logger.info(
            "Novelty filter: %d/%d %s clusters are novel",
            len(novel),
            len(proposals),
            cluster_type,
        )
        return novel

    def filter_novel_relations(
        self,
        proposals: list[dict[str, Any]],
        relation_type: str,
    ) -> list[dict[str, Any]]:
        """
        Filter a list of relation proposals, keeping only novel ones.

        Args:
            proposals: List of relation proposals with 'source_uri' and 'target_uri' keys
            relation_type: Type of relations

        Returns:
            List of novel proposals
        """
        novel = []
        for proposal in proposals:
            source_uri = proposal.get("source_uri", "")
            target_uri = proposal.get("target_uri", "")
            is_novel, details = self.is_novel_relation(source_uri, target_uri, relation_type)
            if is_novel:
                proposal["_novelty_details"] = details
                novel.append(proposal)

        logger.info(
            "Novelty filter: %d/%d %s relations are novel",
            len(novel),
            len(proposals),
            relation_type,
        )
        return novel

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_novelty_stats(self) -> dict[str, Any]:
        """Get statistics about the novelty detection."""
        self._build_caches()

        return {
            "clusters_in_history": len(self._cluster_sets) if self._cluster_sets else 0,
            "relations_in_history": len(self._relation_pairs) // 2 if self._relation_pairs else 0,
            "paths_in_history": len(self._path_sets) if self._path_sets else 0,
            "cluster_overlap_threshold": self.cluster_overlap_threshold,
            "path_overlap_threshold": self.path_overlap_threshold,
        }
