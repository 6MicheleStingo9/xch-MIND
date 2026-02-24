"""
Knowledge History Store - Persistence layer for cross-run memory.

This module persists assertions, clusters, relations, and paths from previous
runs to enable the system to build upon existing knowledge and avoid duplicates.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS FOR HISTORY
# =============================================================================


class StoredCluster(BaseModel):
    """A cluster stored in history."""

    cluster_id: str
    cluster_type: str  # geographic, chronological, typological
    label: str
    member_uris: list[str]
    confidence: float
    run_id: str
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class StoredRelation(BaseModel):
    """A relation stored in history."""

    relation_id: str
    relation_type: str  # nearTo, contemporaryWith, similarTo
    source_uri: str
    target_uri: str
    confidence: float
    run_id: str
    created_at: str
    value: float | None = None  # distance_km, overlap, jaccard, etc.
    metadata: dict[str, Any] = Field(default_factory=dict)


class StoredPath(BaseModel):
    """A thematic path stored in history."""

    path_id: str
    theme: str
    path_type: str
    stop_uris: list[str]
    confidence: float
    run_id: str
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class HistoryStatistics(BaseModel):
    """Statistics about the knowledge history."""

    total_runs: int = 0
    total_clusters: int = 0
    total_relations: int = 0
    total_paths: int = 0
    last_run_id: str | None = None
    last_run_at: str | None = None
    entities_seen: list[str] = Field(default_factory=list)


class KnowledgeHistory(BaseModel):
    """Complete knowledge history from all runs."""

    version: str = "1.0"
    clusters: list[StoredCluster] = Field(default_factory=list)
    relations: list[StoredRelation] = Field(default_factory=list)
    paths: list[StoredPath] = Field(default_factory=list)
    statistics: HistoryStatistics = Field(default_factory=HistoryStatistics)


# =============================================================================
# HISTORY STORE
# =============================================================================


class KnowledgeHistoryStore:
    """
    Manages persistence of knowledge across runs.

    The store maintains a JSON file with all assertions from previous runs,
    enabling the system to:
    1. Avoid proposing duplicate clusters/relations
    2. Provide context to the LLM about existing knowledge
    3. Track statistics about the knowledge base growth
    """

    DEFAULT_FILENAME = ".knowledge_history.json"

    def __init__(self, output_dir: Path | str):
        """
        Initialize the history store.

        Args:
            output_dir: Directory where the history file is stored
        """
        self.output_dir = Path(output_dir)
        self.history_path = self.output_dir / self.DEFAULT_FILENAME
        self._history: KnowledgeHistory | None = None

    @property
    def history(self) -> KnowledgeHistory:
        """Get the current history, loading from disk if needed."""
        if self._history is None:
            self._history = self._load()
        return self._history

    def _load(self) -> KnowledgeHistory:
        """Load history from disk or create empty history."""
        if self.history_path.exists():
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                history = KnowledgeHistory(**data)
                logger.info(
                    "Loaded knowledge history: %d clusters, %d relations, %d paths from %d runs",
                    len(history.clusters),
                    len(history.relations),
                    len(history.paths),
                    history.statistics.total_runs,
                )
                return history
            except Exception as e:
                logger.warning("Failed to load history, starting fresh: %s", e)
                return KnowledgeHistory()
        else:
            logger.info("No existing history found, starting fresh")
            return KnowledgeHistory()

    def _backup_if_exists(self) -> Path | None:
        """Create a timestamped backup of the history file if it exists.

        Returns:
            Path to the backup file, or None if no backup was created.
        """
        if not self.history_path.exists():
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.history_path.with_name(f".knowledge_history_bak_{timestamp}.json")

        try:
            shutil.copy2(self.history_path, backup_path)
            logger.info("Created backup of knowledge history: %s", backup_path)
            return backup_path
        except Exception as e:
            logger.warning("Failed to create history backup: %s", e)
            return None

    def save(self, backup: bool = True) -> None:
        """Save current history to disk.

        Args:
            backup: If True and the file exists, create a timestamped backup first.
        """
        if self._history is None:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create backup of existing history before overwriting
        if backup:
            self._backup_if_exists()

        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(self._history.model_dump(), f, indent=2, ensure_ascii=False)
        logger.info("Saved knowledge history to %s", self.history_path)

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def get_clusters_by_type(self, cluster_type: str) -> list[StoredCluster]:
        """Get all clusters of a specific type."""
        return [c for c in self.history.clusters if c.cluster_type == cluster_type]

    def get_relations_by_type(self, relation_type: str) -> list[StoredRelation]:
        """Get all relations of a specific type."""
        return [r for r in self.history.relations if r.relation_type == relation_type]

    def get_clusters_containing_uri(self, uri: str) -> list[StoredCluster]:
        """Get all clusters that contain a specific URI."""
        return [c for c in self.history.clusters if uri in c.member_uris]

    def get_relations_involving_uri(self, uri: str) -> list[StoredRelation]:
        """Get all relations involving a specific URI."""
        return [r for r in self.history.relations if r.source_uri == uri or r.target_uri == uri]

    def relation_exists(self, source_uri: str, target_uri: str, relation_type: str) -> bool:
        """Check if a specific relation already exists (in either direction)."""
        for r in self.history.relations:
            if r.relation_type != relation_type:
                continue
            if (r.source_uri == source_uri and r.target_uri == target_uri) or (
                r.source_uri == target_uri and r.target_uri == source_uri
            ):
                return True
        return False

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the knowledge history for LLM context."""
        h = self.history

        # Group clusters by type
        clusters_by_type = {}
        for c in h.clusters:
            if c.cluster_type not in clusters_by_type:
                clusters_by_type[c.cluster_type] = []
            clusters_by_type[c.cluster_type].append(
                {
                    "label": c.label,
                    "members": len(c.member_uris),
                    "confidence": c.confidence,
                }
            )

        # Group relations by type
        relations_by_type = {}
        for r in h.relations:
            if r.relation_type not in relations_by_type:
                relations_by_type[r.relation_type] = 0
            relations_by_type[r.relation_type] += 1

        # Path summary
        path_themes = [p.theme for p in h.paths]

        return {
            "total_runs": h.statistics.total_runs,
            "clusters": clusters_by_type,
            "relations": relations_by_type,
            "paths": {
                "count": len(h.paths),
                "themes": path_themes[:10],  # Limit to 10 themes
            },
            "entities_known": len(h.statistics.entities_seen),
        }

    # =========================================================================
    # UPDATE METHODS
    # =========================================================================

    def add_cluster(
        self,
        cluster_id: str,
        cluster_type: str,
        label: str,
        member_uris: list[str],
        confidence: float,
        run_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a cluster to history."""
        cluster = StoredCluster(
            cluster_id=cluster_id,
            cluster_type=cluster_type,
            label=label,
            member_uris=member_uris,
            confidence=confidence,
            run_id=run_id,
            created_at=datetime.now().isoformat(),
            metadata=metadata or {},
        )
        self.history.clusters.append(cluster)
        self.history.statistics.total_clusters += 1

        # Track seen entities
        for uri in member_uris:
            if uri not in self.history.statistics.entities_seen:
                self.history.statistics.entities_seen.append(uri)

    def add_relation(
        self,
        relation_id: str,
        relation_type: str,
        source_uri: str,
        target_uri: str,
        confidence: float,
        run_id: str,
        value: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a relation to history."""
        relation = StoredRelation(
            relation_id=relation_id,
            relation_type=relation_type,
            source_uri=source_uri,
            target_uri=target_uri,
            confidence=confidence,
            run_id=run_id,
            created_at=datetime.now().isoformat(),
            value=value,
            metadata=metadata or {},
        )
        self.history.relations.append(relation)
        self.history.statistics.total_relations += 1

        # Track seen entities
        for uri in [source_uri, target_uri]:
            if uri not in self.history.statistics.entities_seen:
                self.history.statistics.entities_seen.append(uri)

    def add_path(
        self,
        path_id: str,
        theme: str,
        path_type: str,
        stop_uris: list[str],
        confidence: float,
        run_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a path to history."""
        path = StoredPath(
            path_id=path_id,
            theme=theme,
            path_type=path_type,
            stop_uris=stop_uris,
            confidence=confidence,
            run_id=run_id,
            created_at=datetime.now().isoformat(),
            metadata=metadata or {},
        )
        self.history.paths.append(path)
        self.history.statistics.total_paths += 1

        # Track seen entities
        for uri in stop_uris:
            if uri not in self.history.statistics.entities_seen:
                self.history.statistics.entities_seen.append(uri)

    def finalize_run(self, run_id: str) -> None:
        """Mark a run as complete and update statistics."""
        self.history.statistics.total_runs += 1
        self.history.statistics.last_run_id = run_id
        self.history.statistics.last_run_at = datetime.now().isoformat()
        self.save()
        logger.info(
            "Finalized run %s. Total: %d clusters, %d relations, %d paths",
            run_id,
            self.history.statistics.total_clusters,
            self.history.statistics.total_relations,
            self.history.statistics.total_paths,
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def clear(self) -> None:
        """Clear all history (use with caution)."""
        self._history = KnowledgeHistory()
        if self.history_path.exists():
            self.history_path.unlink()
        logger.warning("Cleared all knowledge history")

    def export_for_analysis(self) -> dict[str, Any]:
        """Export history in a format suitable for analysis."""
        return {
            "version": self.history.version,
            "statistics": self.history.statistics.model_dump(),
            "clusters": [c.model_dump() for c in self.history.clusters],
            "relations": [r.model_dump() for r in self.history.relations],
            "paths": [p.model_dump() for p in self.history.paths],
        }
