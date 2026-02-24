"""
Context Builder - Prepares LLM prompts with historical knowledge.

This module builds context sections for LLM prompts that inform the model
about existing knowledge from previous runs, guiding it toward novel discoveries.
"""

import logging
from typing import Any

from src.memory.history_store import KnowledgeHistoryStore

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds context sections for LLM prompts based on knowledge history.

    The context helps the LLM:
    1. Understand what knowledge already exists
    2. Avoid proposing duplicates
    3. Focus on novel discoveries and connections
    """

    def __init__(
        self,
        history_store: KnowledgeHistoryStore,
        max_clusters_per_type: int = 10,
        max_relations_shown: int = 20,
        max_paths_shown: int = 5,
        min_confidence: float = 0.70,
    ):
        """
        Initialize the context builder.

        Args:
            history_store: The knowledge history store
            max_clusters_per_type: Maximum clusters to show per type in context
            max_relations_shown: Maximum relations to show in context
            max_paths_shown: Maximum paths to show in context
            min_confidence: Minimum confidence for assertions to be included in
                LLM context. Assertions below this threshold are kept in the
                history but not shown to the LLM, preventing low-quality
                knowledge from polluting future prompts.
        """
        self.history = history_store
        self.max_clusters_per_type = max_clusters_per_type
        self.max_relations_shown = max_relations_shown
        self.max_paths_shown = max_paths_shown
        self.min_confidence = min_confidence

    def build_geo_context(self) -> str:
        """
        Build context for the GeoAnalyzer.

        Returns:
            Context string to include in the LLM prompt
        """
        h = self.history.history

        if h.statistics.total_runs == 0:
            return ""

        lines = [
            "",
            "## EXISTING KNOWLEDGE FROM PREVIOUS RUNS",
            f"(Based on {h.statistics.total_runs} previous analysis runs)",
            "",
        ]

        # Geographic clusters (filtered by minimum confidence)
        geo_clusters = [
            c
            for c in self.history.get_clusters_by_type("geographic")
            if c.confidence >= self.min_confidence
        ]
        if geo_clusters:
            lines.append("### Already Identified Geographic Clusters:")
            for cluster in geo_clusters[: self.max_clusters_per_type]:
                members_preview = ", ".join(uri.split("/")[-1] for uri in cluster.member_uris[:3])
                if len(cluster.member_uris) > 3:
                    members_preview += f", ... (+{len(cluster.member_uris) - 3} more)"
                lines.append(
                    f"- **{cluster.label}** ({len(cluster.member_uris)} sites): {members_preview}"
                )
            if len(geo_clusters) > self.max_clusters_per_type:
                lines.append(
                    f"  ... and {len(geo_clusters) - self.max_clusters_per_type} more clusters"
                )
            lines.append("")

        # nearTo relations (filtered by minimum confidence)
        near_relations = [
            r
            for r in self.history.get_relations_by_type("nearTo")
            if r.confidence >= self.min_confidence
        ]
        if near_relations:
            lines.append("### Already Established Proximity Relations (nearTo):")
            for rel in near_relations[: self.max_relations_shown]:
                source_name = rel.source_uri.split("/")[-1]
                target_name = rel.target_uri.split("/")[-1]
                distance = f" ({rel.value:.1f} km)" if rel.value else ""
                lines.append(f"- {source_name} ↔ {target_name}{distance}")
            if len(near_relations) > self.max_relations_shown:
                lines.append(
                    f"  ... and {len(near_relations) - self.max_relations_shown} more relations"
                )
            lines.append("")

        # Instructions
        lines.extend(
            [
                "### YOUR TASK:",
                "Propose NEW geographic clusters and proximity relations that ADD VALUE to the existing knowledge.",
                "- DO NOT propose clusters that duplicate the ones above",
                "- Focus on discovering NEW spatial patterns not yet identified",
                "- Consider connections BETWEEN existing clusters",
                "- Propose relations for entity pairs not yet connected",
                "",
            ]
        )

        return "\n".join(lines)

    def build_temporal_context(self) -> str:
        """
        Build context for the TemporalAnalyzer.

        Returns:
            Context string to include in the LLM prompt
        """
        h = self.history.history

        if h.statistics.total_runs == 0:
            return ""

        lines = [
            "",
            "## EXISTING KNOWLEDGE FROM PREVIOUS RUNS",
            f"(Based on {h.statistics.total_runs} previous analysis runs)",
            "",
        ]

        # Chronological clusters (filtered by minimum confidence)
        chrono_clusters = [
            c
            for c in self.history.get_clusters_by_type("chronological")
            if c.confidence >= self.min_confidence
        ]
        if chrono_clusters:
            lines.append("### Already Identified Chronological Clusters:")
            for cluster in chrono_clusters[: self.max_clusters_per_type]:
                members_preview = ", ".join(uri.split("/")[-1] for uri in cluster.member_uris[:3])
                if len(cluster.member_uris) > 3:
                    members_preview += f", ... (+{len(cluster.member_uris) - 3} more)"
                period = cluster.metadata.get("period_label", cluster.label)
                lines.append(
                    f"- **{period}** ({len(cluster.member_uris)} sites): {members_preview}"
                )
            if len(chrono_clusters) > self.max_clusters_per_type:
                lines.append(
                    f"  ... and {len(chrono_clusters) - self.max_clusters_per_type} more clusters"
                )
            lines.append("")

        # contemporaryWith relations (filtered by minimum confidence)
        contemporary_relations = [
            r
            for r in self.history.get_relations_by_type("contemporaryWith")
            if r.confidence >= self.min_confidence
        ]
        if contemporary_relations:
            lines.append("### Already Established Contemporary Relations:")
            for rel in contemporary_relations[: self.max_relations_shown]:
                source_name = rel.source_uri.split("/")[-1]
                target_name = rel.target_uri.split("/")[-1]
                lines.append(f"- {source_name} ↔ {target_name}")
            if len(contemporary_relations) > self.max_relations_shown:
                lines.append(
                    f"  ... and {len(contemporary_relations) - self.max_relations_shown} more relations"
                )
            lines.append("")

        # Instructions
        lines.extend(
            [
                "### YOUR TASK:",
                "Propose NEW chronological clusters and contemporary relations that ADD VALUE.",
                "- DO NOT propose clusters that duplicate the periods above",
                "- Look for finer temporal distinctions within known periods",
                "- Identify contemporary relations not yet established",
                "- Consider cross-period connections or transitions",
                "",
            ]
        )

        return "\n".join(lines)

    def build_type_context(self) -> str:
        """
        Build context for the TypeAnalyzer.

        Returns:
            Context string to include in the LLM prompt
        """
        h = self.history.history

        if h.statistics.total_runs == 0:
            return ""

        lines = [
            "",
            "## EXISTING KNOWLEDGE FROM PREVIOUS RUNS",
            f"(Based on {h.statistics.total_runs} previous analysis runs)",
            "",
        ]

        # Typological clusters (filtered by minimum confidence)
        typo_clusters = [
            c
            for c in self.history.get_clusters_by_type("typological")
            if c.confidence >= self.min_confidence
        ]
        if typo_clusters:
            lines.append("### Already Identified Typological Clusters:")
            for cluster in typo_clusters[: self.max_clusters_per_type]:
                members_preview = ", ".join(uri.split("/")[-1] for uri in cluster.member_uris[:3])
                if len(cluster.member_uris) > 3:
                    members_preview += f", ... (+{len(cluster.member_uris) - 3} more)"
                features = cluster.metadata.get("shared_features", [])
                features_str = f" [features: {', '.join(features[:3])}]" if features else ""
                lines.append(
                    f"- **{cluster.label}** ({len(cluster.member_uris)} sites): {members_preview}{features_str}"
                )
            if len(typo_clusters) > self.max_clusters_per_type:
                lines.append(
                    f"  ... and {len(typo_clusters) - self.max_clusters_per_type} more clusters"
                )
            lines.append("")

        # similarTo relations (filtered by minimum confidence)
        similar_relations = [
            r
            for r in self.history.get_relations_by_type("similarTo")
            if r.confidence >= self.min_confidence
        ]
        if similar_relations:
            lines.append("### Already Established Similarity Relations:")
            for rel in similar_relations[: self.max_relations_shown]:
                source_name = rel.source_uri.split("/")[-1]
                target_name = rel.target_uri.split("/")[-1]
                similarity = f" (score: {rel.value:.2f})" if rel.value else ""
                lines.append(f"- {source_name} ↔ {target_name}{similarity}")
            if len(similar_relations) > self.max_relations_shown:
                lines.append(
                    f"  ... and {len(similar_relations) - self.max_relations_shown} more relations"
                )
            lines.append("")

        # Instructions
        lines.extend(
            [
                "### YOUR TASK:",
                "Propose NEW typological clusters and similarity relations that ADD VALUE.",
                "- DO NOT propose clusters with the same typology as above",
                "- Look for sub-typologies or cross-cutting categories",
                "- Identify similarities between entities not yet connected",
                "- Consider architectural, functional, or contextual features not yet explored",
                "",
            ]
        )

        return "\n".join(lines)

    def build_path_context(self) -> str:
        """
        Build context for the PathGenerator.

        Returns:
            Context string to include in the LLM prompt
        """
        h = self.history.history

        if h.statistics.total_runs == 0:
            return ""

        lines = [
            "",
            "## EXISTING KNOWLEDGE FROM PREVIOUS RUNS",
            f"(Based on {h.statistics.total_runs} previous analysis runs)",
            "",
        ]

        # Existing paths (filtered by minimum confidence)
        qualified_paths = [p for p in h.paths if p.confidence >= self.min_confidence]
        if qualified_paths:
            lines.append("### Already Created Thematic Paths:")
            for path in qualified_paths[: self.max_paths_shown]:
                stops_preview = " → ".join(uri.split("/")[-1] for uri in path.stop_uris[:4])
                if len(path.stop_uris) > 4:
                    stops_preview += f" → ... (+{len(path.stop_uris) - 4} more stops)"
                lines.append(f"- **{path.theme}** ({path.path_type}): {stops_preview}")
            if len(qualified_paths) > self.max_paths_shown:
                lines.append(f"  ... and {len(qualified_paths) - self.max_paths_shown} more paths")
            lines.append("")

            # List themes to avoid
            existing_themes = list(set(p.theme.lower() for p in qualified_paths))
            lines.append(f"**Themes already covered:** {', '.join(existing_themes[:10])}")
            lines.append("")

        # Available clusters for path building
        all_clusters = len(h.clusters)
        if all_clusters > 0:
            lines.append(
                f"**Available clusters for path building:** {all_clusters} clusters across all types"
            )
            lines.append("")

        # Instructions
        lines.extend(
            [
                "### YOUR TASK:",
                "Propose NEW thematic paths that ADD VALUE to the existing collection.",
                "- DO NOT propose paths with themes already covered above",
                "- Create paths with DIFFERENT narratives and perspectives",
                "- Consider combining entities from different clusters",
                "- Explore themes not yet represented in the path collection",
                "",
            ]
        )

        return "\n".join(lines)

    def build_generic_context(self) -> str:
        """
        Build a generic context summary for any worker.

        Returns:
            Context string with overall knowledge summary
        """
        summary = self.history.get_summary()

        if summary["total_runs"] == 0:
            return ""

        lines = [
            "",
            "## KNOWLEDGE BASE SUMMARY",
            f"Total previous runs: {summary['total_runs']}",
            f"Entities in knowledge base: {summary['entities_known']}",
            "",
        ]

        if summary["clusters"]:
            lines.append("**Clusters by type:**")
            for ctype, clusters in summary["clusters"].items():
                lines.append(f"  - {ctype}: {len(clusters)} clusters")

        if summary["relations"]:
            lines.append("**Relations by type:**")
            for rtype, count in summary["relations"].items():
                lines.append(f"  - {rtype}: {count} relations")

        if summary["paths"]["count"] > 0:
            lines.append(f"**Paths:** {summary['paths']['count']} thematic paths")

        lines.append("")
        return "\n".join(lines)

    def get_context_for_worker(self, worker_type: str) -> str:
        """
        Get the appropriate context for a specific worker type.

        Args:
            worker_type: One of 'geo', 'temporal', 'type', 'path'

        Returns:
            Context string for the LLM prompt
        """
        context_builders = {
            "geo": self.build_geo_context,
            "geo_analyzer": self.build_geo_context,
            "temporal": self.build_temporal_context,
            "temporal_analyzer": self.build_temporal_context,
            "type": self.build_type_context,
            "type_analyzer": self.build_type_context,
            "path": self.build_path_context,
            "path_generator": self.build_path_context,
        }

        builder = context_builders.get(worker_type.lower())
        if builder:
            return builder()
        else:
            logger.warning("Unknown worker type '%s', using generic context", worker_type)
            return self.build_generic_context()
