"""
Agents package for the xch-MIND multi-agent system.

Provides:
- BaseAgent: Abstract base class for all agents
- OrchestratorAgent: Coordinates the workflow
- Worker Agents: Specialized analyzers (Geo, Temporal, Type, Path)
- Data models for agent communication
- LLM structured output models for reliable parsing
- Validation layer for algorithmic grounding
- Confidence calculator for hybrid scoring
"""

from .base import BaseAgent
from .models import (
    AgentResult,
    AgentState,
    AgentTask,
    AgentType,
    AssertionType,
    ChronologicalCluster,
    GeographicCluster,
    InterpretiveAssertion,
    PathStop,
    PathType,
    SiteRelation,
    TaskStatus,
    ThematicPath,
    TypologicalCluster,
    # LLM Structured Output Models
    GeoClusterProposal,
    NearRelationProposal,
    GeoAnalysisResponse,
    PeriodNormalization,
    ChronoClusterProposal,
    ContemporaryPairProposal,
    ChronologicalAnalysisResponse,
    ExtractedFeatures,
    TypeClusterProposal,
    SimilarityPairProposal,
    TypologicalAnalysisResponse,
    PathStopProposal,
    ThematicPathProposal,
    NarrativeAnalysisResponse,
)
from .orchestrator import OrchestratorAgent
from .workers import (
    GeoAnalyzerAgent,
    PathGeneratorAgent,
    TemporalAnalyzerAgent,
    TypeAnalyzerAgent,
)
from .validation import (
    ValidationResult,
    haversine_distance,
    validate_coordinates,
    validate_geographic_cluster,
    validate_near_relation,
    validate_chronological_cluster,
    validate_contemporary_relation,
    validate_typological_cluster,
    validate_thematic_path,
)
from .confidence import (
    ConfidenceWeights,
    ConfidenceThresholds,
    calculate_hybrid_confidence,
    calculate_cluster_confidence,
    calculate_relation_confidence,
    calculate_path_confidence,
    filter_by_confidence,
    sort_by_confidence,
    generate_confidence_report,
)

__all__ = [
    # Base
    "BaseAgent",
    # Orchestrator
    "OrchestratorAgent",
    # Worker Agents
    "GeoAnalyzerAgent",
    "TemporalAnalyzerAgent",
    "TypeAnalyzerAgent",
    "PathGeneratorAgent",
    # Models - Enums
    "AgentType",
    "TaskStatus",
    "AssertionType",
    "PathType",
    # Models - Tasks
    "AgentTask",
    "AgentResult",
    # Models - State
    "AgentState",
    # Models - Assertions
    "InterpretiveAssertion",
    "GeographicCluster",
    "ChronologicalCluster",
    "TypologicalCluster",
    "SiteRelation",
    "ThematicPath",
    "PathStop",
    # LLM Structured Output Models
    "GeoClusterProposal",
    "NearRelationProposal",
    "GeoAnalysisResponse",
    "PeriodNormalization",
    "ChronoClusterProposal",
    "ContemporaryPairProposal",
    "ChronologicalAnalysisResponse",
    "ExtractedFeatures",
    "TypeClusterProposal",
    "SimilarityPairProposal",
    "TypologicalAnalysisResponse",
    "PathStopProposal",
    "ThematicPathProposal",
    "NarrativeAnalysisResponse",
    # Validation
    "ValidationResult",
    "haversine_distance",
    "validate_coordinates",
    "validate_geographic_cluster",
    "validate_near_relation",
    "validate_chronological_cluster",
    "validate_contemporary_relation",
    "validate_typological_cluster",
    "validate_thematic_path",
    # Confidence
    "ConfidenceWeights",
    "ConfidenceThresholds",
    "calculate_hybrid_confidence",
    "calculate_cluster_confidence",
    "calculate_relation_confidence",
    "calculate_path_confidence",
    "filter_by_confidence",
    "sort_by_confidence",
    "generate_confidence_report",
]
