"""
Configuration management for xch-MIND project.
"""

import os
from pathlib import Path
from typing import Any, Dict, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings


class GeminiConfig(BaseModel):
    """Gemini LLM configuration."""

    model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_tokens: int = 32768


class OllamaConfig(BaseModel):
    """Ollama LLM configuration."""

    model: str = "deepseek-r1:1.5b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 32768


class RateLimitingConfig(BaseModel):
    """Rate limiting configuration for cloud APIs.

    Default values are for Gemini Free Tier.
    Set enabled=False for paid tiers (Tier 1+) where limits are much higher.
    """

    enabled: bool = True  # Set to False for paid tiers
    requests_per_minute: int = 15  # Free Tier: 15, Tier 1: 2000
    requests_per_day: int = 20  # Free Tier: 20, Tier 1: unlimited (set high value)
    max_retries: int = 5
    base_delay: float = 2.0


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: Literal["gemini", "ollama"] = "gemini"
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    rate_limiting: RateLimitingConfig = Field(default_factory=RateLimitingConfig)


class PathsConfig(BaseModel):
    """Project paths configuration."""

    entities_dir: Path = Path("./entities")
    ontology_dir: Path = Path("./ontology")
    arco_ontology: Path = Path("./ontology/ArCo.owl")
    xch_ontology: Path = Path("./ontology/xch/xch-core.ttl")
    output_dir: Path = Path("./output")


class NamespacesConfig(BaseModel):
    """RDF namespaces configuration."""

    xch: str = "https://w3id.org/xch-mind/ontology/"
    xch_res: str = "https://w3id.org/xch-mind/resource/"
    arco: str = "https://w3id.org/arco/ontology/arco/"
    arco_res: str = "https://w3id.org/arco/resource/"


class AgentConfig(BaseModel):
    """Individual agent configuration."""

    enabled: bool = True
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class GeospatialAgentConfig(AgentConfig):
    """Geospatial agent configuration."""

    max_distance_km: float = 30.0
    clustering_threshold: int = 10
    min_cluster_size: int = 2
    max_cluster_size: int = 6
    max_clusters: int = 15
    max_pairs: int = 20


class ChronologicalAgentConfig(AgentConfig):
    """Chronological agent configuration."""

    period_tolerance_years: int = 100
    min_cluster_size: int = 2
    max_cluster_size: int = 6
    max_clusters: int = 10
    max_pairs: int = 15


class TypologicalAgentConfig(AgentConfig):
    """Typological agent configuration."""

    similarity_threshold: float = 0.7
    embedding_similarity_threshold: float = 0.7
    majority_threshold: float = 0.6
    min_cluster_size: int = 2
    max_cluster_size: int = 6
    max_clusters: int = 10
    max_pairs: int = 20


class NarrativeAgentConfig(AgentConfig):
    """Narrative agent configuration."""

    max_path_length: int = 10
    min_path_length: int = 3
    max_paths: int = 5
    min_stops_per_path: int = 3
    max_stops_per_path: int = 8


class ValidationAgentConfig(AgentConfig):
    """Validation agent configuration."""

    min_confidence_auto_accept: float = 0.7
    require_manual_review_threshold: float = 0.6


class OrchestratorAgentConfig(AgentConfig):
    """Orchestrator agent configuration.

    Controls routing decisions between worker agents.
    In 'autonomous' mode, the orchestrator uses LLM calls to decide routing,
    so the temperature setting matters. In 'comprehensive' mode, routing
    is deterministic and the temperature is ignored.
    """

    pass  # Only inherits enabled + temperature from AgentConfig


class AgentsConfig(BaseModel):
    """All agents configuration."""

    orchestrator: OrchestratorAgentConfig = Field(default_factory=OrchestratorAgentConfig)
    geospatial: GeospatialAgentConfig = Field(default_factory=GeospatialAgentConfig)
    chronological: ChronologicalAgentConfig = Field(default_factory=ChronologicalAgentConfig)
    typological: TypologicalAgentConfig = Field(default_factory=TypologicalAgentConfig)
    narrative: NarrativeAgentConfig = Field(default_factory=NarrativeAgentConfig)
    validation: ValidationAgentConfig = Field(default_factory=ValidationAgentConfig)


class MemoryConfig(BaseModel):
    """Memory / knowledge history configuration."""

    min_confidence: float = 0.70
    cluster_overlap_threshold: float = 0.8
    path_overlap_threshold: float = 0.7


class WorkflowConfig(BaseModel):
    """Workflow configuration."""

    max_iterations: int = 20
    # Analysis mode:
    # - "comprehensive": Run ALL analyzers + PathGenerator (maximize insights)
    # - "autonomous": LLM-guided agent selection, stop when coverage + diversity thresholds met
    mode: Literal["comprehensive", "autonomous"] = "comprehensive"
    min_entity_coverage: float = 0.8  # Used in autonomous mode
    min_diversity: int = 2  # Minimum different assertion types (used in autonomous mode)


class OutputConfig(BaseModel):
    """Output configuration."""

    format: Literal["turtle", "xml", "json-ld"] = "turtle"
    include_provenance: bool = True
    include_confidence_scores: bool = True
    validation_report: bool = True
    manual_review_queue: bool = True


class Settings(BaseSettings):
    """Main configuration class."""

    model_config = ConfigDict(
        env_prefix="XCH_",
        env_nested_delimiter="__",
    )

    llm: LLMConfig = Field(default_factory=LLMConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    namespaces: NamespacesConfig = Field(default_factory=NamespacesConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(config_path: str = "src/config/config.yaml") -> Settings:
    """
    Load configuration from YAML file and environment variables.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Settings object with loaded configuration
    """
    config_file = Path(config_path)

    if config_file.exists():
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = {}

    # Create settings, which will also load from environment variables
    settings = Settings(**config_dict)

    return settings


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = load_config()
    return _settings
