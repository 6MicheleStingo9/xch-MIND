"""
Base agent class for the multi-agent system.

Provides common interface and functionality for all agents.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from src.llm import LLMProvider, create_provider
from src.utils.rate_limiter import RateLimiter

from .models import AgentResult, AgentState, AgentTask, AgentType

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.

    Provides:
    - LLM provider access
    - Configuration management
    - Common logging and error handling
    """

    # Agent type - must be set by subclasses
    agent_type: AgentType

    def __init__(
        self,
        config: dict[str, Any],
        provider: LLMProvider | None = None,
        rate_limiter: RateLimiter | None = None,
    ):
        """
        Initialize the agent.

        Args:
            config: Full configuration dictionary (from config.yaml)
            provider: Optional LLM provider. If not provided, creates one from config.
            rate_limiter: Optional RateLimiter for tracking and throttling
        """
        self.config = config
        self._provider = provider
        self.rate_limiter = rate_limiter

        # Get agent-specific config
        self.agent_config = self._get_agent_config()

        logger.info(
            "Initialized %s with temperature=%.2f",
            self.__class__.__name__,
            self.get_temperature(),
        )

    def _get_agent_config(self) -> dict[str, Any]:
        """Get configuration specific to this agent."""
        agents_config = self.config.get("agents", {})

        # Map agent type to config key
        config_key_map = {
            AgentType.ORCHESTRATOR: "orchestrator",
            AgentType.GEO_ANALYZER: "geospatial",
            AgentType.TEMPORAL_ANALYZER: "chronological",
            AgentType.TYPE_ANALYZER: "typological",
            AgentType.PATH_GENERATOR: "narrative",
            AgentType.TRIPLE_GENERATOR: "triple_generator",
            AgentType.VALIDATOR: "validation",
        }

        config_key = config_key_map.get(self.agent_type, "")
        return agents_config.get(config_key, {})

    def get_temperature(self) -> float:
        """Get the temperature setting for this agent."""
        return self.agent_config.get("temperature", 0.0)

    def is_enabled(self) -> bool:
        """Check if this agent is enabled in configuration."""
        return self.agent_config.get("enabled", True)

    @property
    def provider(self) -> LLMProvider:
        """Get or create the LLM provider with agent-specific temperature."""
        if self._provider is None:
            llm_config = self.config.get("llm", {})
            provider_type = llm_config.get("provider", "gemini")
            provider_config = llm_config.get(provider_type, {}).copy()

            # Override temperature with agent-specific value
            provider_config["temperature"] = self.get_temperature()
            provider_config["provider"] = provider_type

            self._provider = create_provider(
                provider_type=provider_type,
                model_name=provider_config.get("model"),
                temperature=self.get_temperature(),
                max_tokens=provider_config.get("max_tokens", 4096),
                rate_limiter=self.rate_limiter,
            )

        return self._provider

    @abstractmethod
    def analyze(self, state: AgentState, task: AgentTask) -> AgentResult:
        """
        Execute the agent's analysis on the current state.

        Args:
            state: Current system state
            task: Task assigned to this agent

        Returns:
            AgentResult with generated assertions or error
        """
        pass

    def _create_result(
        self,
        task: AgentTask,
        assertions: list | None = None,
        error: str | None = None,
        metadata: dict | None = None,
    ) -> AgentResult:
        """Helper to create a standardized AgentResult."""
        from .models import TaskStatus

        return AgentResult(
            agent=self.agent_type,
            task_id=task.task_id,
            status=TaskStatus.COMPLETED if error is None else TaskStatus.FAILED,
            assertions=assertions or [],
            metadata=metadata or {},
            error=error,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.agent_type}, temp={self.get_temperature()})"
