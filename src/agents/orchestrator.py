"""
Orchestrator Agent - Coordinates the multi-agent workflow.

The Orchestrator decides which worker agent to activate based on
the current state and routes tasks accordingly.
"""

import json
import logging
import uuid
from typing import Any

from src.llm import PromptLibrary

from .base import BaseAgent
from .models import (
    AgentResult,
    AgentState,
    AgentTask,
    AgentType,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseAgent):
    """
    Orchestrator agent that coordinates the multi-agent workflow.

    Responsibilities:
    - Decide which worker agent to activate next
    - Create and assign tasks to workers
    - Determine when to terminate the workflow
    - Track overall progress
    """

    agent_type = AgentType.ORCHESTRATOR

    # Mapping of agent names (from LLM) to AgentType
    AGENT_NAME_MAP = {
        "geo_analyzer": AgentType.GEO_ANALYZER,
        "geospatial": AgentType.GEO_ANALYZER,
        "temporal_analyzer": AgentType.TEMPORAL_ANALYZER,
        "chronological": AgentType.TEMPORAL_ANALYZER,
        "type_analyzer": AgentType.TYPE_ANALYZER,
        "typological": AgentType.TYPE_ANALYZER,
        "path_generator": AgentType.PATH_GENERATOR,
        "narrative": AgentType.PATH_GENERATOR,
        "triple_generator": AgentType.TRIPLE_GENERATOR,
        "validator": AgentType.VALIDATOR,
        "validation": AgentType.VALIDATOR,
        "terminate": None,  # Special case: workflow should end
    }

    def analyze(self, state: AgentState, task: AgentTask) -> AgentResult:
        """
        Orchestrator doesn't analyze - it routes.

        This method is implemented for interface compatibility but
        the main logic is in decide_next_agent().
        """
        return self._create_result(
            task=task,
            metadata={"message": "Orchestrator routes, not analyzes"},
        )

    def decide_next_agent(self, state: AgentState, entities_summary: str) -> AgentTask | None:
        """
        Decide which agent should be activated next.

        Uses LLM to analyze current state and determine the best next step.

        Args:
            state: Current system state
            entities_summary: Summary of available entities for context

        Returns:
            AgentTask for the next agent, or None if workflow should terminate
        """
        # Check termination conditions first
        if state.should_terminate():
            logger.info("Orchestrator: Termination condition met")
            return None

        # Check workflow mode
        workflow_mode = self.config.get("workflow", {}).get("mode", "comprehensive")
        if workflow_mode == "comprehensive":
            logger.info(
                "Orchestrator: Comprehensive mode - using deterministic routing for all analyzers"
            )
            return self._fallback_decision(state)

        # Prepare state summary for LLM
        state_summary = self._format_state_for_llm(state)

        # Determine goal based on what's not yet done
        goal = self._determine_goal(state)

        if goal is None:
            logger.info("Orchestrator: All goals completed")
            return None

        logger.info("Orchestrator: Current goal is to %s", goal)

        # Ask LLM for next action
        try:
            response = self.provider.invoke_with_prompt(
                system_prompt=PromptLibrary.ORCHESTRATOR_SYSTEM,
                human_prompt=PromptLibrary.ORCHESTRATOR_TASK,
                variables={
                    "entity_count": state.entity_count,
                    "current_state": state_summary,
                    "goal": goal,
                },
            )

            # Parse LLM response
            task = self._parse_llm_response(response, state)
            if task:
                logger.info(
                    "Orchestrator: Selected agent '%s' to achieve goal: %s", task.agent.value, goal
                )
            return task

        except Exception as e:
            logger.error("Orchestrator LLM invocation failed: %s", e)
            # Fallback to rule-based decision
            return self._fallback_decision(state)

    def _format_state_for_llm(self, state: AgentState) -> str:
        """Format current state as a string for LLM context."""
        summary = state.get_summary()
        return json.dumps(summary, indent=2)

    def _determine_goal(self, state: AgentState) -> str | None:
        """Determine the current goal based on state."""
        if not state.geo_analysis_done:
            return "Identify geographic clusters and proximity relationships between dolmen sites"

        if not state.temporal_analysis_done:
            return "Identify chronological relationships and group sites by period"

        if not state.type_analysis_done:
            return "Identify typological similarities between sites"

        if not state.path_generation_done:
            return "Generate thematic paths connecting related sites"

        return None  # All done

    def _parse_llm_response(self, response: str, state: AgentState) -> AgentTask | None:
        """Parse LLM response and create appropriate task."""
        try:
            # Try to extract JSON from response
            response = response.strip()

            # Handle markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            data = json.loads(response)

            agent_name = data.get("agent", "").lower().replace(" ", "_")
            task_desc = data.get("task", "Perform analysis")
            params = data.get("params", {})

            # Map agent name to type
            agent_type = self.AGENT_NAME_MAP.get(agent_name)

            if agent_type is None:
                if agent_name == "terminate":
                    logger.info("Orchestrator: LLM requested termination")
                    return None
                logger.warning("Unknown agent name from LLM: %s", agent_name)
                return self._fallback_decision(state)

            # Create task
            return AgentTask(
                agent=agent_type,
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                description=task_desc,
                params=params,
                priority=self._get_priority(agent_type, state),
            )

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM response as JSON: %s", e)
            task = self._fallback_decision(state)
            if task:
                logger.info(
                    "Orchestrator: Falling back to rule-based decision: %s", task.agent.value
                )
            return task

    def _fallback_decision(self, state: AgentState) -> AgentTask | None:
        """
        Rule-based fallback when LLM fails.

        Follows a deterministic order:
        1. Geographic analysis
        2. Temporal analysis
        3. Typological analysis
        4. Path generation
        """
        if not state.geo_analysis_done:
            geo_config = self.config.get("agents", {}).get("geospatial", {})
            return AgentTask(
                agent=AgentType.GEO_ANALYZER,
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                description="Analyze geographic proximity and create spatial clusters",
                params={
                    "max_distance_km": geo_config.get("max_distance_km", 50),
                    "min_cluster_size": geo_config.get("min_cluster_size"),
                    "max_cluster_size": geo_config.get("max_cluster_size"),
                    "max_clusters": geo_config.get("max_clusters"),
                    "max_pairs": geo_config.get("max_pairs"),
                },
                priority=10,
            )

        if not state.temporal_analysis_done:
            chrono_config = self.config.get("agents", {}).get("chronological", {})
            return AgentTask(
                agent=AgentType.TEMPORAL_ANALYZER,
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                description="Analyze chronological relationships between sites",
                params={
                    "period_tolerance_years": chrono_config.get("period_tolerance_years", 100),
                    "min_cluster_size": chrono_config.get("min_cluster_size"),
                    "max_cluster_size": chrono_config.get("max_cluster_size"),
                    "max_clusters": chrono_config.get("max_clusters"),
                    "max_pairs": chrono_config.get("max_pairs"),
                },
                priority=9,
            )

        if not state.type_analysis_done:
            typo_config = self.config.get("agents", {}).get("typological", {})
            return AgentTask(
                agent=AgentType.TYPE_ANALYZER,
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                description="Identify typological similarities between sites",
                params={
                    "embedding_similarity_threshold": typo_config.get(
                        "embedding_similarity_threshold", 0.7
                    ),
                    "majority_threshold": typo_config.get("majority_threshold", 0.6),
                    "min_cluster_size": typo_config.get("min_cluster_size") or 2,
                    "max_cluster_size": typo_config.get("max_cluster_size") or 6,
                    "max_clusters": typo_config.get("max_clusters") or 10,
                    "max_pairs": typo_config.get("max_pairs") or 20,
                },
                priority=8,
            )

        if not state.path_generation_done:
            narrative_config = self.config.get("agents", {}).get("narrative", {})
            return AgentTask(
                agent=AgentType.PATH_GENERATOR,
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                description="Generate thematic paths based on identified relationships",
                params={
                    "max_path_length": narrative_config.get("max_path_length", 10),
                    "min_path_length": narrative_config.get("min_path_length", 3),
                    "max_paths": narrative_config.get("max_paths"),
                },
                priority=7,
            )

        return None

    def _get_priority(self, agent_type: AgentType, state: AgentState) -> int:
        """Assign priority based on agent type and state."""
        priority_map = {
            AgentType.GEO_ANALYZER: 10,
            AgentType.TEMPORAL_ANALYZER: 9,
            AgentType.TYPE_ANALYZER: 8,
            AgentType.PATH_GENERATOR: 7,
            AgentType.TRIPLE_GENERATOR: 6,
            AgentType.VALIDATOR: 5,
        }
        return priority_map.get(agent_type, 5)

    def route(
        self, state: AgentState, available_workers: dict[AgentType, "BaseAgent"]
    ) -> AgentType | None:
        """
        Route to the next agent based on current state.

        This is a simplified routing method for integration with LangGraph.

        Args:
            state: Current system state
            available_workers: Dictionary of available worker agents

        Returns:
            AgentType of next agent to activate, or None to terminate
        """
        task = self.decide_next_agent(state, "")

        if task is None:
            return None

        # Verify the agent is available
        if task.agent not in available_workers:
            logger.warning("Requested agent %s not available", task.agent)
            return self._fallback_route(state, available_workers)

        return task.agent

    def _fallback_route(
        self, state: AgentState, available_workers: dict[AgentType, "BaseAgent"]
    ) -> AgentType | None:
        """Fallback routing when preferred agent unavailable."""
        # Try agents in order of priority
        order = [
            (AgentType.GEO_ANALYZER, not state.geo_analysis_done),
            (AgentType.TEMPORAL_ANALYZER, not state.temporal_analysis_done),
            (AgentType.TYPE_ANALYZER, not state.type_analysis_done),
            (AgentType.PATH_GENERATOR, not state.path_generation_done),
        ]

        for agent_type, needed in order:
            if needed and agent_type in available_workers:
                return agent_type

        return None
