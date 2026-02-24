"""
Workflow Module - LangGraph-based multi-agent orchestration.

This module provides the complete workflow infrastructure for
analyzing dolmen entities using a coordinated multi-agent system.

Components:
- graph.py: Main workflow graph definition
- nodes.py: Node functions for each agent
- conditions.py: Routing and termination conditions
"""

from .conditions import route_from_orchestrator, should_continue
from .graph import DolmenWorkflow, build_workflow, create_initial_state
from .nodes import NodeContext

__all__ = [
    # Main workflow class
    "DolmenWorkflow",
    # Graph building
    "build_workflow",
    "create_initial_state",
    # Context
    "NodeContext",
    # Conditions
    "route_from_orchestrator",
    "should_continue",
]
