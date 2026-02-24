"""
Memory module for cross-run knowledge persistence.

This module implements the Knowledge History Store (Option B) that enables
the system to remember assertions from previous runs and avoid proposing
duplicates while guiding the LLM toward novel discoveries.

Components:
- KnowledgeHistoryStore: Persists assertions across runs
- NoveltyDetector: Identifies duplicate/overlapping proposals
- ContextBuilder: Prepares LLM prompts with historical context
"""

from src.memory.history_store import KnowledgeHistoryStore
from src.memory.novelty_detector import NoveltyDetector
from src.memory.context_builder import ContextBuilder

__all__ = [
    "KnowledgeHistoryStore",
    "NoveltyDetector",
    "ContextBuilder",
]
