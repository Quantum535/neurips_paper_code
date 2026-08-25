"""Semantic Conflicts research artifact.

Gold = human-adjudicated labels.
Silver = LLM judge labels.
Heuristic = deterministic candidate-pool flags.
"""

from semantic_conflicts.config import Settings, load_settings

__version__ = "0.1.0"
__all__ = ["Settings", "load_settings", "__version__"]
