"""Blinded human annotation / adjudication.

Gold labels come ONLY from human annotation and adjudication.
LLM outputs are silver. Heuristic pool flags are not gold.
"""

from semantic_conflicts.annotation.agreement import compute_agreement
from semantic_conflicts.annotation.io_labels import import_labels, write_gold

__all__ = ["compute_agreement", "import_labels", "write_gold"]
