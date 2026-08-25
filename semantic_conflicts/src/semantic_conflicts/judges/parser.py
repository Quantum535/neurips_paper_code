"""Parse rubric JSON from model text. Invalid output -> None (logged, not coerced)."""

from __future__ import annotations

import json
import re
from typing import Any

from semantic_conflicts.schemas import VALID_CATEGORIES

_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_judge_output(text: str, *, categories: set[str] | None = None) -> dict[str, Any] | None:
    if not text or not str(text).strip():
        return None
    cats = categories or set(VALID_CATEGORIES)
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    cat = str(obj.get("category", "")).strip()
    if cat not in cats:
        return None
    try:
        conf = int(obj.get("confidence", 0))
    except (TypeError, ValueError):
        return None
    if conf not in (1, 2, 3):
        return None
    evidence = str(obj.get("evidence", ""))[:300]
    return {"category": cat, "confidence": conf, "evidence": evidence}
