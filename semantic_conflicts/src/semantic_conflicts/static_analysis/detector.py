"""Execution-free static semantic-interference detector.

Depth-2 directory co-location is a cheap heuristic baseline (pools.near_miss).
This detector looks for def/use, delete/use, export/use, config-key, env,
schema, import-path, and dependency-file relations across two disjoint-file PRs.

Every decision is inspectable structured evidence. The machine verdict is never gold.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Iterable

from semantic_conflicts.static_analysis.extractors import (
    SymbolEvent,
    extract_from_files,
    extract_from_patch,
)

STOP = {
    "self",
    "this",
    "that",
    "init",
    "main",
    "test",
    "data",
    "item",
    "value",
    "type",
    "name",
    "info",
    "get",
    "set",
    "len",
    "str",
    "int",
    "list",
    "dict",
    "true",
    "false",
    "none",
    "null",
    "return",
    "print",
    "log",
}


@dataclass
class InterferenceEvidence:
    prediction: bool
    direction: str
    relation: str
    symbol: str
    definition_file: str
    consumer_file: str
    confidence: float
    details: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


def _index(events: Iterable[SymbolEvent]) -> dict[str, dict[str, list[SymbolEvent]]]:
    out: dict[str, dict[str, list[SymbolEvent]]] = defaultdict(lambda: defaultdict(list))
    for e in events:
        if e.name.lower() in STOP:
            continue
        out[e.kind][e.name].append(e)
        # also index without deleted_ prefix under a parallel map
        if e.kind.startswith("deleted_"):
            out["deleted"][e.name].append(e)
            base = e.kind[len("deleted_") :]
            out[f"deleted_{base}"][e.name].append(e)
    return out


def _names(idx, *kinds) -> set[str]:
    s: set[str] = set()
    for k in kinds:
        s |= set(idx.get(k, {}))
    return s


def _first(idx, kind, name) -> SymbolEvent | None:
    lst = idx.get(kind, {}).get(name) or idx.get("deleted_" + kind, {}).get(name)
    return lst[0] if lst else None


def detect_pair(
    events_a: list[SymbolEvent],
    events_b: list[SymbolEvent],
    *,
    files_overlap: bool = False,
) -> list[InterferenceEvidence]:
    """Return ranked evidence list. prediction=True if any high-confidence relation exists."""
    ia, ib = _index(events_a), _index(events_b)
    evidence: list[InterferenceEvidence] = []

    def add(direction, relation, symbol, def_kind, use_kind, src, dst, conf, **det):
        de = _first(src, def_kind, symbol) or _first(src, "deleted_" + def_kind if not def_kind.startswith("deleted") else def_kind, symbol)
        ue = _first(dst, use_kind, symbol)
        evidence.append(
            InterferenceEvidence(
                prediction=conf >= 0.5,
                direction=direction,
                relation=relation,
                symbol=symbol,
                definition_file=de.file if de else "",
                consumer_file=ue.file if ue else "",
                confidence=conf,
                details=det,
            )
        )

    def_kinds = ("def", "class", "export")
    use_kinds = ("use", "import")
    defs_a, defs_b = _names(ia, *def_kinds), _names(ib, *def_kinds)
    uses_a, uses_b = _names(ia, *use_kinds), _names(ib, *use_kinds)
    del_a, del_b = _names(ia, "deleted_def", "deleted_class", "deleted_export", "deleted"), _names(
        ib, "deleted_def", "deleted_class", "deleted_export", "deleted"
    )
    for sym in sorted(defs_a & uses_b):
        add("A->B", "changed_definition_used_by_partner", sym, "def", "use", ia, ib, 0.91)
    for sym in sorted(defs_b & uses_a):
        add("B->A", "changed_definition_used_by_partner", sym, "def", "use", ib, ia, 0.91)
    for sym in sorted(del_a & uses_b):
        add("A->B", "deleted_definition_used_by_partner", sym, "deleted_def", "use", ia, ib, 0.93)
    for sym in sorted(del_b & uses_a):
        add("B->A", "deleted_definition_used_by_partner", sym, "deleted_def", "use", ib, ia, 0.93)

    exp_a, exp_b = _names(ia, "export"), _names(ib, "export")
    for sym in sorted(exp_a & uses_b):
        add("A->B", "exported_change_used_by_partner", sym, "export", "use", ia, ib, 0.88)
    for sym in sorted(exp_b & uses_a):
        add("B->A", "exported_change_used_by_partner", sym, "export", "use", ib, ia, 0.88)

    cfg_use_b = _names(ib, "config_key", "use", "env")
    cfg_use_a = _names(ia, "config_key", "use", "env")
    for sym in sorted(_names(ia, "config_key", "deleted_config_key") & cfg_use_b):
        if sym.lower() in STOP:
            continue
        add("A->B", "config_key_producer_consumer", sym, "config_key", "config_key", ia, ib, 0.8)
    for sym in sorted(_names(ib, "config_key", "deleted_config_key") & cfg_use_a):
        add("B->A", "config_key_producer_consumer", sym, "config_key", "config_key", ib, ia, 0.8)

    for sym in sorted(_names(ia, "env", "deleted_env") & _names(ib, "env", "use")):
        add("A->B", "env_var_producer_consumer", sym, "env", "env", ia, ib, 0.85)
    for sym in sorted(_names(ib, "env", "deleted_env") & _names(ia, "env", "use")):
        add("B->A", "env_var_producer_consumer", sym, "env", "env", ib, ia, 0.85)

    for sym in sorted(_names(ia, "schema", "deleted_schema") & _names(ib, "schema", "use", "ident")):
        add("A->B", "schema_field_intersection", sym, "schema", "use", ia, ib, 0.8)
    for sym in sorted(_names(ib, "schema", "deleted_schema") & _names(ia, "schema", "use", "ident")):
        add("B->A", "schema_field_intersection", sym, "schema", "use", ib, ia, 0.8)

    imp_a, imp_b = _names(ia, "import"), _names(ib, "import")
    # import-path dependency: A changes a module whose path is imported by B
    files_a = {e.file for e in events_a}
    files_b = {e.file for e in events_b}
    for imp in sorted(imp_b):
        for fa in files_a:
            stem = fa.replace("\\", "/").rsplit(".", 1)[0]
            if imp.replace(".", "/").replace("::", "/") in stem or stem.split("/")[-1] == imp.split(".")[-1]:
                add("A->B", "import_path_dependency", imp, "def", "import", ia, ib, 0.7, import_path=imp, changed_file=fa)
    for imp in sorted(imp_a):
        for fb in files_b:
            stem = fb.replace("\\", "/").rsplit(".", 1)[0]
            if imp.replace(".", "/").replace("::", "/") in stem or stem.split("/")[-1] == imp.split(".")[-1]:
                add("B->A", "import_path_dependency", imp, "def", "import", ib, ia, 0.7, import_path=imp, changed_file=fb)

    # Weak identifier fallback (low confidence, never sole official I without other evidence)
    id_a = {e.name for e in events_a if e.kind in {"ident", "def", "class"} and len(e.name) >= 6}
    id_b = {e.name for e in events_b if e.kind in {"use", "ident"} and len(e.name) >= 6}
    weak = sorted((id_a & id_b) - STOP)
    for sym in weak[:8]:
        add("A->B", "identifier_similarity_fallback", sym, "def", "use", ia, ib, 0.35, weak=True)

    evidence.sort(key=lambda e: -e.confidence)
    # Drop weak-only if stronger evidence exists
    strong = [e for e in evidence if e.confidence >= 0.5]
    return strong or evidence[:3]


def detect_from_patches(patches_a: dict[str, str], patches_b: dict[str, str], *, files_overlap: bool = False) -> list[InterferenceEvidence]:
    ea, eb = [], []
    for f, p in patches_a.items():
        ea.extend(extract_from_patch(p, f))
    for f, p in patches_b.items():
        eb.extend(extract_from_patch(p, f))
    return detect_pair(ea, eb, files_overlap=files_overlap)


def detect_from_file_texts(files_a: dict[str, str], files_b: dict[str, str]) -> list[InterferenceEvidence]:
    return detect_pair(extract_from_files(files_a), extract_from_files(files_b))


def pair_prediction(evidence: list[InterferenceEvidence]) -> dict:
    strong = [e for e in evidence if e.confidence >= 0.5]
    if not strong:
        return {
            "prediction": False,
            "direction": "none",
            "relation": "none",
            "symbol": "",
            "confidence": float(evidence[0].confidence) if evidence else 0.0,
            "n_evidence": len(evidence),
            "evidence": [e.to_json() for e in evidence],
        }
    top = strong[0]
    dirs = {e.direction for e in strong}
    direction = "both" if ("A->B" in dirs and "B->A" in dirs) else top.direction
    return {
        "prediction": True,
        "direction": direction,
        "relation": top.relation,
        "symbol": top.symbol,
        "definition_file": top.definition_file,
        "consumer_file": top.consumer_file,
        "confidence": top.confidence,
        "n_evidence": len(strong),
        "evidence": [e.to_json() for e in strong],
    }
