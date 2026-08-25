from __future__ import annotations

from pathlib import Path

import pandas as pd

from semantic_conflicts.config import Settings
from semantic_conflicts.io import write_csv
from semantic_conflicts.schemas import InvariantError, validate_labels

ID_COL_CANDIDATES = ("anon_id", "frame_id")


def _id_col(df: pd.DataFrame) -> str:
    for c in ID_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise InvariantError("annotation sheet needs anon_id or frame_id")


def blinded_export(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    hidden = set(settings.annotation.hidden_fields)
    drop = [c for c in df.columns if c in hidden]
    out = df.drop(columns=drop, errors="ignore").copy()
    # Never leak merge/outcome/pool/LLM even if the caller forgot the config list.
    extra_leak = [
        c
        for c in out.columns
        if any(
            k in c.lower()
            for k in (
                "merged",
                "closed",
                "pool",
                "stratum",
                "incl_prob",
                "pi_",
                "llm",
                "judge",
                "silver",
                "gold",
                "provisional",
            )
        )
    ]
    out = out.drop(columns=extra_leak, errors="ignore")
    for c in ("label_category", "confidence", "evidence", "ambiguous", "annotator", "timestamp"):
        if c not in out.columns:
            out[c] = ""
    return out


def import_labels(src: Path, dest: Path, *, annotator: str, settings: Settings) -> int:
    df = pd.read_csv(src)
    col = "label_category" if "label_category" in df.columns else None
    if col is None:
        matches = [c for c in df.columns if "label_category" in c]
        if not matches:
            raise InvariantError("no label_category column")
        col = matches[0]
        df = df.rename(columns={col: "label_category"})
    df["label_category"] = df["label_category"].astype(str).str.strip()
    df = df[df["label_category"].ne("") & df["label_category"].ne("nan")]
    validate_labels(df["label_category"], name="imported", strict=True)
    if "confidence" in df.columns:
        conf = pd.to_numeric(df["confidence"], errors="coerce")
        bad = int((~conf.isna() & ~conf.isin(settings.annotation.confidence)).sum())
        if bad:
            raise InvariantError(f"{bad} confidence values outside {settings.annotation.confidence}")
    df["annotator"] = annotator
    df["rubric_version"] = settings.annotation.rubric_version
    df["annotation_version"] = settings.version
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_csv(dest, df)
    return int(len(df))


def adjudicate(
    annotator_frames: list[pd.DataFrame],
    adjudication: pd.DataFrame | None,
    settings: Settings,
) -> pd.DataFrame:
    if not annotator_frames:
        raise InvariantError("no annotator frames")
    idc = _id_col(annotator_frames[0])
    merged = None
    for i, fr in enumerate(annotator_frames, 1):
        sub = fr[[idc, "label_category"]].rename(columns={"label_category": f"ann{i}"})
        if "confidence" in fr.columns:
            sub[f"conf{i}"] = fr["confidence"]
        if "evidence" in fr.columns:
            sub[f"evidence{i}"] = fr["evidence"]
        merged = sub if merged is None else merged.merge(sub, on=idc, how="outer")
    lab_cols = [c for c in merged.columns if c.startswith("ann")]
    if len(lab_cols) >= 2:
        agree = merged[lab_cols[0]] == merged[lab_cols[1]]
        merged["agreed"] = agree
        merged["gold"] = merged[lab_cols[0]].where(agree)
    else:
        merged["agreed"] = True
        merged["gold"] = merged[lab_cols[0]]
    if adjudication is not None and len(adjudication):
        adj = adjudication.copy()
        gcol = "gold" if "gold" in adj.columns else "label_category"
        adj = adj.rename(columns={gcol: "adj_gold"})
        if "adjudicator_notes" not in adj.columns:
            adj["adjudicator_notes"] = ""
        merged = merged.merge(adj[[idc, "adj_gold", "adjudicator_notes"]], on=idc, how="left")
        merged["gold"] = merged["adj_gold"].where(merged["adj_gold"].notna(), merged["gold"])
    gold = merged.dropna(subset=["gold"]).copy()
    gold["label_category"] = gold["gold"].astype(str).str.strip()
    validate_labels(gold["label_category"], name="gold", strict=True)
    gold["label_source"] = "human-adjudicated"
    gold["dataset_version"] = settings.version
    return gold


def write_gold(path: Path, gold: pd.DataFrame, settings: Settings) -> None:
    cols = [c for c in gold.columns if c in gold.columns]
    write_csv(path, gold[cols])
