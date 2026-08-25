"""Error-analysis bundles. Blinded: no merge/outcome fields."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from semantic_conflicts.io import write_csv, write_json

WAITING = "WAITING_FOR_HUMAN_LABELS"


def generate_error_bundles(
    gold: pd.DataFrame | None,
    predictions: pd.DataFrame | None,
    frame: pd.DataFrame,
    out_dir: Path,
) -> dict:
    dest = out_dir / "error_analysis"
    dest.mkdir(parents=True, exist_ok=True)
    if gold is None or not len(gold) or predictions is None or not len(predictions):
        msg = {"status": WAITING, "detail": "NOT AVAILABLE: human gold labels required"}
        write_json(dest / "status.json", msg)
        return msg
    idc = "frame_id" if "frame_id" in gold.columns else "anon_id"
    g = gold.merge(predictions, on=idc, how="inner", suffixes=("_gold", "_pred"))
    if "label_category" in g.columns:
        g["gold"] = g["label_category"]
    pred_col = "category" if "category" in g.columns else "provisional_category"
    g["pred"] = g[pred_col]
    g["disagree"] = g["gold"].astype(str) != g["pred"].astype(str)
    fp = g[g.disagree & (g.pred != "none") & (g.gold.isin(["none", "B"]))]
    fn = g[g.disagree & (g.gold.isin(["D", "C", "H", "I"])) & (g.pred.isin(["none", "B"]))]
    keep = [c for c in g.columns if not any(x in c.lower() for x in ("merged", "closed", "pool", "pi_", "incl"))]
    write_csv(dest / "high_confidence_false_positives.csv", fp[keep].head(50) if len(fp) else fp)
    write_csv(dest / "high_confidence_false_negatives.csv", fn[keep].head(50) if len(fn) else fn)
    write_json(
        dest / "status.json",
        {"n_gold": int(len(gold)), "n_fp": int(len(fp)), "n_fn": int(len(fn)), "status": "ok"},
    )
    return {"n_fp": int(len(fp)), "n_fn": int(len(fn))}
