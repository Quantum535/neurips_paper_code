"""How good is file overlap at detecting HUMAN-VALIDATED harmful semantic interference?"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from semantic_conflicts.io import write_json
from semantic_conflicts.schemas import VALID_CATEGORIES

HARMFUL = {"D", "C", "H", "I"}
BENIGN = {"B", "none"}


def file_overlap_confusion(gold: pd.DataFrame, conflict_col: str = "conflict") -> dict:
    if "label_category" not in gold.columns:
        return {"status": "NOT AVAILABLE: human gold labels required"}
    y = gold["label_category"].astype(str)
    fo = gold[conflict_col].astype(bool)
    harmful = y.isin(HARMFUL)
    tp = int((fo & harmful).sum())
    fp = int((fo & ~harmful).sum())
    fn = int((~fo & harmful).sum())
    tn = int((~fo & ~harmful).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and (prec + rec)) else float("nan")
    by_cat = {}
    for cat in sorted(VALID_CATEGORIES):
        sub = gold[y == cat]
        by_cat[cat] = {
            "n": int(len(sub)),
            "file_overlap": int(sub[conflict_col].astype(bool).sum()) if len(sub) else 0,
            "no_file_overlap": int((~sub[conflict_col].astype(bool)).sum()) if len(sub) else 0,
        }
    return {
        "mapping": {"harmful": sorted(HARMFUL), "not_harmful": sorted(BENIGN)},
        "table": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else float("nan"),
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else float("nan"),
        "by_category": by_cat,
        "n": int(len(gold)),
        "label_source": "human-adjudicated-gold",
    }


def write_file_overlap_report(path: Path, report: dict) -> None:
    write_json(path, report)
