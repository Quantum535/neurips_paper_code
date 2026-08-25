#!/usr/bin/env python3
"""Score silver judge labels against human gold when available."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_conflicts.config import load_settings
from semantic_conflicts.io import read_csv, write_json
from semantic_conflicts.paths import results_dir
from semantic_conflicts.statistics import cohen_kappa, multiclass_metrics, wilson_interval


def main() -> None:
    settings = load_settings()
    out = results_dir(settings.version)
    gold_p = out / "gold_labels.csv"
    if not gold_p.exists():
        write_json(
            out / "judge_report.json",
            {"status": "WAITING_FOR_HUMAN_LABELS", "note": "LLM labels are silver, never gold."},
        )
        print("WAITING_FOR_HUMAN_LABELS")
        return
    gold = read_csv(gold_p)
    runs = sorted((out / "judges").glob("*/predictions.csv")) if (out / "judges").exists() else []
    if not runs:
        write_json(out / "judge_report.json", {"status": "WAITING_FOR_JUDGE_RUN", "gold_n": int(len(gold))})
        print("WAITING_FOR_JUDGE_RUN")
        return
    pred = read_csv(runs[-1])
    idc = "frame_id" if "frame_id" in gold.columns else "anon_id"
    m = gold.merge(pred, on=idc, how="inner", suffixes=("_gold", "_pred"))
    gcol = "label_category" if "label_category" in m.columns else "gold"
    pcol = "category" if "category" in m.columns else "provisional_category"
    rep = {
        "label_source_gold": "human-adjudicated",
        "label_source_silver": "llm",
        "n": int(len(m)),
        "kappa": cohen_kappa(m[gcol], m[pcol]),
        "multiclass": multiclass_metrics(m[gcol], m[pcol], settings.annotation.labels),
    }
    write_json(out / "judge_report.json", rep)
    print(json.dumps({k: rep[k] for k in ("n", "kappa")}, indent=2))


if __name__ == "__main__":
    main()
