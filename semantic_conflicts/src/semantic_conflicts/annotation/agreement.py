from __future__ import annotations

from pathlib import Path

import pandas as pd

from semantic_conflicts.config import Settings
from semantic_conflicts.io import write_csv, write_json
from semantic_conflicts.statistics import (
    cohen_kappa,
    confusion_matrix,
    fleiss_kappa,
    krippendorff_alpha,
    per_class_prf,
)


def compute_agreement(frames: list[pd.DataFrame], settings: Settings) -> tuple[dict, pd.DataFrame]:
    idc = "anon_id" if "anon_id" in frames[0].columns else "frame_id"
    wide = frames[0][[idc, "label_category"]].rename(columns={"label_category": "a1"})
    for i, fr in enumerate(frames[1:], 2):
        wide = wide.merge(fr[[idc, "label_category"]].rename(columns={"label_category": f"a{i}"}), on=idc, how="inner")
    cols = [c for c in wide.columns if c.startswith("a")]
    y1, y2 = wide[cols[0]], wide[cols[1]]
    raw = float((y1 == y2).mean()) if len(wide) else float("nan")
    report = {
        "n_annotators": len(frames),
        "n_shared_items": int(len(wide)),
        "raw_agreement": raw,
        "cohen_kappa": cohen_kappa(y1, y2) if len(cols) >= 2 else None,
        "labels": list(settings.annotation.labels),
        "label_source_note": "These are human labels, not gold until adjudicated.",
    }
    prf = per_class_prf(y1, y2, settings.annotation.labels)
    report["per_class_precision_recall_ann1_as_true"] = prf.to_dict(orient="records")
    report["macro_class_agreement"] = float(prf["f1"].mean(numeric_only=True)) if len(prf) else None
    cm = confusion_matrix(y1, y2, settings.annotation.labels)
    if len(cols) > 2:
        rat = wide[cols]
        report["fleiss_kappa"] = fleiss_kappa(rat)
        report["krippendorff_alpha"] = krippendorff_alpha(rat)
    return report, cm.reset_index()


def write_agreement_artifacts(out_dir: Path, report: dict, confusion: pd.DataFrame) -> None:
    write_json(out_dir / "human_agreement.json", report)
    write_csv(out_dir / "human_confusion.csv", confusion)
