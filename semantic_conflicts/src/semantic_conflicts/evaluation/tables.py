"""Paper table CSVs. Missing gold -> explicit NOT AVAILABLE status, never fabricated numbers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from semantic_conflicts.io import write_csv, write_json

WAITING = "NOT AVAILABLE: human gold labels required"


def export_deterministic_tables(counts: dict, outcomes: dict, design: dict, out_dir: Path, settings) -> None:
    pool_rows = [
        {"pool": k, "n": v, "dataset_version": settings.version}
        for k, v in counts.items()
    ]
    write_csv(out_dir / "table_pool_validation.csv", pd.DataFrame(pool_rows))
    write_csv(
        out_dir / "table_prevalence.csv",
        pd.DataFrame(
            [
                {
                    "status": WAITING,
                    "note": "Prevalence from gold-corrected weights requires human labels.",
                    "design": "option_b_prevalence Hajek + cluster bootstrap",
                }
            ]
        ),
    )
    write_csv(
        out_dir / "table_gold_agreement.csv",
        pd.DataFrame([{"status": WAITING}]),
    )
    write_csv(
        out_dir / "table_baselines.csv",
        pd.DataFrame([{"status": WAITING, "note": "Baselines evaluate against gold test."}]),
    )
    write_csv(
        out_dir / "table_static_detector.csv",
        pd.DataFrame([{"status": WAITING, "note": "Static detector vs human gold."}]),
    )
    xa = counts.get("cross_agent_candidate")
    write_csv(
        out_dir / "table_cross_agent.csv",
        pd.DataFrame(
            [{"metric": "cross_agent_candidates", "n": xa, "dataset_version": settings.version}]
        ),
    )
    write_json(out_dir / "tables_status.json", {"ok": True, "gold": WAITING})
