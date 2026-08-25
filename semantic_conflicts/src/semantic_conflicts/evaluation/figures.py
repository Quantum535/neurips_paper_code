"""Deterministic figures that do not require gold labels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from semantic_conflicts.io import write_json

WAITING = "NOT AVAILABLE: human gold labels required"


def export_deterministic_figures(flags: pd.DataFrame, outcomes: dict, out_dir: Path, settings) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # Pool sizes
    keys = ["dup_title", "revert", "fix_in_flight", "near_miss", "bm_shared"]
    vals = [int(flags[k].sum()) for k in keys]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.barh(keys, vals, color="#0072B2")
    ax.set_xlabel("pairs")
    fig.savefig(out_dir / "fig_sampling_coverage.pdf", bbox_inches="tight")
    plt.close(fig)

    # Category prevalence placeholder
    for name in (
        "fig_file_overlap_confusion.pdf",
        "fig_category_prevalence.pdf",
        "fig_baseline_pr.pdf",
        "fig_static_vs_directory.pdf",
        "fig_error_breakdown.pdf",
    ):
        fig, ax = plt.subplots(figsize=(4.0, 2.4))
        ax.text(0.5, 0.5, WAITING, ha="center", va="center", wrap=True)
        ax.axis("off")
        fig.savefig(out_dir / name, bbox_inches="tight")
        plt.close(fig)
    write_json(out_dir / "figures_status.json", {"ok": True, "gold_dependent": WAITING})
