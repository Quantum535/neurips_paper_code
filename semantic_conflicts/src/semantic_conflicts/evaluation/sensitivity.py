"""Sensitivity of heuristic thresholds. Never fabricates gold metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from semantic_conflicts.io import write_csv, write_json
from semantic_conflicts.pools import is_duplicate_title, parent_dir_prefix


def duplicate_title_sweep(pairs: pd.DataFrame, titles_a, titles_b, thresholds=None) -> pd.DataFrame:
    thresholds = thresholds or [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    rows = []
    for t in thresholds:
        n = sum(
            is_duplicate_title(a, b, jaccard_threshold=t)
            for a, b in zip(titles_a, titles_b)
        )
        rows.append({"jaccard_threshold": t, "n_candidates": int(n)})
    return pd.DataFrame(rows)


def near_miss_sweep(pairs: pd.DataFrame, files_idx: dict) -> pd.DataFrame:
    rows = []
    for depth in (1, 2, 3):
        n = 0
        for r, a, b, c in zip(pairs.repo, pairs.pr_a, pairs.pr_b, pairs.conflict):
            if bool(c):
                continue
            ka = {parent_dir_prefix(f, max_dir_components=depth) for f in files_idx.get((str(r), int(a)), set())}
            kb = {parent_dir_prefix(f, max_dir_components=depth) for f in files_idx.get((str(r), int(b)), set())}
            ka.discard(None)
            kb.discard(None)
            n += int(bool(ka & kb))
        rows.append({"variant": f"dir_prefix_depth{depth}", "n_candidates": n})
    # same parent dir (full parent, any depth)
    n_parent = 0
    for r, a, b, c in zip(pairs.repo, pairs.pr_a, pairs.pr_b, pairs.conflict):
        if bool(c):
            continue
        def parents(fs):
            out = set()
            for f in fs:
                parts = str(f).split("/")
                if len(parts) >= 2:
                    out.add("/".join(parts[:-1]))
            return out
        if parents(files_idx.get((str(r), int(a)), set())) & parents(files_idx.get((str(r), int(b)), set())):
            n_parent += 1
    rows.append({"variant": "same_parent_dir", "n_candidates": n_parent})
    return pd.DataFrame(rows)


def write_sensitivity(out_dir: Path, dup: pd.DataFrame, near: pd.DataFrame) -> None:
    write_csv(out_dir / "sensitivity_duplicate_title.csv", dup)
    write_csv(out_dir / "sensitivity_near_miss.csv", near)
    write_json(
        out_dir / "sensitivity_status.json",
        {
            "gold_precision_recall": "NOT AVAILABLE: human gold labels required"
            if True
            else "computed",
            "note": "Candidate counts are deterministic. Gold PR/RC require human labels.",
        },
    )
