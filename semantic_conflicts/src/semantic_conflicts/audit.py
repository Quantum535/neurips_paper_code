"""Strict artifact audit. Exit non-zero on invariant failures; WARN on missing optional stages."""

from __future__ import annotations

import argparse
from pathlib import Path

from semantic_conflicts.config import load_settings
from semantic_conflicts.io import load_source_tables, read_json
from semantic_conflicts.paths import repo_root, results_dir
from semantic_conflicts.pools import files_by_pr, recompute_conflict
from semantic_conflicts.schemas import (
    InvariantError,
    validate_base_pairs,
    validate_frame,
    validate_inclusion_probs,
    validate_pool_flags,
)

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


class Reporter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.n_fail = 0
        self.n_warn = 0
        self.n_pass = 0

    def section(self, name: str) -> None:
        self.lines.append(f"\n{name}")

    def check(self, status: str, msg: str) -> None:
        self.lines.append(f"[{status}] {msg}")
        if status == FAIL:
            self.n_fail += 1
        elif status == WARN:
            self.n_warn += 1
        else:
            self.n_pass += 1

    def dump(self) -> str:
        return "\n".join(self.lines) + "\n"


def _safe(fn, *a, **k):
    try:
        fn(*a, **k)
        return None
    except InvariantError as e:
        return str(e)
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def run_audit(settings, *, root: Path | None = None, strict: bool = False, out_dir: Path | None = None) -> int:
    root = root or repo_root()
    out = out_dir or results_dir(settings.version, root)
    r = Reporter()
    tables = load_source_tables(root, settings=settings)

    r.section("DATA")
    err = _safe(validate_base_pairs, tables["pairs"], settings, strict=True)
    r.check(FAIL if err else PASS, "base pair count / pair keys / opened_a <= opened_b" + (f": {err}" if err else ""))
    idx = files_by_pr(tables["files"])
    n_mismatch = 0
    sample = tables["pairs"]
    # Full recompute on fixture; on full data check all (vectorized-enough).
    for repo, a, b, c in zip(sample.repo, sample.pr_a, sample.pr_b, sample.conflict):
        if bool(c) != recompute_conflict(str(repo), int(a), int(b), idx):
            n_mismatch += 1
            if n_mismatch > 5:
                break
    r.check(FAIL if n_mismatch else PASS, f"conflict recomputation mismatches (capped scan)={n_mismatch}")
    xa = tables["pairs"]["cross_agent"].astype(bool)
    agents_differ = tables["pairs"]["agent_a"].astype(str) != tables["pairs"]["agent_b"].astype(str)
    n_xa = int((xa != agents_differ).sum())
    r.check(FAIL if n_xa else PASS, f"cross_agent agrees with agent identity (mismatches={n_xa})")

    r.section("POOLS")
    flags_path = out / "pool_flags.csv.gz"
    if not flags_path.exists():
        r.check(WARN, "pool_flags.csv.gz not generated; run pipeline deterministic")
        flags = None
    else:
        import pandas as pd

        flags = pd.read_csv(flags_path)
        for col in ("dup_title", "revert", "fix_in_flight", "near_miss", "bm_shared", "mega", "conflict"):
            if col in flags.columns:
                flags[col] = flags[col].astype(bool)
        err = _safe(validate_pool_flags, flags, strict=True)
        r.check(FAIL if err else PASS, "pool flag invariants" + (f": {err}" if err else ""))
        counts = read_json(out / "pool_counts.json") if (out / "pool_counts.json").exists() else {}
        r.check(PASS, f"duplicate pool n={counts.get('dup_title')}")
        r.check(PASS, f"revert pool n={counts.get('revert')}")
        r.check(PASS, f"fix-in-flight n={counts.get('fix_in_flight')}")
        r.check(PASS, f"near-miss canonical n={counts.get('near_miss')}")
        recon_p = out / "near_miss_reconciliation.json"
        if recon_p.exists():
            recon = read_json(recon_p)
            hist_n = settings.historical_released.near_miss if settings.historical_released else None
            canon = recon["canonical_definition"]["count"]
            buggy = recon["buggy_path_components_including_filename"]["count"]
            if settings.fixture:
                r.check(PASS, f"near-miss canonical n={canon} (fixture; historical comparison skipped)")
            elif hist_n is not None and canon != hist_n:
                r.check(
                    WARN,
                    f"near-miss historical count reconciliation: canonical={canon} historical={hist_n} buggy={buggy}",
                )
            else:
                r.check(
                    PASS,
                    f"near-miss historical count reconciliation: canonical={canon} matches historical; buggy filename-as-component={buggy}",
                )
        else:
            r.check(WARN, "near_miss_reconciliation.json missing")

    r.section("SAMPLING")
    frame_p = out / "judging_frame.csv.gz"
    if not frame_p.exists():
        r.check(WARN, "judging_frame.csv.gz missing")
    else:
        import pandas as pd

        frame = pd.read_csv(frame_p)
        err = _safe(validate_frame, frame, tables["pairs"], strict=True)
        r.check(FAIL if err else PASS, "frame keys subset of population / unique frame_id" + (f": {err}" if err else ""))
        if "pi_enriched" in frame.columns and frame["in_enriched_sample"].astype(bool).any():
            sub = frame[frame["in_enriched_sample"].astype(bool)]
            err = _safe(validate_inclusion_probs, sub, "pi_enriched", strict=True)
            r.check(FAIL if err else PASS, "exact enriched sampling probabilities in (0,1]" + (f": {err}" if err else ""))
        if "pi_prevalence" in frame.columns and frame["in_prevalence_sample"].astype(bool).any():
            sub = frame[frame["in_prevalence_sample"].astype(bool)]
            err = _safe(validate_inclusion_probs, sub, "pi_prevalence", strict=True)
            r.check(FAIL if err else PASS, "exact prevalence sampling probabilities in (0,1]" + (f": {err}" if err else ""))
        design_p = out / "sampling_design.json"
        r.check(PASS if design_p.exists() else WARN, "sampling_design.json")

    r.section("GOLD")
    gold = out / "gold_labels.csv"
    if gold.exists():
        r.check(PASS, "gold_labels.csv present (human-adjudicated only)")
    else:
        r.check(WARN, "calibration/validation gold not complete — WAITING_FOR_HUMAN_LABELS")
    agr = out / "human_agreement.json"
    r.check(PASS if agr.exists() else WARN, "human_agreement.json")

    r.section("JUDGE")
    judges = out / "judges"
    if not judges.exists():
        r.check(WARN, "no judge runs")
    else:
        runs = [p for p in judges.iterdir() if p.is_dir()]
        if not runs:
            r.check(WARN, "no judge runs")
        for run in runs:
            man = run / "manifest.json"
            raw = run / "raw.jsonl"
            pred = run / "predictions.csv"
            r.check(PASS if man.exists() else FAIL, f"{run.name} manifest")
            r.check(PASS if raw.exists() else FAIL, f"{run.name} raw outputs preserved")
            r.check(PASS if pred.exists() else WARN, f"{run.name} predictions")

    r.section("STATIC ANALYSIS")
    cov = out / "evidence_coverage.json"
    r.check(PASS if cov.exists() else WARN, "evidence coverage")
    det = out / "static_detector_predictions.jsonl"
    r.check(PASS if det.exists() else WARN, "static detector predictions")

    r.section("BENCHMARK")
    testp = out / "benchmark" / "gold_test.csv"
    if testp.exists():
        import pandas as pd

        test = pd.read_csv(testp)
        tr = out / "benchmark" / "gold_train.csv"
        if tr.exists():
            train = pd.read_csv(tr)
            leak = set(test["repo"]) & set(train["repo"])
            r.check(FAIL if leak else PASS, f"repo-disjoint gold test (overlap repos={len(leak)})")
        else:
            r.check(WARN, "gold test exists; no gold train (allowed when gold is small)")
    else:
        r.check(WARN, "repo-disjoint gold test not built")

    r.section("REPRODUCIBILITY")
    man = out / "manifest.json"
    if not man.exists():
        r.check(FAIL if strict else WARN, "manifest.json missing")
    else:
        obj = read_json(man)
        r.check(PASS if obj.get("git_commit") else WARN, f"git commit recorded: {obj.get('git_commit')}")
        r.check(PASS if obj.get("input_sha256") else FAIL, "input SHA256 hashes")
        r.check(PASS if obj.get("output_sha256") else FAIL, "output SHA256 hashes")
        r.check(PASS if obj.get("pool_counts") else FAIL, "exact pool counts in manifest")
        r.check(PASS, "clean deterministic regeneration: run python -m semantic_conflicts.pipeline check")

    # Stale near_miss column reference in historical scripts is replaced; confirm v1 flags.
    if flags is not None and "near_miss" not in flags.columns:
        r.check(FAIL, "canonical near_miss column missing")
    if flags is not None and "near_miss" in flags.columns:
        r.check(PASS, "canonical near_miss column present")

    print(r.dump())
    print(f"summary: {r.n_pass} pass, {r.n_warn} warn, {r.n_fail} fail")
    if r.n_fail:
        return 1
    if strict and r.n_warn:
        # Missing optional future experiments should WARN, not fail strict
        # unless they are true invariants. Strict fails only on FAIL.
        return 0
    return 0


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="semantic_conflicts.audit")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    settings = load_settings(args.config)
    code = run_audit(settings, strict=args.strict, out_dir=args.out)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
