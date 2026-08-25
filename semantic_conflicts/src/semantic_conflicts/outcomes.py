"""Outcome-association analyses (not causal).

Later-PR merge vs exposure, with:
  - pair-level 2x2 / crude OR
  - repo-stratified Mantel–Haenszel OR + RBG CI
  - later-PR dedup (one row per distinct later PR)
  - mega-repo exclusion robustness
"""

from __future__ import annotations

import pandas as pd

from semantic_conflicts.config import Settings
from semantic_conflicts.statistics import mantel_haenszel_or, wilson_interval


def _later_is_b(df: pd.DataFrame) -> pd.Series:
    oa = pd.to_datetime(df["opened_a"], utc=True)
    ob = pd.to_datetime(df["opened_b"], utc=True)
    return ob >= oa


def with_later_outcome(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    later_b = _later_is_b(out)
    out["merged_later"] = out["merged_b"].where(later_b, out["merged_a"]).astype(bool)
    out["pr_later"] = out["pr_b"].where(later_b, out["pr_a"])
    return out


def _mh(df: pd.DataFrame, exposed: str, settings: Settings) -> dict:
    return mantel_haenszel_or(df, stratum="repo", exposed=exposed, outcome="merged_later")


def outcome_analyses(flags: pd.DataFrame, settings: Settings) -> dict:
    df = with_later_outcome(flags)
    z = settings.statistics.wilson_z
    reports: dict = {}

    a = _mh(df, "conflict", settings)
    a["exposure"] = "file-overlap conflict"
    a["control"] = "no shared file (all other pairs)"
    reports["a_conflict_vs_none"] = a

    b = _mh(df.assign(dup=df["dup_title"].astype(bool)), "dup", settings)
    b["exposure"] = "duplicate-title pair"
    b["control"] = "all non-duplicate pairs"
    reports["b_dup_title_vs_rest"] = b

    conf = df[df["conflict"].astype(bool)].copy()
    if len(conf):
        c = _mh(conf, "fix_in_flight", settings)
        c["exposure"] = "fix-in-flight (later title has fix/bug/patch token)"
        c["control"] = "other conflicting pairs"
        reports["c_fif_vs_other_conflict"] = c

    disjoint = df[~df["conflict"].astype(bool)].copy()
    if len(disjoint):
        d = _mh(disjoint, "near_miss", settings)
        d["exposure"] = "near-miss (shared depth-2 dir, no shared file)"
        d["control"] = "no shared file and no shared depth-2 dir"
        reports["d_nearmiss_vs_nothing"] = d

    reports["e_both_merged_shared_file"] = {
        "excluded": True,
        "reason": (
            "Pool is outcome-conditioned: membership requires merged_a AND merged_b, "
            "so the later-PR merge outcome is deterministically 1 for every exposed pair."
        ),
    }

    # PR-level dedup: one row per (repo, later PR)
    pl = (
        df.groupby(["repo", "pr_later"], as_index=False)
        .agg(ever_conf=("conflict", "max"), merged_later=("merged_later", "first"))
    )
    pl["ever_conf"] = pl["ever_conf"].astype(bool)
    reports["a_dedup_pr_level"] = _mh(pl, "ever_conf", settings)
    reports["a_dedup_pr_level"]["exposure"] = "later PR ever in a conflicting pair"
    reports["a_dedup_pr_level"]["control"] = "later PR never in a conflicting pair"

    pld = (
        df.groupby(["repo", "pr_later"], as_index=False)
        .agg(ever_dup=("dup_title", "max"), merged_later=("merged_later", "first"))
    )
    pld["ever_dup"] = pld["ever_dup"].astype(bool)
    reports["b_dedup_pr_level"] = _mh(pld, "ever_dup", settings)
    reports["b_dedup_pr_level"]["exposure"] = "later PR ever in a duplicate-title pair"
    reports["b_dedup_pr_level"]["control"] = "later PR never in a duplicate-title pair"

    nomega = df[~df["mega"].astype(bool)]
    reports["robustness_excl_mega"] = {
        "n_pairs": int(len(nomega)),
        "a_conflict_vs_none": _mh(nomega, "conflict", settings),
    }

    # Headline binomial CIs (descriptive; pairs are clustered by repo)
    n = len(df)
    k_conf = int(df["conflict"].astype(bool).sum())
    ca = df[df["cross_agent"].astype(bool)]
    headlines = {
        "conflict_rate_all_pairs": wilson_interval(k_conf, n, z),
        "conflict_rate_cross_agent": wilson_interval(int(ca["conflict"].sum()), len(ca), z) if len(ca) else None,
        "n_pairs": n,
        "n_repos": int(df["repo"].nunique()),
        "note": "Wilson intervals treat pairs as iid; cluster-bootstrap CIs are attached below",
    }
    from semantic_conflicts.statistics import cluster_bootstrap_mean

    n_boot = int(settings.statistics.bootstrap_iterations)
    seed = int(settings.statistics.bootstrap_seed)
    headlines["conflict_rate_cluster_bootstrap"] = cluster_bootstrap_mean(
        df, cluster_col="repo", y_col="conflict", n_iter=n_boot, seed=seed
    )
    if len(disjoint):
        headlines["near_miss_rate_among_disjoint_cluster_bootstrap"] = cluster_bootstrap_mean(
            disjoint,
            cluster_col="repo",
            y_col="near_miss",
            n_iter=n_boot,
            seed=seed,
        )
    dup = df[df["dup_title"].astype(bool)]
    if len(dup):
        headlines["dup_both_merged"] = wilson_interval(int(dup["both_merged"].sum()), len(dup), z)
        headlines["dup_exactly_one_merged"] = wilson_interval(
            int((dup["merged_a"] ^ dup["merged_b"]).sum()), len(dup), z
        )
        headlines["dup_neither_merged"] = wilson_interval(
            int((~dup["merged_a"] & ~dup["merged_b"]).sum()), len(dup), z
        )
        headlines["dup_no_shared_file"] = wilson_interval(int((~dup["conflict"].astype(bool)).sum()), len(dup), z)
    bm = df[df["bm_shared"].astype(bool)]
    if len(bm) and "closed_a" in bm.columns:
        ca_t = pd.to_datetime(bm["closed_a"], utc=True)
        cb_t = pd.to_datetime(bm["closed_b"], utc=True)
        gap_h = (cb_t - ca_t).abs().dt.total_seconds() / 3600.0
        headlines["both_merged_shared_gap_lt24h"] = wilson_interval(int((gap_h < 24).sum()), len(bm), z)
        headlines["bm_shared_n"] = len(bm)
    return {"outcomes": reports, "headlines": headlines}
