"""Statistics that do not assume pairs are IID.

Estimators
----------
Wilson interval
    Standard binomial score interval. Appropriate for iid Bernoulli draws or
    as a *descriptive* interval. Not a substitute for cluster-robust CIs.

Mantel–Haenszel odds ratio
    Repo-stratified OR. Variance of log OR uses the Robins–Breslow–Greenland
    estimator. This is an association measure, not a causal effect.

Hájek prevalence (default)
    For a probability sample with inclusion probabilities π_i > 0,
        Ŷ_Hájek = (Σ_{i∈s} y_i / π_i) / (Σ_{i∈s} 1/π_i)
    This is a ratio estimator of the population mean. It is preferred to
    Horvitz–Thompson for a bounded mean because it is invariant to scaling of
    weights and typically more stable.

Horvitz–Thompson mean
        Ŷ_HT = (1/N) Σ_{i∈s} y_i / π_i
    Unbiased for the population mean when π_i are exact design probabilities
    and N is known. Reported alongside Hájek; default published number is Hájek.

Cluster bootstrap
    Repositories are the clusters. Draw R repos with replacement, keep all
    sampled pairs (and their design weights) from each drawn repo, recompute
    the estimator. Percentile CIs. Deterministic given seed.

Effective sample number
        n_eff = (Σ w_i)^2 / Σ w_i^2
    Kish's approximation. Extreme weights are flagged, never silently clipped.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from semantic_conflicts.schemas import VALID_CATEGORIES


def wilson_interval(k: int | float, n: int | float, z: float = 1.959963984540054) -> dict:
    k, n = float(k), float(n)
    if n <= 0:
        return {"p": float("nan"), "lo": float("nan"), "hi": float("nan"), "k": k, "n": n}
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return {"p": float(p), "lo": float(center - half), "hi": float(center + half), "k": k, "n": n}


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    a_s, b_s = pd.Series(list(a)), pd.Series(list(b))
    if len(a_s) == 0:
        return float("nan")
    po = float((a_s.values == b_s.values).mean())
    cats = sorted(set(a_s) | set(b_s))
    pe = sum(float((a_s == c).mean()) * float((b_s == c).mean()) for c in cats)
    return float((po - pe) / (1.0 - pe)) if pe < 1.0 else float("nan")


def fleiss_kappa(ratings: pd.DataFrame) -> float:
    """ratings: rows = items, columns = annotators, values = categories."""
    n_items = len(ratings)
    n_raters = ratings.shape[1]
    if n_items == 0 or n_raters < 2:
        return float("nan")
    cats = sorted({v for col in ratings.columns for v in ratings[col].dropna().unique()})
    N = n_items
    n = n_raters
    p_j = []
    P_i = []
    for cat in cats:
        counts = (ratings == cat).sum(axis=1)
        p_j.append(float(counts.sum()) / (N * n))
    p_j = np.array(p_j)
    for i in range(N):
        row = ratings.iloc[i]
        counts = np.array([(row == cat).sum() for cat in cats], dtype=float)
        P_i.append((counts * (counts - 1)).sum() / (n * (n - 1)))
    P_bar = float(np.mean(P_i))
    P_e = float((p_j**2).sum())
    return float((P_bar - P_e) / (1.0 - P_e)) if P_e < 1.0 else float("nan")


def krippendorff_alpha(ratings: pd.DataFrame, *, nominal: bool = True) -> float:
    """Nominal Krippendorff's α for two-or-more annotators (listwise skip of NaN)."""
    cats = sorted({v for col in ratings.columns for v in ratings[col].dropna().unique()})
    if len(cats) < 2:
        return float("nan")
    cat_index = {c: i for i, c in enumerate(cats)}
    coincidence = np.zeros((len(cats), len(cats)), dtype=float)
    for _, row in ratings.iterrows():
        vals = [v for v in row.tolist() if pd.notna(v)]
        m = len(vals)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                coincidence[cat_index[vals[i]], cat_index[vals[j]]] += 1.0 / (m - 1)
    n = coincidence.sum()
    if n <= 0:
        return float("nan")
    n_c = coincidence.sum(axis=1)
    de = 0.0
    do = 0.0
    for c in range(len(cats)):
        for k in range(len(cats)):
            delta = 0.0 if c == k else 1.0
            do += coincidence[c, k] * delta
            de += n_c[c] * n_c[k] * delta
    de = de / (n - 1) if n > 1 else 0.0
    return float(1.0 - do / de) if de else float("nan")


def confusion_matrix(y_true: Sequence, y_pred: Sequence, labels: Sequence[str] | None = None) -> pd.DataFrame:
    labels = list(labels) if labels is not None else sorted(VALID_CATEGORIES)
    t = pd.Series(list(y_true)).astype(str)
    p = pd.Series(list(y_pred)).astype(str)
    mat = pd.crosstab(t, p, dropna=False)
    mat = mat.reindex(index=labels, columns=labels, fill_value=0)
    mat.index.name = "true"
    mat.columns.name = "pred"
    return mat


def per_class_prf(y_true: Sequence, y_pred: Sequence, labels: Sequence[str] | None = None) -> pd.DataFrame:
    labels = list(labels) if labels is not None else sorted(VALID_CATEGORIES)
    cm = confusion_matrix(y_true, y_pred, labels)
    rows = []
    for lab in labels:
        tp = int(cm.loc[lab, lab]) if lab in cm.index and lab in cm.columns else 0
        fp = int(cm[lab].sum() - tp) if lab in cm.columns else 0
        fn = int(cm.loc[lab].sum() - tp) if lab in cm.index else 0
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (2 * prec * rec / (prec + rec)) if (prec == prec and rec == rec and (prec + rec)) else float("nan")
        rows.append(
            {
                "label": lab,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": int((pd.Series(list(y_true)).astype(str) == lab).sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    return pd.DataFrame(rows)


def mantel_haenszel_or(
    table: pd.DataFrame,
    *,
    stratum: str,
    exposed: str,
    outcome: str,
) -> dict:
    """Repo-stratified MH odds ratio with Robins–Breslow–Greenland 95% CI.

    Cells per stratum (standard 2x2):
        a = exposed & outcome, b = exposed & ~outcome,
        c = ~exposed & outcome, d = ~exposed & ~outcome.
    Strata with a zero margin that make R+S undefined are skipped.
    """
    z = 1.959963984540054
    R = S = 0.0
    num_var_p = num_var_q = num_var_mid = 0.0
    n_used = n_disc = 0
    a_tot = b_tot = c_tot = d_tot = 0
    for _, g in table.groupby(stratum, dropna=False):
        exp = g[exposed].astype(bool).to_numpy()
        out = g[outcome].astype(bool).to_numpy()
        a = float((exp & out).sum())
        b = float((exp & ~out).sum())
        c = float((~exp & out).sum())
        d = float((~exp & ~out).sum())
        n = a + b + c + d
        if n <= 0:
            continue
        if (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
            continue
        n_used += 1
        if (a > 0 and d > 0) or (b > 0 and c > 0):
            n_disc += 1
        a_tot += a
        b_tot += b
        c_tot += c
        d_tot += d
        Ri = a * d / n
        Si = b * c / n
        R += Ri
        S += Si
        Pi = (a + d) / n
        Qi = (b + c) / n
        num_var_p += Pi * Ri
        num_var_q += Qi * Si
        num_var_mid += Pi * Si + Qi * Ri
    if S == 0:
        or_mh = float("inf") if R > 0 else float("nan")
        se = float("nan")
        lo = hi = float("nan")
    else:
        or_mh = R / S
        # Robins–Breslow–Greenland variance of log MH-OR
        se = np.sqrt(num_var_p / (2 * R * R) + num_var_mid / (2 * R * S) + num_var_q / (2 * S * S))
        lo = float(np.exp(np.log(or_mh) - z * se))
        hi = float(np.exp(np.log(or_mh) + z * se))
    p1 = a_tot / (a_tot + b_tot) if (a_tot + b_tot) else float("nan")
    p0 = c_tot / (c_tot + d_tot) if (c_tot + d_tot) else float("nan")
    or_crude = (a_tot * d_tot) / (b_tot * c_tot) if (b_tot * c_tot) else float("nan")
    return {
        "or_mh": float(or_mh),
        "ci_lo": lo,
        "ci_hi": hi,
        "se_log": float(se) if se == se else None,
        "n_strata_used": n_used,
        "n_strata_discordant": n_disc,
        "n_exposed": int(a_tot + b_tot),
        "n_control": int(c_tot + d_tot),
        "merged_rate_exposed": float(p1) if p1 == p1 else None,
        "merged_rate_control": float(p0) if p0 == p0 else None,
        "crude": {
            "or_crude": float(or_crude) if or_crude == or_crude else None,
            "cells": {"a": int(a_tot), "b": int(b_tot), "c": int(c_tot), "d": int(d_tot)},
        },
    }


def kish_ess(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return float("nan")
    return float((w.sum() ** 2) / (w**2).sum())


def hajek_mean(y: np.ndarray, pi: np.ndarray) -> float:
    """Hájek ratio estimator of a population mean. π_i must be exact design probs in (0,1]."""
    y = np.asarray(y, dtype=float)
    pi = np.asarray(pi, dtype=float)
    if (pi <= 0).any() or (pi > 1).any():
        raise ValueError("inclusion probabilities must lie in (0, 1]")
    w = 1.0 / pi
    return float((w * y).sum() / w.sum())


def horvitz_thompson_mean(y: np.ndarray, pi: np.ndarray, n_pop: int) -> float:
    y = np.asarray(y, dtype=float)
    pi = np.asarray(pi, dtype=float)
    if n_pop <= 0:
        raise ValueError("n_pop must be positive")
    if (pi <= 0).any() or (pi > 1).any():
        raise ValueError("inclusion probabilities must lie in (0, 1]")
    return float(((y / pi).sum()) / n_pop)


def weighted_prevalence(
    y: np.ndarray,
    pi: np.ndarray,
    *,
    n_pop: int,
    estimator: str = "hajek",
) -> dict:
    y = np.asarray(y, dtype=float)
    pi = np.asarray(pi, dtype=float)
    w = 1.0 / pi
    hajek = hajek_mean(y, pi)
    ht = horvitz_thompson_mean(y, pi, n_pop)
    value = hajek if estimator == "hajek" else ht
    return {
        "estimator": estimator,
        "value": float(value),
        "hajek": float(hajek),
        "horvitz_thompson": float(ht),
        "n_sample": int(len(y)),
        "n_pop": int(n_pop),
        "n_eff": kish_ess(w),
        "weight_min": float(w.min()) if len(w) else None,
        "weight_max": float(w.max()) if len(w) else None,
        "weight_median": float(np.median(w)) if len(w) else None,
    }


def flag_extreme_weights(pi: np.ndarray, *, ratio: float = 50.0) -> dict:
    w = 1.0 / np.asarray(pi, dtype=float)
    med = float(np.median(w)) if len(w) else float("nan")
    thresh = med * ratio if med == med else float("inf")
    extreme = w >= thresh
    return {
        "n_extreme": int(extreme.sum()),
        "threshold": thresh,
        "ratio": ratio,
        "max_weight": float(w.max()) if len(w) else None,
        "median_weight": med,
        "note": "Weights are never clipped; this is a diagnostic only.",
    }


def cluster_bootstrap_mean(
    df: pd.DataFrame,
    *,
    cluster_col: str,
    y_col: str,
    n_iter: int,
    seed: int,
) -> dict:
    """Fast cluster bootstrap of a pair-level mean using per-cluster sums."""
    y = df[y_col].astype(float)
    g = df.assign(_y=y).groupby(cluster_col, sort=False).agg(sy=("_y", "sum"), n=("_y", "size"))
    sy = g["sy"].to_numpy(dtype=float)
    n = g["n"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    n_c = len(g)
    stats = np.empty(n_iter, dtype=float)
    if n_c == 0:
        return {"value": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_iter": 0, "n_clusters": 0}
    for i in range(n_iter):
        draw = rng.integers(0, n_c, size=n_c)
        stats[i] = sy[draw].sum() / n[draw].sum()
    point = float(y.mean())
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return {
        "value": point,
        "lo": float(lo),
        "hi": float(hi),
        "n_iter": n_iter,
        "n_clusters": n_c,
        "seed": seed,
        "method": "cluster_bootstrap_mean_percentile",
        "cluster": cluster_col,
    }


def cluster_bootstrap(
    df: pd.DataFrame,
    *,
    cluster_col: str,
    statistic: Callable[[pd.DataFrame], float],
    n_iter: int,
    seed: int,
) -> dict:
    """Resample clusters (repositories) with replacement; percentile 95% CI.

    Each replicate keeps all rows belonging to the drawn clusters. If a repo is
    drawn k times its rows appear k times (standard cluster bootstrap).
    """
    rng = np.random.default_rng(seed)
    clusters = df[cluster_col].astype(str).unique()
    n_c = len(clusters)
    if n_c == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_iter": 0, "n_clusters": 0}
    grouped = {c: g for c, g in df.groupby(cluster_col, sort=False)}
    stats = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        draw = rng.choice(clusters, size=n_c, replace=True)
        parts = [grouped[c] for c in draw]
        boot = pd.concat(parts, ignore_index=True)
        stats[i] = statistic(boot)
    point = float(statistic(df))
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return {
        "value": point,
        "lo": float(lo),
        "hi": float(hi),
        "n_iter": n_iter,
        "n_clusters": n_c,
        "seed": seed,
        "method": "cluster_bootstrap_percentile",
        "cluster": cluster_col,
    }


def multiclass_metrics(y_true: Sequence, y_pred: Sequence, labels: Sequence[str] | None = None) -> dict:
    labels = list(labels) if labels is not None else sorted(VALID_CATEGORIES)
    y_true = pd.Series(list(y_true)).astype(str)
    y_pred = pd.Series(list(y_pred)).astype(str)
    prf = per_class_prf(y_true, y_pred, labels)
    acc = float((y_true.values == y_pred.values).mean()) if len(y_true) else float("nan")
    recs = prf["recall"].to_numpy(dtype=float)
    balanced = float(np.nanmean(recs)) if len(recs) else float("nan")
    macro_f1 = float(np.nanmean(prf["f1"].to_numpy(dtype=float)))
    cm = confusion_matrix(y_true, y_pred, labels)
    tp = int(np.trace(cm.to_numpy()))
    total = int(cm.to_numpy().sum())
    micro_f1 = tp / total if total else float("nan")  # for a partition, micro-F1 = accuracy
    return {
        "n": int(len(y_true)),
        "accuracy": acc,
        "balanced_accuracy": balanced,
        "macro_f1": macro_f1,
        "micro_f1": float(micro_f1),
        "per_class": prf.to_dict(orient="records"),
        "confusion": cm.astype(int).to_dict(orient="index"),
    }


def binary_scores(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """AUROC / AUPRC / Brier for a binary label with a score in [0,1] (or any real)."""
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    out: dict = {"n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
    if y.min() == y.max():
        out.update({"auroc": None, "auprc": None, "note": "single class; ROC undefined"})
    else:
        out["auroc"] = float(roc_auc_score(y, s))
        out["auprc"] = float(average_precision_score(y, s))
    pred = (s >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    out["precision_at_0.5"] = float(p)
    out["recall_at_0.5"] = float(r)
    out["f1_at_0.5"] = float(f1)
    try:
        out["brier"] = float(brier_score_loss(y, np.clip(s, 0, 1)))
    except Exception:
        out["brier"] = None
    return out
