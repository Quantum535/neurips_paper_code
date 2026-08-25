"""Probability sampling for the judging frame.

Design (two coordinated samples; neither uses heuristic 1-∏(1-p_i) weights)
===========================================================================

Option A — disjoint membership strata (enriched / precision sample)
    Every pair is assigned exactly one ``primary_stratum`` (see pools.py):
        revert > dup > fix_in_flight > bm_shared > overlap_other > near_miss > control
    Nested cells are primary_stratum × mega × cross_agent × both_merged.
    Within cell s we draw n_s without replacement. The inclusion probability
    is exactly π_i = n_s / N_s for every unit in the cell (π_i = 1 if census).

    Rare scientifically important cells (all revert; all cross-agent candidates;
    any cell with N_s ≤ census_if_n_le) are taken in full.

    Pool-level precision is a *domain* Hájek mean: restrict to sampled units
    with the pool flag, using their disjoint-stratum π_i. This is not a simple
    random sample of the pool unless the pool coincides with a single cell.

Option B — population prevalence sample
    Independent stratified sample of the full 577,045-pair population.
    Strata: mega × cross_agent × conflict. π_i = n_s / N_s exactly.
    Prevalence estimates MUST use this sample (or a documented union with
    known joint inclusion probabilities). They must not use Option A weights
    as if A were an SRS of the population.

The released v0 frame used overlapping pool samples and
``1 - prod(1 - p_i)`` as a union probability. That formula is exact only for
independent Bernoulli (Poisson) sampling. The v0 draws were without-replacement
stratified samples from overlapping frames, so those weights were heuristic.
They are not reused.

Calibration / validation sheets are subsamples of the judging frame with a
fresh Generator stream (same seed documented in sampling_design.json).
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from semantic_conflicts.config import Settings
from semantic_conflicts.schemas import validate_frame, validate_inclusion_probs

NEST_COLS = ("mega", "cross_agent", "both_merged")
PREV_COLS = ("mega", "cross_agent", "conflict")


def _as_bool(s: pd.Series) -> pd.Series:
    return s.astype(bool)


def _cell_key(row: pd.Series, cols: Iterable[str]) -> tuple:
    return tuple(bool(row[c]) if c in row.index else False for c in cols)


def _hamilton_allocate(sizes: pd.Series, n: int) -> pd.Series:
    """Largest-remainder allocation of n seats to groups, clipped to group size."""
    sizes = sizes.astype(int)
    if sizes.sum() <= n:
        return sizes.copy()
    if n <= 0:
        return pd.Series(0, index=sizes.index, dtype=int)
    raw = sizes / sizes.sum() * n
    floor = np.floor(raw).astype(int).clip(upper=sizes)
    leftover = int(n - floor.sum())
    remainder = (raw - np.floor(raw)).sort_values(ascending=False)
    alloc = floor.copy()
    for idx in remainder.index:
        if leftover <= 0:
            break
        if alloc.loc[idx] < sizes.loc[idx]:
            alloc.loc[idx] += 1
            leftover -= 1
    # If clipping left seats, dump into largest remaining capacities.
    if leftover > 0:
        cap = (sizes - alloc).sort_values(ascending=False)
        for idx in cap.index:
            if leftover <= 0:
                break
            take = min(int(cap.loc[idx]), leftover)
            alloc.loc[idx] += take
            leftover -= take
    return alloc.astype(int)


def _sample_cells(
    df: pd.DataFrame,
    *,
    n_target: int | str,
    nest_cols: list[str],
    rng: np.random.Generator,
    census_if_n_le: int,
    force_census_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample within nested cells. Returns (picked, audit_rows)."""
    if len(df) == 0:
        empty = df.copy()
        empty["pi"] = pd.Series(dtype=float)
        return empty, pd.DataFrame()
    work = df.copy()
    work["_force"] = force_census_mask.reindex(work.index).fillna(False).astype(bool) if force_census_mask is not None else False
    for c in nest_cols:
        if c not in work.columns:
            work[c] = False
        work[c] = _as_bool(work[c])

    gcols = list(nest_cols)
    sizes = work.groupby(gcols, dropna=False).size() if gcols else pd.Series({(): len(work)})
    small_cells = set(sizes[sizes <= census_if_n_le].index.tolist())

    census_mask = work["_force"].copy()
    if gcols:
        keys = list(zip(*[work[c].astype(bool) for c in gcols]))
        census_mask = census_mask | pd.Series([k in small_cells for k in keys], index=work.index)
    else:
        if len(work) <= census_if_n_le:
            census_mask[:] = True

    picked_parts = []
    audit = []
    # Census portion
    cens = work[census_mask]
    if len(cens):
        part = cens.copy()
        part["pi"] = 1.0
        picked_parts.append(part)
        for key, sub in cens.groupby(gcols, dropna=False) if gcols else [((), cens)]:
            audit.append(
                {
                    "cell": str(key),
                    "N": int(len(sub)),
                    "n": int(len(sub)),
                    "pi": 1.0,
                    "mode": "census",
                }
            )
    rest = work[~census_mask]
    remaining_target = n_target
    if remaining_target == "census":
        remaining_target = len(rest)
    if isinstance(remaining_target, str):
        raise TypeError(n_target)
    remaining_target = max(0, int(remaining_target) - len(cens))
    if len(rest) and remaining_target > 0:
        if gcols:
            gsizes = rest.groupby(gcols, dropna=False).size()
            alloc = _hamilton_allocate(gsizes, min(remaining_target, int(gsizes.sum())))
            for key, sub in rest.groupby(gcols, dropna=False):
                n_k = int(alloc.loc[key]) if key in alloc.index else 0
                N_k = len(sub)
                n_k = min(n_k, N_k)
                if n_k <= 0:
                    audit.append({"cell": str(key), "N": N_k, "n": 0, "pi": float("nan"), "mode": "none"})
                    continue
                take = sub.iloc[rng.choice(N_k, size=n_k, replace=False)]
                take = take.copy()
                take["pi"] = n_k / N_k
                picked_parts.append(take)
                audit.append({"cell": str(key), "N": N_k, "n": n_k, "pi": n_k / N_k, "mode": "srs_wor"})
        else:
            N = len(rest)
            n = min(remaining_target, N)
            take = rest.iloc[rng.choice(N, size=n, replace=False)].copy()
            take["pi"] = n / N
            picked_parts.append(take)
            audit.append({"cell": "all", "N": N, "n": n, "pi": n / N, "mode": "srs_wor"})
    elif len(rest) == 0:
        pass
    picked = pd.concat(picked_parts, ignore_index=False) if picked_parts else work.iloc[[]].copy()
    return picked, pd.DataFrame(audit)


def build_prevalence_sample(
    flags: pd.DataFrame,
    settings: Settings,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cfg = settings.sampling.prevalence
    work = flags.copy()
    work["mega"] = _as_bool(work["mega"])
    work["cross_agent"] = _as_bool(work["cross_agent"])
    work["conflict"] = _as_bool(work["conflict"])
    gcols = list(cfg.strata)
    N = len(work)
    sizes = work.groupby(gcols, dropna=False).size()
    # Guarantee min_per_stratum (or census) then allocate remainder by Hamilton.
    mins = sizes.clip(upper=cfg.min_per_stratum)
    mins = pd.Series({k: min(int(sizes.loc[k]), int(cfg.min_per_stratum)) for k in sizes.index})
    floor_total = int(mins.sum())
    leftover = max(0, int(cfg.n) - floor_total)
    extra_cap = sizes - mins
    extra = _hamilton_allocate(extra_cap.clip(lower=0), leftover) if leftover else extra_cap * 0
    alloc = mins + extra
    alloc = alloc.clip(upper=sizes).astype(int)

    parts = []
    audit = []
    for key, sub in work.groupby(gcols, dropna=False):
        N_k = len(sub)
        n_k = int(alloc.loc[key]) if key in alloc.index else 0
        n_k = min(n_k, N_k)
        if n_k <= 0:
            audit.append({"stratum": str(key), "N": N_k, "n": 0, "pi": float("nan")})
            continue
        take = sub.iloc[rng.choice(N_k, size=n_k, replace=False)].copy()
        take["pi_prevalence"] = n_k / N_k
        parts.append(take)
        audit.append({"stratum": str(key), "N": N_k, "n": n_k, "pi": n_k / N_k})
    sample = pd.concat(parts, ignore_index=True) if parts else work.iloc[0:0].copy()
    sample["sample_role"] = "prevalence"
    meta = {
        "n_pop": N,
        "n_sample": int(len(sample)),
        "target_n": int(cfg.n),
        "estimator": settings.statistics.prevalence_estimator,
        "strata": gcols,
        "inclusion": "exact n_s / N_s within mega × cross_agent × conflict",
    }
    return sample, pd.DataFrame(audit), meta


def build_enriched_sample(
    flags: pd.DataFrame,
    settings: Settings,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cfg = settings.sampling.enriched
    work = flags.copy()
    work["mega"] = _as_bool(work["mega"])
    work["cross_agent"] = _as_bool(work["cross_agent"])
    work["both_merged"] = _as_bool(work["both_merged"])
    nest = [c for c in cfg.substrata if c in work.columns]
    force_all = pd.Series(False, index=work.index)
    if "cross_agent_candidate" in cfg.force_census_if and "cross_agent_candidate" in work.columns:
        force_all = force_all | _as_bool(work["cross_agent_candidate"])
    if "revert" in cfg.force_census_if:
        force_all = force_all | _as_bool(work["revert"])

    parts = []
    audits = []
    realized = {}
    for stratum, target in cfg.targets.items():
        sub = work[work["primary_stratum"].astype(str) == stratum]
        n_target: int | str
        if target == "census":
            n_target = "census"
            n_int = len(sub)
        else:
            n_int = int(target)
            n_target = n_int
        force = force_all.reindex(sub.index).fillna(False)
        picked, audit = _sample_cells(
            sub,
            n_target=n_int if n_target != "census" else len(sub),
            nest_cols=nest,
            rng=rng,
            census_if_n_le=int(cfg.census_if_n_le),
            force_census_mask=force,
        )
        if n_target == "census" and len(sub):
            picked = sub.copy()
            picked["pi"] = 1.0
            audit = pd.DataFrame(
                [{"cell": "census", "N": len(sub), "n": len(sub), "pi": 1.0, "mode": "census"}]
            )
        if len(picked):
            picked = picked.copy()
            picked["pi_enriched"] = picked["pi"].astype(float)
            parts.append(picked)
        audit = audit.copy()
        audit["primary_stratum"] = stratum
        audits.append(audit)
        realized[stratum] = int(len(picked))

    sample = pd.concat(parts, ignore_index=True) if parts else work.iloc[0:0].copy()
    # A pair might be force-included in revert and also appear... but primary_stratum is exclusive,
    # so keys are unique. Force-census of cross_agent_candidate can duplicate if we also sample
    # their primary stratum. Dedup by pair key keeping the *smaller* (more conservative? No:
    # exact π is the cell they were drawn from. Force-included units should have π=1.
    key = ["repo", "pr_a", "pr_b"]
    if len(sample):
        sample["_force_pi1"] = sample.index.isin(sample.index)  # placeholder
        sample = sample.sort_values("pi_enriched")  # keep smallest π? WRONG.
        # If a unit is in two draws, the actual inclusion is not independent.
        # Because primary strata are disjoint, the only duplication is force-census
        # of cross_agent_candidate that also belong to a sampled primary cell.
        # Those units are in the force mask of their own primary cell, so they
        # were already taken with π=1 in that cell. Drop duplicates keeping π=1.
        sample = sample.sort_values("pi_enriched", ascending=False).drop_duplicates(key, keep="first")
    sample["sample_role"] = "enriched"
    meta = {
        "n_sample": int(len(sample)),
        "realized_by_stratum": realized,
        "inclusion": "exact n_s / N_s in disjoint primary_stratum × mega × cross_agent × both_merged cells",
        "force_census": cfg.force_census_if,
    }
    audit_df = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    return sample, audit_df, meta


def attach_judge_inputs(
    frame: pd.DataFrame,
    texts: pd.DataFrame,
    files: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    title = {(str(r), int(p)): (t if isinstance(t, str) else "") for r, p, t in zip(texts.repo, texts.pr, texts.title)}
    body = {(str(r), int(p)): (t if isinstance(t, str) else "") for r, p, t in zip(texts.repo, texts.pr, texts.body)}
    cap = settings.body_truncation_chars
    flist: dict[tuple[str, int], list[str]] = {}
    for r, p, f in zip(files.repo, files.pr, files.filepath):
        flist.setdefault((str(r), int(p)), []).append(str(f))
    out = frame.copy()
    ncap = settings.files_list_cap
    for side, pcol in (("a", "pr_a"), ("b", "pr_b")):
        keys = [(str(r), int(p)) for r, p in zip(out["repo"], out[pcol])]
        out[f"title_{side}"] = [title.get(k, "") for k in keys]
        out[f"body_{side}"] = [str(body.get(k, ""))[:cap] for k in keys]
        out[f"files_{side}"] = ["\n".join(sorted(flist.get(k, []))[:ncap]) for k in keys]
        out[f"n_files_{side}"] = [len(flist.get(k, [])) for k in keys]
    shared = []
    for r, a, b in zip(out["repo"], out["pr_a"], out["pr_b"]):
        sa = set(flist.get((str(r), int(a)), []))
        sb = set(flist.get((str(r), int(b)), []))
        shared.append("\n".join(sorted(sa & sb)[:ncap]))
    out["files_shared"] = shared
    out["has_diff"] = False
    return out


def build_calibration(
    frame: pd.DataFrame,
    settings: Settings,
    rng: np.random.Generator,
) -> pd.DataFrame:
    quotas = settings.sampling.calibration.quotas
    n = int(settings.sampling.calibration.n)
    parts = []

    def take(mask: pd.Series, k: int) -> pd.DataFrame:
        sub = frame.loc[mask]
        if len(sub) == 0 or k <= 0:
            return sub.iloc[0:0]
        k = min(k, len(sub))
        return sub.iloc[rng.choice(len(sub), size=k, replace=False)]

    mapping = {
        "overlap": frame["conflict"].astype(bool) if "conflict" in frame.columns else pd.Series(False, index=frame.index),
        "dup": frame["primary_stratum"].astype(str).eq("dup") if "primary_stratum" in frame.columns else pd.Series(False, index=frame.index),
        "revert": frame["primary_stratum"].astype(str).eq("revert") if "primary_stratum" in frame.columns else pd.Series(False, index=frame.index),
        "near_miss": frame["primary_stratum"].astype(str).eq("near_miss") if "primary_stratum" in frame.columns else pd.Series(False, index=frame.index),
        "control": frame["primary_stratum"].astype(str).eq("control") if "primary_stratum" in frame.columns else pd.Series(False, index=frame.index),
    }
    for name, k in quotas.items():
        if name in mapping:
            parts.append(take(mapping[name], int(k)))
    cal = pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0]
    cal = cal.drop_duplicates(["repo", "pr_a", "pr_b"])
    if len(cal) > n:
        cal = cal.iloc[rng.choice(len(cal), size=n, replace=False)]
    elif len(cal) < n:
        leftover = frame.drop(index=cal.index, errors="ignore")
        need = min(n - len(cal), len(leftover))
        if need:
            extra = leftover.iloc[rng.choice(len(leftover), size=need, replace=False)]
            cal = pd.concat([cal, extra], ignore_index=True)
    return cal.reset_index(drop=True)


def assemble_frame(
    flags: pd.DataFrame,
    texts: pd.DataFrame,
    files: pd.DataFrame,
    settings: Settings,
) -> dict[str, Any]:
    rng = np.random.default_rng(settings.sampling.seed)
    prev, prev_audit, prev_meta = build_prevalence_sample(flags, settings, rng)
    enr, enr_audit, enr_meta = build_enriched_sample(flags, settings, rng)

    prev_keys = set(map(tuple, prev[["repo", "pr_a", "pr_b"]].itertuples(index=False, name=None))) if len(prev) else set()
    enr_keys = set(map(tuple, enr[["repo", "pr_a", "pr_b"]].itertuples(index=False, name=None))) if len(enr) else set()
    union_keys = list(prev_keys | enr_keys)
    if not union_keys:
        frame = flags.iloc[0:0].copy()
    else:
        indexed = flags.set_index(["repo", "pr_a", "pr_b"])
        frame = indexed.loc[union_keys].reset_index()

    prev_pi = (
        prev.set_index(["repo", "pr_a", "pr_b"])["pi_prevalence"] if len(prev) and "pi_prevalence" in prev.columns else pd.Series(dtype=float)
    )
    enr_pi = (
        enr.set_index(["repo", "pr_a", "pr_b"])["pi_enriched"] if len(enr) and "pi_enriched" in enr.columns else pd.Series(dtype=float)
    )
    keys = list(zip(frame["repo"], frame["pr_a"], frame["pr_b"]))
    frame["in_prevalence_sample"] = [(k in prev_keys) for k in keys]
    frame["in_enriched_sample"] = [(k in enr_keys) for k in keys]
    frame["pi_prevalence"] = [float(prev_pi.loc[k]) if k in prev_pi.index else np.nan for k in keys]
    frame["pi_enriched"] = [float(enr_pi.loc[k]) if k in enr_pi.index else np.nan for k in keys]
    # Analysis-specific inclusion: do not invent a union probability.
    frame["incl_prob"] = frame["pi_enriched"]  # default column for precision analyses
    frame = attach_judge_inputs(frame, texts, files, settings)
    frame = frame.sample(frac=1.0, random_state=int(settings.seed), replace=False).reset_index(drop=True)
    frame["frame_id"] = np.arange(len(frame), dtype=int)
    frame["dataset_version"] = settings.version
    frame["config_version"] = settings.version

    cal = build_calibration(frame, settings, np.random.default_rng(settings.sampling.seed + 1))
    val_n = int(settings.sampling.validation.n) if hasattr(settings.sampling.validation, "n") else int(settings.sampling.validation.get("n", 0))
    leftover = frame[~frame["frame_id"].isin(set(cal["frame_id"]))] if len(cal) and "frame_id" in cal.columns else frame
    rng_v = np.random.default_rng(settings.sampling.seed + 2)
    n_val = min(val_n, len(leftover))
    validation = leftover.iloc[rng_v.choice(len(leftover), size=n_val, replace=False)].copy() if n_val else leftover.iloc[0:0]

    validate_frame(frame, flags, strict=True)
    if frame["in_enriched_sample"].any():
        validate_inclusion_probs(frame.loc[frame["in_enriched_sample"]], "pi_enriched", strict=True)
    if frame["in_prevalence_sample"].any():
        validate_inclusion_probs(frame.loc[frame["in_prevalence_sample"]], "pi_prevalence", strict=True)

    from semantic_conflicts.statistics import flag_extreme_weights, kish_ess

    prev_pi_arr = frame.loc[frame["in_prevalence_sample"], "pi_prevalence"].to_numpy(dtype=float)
    enr_pi_arr = frame.loc[frame["in_enriched_sample"], "pi_enriched"].to_numpy(dtype=float)
    design = {
        "seed": settings.sampling.seed,
        "config_version": settings.version,
        "option_a_enriched": {
            **enr_meta,
            "n_eff": kish_ess(1.0 / enr_pi_arr) if len(enr_pi_arr) else None,
            "extreme_weights": flag_extreme_weights(enr_pi_arr, ratio=settings.statistics.extreme_weight_ratio)
            if len(enr_pi_arr)
            else None,
        },
        "option_b_prevalence": {
            **prev_meta,
            "n_eff": kish_ess(1.0 / prev_pi_arr) if len(prev_pi_arr) else None,
            "extreme_weights": flag_extreme_weights(prev_pi_arr, ratio=settings.statistics.extreme_weight_ratio)
            if len(prev_pi_arr)
            else None,
        },
        "union_frame_size": int(len(frame)),
        "calibration_n": int(len(cal)),
        "validation_n": int(len(validation)),
        "v0_heuristic_union_probability_not_used": True,
        "notes": (
            "Prevalence estimation must use in_prevalence_sample with pi_prevalence. "
            "Pool precision must use in_enriched_sample with pi_enriched as domain weights. "
            "Do not use 1-prod(1-p_i)."
        ),
    }
    return {
        "frame": frame,
        "calibration": cal,
        "validation": validation,
        "prevalence_audit": prev_audit,
        "enriched_audit": enr_audit,
        "design": design,
        "prevalence_meta": prev_meta,
        "enriched_meta": enr_meta,
    }
