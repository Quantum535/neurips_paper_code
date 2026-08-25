"""Strict schemas and invariant checks for Semantic Conflicts artifacts."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from semantic_conflicts.config import Settings

PAIR_KEY = ["repo", "pr_a", "pr_b"]
VALID_CATEGORIES = frozenset({"D", "C", "H", "B", "I", "none"})
PRIMARY_STRATA = (
    "revert",
    "dup",
    "fix_in_flight",
    "bm_shared",
    "overlap_other",
    "near_miss",
    "control",
)
POOL_FLAG_COLUMNS = (
    "dup_title",
    "revert",
    "fix_in_flight",
    "near_miss",
    "both_merged",
    "bm_shared",
    "mega",
    "cross_agent",
    "conflict",
)


class InvariantError(ValueError):
    """Raised when a dataset invariant is violated."""


def _fail(errors: list[str], *, strict: bool) -> list[str]:
    if errors and strict:
        raise InvariantError("Invariant violations:\n- " + "\n- ".join(errors))
    return errors


def require_columns(df: pd.DataFrame, cols: Iterable[str], name: str = "frame") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise InvariantError(f"{name} missing columns: {missing}")


def validate_pair_keys(df: pd.DataFrame, *, name: str = "pairs", strict: bool = True) -> list[str]:
    require_columns(df, PAIR_KEY, name)
    errors: list[str] = []
    dup = df.duplicated(PAIR_KEY).sum()
    if dup:
        errors.append(f"{name}: {dup} duplicate (repo, pr_a, pr_b) keys")
    same = int((df["pr_a"] == df["pr_b"]).sum())
    if same:
        errors.append(f"{name}: {same} rows with pr_a == pr_b")
    if df[PAIR_KEY].isna().any().any():
        errors.append(f"{name}: null values in pair key")
    return _fail(errors, strict=strict)


def validate_base_pairs(
    pairs: pd.DataFrame,
    settings: Settings,
    *,
    strict: bool = True,
) -> list[str]:
    errors = validate_pair_keys(pairs, name="base_pairs", strict=False)
    n = len(pairs)
    expected = settings.expected_n_pairs
    if expected is not None and n != expected:
        errors.append(f"base pair count {n} != expected_n_pairs {expected}")
    require_columns(pairs, ["conflict", "cross_agent", "opened_a", "opened_b"], "base_pairs")
    if "opened_a" in pairs.columns and "opened_b" in pairs.columns:
        oa = pd.to_datetime(pairs["opened_a"], utc=True, errors="coerce")
        ob = pd.to_datetime(pairs["opened_b"], utc=True, errors="coerce")
        bad = int((ob < oa).sum())
        if bad:
            errors.append(f"{bad} rows with opened_b < opened_a")
    if pairs["conflict"].isna().any():
        errors.append("conflict has nulls")
    if pairs["cross_agent"].isna().any():
        errors.append("cross_agent has nulls")
    return _fail(errors, strict=strict)


def validate_pool_flags(df: pd.DataFrame, *, strict: bool = True) -> list[str]:
    errors = validate_pair_keys(df, name="pool_flags", strict=False)
    for col in POOL_FLAG_COLUMNS:
        if col not in df.columns:
            errors.append(f"pool_flags missing boolean column {col}")
            continue
        s = df[col]
        if s.dtype != bool and set(pd.unique(s.dropna())) - {0, 1, True, False}:
            errors.append(f"{col} is not boolean-valued")
        if s.isna().any():
            errors.append(f"{col} has nulls")
    if "near_miss" in df.columns and "conflict" in df.columns:
        leak = int((df["near_miss"] & df["conflict"]).sum())
        if leak:
            errors.append(f"near_miss must be disjoint-file; {leak} overlapping-file rows")
    if "bm_shared" in df.columns and "conflict" in df.columns:
        bad = int((df["bm_shared"] & ~df["conflict"].astype(bool)).sum())
        if bad:
            errors.append(f"bm_shared implies conflict; {bad} violations")
    if "fix_in_flight" in df.columns and "conflict" in df.columns:
        bad = int((df["fix_in_flight"] & ~df["conflict"].astype(bool)).sum())
        if bad:
            errors.append(f"fix_in_flight implies conflict; {bad} violations")
    if "primary_stratum" in df.columns:
        unknown = set(df["primary_stratum"].astype(str).unique()) - set(PRIMARY_STRATA)
        if unknown:
            errors.append(f"invalid primary_stratum values: {sorted(unknown)}")
    return _fail(errors, strict=strict)


def validate_inclusion_probs(df: pd.DataFrame, col: str, *, strict: bool = True) -> list[str]:
    errors: list[str] = []
    if col not in df.columns:
        errors.append(f"missing inclusion probability column {col}")
        return _fail(errors, strict=strict)
    s = pd.to_numeric(df[col], errors="coerce")
    if s.isna().any():
        errors.append(f"{col} has non-numeric/null values")
    bad = int(((s <= 0) | (s > 1)).sum())
    if bad:
        errors.append(f"{col}: {bad} values not in (0, 1]")
    return _fail(errors, strict=strict)


def validate_frame(frame: pd.DataFrame, base: pd.DataFrame, *, strict: bool = True) -> list[str]:
    errors = validate_pair_keys(frame, name="frame", strict=False)
    if "frame_id" in frame.columns:
        if frame["frame_id"].duplicated().any():
            errors.append("frame_id is not unique")
        if frame["frame_id"].isna().any():
            errors.append("frame_id has nulls")
    base_keys = set(map(tuple, base[PAIR_KEY].itertuples(index=False, name=None)))
    missing = [
        k
        for k in map(tuple, frame[PAIR_KEY].itertuples(index=False, name=None))
        if k not in base_keys
    ]
    if missing:
        errors.append(f"{len(missing)} frame pairs not in base population (e.g. {missing[0]})")
    for col in ("pi_enriched", "pi_prevalence", "incl_prob"):
        if col in frame.columns:
            errors.extend(validate_inclusion_probs(frame.dropna(subset=[col]), col, strict=False))
    return _fail(errors, strict=strict)


def validate_labels(series: pd.Series, *, name: str = "label", strict: bool = True) -> list[str]:
    vals = series.dropna().astype(str).str.strip()
    bad = sorted(set(vals) - VALID_CATEGORIES)
    errors = [f"{name}: invalid categories {bad}"] if bad else []
    return _fail(errors, strict=strict)


def record_versions(row: dict[str, Any], settings: Settings) -> dict[str, Any]:
    out = dict(row)
    out.setdefault("dataset_version", settings.version)
    out.setdefault("config_version", settings.version)
    return out
