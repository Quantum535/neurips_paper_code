"""Leakage firewall: prediction features must not include future/outcome/label columns."""

from __future__ import annotations

FORBIDDEN_FEATURE_COLUMNS = frozenset(
    {
        "merged",
        "merged_a",
        "merged_b",
        "both_merged",
        "merged_later",
        "closed_a",
        "closed_b",
        "state_a",
        "state_b",
        "gold",
        "gold_label",
        "label_category",
        "judge_category",
        "category",
        "provisional_category",
        "silver_label",
        "llm_category",
    }
)

ALLOWED_RETROSPECTIVE = frozenset(FORBIDDEN_FEATURE_COLUMNS)


def assert_no_leakage(columns: list[str], *, retrospective: bool = False) -> None:
    cols = set(columns)
    if retrospective:
        return
    leak = sorted(cols & FORBIDDEN_FEATURE_COLUMNS)
    if leak:
        raise ValueError(f"prediction features leak outcome/label columns: {leak}")
