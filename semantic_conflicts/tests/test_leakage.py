import pytest

from semantic_conflicts.evaluation.baselines import hybrid_features
from semantic_conflicts.leakage import FORBIDDEN_FEATURE_COLUMNS, assert_no_leakage


def test_hybrid_features_reject_outcome_columns(tiny_tables, settings_ci):
    from semantic_conflicts.pools import add_pool_flags, files_by_pr

    flags = add_pool_flags(tiny_tables["pairs"], tiny_tables["texts"], tiny_tables["files"], settings_ci, validate=False)
    idx = files_by_pr(tiny_tables["files"])
    title = {(r, p): t for r, p, t in zip(tiny_tables["texts"].repo, tiny_tables["texts"].pr, tiny_tables["texts"].title)}
    body = {(r, p): b for r, p, b in zip(tiny_tables["texts"].repo, tiny_tables["texts"].pr, tiny_tables["texts"].body)}
    ta = [title.get((r, a), "") for r, a in zip(flags.repo, flags.pr_a)]
    tb = [title.get((r, b), "") for r, b in zip(flags.repo, flags.pr_b)]
    ba = [body.get((r, a), "") for r, a in zip(flags.repo, flags.pr_a)]
    bb = [body.get((r, b), "") for r, b in zip(flags.repo, flags.pr_b)]
    feat = hybrid_features(flags, idx, ta, tb, ba, bb)
    assert_no_leakage(list(feat.columns))
    assert not (set(feat.columns) & FORBIDDEN_FEATURE_COLUMNS)


def test_assert_no_leakage_raises():
    with pytest.raises(ValueError):
        assert_no_leakage(["title_jaccard", "merged_b"])
    assert_no_leakage(["title_jaccard", "merged_b"], retrospective=True)
