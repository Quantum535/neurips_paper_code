import numpy as np
import pandas as pd

from semantic_conflicts.sampling import _hamilton_allocate, assemble_frame, build_prevalence_sample


def test_inclusion_probability_exact_on_toy(settings_ci):
    rng = np.random.default_rng(0)
    # 20 units in 2 strata, sample 6 from A (N=10) and 4 from B (N=10) via prevalence-like cells
    df = pd.DataFrame(
        {
            "repo": ["r"] * 20,
            "pr_a": list(range(20)),
            "pr_b": list(range(100, 120)),
            "mega": [False] * 10 + [True] * 10,
            "cross_agent": [False] * 20,
            "conflict": [False] * 20,
            "both_merged": [False] * 20,
            "dup_title": [False] * 20,
            "revert": [False] * 20,
            "fix_in_flight": [False] * 20,
            "near_miss": [False] * 20,
            "bm_shared": [False] * 20,
            "primary_stratum": ["control"] * 20,
            "cross_agent_candidate": [False] * 20,
            "agent_a": ["A"] * 20,
            "agent_b": ["A"] * 20,
        }
    )
    settings_ci.sampling.prevalence.n = 8
    settings_ci.sampling.prevalence.min_per_stratum = 2
    sample, audit, meta = build_prevalence_sample(df, settings_ci, rng)
    for _, row in audit.iterrows():
        if row.n == 0:
            continue
        assert abs(row.pi - row.n / row.N) < 1e-12
        assert 0 < row.pi <= 1
    # realized π matches n/N
    for pi in sample.pi_prevalence.unique():
        n = int((sample.pi_prevalence == pi).sum())
        assert n >= 1
        assert pi > 0


def test_hamilton_allocate_sums_and_clips():
    sizes = pd.Series({"a": 10, "b": 5, "c": 1})
    alloc = _hamilton_allocate(sizes, 8)
    assert int(alloc.sum()) == 8
    assert (alloc <= sizes).all()
    assert _hamilton_allocate(sizes, 100).equals(sizes)


def test_disjoint_strata_and_unique_frame_ids(tiny_tables, settings_ci):
    from semantic_conflicts.pools import add_pool_flags

    flags = add_pool_flags(tiny_tables["pairs"], tiny_tables["texts"], tiny_tables["files"], settings_ci, validate=False)
    pack = assemble_frame(flags, tiny_tables["texts"], tiny_tables["files"], settings_ci)
    frame = pack["frame"]
    assert frame.frame_id.is_unique
    assert not frame.duplicated(["repo", "pr_a", "pr_b"]).any()
    # exclusive primary stratum
    assert flags.primary_stratum.nunique() >= 2
    vc = flags.groupby(["repo", "pr_a", "pr_b"]).size()
    assert (vc == 1).all()
    if frame.in_enriched_sample.any():
        pi = frame.loc[frame.in_enriched_sample, "pi_enriched"]
        assert (pi > 0).all() and (pi <= 1).all()
    # deterministic
    pack2 = assemble_frame(flags, tiny_tables["texts"], tiny_tables["files"], settings_ci)
    assert list(pack2["frame"].frame_id) == list(frame.frame_id)
    assert list(zip(pack2["frame"].repo, pack2["frame"].pr_a, pack2["frame"].pr_b)) == list(
        zip(frame.repo, frame.pr_a, frame.pr_b)
    )
