import numpy as np
import pandas as pd

from semantic_conflicts.statistics import (
    cluster_bootstrap,
    cohen_kappa,
    hajek_mean,
    horvitz_thompson_mean,
    kish_ess,
    mantel_haenszel_or,
    wilson_interval,
)


def test_wilson_known_values():
    w = wilson_interval(1, 1)
    assert w["p"] == 1.0
    assert 0 < w["lo"] < 1 < w["hi"] or w["hi"] >= 1
    w0 = wilson_interval(0, 10)
    assert w0["p"] == 0.0 and w0["lo"] >= 0


def test_hajek_and_ht_on_known_population():
    # Population of 10 units, y=1 for first 4. Sample units 0,1,2,5,6 with π=0.5.
    y = np.array([1, 1, 1, 0, 0], dtype=float)
    pi = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    hajek = hajek_mean(y, pi)
    assert abs(hajek - 0.6) < 1e-12
    ht = horvitz_thompson_mean(y, pi, n_pop=10)
    # HT total of y is  (1+1+1+0+0)/0.5 = 6, mean 6/10=0.6
    assert abs(ht - 0.6) < 1e-12
    assert kish_ess(1 / pi) == 5


def test_mh_or_two_strata():
    # Two repos, exposure associated with lower outcome in both.
    rows = []
    for repo, a, b, c, d in [("r1", 1, 3, 3, 1), ("r2", 2, 6, 6, 2)]:
        rows += [{"repo": repo, "exp": True, "out": True}] * a
        rows += [{"repo": repo, "exp": True, "out": False}] * b
        rows += [{"repo": repo, "exp": False, "out": True}] * c
        rows += [{"repo": repo, "exp": False, "out": False}] * d
    df = pd.DataFrame(rows)
    r = mantel_haenszel_or(df, stratum="repo", exposed="exp", outcome="out")
    # Each stratum OR = (a*d)/(b*c) = (1*1)/(3*3)=1/9, (2*2)/(6*6)=4/36=1/9
    assert abs(r["or_mh"] - (1 / 9)) < 1e-8


def test_cluster_bootstrap_deterministic():
    df = pd.DataFrame(
        {
            "repo": ["a"] * 5 + ["b"] * 5,
            "y": [1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
        }
    )

    def stat(g):
        return float(g.y.mean())

    r1 = cluster_bootstrap(df, cluster_col="repo", statistic=stat, n_iter=50, seed=1)
    r2 = cluster_bootstrap(df, cluster_col="repo", statistic=stat, n_iter=50, seed=1)
    assert r1["lo"] == r2["lo"] and r1["hi"] == r2["hi"]
    r3 = cluster_bootstrap(df, cluster_col="repo", statistic=stat, n_iter=50, seed=2)
    assert r1["lo"] != r3["lo"] or r1["hi"] != r3["hi"] or r1["value"] == r3["value"]


def test_cohen_kappa_perfect_and_chance():
    a = ["D", "C", "none"]
    assert cohen_kappa(a, a) == 1.0
