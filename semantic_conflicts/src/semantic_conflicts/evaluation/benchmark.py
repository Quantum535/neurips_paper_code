"""Repo-disjoint gold/silver benchmark manifests. Silver never enters gold test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from semantic_conflicts.config import Settings
from semantic_conflicts.io import write_csv, write_json
from semantic_conflicts.schemas import InvariantError


def _no_pr_leakage(train: pd.DataFrame, other: pd.DataFrame) -> None:
    def prs(df):
        s = set(zip(df.repo.astype(str), df.pr_a.astype(int)))
        s |= set(zip(df.repo.astype(str), df.pr_b.astype(int)))
        return s

    leak = prs(train) & prs(other)
    if leak:
        raise InvariantError(f"PR leakage across splits: {list(leak)[:5]}")


def _no_reversal(train: pd.DataFrame, other: pd.DataFrame) -> None:
    def keys(df):
        return set(zip(df.repo.astype(str), df.pr_a.astype(int), df.pr_b.astype(int))) | set(
            zip(df.repo.astype(str), df.pr_b.astype(int), df.pr_a.astype(int))
        )

    if keys(train) & keys(other):
        raise InvariantError("pair-reversal leakage across splits")


def split_gold(
    gold: pd.DataFrame,
    settings: Settings,
    *,
    min_train: int | None = None,
) -> dict[str, pd.DataFrame]:
    min_train = settings.benchmark.min_gold_for_train if min_train is None else min_train
    rng = np.random.default_rng(settings.benchmark.split_seed)
    repos = rng.permutation(np.unique(gold["repo"].astype(str).to_numpy()))
    n = len(gold)
    if n < min_train:
        # Gold is too small for a meaningful train split.
        n_test = max(1, int(round(n * settings.benchmark.gold_test_frac)))
        n_test = min(n_test, n)
        # Split repos, not rows, still.
        test_repos, dev_repos = set(), set()
        # Assign repos greedily to test then dev
        counts = gold.groupby("repo").size().to_dict()
        order = list(repos)
        test_n = 0
        for r in order:
            if test_n < n_test:
                test_repos.add(r)
                test_n += counts.get(r, 0)
            else:
                dev_repos.add(r)
        test = gold[gold.repo.isin(test_repos)].copy()
        dev = gold[gold.repo.isin(dev_repos)].copy()
        train = gold.iloc[0:0].copy()
        policy = "gold_dev_test_only"
    else:
        n_test = int(round(n * settings.benchmark.gold_test_frac))
        n_dev = int(round(n * settings.benchmark.gold_dev_frac))
        test_repos, dev_repos, train_repos = set(), set(), set()
        counts = gold.groupby("repo").size().to_dict()
        test_n = dev_n = 0
        for r in repos:
            c = counts[r]
            if test_n < n_test:
                test_repos.add(r)
                test_n += c
            elif dev_n < n_dev:
                dev_repos.add(r)
                dev_n += c
            else:
                train_repos.add(r)
        test = gold[gold.repo.isin(test_repos)].copy()
        dev = gold[gold.repo.isin(dev_repos)].copy()
        train = gold[gold.repo.isin(train_repos)].copy()
        policy = "repo_disjoint_train_dev_test"
        _no_pr_leakage(train, test)
        _no_pr_leakage(train, dev)
        _no_reversal(train, test)

    leak_td = set(test.repo) & set(dev.repo)
    if leak_td:
        raise InvariantError(f"repo leakage test∩dev: {leak_td}")
    if len(train) and (set(train.repo) & set(test.repo)):
        raise InvariantError("repo leakage train∩test")
    return {"gold_train": train, "gold_dev": dev, "gold_test": test, "policy": policy}


def write_benchmark(out_dir: Path, splits: dict[str, pd.DataFrame], *, silver: pd.DataFrame | None, settings: Settings) -> None:
    bdir = out_dir / "benchmark"
    bdir.mkdir(parents=True, exist_ok=True)
    support = {}
    for name in ("gold_train", "gold_dev", "gold_test"):
        df = splits[name]
        write_csv(bdir / f"{name}.csv", df)
        if "label_category" in df.columns:
            support[name] = df["label_category"].value_counts().astype(int).to_dict()
        else:
            support[name] = {"n": int(len(df))}
    if silver is not None and len(silver):
        # Silver may train; never write into gold_test
        write_csv(bdir / "silver_train.csv", silver)
        if len(splits["gold_test"]) and "repo" in silver.columns:
            overlap = set(silver.repo) & set(splits["gold_test"].repo)
            support["silver_train_repos_in_gold_test"] = sorted(overlap)
    write_json(
        bdir / "split_meta.json",
        {
            "policy": splits.get("policy"),
            "seed": settings.benchmark.split_seed,
            "support": support,
            "label_source_gold": "human-adjudicated",
            "label_source_silver": "llm-never-gold",
            "dataset_version": settings.version,
        },
    )
