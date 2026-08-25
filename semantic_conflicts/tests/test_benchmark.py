
from semantic_conflicts.evaluation.benchmark import split_gold
import pandas as pd


def test_repo_disjoint_splits(settings_ci):
    gold = pd.DataFrame(
        {
            "repo": ["a"] * 5 + ["b"] * 5 + ["c"] * 5,
            "pr_a": list(range(15)),
            "pr_b": list(range(100, 115)),
            "label_category": ["D"] * 15,
        }
    )
    settings_ci.benchmark.min_gold_for_train = 10
    splits = split_gold(gold, settings_ci)
    assert set(splits["gold_test"].repo).isdisjoint(set(splits["gold_dev"].repo))
    if len(splits["gold_train"]):
        assert set(splits["gold_train"].repo).isdisjoint(set(splits["gold_test"].repo))


def test_small_gold_skips_train(settings_ci):
    gold = pd.DataFrame(
        {
            "repo": ["a"] * 3 + ["b"] * 3,
            "pr_a": list(range(6)),
            "pr_b": list(range(10, 16)),
            "label_category": ["none"] * 6,
        }
    )
    settings_ci.benchmark.min_gold_for_train = 200
    splits = split_gold(gold, settings_ci)
    assert splits["policy"] == "gold_dev_test_only"
    assert len(splits["gold_train"]) == 0
