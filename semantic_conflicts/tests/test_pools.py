from semantic_conflicts.pools import (
    is_duplicate_title,
    normalize_title,
    parent_dir_prefix,
    prefix_set,
)
from semantic_conflicts.config import NearMissConfig


def test_normalize_title_strips_tags_and_punct():
    assert normalize_title("[Codex] Fix: Login!") == "fix login"


def test_duplicate_jaccard_and_guard():
    assert is_duplicate_title("add user login flow", "add user login form")
    assert not is_duplicate_title("fix", "fix it")  # min token guard
    assert is_duplicate_title("enable any llm to run test functionality", "enable any llm to run test functionality extra")


def test_containment_duplicate():
    a = "implement outbox pattern for products"
    b = "implement outbox pattern for products in the payments service"
    assert is_duplicate_title(a, b)


def test_canonical_near_miss_uses_parent_dir_not_filename():
    cfg = NearMissConfig(max_dir_components=2, exclude_root_files=True, include_filename_as_component=False)
    assert parent_dir_prefix("src/foo.py", include_filename_as_component=False) == "src"
    assert parent_dir_prefix("src/util/foo.py") == "src/util"
    assert parent_dir_prefix("README.md") is None
    a = {"src/auth.py"}
    b = {"src/login.py"}
    assert prefix_set(a, cfg) & prefix_set(b, cfg) == {"src"}
    buggy = NearMissConfig(max_dir_components=2, exclude_root_files=True, include_filename_as_component=True)
    assert not (prefix_set(a, buggy) & prefix_set(b, buggy))


def test_pool_flags_on_tiny(tiny_tables, settings_ci):
    from semantic_conflicts.pools import add_pool_flags

    flags = add_pool_flags(tiny_tables["pairs"], tiny_tables["texts"], tiny_tables["files"], settings_ci, validate=False)
    # pair 1-2: dup + near_miss, not conflict
    r = flags[(flags.pr_a == 1) & (flags.pr_b == 2)].iloc[0]
    assert bool(r.dup_title)
    assert bool(r.near_miss)
    assert not bool(r.conflict)
    # pair 1-4: conflict + fix-in-flight
    r = flags[(flags.pr_a == 1) & (flags.pr_b == 4)].iloc[0]
    assert bool(r.conflict) and bool(r.fix_in_flight)
    # pair 1-3: revert
    r = flags[(flags.pr_a == 1) & (flags.pr_b == 3)].iloc[0]
    assert bool(r.revert) and bool(r.conflict)
    # pair 10-11 both merged shared
    r = flags[(flags.pr_a == 10) & (flags.pr_b == 11)].iloc[0]
    assert bool(r.bm_shared)
    # exclusive strata
    assert flags.primary_stratum.notna().all()
    assert flags.groupby(["repo", "pr_a", "pr_b"]).size().max() == 1
