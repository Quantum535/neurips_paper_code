"""Deterministic candidate-pool flags.

Canonical near-miss definition (Category I *heuristic baseline*, not a detector)
=============================================================================
For each changed file, take the parent directory path (all components except the
filename) and truncate it to the first ``max_dir_components`` directory
components. Files that live at repo root (no directory) contribute no key.
A pair is a near-miss iff it has no shared changed file and the two PRs' key
sets intersect.

This is the rule recorded in ``common/verification_log.md`` (176,756 pairs on
the frozen 577,045-pair artifact). It is **not** "first two path components
including the filename": that buggy reading maps ``src/foo.py`` to
``src/foo.py``, which cannot match a partner's different file in the same
folder, collapsing the pool to 171,067 and making ``dir2`` identical to
``dirs_only``.

Historical counts live in config ``historical_released`` and in the
reconciliation report; they are never hardcoded into the membership test.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from semantic_conflicts.config import NearMissConfig, Settings, load_settings
from semantic_conflicts.io import load_source_tables, write_csv, write_json
from semantic_conflicts.paths import repo_root, results_dir
from semantic_conflicts.schemas import validate_base_pairs, validate_pool_flags

# ---------------------------------------------------------------------------
# Title / keyword rules
# ---------------------------------------------------------------------------

_BRACKET = re.compile(r"\[[^\]]*\]")
_NONALNUM = re.compile(r"[^a-z0-9\s]")


def normalize_title(title: str) -> str:
    t = (title or "").lower()
    t = _BRACKET.sub(" ", t)
    t = _NONALNUM.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def title_tokens(title: str) -> set[str]:
    n = normalize_title(title)
    return set(n.split()) if n else set()


def is_duplicate_title(
    title_a: str,
    title_b: str,
    *,
    min_tokens: int = 3,
    jaccard_threshold: float = 0.5,
    containment_min_chars: int = 15,
) -> bool:
    na, nb = normalize_title(title_a), normalize_title(title_b)
    a, b = set(na.split()), set(nb.split())
    if len(a) < min_tokens or len(b) < min_tokens:
        return False
    union = a | b
    j = (len(a & b) / len(union)) if union else 0.0
    if j >= jaccard_threshold:
        return True
    if (
        len(na) >= containment_min_chars
        and len(nb) >= containment_min_chars
        and (na in nb or nb in na)
    ):
        return True
    return False


def title_jaccard(title_a: str, title_b: str) -> float:
    a, b = title_tokens(title_a), title_tokens(title_b)
    if not a and not b:
        return 1.0
    union = a | b
    return (len(a & b) / len(union)) if union else 0.0


# ---------------------------------------------------------------------------
# Path / near-miss keys
# ---------------------------------------------------------------------------

def parent_dir_prefix(
    filepath: str,
    *,
    max_dir_components: int = 2,
    exclude_root_files: bool = True,
    include_filename_as_component: bool = False,
) -> str | None:
    """Return the truncated directory prefix used as a near-miss key, or None.

    Canonical (include_filename_as_component=False):
        README.md              -> None (root)
        src/foo.py             -> src
        src/util/foo.py        -> src/util
        a/b/c/d.py             -> a/b

    Buggy (include_filename_as_component=True):
        src/foo.py             -> src/foo.py   # filename treated as a component
        src/util/foo.py        -> src/util
    """
    parts = str(filepath).replace("\\", "/").split("/")
    if include_filename_as_component:
        if len(parts) < 2:
            return None if exclude_root_files else "<root>"
        return "/".join(parts[:max_dir_components])
    if len(parts) <= 1:
        return None if exclude_root_files else "<root>"
    dirs = parts[:-1]
    return "/".join(dirs[:max_dir_components])


def prefix_set(filepaths: Iterable[str], cfg: NearMissConfig) -> set[str]:
    out: set[str] = set()
    for f in filepaths:
        key = parent_dir_prefix(
            f,
            max_dir_components=cfg.max_dir_components,
            exclude_root_files=cfg.exclude_root_files,
            include_filename_as_component=cfg.include_filename_as_component,
        )
        if key:
            out.add(key)
    return out


def near_miss_intersect(files_a: Iterable[str], files_b: Iterable[str], cfg: NearMissConfig) -> bool:
    return bool(prefix_set(files_a, cfg) & prefix_set(files_b, cfg))


# ---------------------------------------------------------------------------
# Exclusive primary stratum (Option A)
# ---------------------------------------------------------------------------

def primary_stratum_row(
    *,
    revert: bool,
    dup_title: bool,
    fix_in_flight: bool,
    bm_shared: bool,
    conflict: bool,
    near_miss: bool,
) -> str:
    """Mutually exclusive, exhaustive membership label.

    Priority is scientific (rare/intent-level first), not sampling convenience.
    Every pair is in exactly one primary stratum.
    """
    if revert:
        return "revert"
    if dup_title:
        return "dup"
    if fix_in_flight:
        return "fix_in_flight"
    if bm_shared:
        return "bm_shared"
    if conflict:
        return "overlap_other"
    if near_miss:
        return "near_miss"
    return "control"


# ---------------------------------------------------------------------------
# File index
# ---------------------------------------------------------------------------

def files_by_pr(files: pd.DataFrame) -> dict[tuple[str, int], set[str]]:
    idx: dict[tuple[str, int], set[str]] = defaultdict(set)
    for repo, pr, fp in zip(files["repo"].tolist(), files["pr"].tolist(), files["filepath"].tolist()):
        idx[(str(repo), int(pr))].add(str(fp))
    return idx


def recompute_conflict(repo: str, pr_a: int, pr_b: int, idx: Mapping[tuple[str, int], set[str]]) -> bool:
    return bool(idx.get((repo, int(pr_a)), set()) & idx.get((repo, int(pr_b)), set()))


# ---------------------------------------------------------------------------
# Main flag builder
# ---------------------------------------------------------------------------

def add_pool_flags(
    pairs: pd.DataFrame,
    texts: pd.DataFrame,
    files: pd.DataFrame,
    settings: Settings,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    if validate:
        validate_base_pairs(pairs, settings, strict=True)
    cfg = settings
    title = {
        (str(r), int(p)): (t if isinstance(t, str) else "")
        for r, p, t in zip(texts["repo"], texts["pr"], texts["title"])
    }
    merged = {
        (str(r), int(p)): bool(m)
        for r, p, m in zip(texts["repo"], texts["pr"], texts["merged"])
    }
    idx = files_by_pr(files)
    revert_re = re.compile(cfg.revert.pattern)
    fix_kw = set(cfg.fix_in_flight.keywords)
    mega = set(cfg.mega_repos)
    nm_cfg = cfg.near_miss
    dup_cfg = cfg.duplicate_title

    ta = [title.get((str(r), int(p)), "") for r, p in zip(pairs["repo"], pairs["pr_a"])]
    tb = [title.get((str(r), int(p)), "") for r, p in zip(pairs["repo"], pairs["pr_b"])]
    out = pairs.copy()
    out["dup_title"] = [
        is_duplicate_title(
            a,
            b,
            min_tokens=dup_cfg.min_tokens,
            jaccard_threshold=dup_cfg.jaccard_threshold,
            containment_min_chars=dup_cfg.containment_min_chars,
        )
        for a, b in zip(ta, tb)
    ]
    out["revert"] = [bool(revert_re.search(a.lower()) or revert_re.search(b.lower())) for a, b in zip(ta, tb)]
    out["fix_in_flight"] = [
        bool(c) and bool(set(normalize_title(b).split()) & fix_kw)
        for c, b in zip(out["conflict"].astype(bool), tb)
    ]
    out["merged_a"] = [merged.get((str(r), int(p)), False) for r, p in zip(out["repo"], out["pr_a"])]
    out["merged_b"] = [merged.get((str(r), int(p)), False) for r, p in zip(out["repo"], out["pr_b"])]
    out["both_merged"] = out["merged_a"] & out["merged_b"]
    out["bm_shared"] = out["both_merged"] & out["conflict"].astype(bool)
    out["mega"] = out["repo"].isin(mega)
    out["near_miss"] = [
        (not bool(c))
        and near_miss_intersect(
            idx.get((str(r), int(a)), set()),
            idx.get((str(r), int(b)), set()),
            nm_cfg,
        )
        for r, a, b, c in zip(out["repo"], out["pr_a"], out["pr_b"], out["conflict"])
    ]
    out["primary_stratum"] = [
        primary_stratum_row(
            revert=bool(rv),
            dup_title=bool(d),
            fix_in_flight=bool(f),
            bm_shared=bool(bm),
            conflict=bool(c),
            near_miss=bool(nm),
        )
        for rv, d, f, bm, c, nm in zip(
            out["revert"],
            out["dup_title"],
            out["fix_in_flight"],
            out["bm_shared"],
            out["conflict"],
            out["near_miss"],
        )
    ]
    out["cross_agent_candidate"] = out["cross_agent"].astype(bool) & (
        out["bm_shared"] | out["dup_title"] | out["revert"]
    )
    if validate:
        validate_pool_flags(out, strict=True)
    return out


def pool_counts(df: pd.DataFrame) -> dict[str, int]:
    return {
        "n_pairs": int(len(df)),
        "conflict": int(df["conflict"].astype(bool).sum()),
        "dup_title": int(df["dup_title"].sum()),
        "revert": int(df["revert"].sum()),
        "fix_in_flight": int(df["fix_in_flight"].sum()),
        "near_miss": int(df["near_miss"].sum()),
        "both_merged": int(df["both_merged"].sum()),
        "bm_shared": int(df["bm_shared"].sum()),
        "cross_agent": int(df["cross_agent"].astype(bool).sum()),
        "cross_agent_candidate": int(df["cross_agent_candidate"].sum()),
        "mega": int(df["mega"].sum()),
        "dup_title_cross_agent": int((df["dup_title"] & df["cross_agent"].astype(bool)).sum()),
        "revert_cross_agent": int((df["revert"] & df["cross_agent"].astype(bool)).sum()),
        "bm_shared_cross_agent": int((df["bm_shared"] & df["cross_agent"].astype(bool)).sum()),
        "near_miss_cross_agent": int((df["near_miss"] & df["cross_agent"].astype(bool)).sum()),
    }


def reconcile_near_miss(
    pairs: pd.DataFrame,
    files: pd.DataFrame,
    settings: Settings,
) -> dict:
    """Explain 176,756 vs 171,067 without baking either number into membership.

    Historical 176,756 is the canonical parent-dir prefix rule.
    171,067 is the buggy 'first two path components including filename' reading
    of the same rule, which is also identical to requiring >=2 directory
    components (dirs_only) on disjoint-file pairs.
    """
    idx = files_by_pr(files)
    canonical = NearMissConfig(
        name="canonical",
        max_dir_components=settings.near_miss.max_dir_components,
        exclude_root_files=True,
        include_filename_as_component=False,
    )
    buggy = NearMissConfig(
        name="buggy_filename_as_component",
        max_dir_components=2,
        exclude_root_files=True,
        include_filename_as_component=True,
    )

    def d2_dirs_only(fs: Iterable[str]) -> set[str]:
        out: set[str] = set()
        for f in fs:
            parts = str(f).replace("\\", "/").split("/")
            if len(parts) >= 3:
                out.add("/".join(parts[:2]))
        return out

    def keys_depth(fs: Iterable[str], depth: int, with_root: bool) -> set[str]:
        out: set[str] = set()
        for f in fs:
            parts = str(f).replace("\\", "/").split("/")
            if len(parts) <= 1:
                if with_root:
                    out.add("<root>")
                continue
            out.add("/".join(parts[:-1][:depth]))
        return out

    n_canonical = n_buggy = n_dirs_only = 0
    n_d1_noroot = n_d2_root = n_d3 = 0
    n_only_canonical = 0
    examples: list[dict] = []
    disjoint = 0
    for r, a, b, c in zip(pairs["repo"], pairs["pr_a"], pairs["pr_b"], pairs["conflict"]):
        if bool(c):
            continue
        disjoint += 1
        fa = idx.get((str(r), int(a)), set())
        fb = idx.get((str(r), int(b)), set())
        can = bool(prefix_set(fa, canonical) & prefix_set(fb, canonical))
        bug = bool(prefix_set(fa, buggy) & prefix_set(fb, buggy))
        donly = bool(d2_dirs_only(fa) & d2_dirs_only(fb))
        n_canonical += int(can)
        n_buggy += int(bug)
        n_dirs_only += int(donly)
        n_d1_noroot += int(bool(keys_depth(fa, 1, False) & keys_depth(fb, 1, False)))
        n_d2_root += int(bool(keys_depth(fa, 2, True) & keys_depth(fb, 2, True)))
        n_d3 += int(bool(keys_depth(fa, 3, False) & keys_depth(fb, 3, False)))
        if can and not bug:
            n_only_canonical += 1
            if len(examples) < 15:
                inter = sorted(prefix_set(fa, canonical) & prefix_set(fb, canonical))
                examples.append(
                    {
                        "repo": str(r),
                        "pr_a": int(a),
                        "pr_b": int(b),
                        "canonical_intersect": inter[:8],
                        "files_a_sample": sorted(fa)[:6],
                        "files_b_sample": sorted(fb)[:6],
                        "why": (
                            "Both PRs change files whose parent-directory prefix matches "
                            "(typically a depth-1 directory such as src/foo.py -> src). "
                            "The buggy rule maps src/foo.py to src/foo.py, so disjoint "
                            "files in the same folder never match."
                        ),
                    }
                )

    hist = settings.historical_released
    report = {
        "canonical_definition": {
            "name": settings.near_miss.name,
            "rule": (
                "parent directory of each changed file, truncated to the first "
                f"{settings.near_miss.max_dir_components} directory components; "
                "filename is not a component; repo-root files excluded"
            ),
            "include_filename_as_component": False,
            "count": n_canonical,
            "share_of_disjoint_file_pairs": (n_canonical / disjoint) if disjoint else None,
        },
        "buggy_path_components_including_filename": {
            "rule": "first two path components including the filename; root files skipped",
            "count": n_buggy,
            "identical_to_dirs_only": n_buggy == n_dirs_only,
            "dirs_only_count": n_dirs_only,
        },
        "cross_checks": {
            "n_disjoint_file_pairs": disjoint,
            "depth1_noroot": n_d1_noroot,
            "depth2_with_root": n_d2_root,
            "depth3_noroot": n_d3,
        },
        "difference": {
            "canonical_minus_buggy": n_canonical - n_buggy,
            "n_only_in_canonical": n_only_canonical,
            "reason": (
                "The buggy reading treats the filename as a path component, so a "
                "file living directly in a directory (src/foo.py) contributes the "
                "key 'src/foo.py' rather than 'src'. Among disjoint-file pairs that "
                "key cannot intersect. Those pairs still match the canonical "
                "parent-directory prefix. Empirically this also equals the "
                "'dirs_only' variant (keys emitted only for files with >=2 directory "
                "components)."
            ),
            "examples": examples,
        },
        "historical_released": {
            "near_miss_count": hist.near_miss if hist else None,
            "buggy_count": hist.buggy_near_miss_path_components_including_filename if hist else None,
            "depth1_noroot": hist.near_miss_depth1_noroot if hist else None,
            "depth2_with_root": hist.near_miss_depth2_with_root if hist else None,
            "source": hist.source if hist else None,
            "note": "Historical numbers are recorded for reconciliation; they are not inputs to membership.",
        },
        "match_historical_canonical": (hist.near_miss == n_canonical) if hist else None,
        "match_historical_buggy": (
            hist.buggy_near_miss_path_components_including_filename == n_buggy
        )
        if hist
        else None,
        "canonical_is_used_for_all_downstream_flags": True,
    }
    return report


def write_pools(
    settings: Settings,
    *,
    root: Path | None = None,
    out_dir: Path | None = None,
    tables: dict | None = None,
) -> dict:
    root = root or repo_root()
    out_dir = out_dir or results_dir(settings.version, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = tables or load_source_tables(root, settings=settings)
    flags = add_pool_flags(tables["pairs"], tables["texts"], tables["files"], settings)
    counts = pool_counts(flags)
    recon = reconcile_near_miss(tables["pairs"], tables["files"], settings)
    write_csv(out_dir / "pool_flags.csv.gz", flags)
    write_json(out_dir / "pool_counts.json", counts)
    write_json(out_dir / "near_miss_reconciliation.json", recon)
    return {"flags": flags, "counts": counts, "reconciliation": recon, "out_dir": out_dir}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Recompute canonical candidate-pool flags")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    settings = load_settings(args.config)
    result = write_pools(settings, out_dir=args.out)
    counts = result["counts"]
    recon = result["reconciliation"]
    print("pool                     n")
    for k in ("dup_title", "revert", "fix_in_flight", "near_miss", "bm_shared"):
        print(f"{k:22s} {counts[k]:>8}")
    print(
        "near-miss canonical",
        recon["canonical_definition"]["count"],
        "buggy filename-as-component",
        recon["buggy_path_components_including_filename"]["count"],
        "delta",
        recon["difference"]["canonical_minus_buggy"],
    )


if __name__ == "__main__":
    main()
