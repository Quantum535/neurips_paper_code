"""Fetch public GitHub PR metadata and file patches into a local cache.

Prefer authenticated ``gh`` CLI. Fall back to REST with GITHUB_TOKEN if set.
Never print or write the token. Cache is gitignored; coverage/hash manifests
are written to the results directory.

Distinguishes:
  - unavailable (private/deleted/rate-limit/network)
  - empty_diff (endpoint succeeded, no files)
  - truncated (GitHub patch truncation)
  - ok
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from semantic_conflicts.config import load_settings
from semantic_conflicts.io import sha256_file, write_json
from semantic_conflicts.paths import cache_dir, results_dir


def _pr_cache_dir(cache: Path, repo: str, pr: int) -> Path:
    safe = repo.replace("/", "__")
    return cache / "github" / safe / str(int(pr))


def _gh_api(path: str) -> tuple[int, str]:
    if shutil.which("gh") is None:
        return 127, "gh not found"
    p = subprocess.run(["gh", "api", path], capture_output=True, text=True)
    return p.returncode, p.stdout if p.returncode == 0 else p.stderr


def _rest(url: str, token: str | None) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "semantic-conflicts"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, body
    except urllib.error.URLError as e:
        return 0, str(e)


def fetch_one(repo: str, pr: int, dest: Path, *, sleep_s: float = 0.2) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    meta_p = dest / "pr.json"
    files_p = dest / "files.json"
    status = {
        "repo": repo,
        "pr": int(pr),
        "status": "ok",
        "unavailable_reason": None,
        "n_files": None,
        "truncated": False,
        "empty_diff": False,
        "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if (dest / "DONE").exists() and meta_p.exists() and files_p.exists():
        status["status"] = "cached"
        try:
            files = json.loads(files_p.read_text(encoding="utf-8"))
            status["n_files"] = len(files) if isinstance(files, list) else None
            status["empty_diff"] = isinstance(files, list) and len(files) == 0
        except json.JSONDecodeError:
            status["status"] = "unavailable"
            status["unavailable_reason"] = "corrupt_cache"
        return status

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    rc, raw_pr = _gh_api(f"repos/{repo}/pulls/{pr}")
    if rc != 0:
        code, raw_pr = _rest(f"https://api.github.com/repos/{repo}/pulls/{pr}", token)
        if code != 200:
            status["status"] = "unavailable"
            status["unavailable_reason"] = f"pr_http_{code}:{raw_pr[:200]}"
            (dest / "error.txt").write_text(raw_pr[:2000], encoding="utf-8")
            return status
    try:
        prj = json.loads(raw_pr)
    except json.JSONDecodeError:
        status["status"] = "unavailable"
        status["unavailable_reason"] = "pr_json"
        return status
    meta_p.write_text(json.dumps(prj, indent=2)[:200_000], encoding="utf-8")

    rc, raw_files = _gh_api(f"repos/{repo}/pulls/{pr}/files?per_page=100")
    if rc != 0:
        code, raw_files = _rest(f"https://api.github.com/repos/{repo}/pulls/{pr}/files?per_page=100", token)
        if code != 200:
            status["status"] = "unavailable"
            status["unavailable_reason"] = f"files_http_{code}"
            return status
    try:
        files = json.loads(raw_files)
    except json.JSONDecodeError:
        status["status"] = "unavailable"
        status["unavailable_reason"] = "files_json"
        return status
    if not isinstance(files, list):
        files = []
    files_p.write_text(json.dumps(files)[:4_000_000], encoding="utf-8")
    status["n_files"] = len(files)
    status["empty_diff"] = len(files) == 0
    status["truncated"] = any(bool(f.get("patch") is None and f.get("changes", 0) > 0) for f in files if isinstance(f, dict))
    sha_p = dest / "HASHES.json"
    hashes = {}
    for name, path in (("pr.json", meta_p), ("files.json", files_p)):
        if path.exists():
            hashes[name] = sha256_file(path)
    sha_p.write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    (dest / "DONE").write_text("ok\n", encoding="utf-8")
    time.sleep(sleep_s)
    return status


def fetch_pr_evidence(
    manifest: Path,
    cache: Path,
    *,
    settings=None,
    max_prs: int | None = None,
    coverage_out: Path | None = None,
) -> dict:
    df = pd.read_csv(manifest)
    pairs = []
    for _, r in df.iterrows():
        pairs.append((str(r["repo"]), int(r["pr_a"])))
        pairs.append((str(r["repo"]), int(r["pr_b"])))
    uniq = sorted(set(pairs))
    if max_prs is not None:
        uniq = uniq[: int(max_prs)]
    rows = []
    for i, (repo, pr) in enumerate(uniq, 1):
        dest = _pr_cache_dir(cache, repo, pr)
        st = fetch_one(repo, pr, dest)
        rows.append(st)
        if i % 25 == 0:
            print(f"[github] {i}/{len(uniq)}")
    cov = {
        "n_unique_prs": len(uniq),
        "n_ok": sum(1 for x in rows if x["status"] in {"ok", "cached"}),
        "n_unavailable": sum(1 for x in rows if x["status"] == "unavailable"),
        "n_empty_diff": sum(1 for x in rows if x.get("empty_diff")),
        "n_truncated": sum(1 for x in rows if x.get("truncated")),
        "by_status": {},
        "note": "Raw patches stay in the gitignored cache. This file is coverage only.",
        "rows": rows,
    }
    for x in rows:
        cov["by_status"][x["status"]] = cov["by_status"].get(x["status"], 0) + 1
    if coverage_out:
        write_json(coverage_out, cov)
    return cov


def coverage_by_slice(frame: pd.DataFrame, status_rows: list[dict]) -> dict:
    st = {(x["repo"], x["pr"]): x["status"] for x in status_rows}
    out = {"overall": {}}
    def both_ok(r):
        return st.get((r.repo, int(r.pr_a))) in {"ok", "cached"} and st.get((r.repo, int(r.pr_b))) in {"ok", "cached"}
    out["overall"]["n"] = int(len(frame))
    out["overall"]["both_ok"] = int(sum(1 for r in frame.itertuples() if both_ok(r)))
    for col in ("primary_stratum", "cross_agent", "mega"):
        if col not in frame.columns:
            continue
        by = {}
        for key, sub in frame.groupby(col):
            by[str(key)] = {
                "n": int(len(sub)),
                "both_ok": int(sum(1 for r in sub.itertuples() if both_ok(r))),
            }
        out[col] = by
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="semantic_conflicts.github.fetch_pr_evidence")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--max-prs", type=int, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    settings = load_settings(args.config)
    cache = args.cache or cache_dir()
    out_dir = args.out or results_dir(settings.version)
    cov = fetch_pr_evidence(
        args.manifest,
        cache,
        settings=settings,
        max_prs=args.max_prs,
        coverage_out=out_dir / "evidence_coverage.json",
    )
    print(json.dumps({k: cov[k] for k in cov if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
