#!/usr/bin/env python3
"""01b_enumerate_pairs.py -- tailored to the actual AIDev-pop schema.

Run on your machine, in the same folder as before:

    python3 01b_enumerate_pairs.py --outdir data

Joins performed (matching the schema you printed):
  pull_request.repo_id -> repository.id -> full_name        (the repo name)
  pull_request.id -> pr_commits.pr_id -> sha
      -> pr_commit_details.sha -> filename/additions/deletions

Outputs in data/:
  pairs_coactive.csv.gz   repo, pr_a, pr_b, agents, open/close timestamps
  pr_texts.csv.gz         repo, pr, agent, state, merged, title, body_1200
  pr_files.csv.gz         repo, pr, filepath, additions, deletions
  repo_stats.csv          repo, language, stars, n_agent_prs
Then zip the data folder and upload it to the chat.
"""
import argparse, gzip, os, sys
import pandas as pd

HF = "hf://datasets/hao-li/AIDev/"


def cols_of(path):
    """Read parquet column names WITHOUT loading data (patch columns are
    huge; we must select columns before loading)."""
    import pyarrow.parquet as pq
    import fsspec
    with fsspec.open(path, "rb") as f:
        return pq.ParquetFile(f).schema_arrow.names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    # ---- pull_request + repository -> repo full name -------------------
    pr = pd.read_parquet(HF + "pull_request.parquet",
                         columns=["id", "number", "title", "body", "agent",
                                  "state", "created_at", "closed_at",
                                  "merged_at", "repo_id", "repo_url"])
    rp = pd.read_parquet(HF + "repository.parquet",
                         columns=["id", "full_name", "language", "stars"])
    pr = pr.merge(rp.rename(columns={"id": "repo_id",
                                     "full_name": "repo"}),
                  on="repo_id", how="left")
    # fallback: derive owner/name from repo_url if the join missed
    miss = pr.repo.isna()
    if miss.any():
        pr.loc[miss, "repo"] = (pr.loc[miss, "repo_url"]
                                .str.extract(r"repos/([^/]+/[^/]+)")[0])
    pr = pr.dropna(subset=["repo"])
    print(f"[info] {len(pr)} PRs across {pr.repo.nunique()} repos "
          f"(paper population: 33,596 / 2,807)")

    pr["_open"] = pd.to_datetime(pr.created_at, utc=True, errors="coerce")
    pr["_close"] = pd.to_datetime(pr.closed_at, utc=True, errors="coerce")
    cutoff = max(pr["_open"].max(), pr["_close"].max())
    pr["_close"] = pr["_close"].fillna(cutoff)
    pr = pr.dropna(subset=["_open"])
    print(f"[info] cutoff = {cutoff}")

    # ---- k=0 co-active pair enumeration (sweep line per repo) ----------
    rows = []
    for repo, g in pr.sort_values("_open").groupby("repo"):
        g = g.reset_index(drop=True)
        active = []
        for i, r in g.iterrows():
            active = [(c, j) for c, j in active if c > r["_open"]]
            for _, j in active:
                o = g.loc[j]
                rows.append((repo, int(o.number), int(r.number),
                             str(o.agent), str(r.agent),
                             o._open.isoformat(), o._close.isoformat(),
                             r._open.isoformat(), r._close.isoformat()))
            active.append((r["_close"], i))
    pairs = pd.DataFrame(rows, columns=[
        "repo", "pr_a", "pr_b", "agent_a", "agent_b",
        "opened_a", "closed_a", "opened_b", "closed_b"])
    pairs["cross_agent"] = pairs.agent_a != pairs.agent_b
    print(f"[pairs] {len(pairs):,} co-active pairs "
          f"({pairs.cross_agent.mean():.3%} cross-agent) in "
          f"{pairs.repo.nunique():,} repos "
          f"(paper: 580,913 / 0.50% / 1,129 -- expect close)")
    with gzip.open(f"{a.outdir}/pairs_coactive.csv.gz", "wt") as f:
        pairs.to_csv(f, index=False)

    # ---- PR texts ------------------------------------------------------
    tx = pr[["repo", "number", "agent", "state", "merged_at",
             "title", "body"]].copy()
    tx.columns = ["repo", "pr", "agent", "state", "merged_at",
                  "title", "body"]
    tx["merged"] = tx.merged_at.notna()
    tx["body"] = tx.body.fillna("").str.slice(0, 1200)
    tx = tx.drop(columns=["merged_at"])
    with gzip.open(f"{a.outdir}/pr_texts.csv.gz", "wt") as f:
        tx.to_csv(f, index=False)
    print(f"[texts] {len(tx):,} rows")

    # ---- file changes: pr -> commits -> commit details -----------------
    cm = pd.read_parquet(HF + "pr_commits.parquet",
                         columns=["sha", "pr_id"])
    det_cols = cols_of(HF + "pr_commit_details.parquet")
    print(f"[schema] pr_commit_details columns: {det_cols}")
    fcol = next((c for c in ("filename", "file_path", "path", "file")
                 if c in det_cols), None)
    if fcol is None:
        sys.exit(f"[fatal] no filename-like column in {det_cols}; "
                 f"paste this line into the chat for a fix.")
    keycol = "sha" if "sha" in det_cols else (
        "pr_id" if "pr_id" in det_cols else None)
    want = [c for c in (keycol, fcol, "additions", "deletions", "status")
            if c and c in det_cols]
    det = pd.read_parquet(HF + "pr_commit_details.parquet", columns=want)
    if keycol == "sha":
        det = det.merge(cm, on="sha", how="inner")
    fl = det.merge(pr[["id", "repo", "number"]]
                   .rename(columns={"id": "pr_id", "number": "pr"}),
                   on="pr_id", how="inner")
    agg = {"additions": "sum", "deletions": "sum"}
    agg = {k: v for k, v in agg.items() if k in fl.columns}
    fl = (fl.groupby(["repo", "pr", fcol], as_index=False).agg(agg)
            .rename(columns={fcol: "filepath"}))
    with gzip.open(f"{a.outdir}/pr_files.csv.gz", "wt") as f:
        fl.to_csv(f, index=False)
    print(f"[files] {len(fl):,} (repo, pr, file) rows for "
          f"{fl.pr.nunique():,} PRs")

    # ---- repo stats ----------------------------------------------------
    st = (pr.groupby("repo").size().rename("n_agent_prs").reset_index()
            .merge(rp.rename(columns={"full_name": "repo"})
                     [["repo", "language", "stars"]],
                   on="repo", how="left"))
    st.to_csv(f"{a.outdir}/repo_stats.csv", index=False)

    total = sum(os.path.getsize(f"{a.outdir}/{f}")
                for f in os.listdir(a.outdir)) / 1e6
    print(f"[done] data/ = {total:.1f} MB -> zip the data folder and "
          f"upload it to the chat (skip pairs_to_replay.parquet if present)")


if __name__ == "__main__":
    main()
