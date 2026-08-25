# Code and data for two NeurIPS 2026 workshop submissions

Both papers analyze the same corpus: 577,045 co-active agent-PR pairs mined from AIDev-pop
(`derived/pairs_labeled.csv.gz`). Everything corpus-level lives in `data/`, `derived/`, `common/`;
everything paper-specific lives in that paper's folder.

| Folder | Paper | Submitted PDF name | Internal plan name |
|---|---|---|---|
| `mergegym/` | **MergeGym: Predicting, Resolving, and Scheduling Around Conflicts Between Concurrent AI Coding Agent Pull Requests** (benchmark: tracks T1/T2/T3) | `neurips_paper_1.pdf` | "paper 2" in `paper_plans/` |
| `semantic_conflicts/` | **Beyond File Overlap: Semantic Conflicts Between Concurrent AI Coding Agent Pull Requests** (measurement: pools, outcome coupling, judge pipeline) | `neurips_paper_2.pdf` | "paper 1" in `paper_plans/` |

(The two labelings disagree; the folder names are unambiguous. Use them.)

## Layout

```
data/                      AIDev-pop extraction (pr_files, pr_texts, pairs_coactive, repo_stats)  [redistributed under AIDev's license]
derived/                   shared derived tables: pairs_labeled (577,045 labeled pairs), pairs_temporal, conflict_pairs_enriched
common/                    01b_enumerate_pairs.py (co-active pair enumeration), regen_figs.py, verification_log.md (55-claim audit)

mergegym/                  ---- MergeGym (neurips_paper_1.pdf) ----
  scripts/
    04b_llm_t1_claude_code.py   T1 zero-shot LLM baseline (Claude Sonnet via Claude Code headless)
    step1_join.py, step2_baselines.py, step3_ensemble.py   T1 feature join, baseline ladder, fused/OOF rows
    figs_extra_p2.py            recomputes T1 ROC / within-repo / calibration / T2 composition / T3 episode figures
    t2_replay.py                T2: fetch refs/pull/N/head into partial clones, git merge-tree replay (all 6,387 candidates)
    t2_groundtruth.py           T2: conflict hunks, repository reconciliation, trivial resolvers (ours/theirs/union)
    07_t2_llm_resolver_claude_code.py   T2: zero-shot LLM resolver pilot over the hunk bundle
    t3_simulator.py             T3: replay simulator (observed / FIFO / oracle / score-gated); `--check` reproduces
                                t3_episode_results.csv exactly for observed/FIFO/oracle on all 97 episodes
  results/
    test_pairs_for_llm.csv.gz, llm_scores.csv, t1_leaderboard.csv, t1_score_vectors.csv.gz, t1_train_manifest.csv.gz, t1_manifest_summary.json, per_repo_auc.csv
    t2_candidates.csv.gz, t2_replay_results.csv, t2_yield_by_language*.csv, t2_summary*.json, t2_hunks.csv.gz, t2_hunk_bundle.jsonl.gz
    t3_top50.csv, t3_episode_results.csv, t3_summary.json, t3_assumptions.md
  paper/                    main.tex, references.bib, figures

semantic_conflicts/        ---- Beyond File Overlap (neurips_paper_2.pdf) ----
  scripts/
    anat1.py, anat2.py, anat3.py   conflict anatomy, file-class composition, concurrency/duration dose-response, outcomes + MH odds ratios
    pools.py                  the five deterministic candidate pools (exact definitions; reproduces all printed counts)
    build_judging_frame.py    §4.3 judging frame (8,949 pairs, inclusion probabilities) + 100-pair calibration sheet
    05_judge_p2_claude_code.py   rubric-guided LLM judge (Claude Code headless), demotion rule applied
    06_score_judge.py         Cohen's kappa, per-pool precision (Wilson CIs), IPW prevalence  -> judge_report.json
    figs_extra_p1.py          agent matrix, forest plot, dose-response, hot files, concentration, duplicate close-gap figures
  results/
    pool_flags.csv.gz (all 577,045 pairs x pool flags), judging_frame.csv.gz, calibration_100.csv, frame_summary.json
    cross_agent_cases.csv (all 99), dup_detail.json, p1_outcomes.json, p1_headline_cis.json, mh_robustness.json
  paper/                    main.tex, references.bib, figures
```

## Running

All scripts resolve paths from the repository root (override with `MG_ROOT=/path/to/repo`). Python 3.10+, `pandas numpy`;
`matplotlib` for figures; `git >= 2.38` for T2 replay; Claude Code logged in for the three `*_claude_code.py` scripts
(`unset ANTHROPIC_API_KEY` so they use the subscription login).

```
python3 semantic_conflicts/scripts/pools.py                 # prints each pool size next to the paper's number
python3 semantic_conflicts/scripts/build_judging_frame.py   # rebuilds the frame (seed 20260824)
python3 semantic_conflicts/scripts/05_judge_p2_claude_code.py --only calibration --model sonnet
python3 semantic_conflicts/scripts/06_score_judge.py

python3 mergegym/scripts/t3_simulator.py --check            # 97 episodes; observed/FIFO/oracle regression vs released results
python3 mergegym/scripts/t2_replay.py --subset core         # clones ~260 repos (blob-less) into mergegym/t2_work/
python3 mergegym/scripts/t2_groundtruth.py --core
python3 mergegym/scripts/07_t2_llm_resolver_claude_code.py --model sonnet
```

## What is not here
- The T3 *predictor-gate* training script (TF-IDF cosine + title Jaccard + same-agent + trailing repo conflict rate, IRLS
  logistic) was not in the working directory this repo was assembled from; `t3_simulator.py` implements the gate mechanics
  with a pluggable `score_fn` and the fitted weights are recorded in `results/t3_summary.json`
  (`predictor_weights_intercept_cos_jac_same_trail`). The predictor-gated rows of `t3_episode_results.csv` are the released run.
- AIDev's per-commit patch table (not redistributed); the judge therefore runs on title + body + file lists in v1.
- `mergegym/t2_work/` (partial clones, ~4 GB) is regenerated by `t2_replay.py`.

## Status of the two papers' pending computations
- MergeGym T2: replay and trivial-resolver floors are done (`mergegym/results/t2_*`); the LLM resolver pilot row is produced by script 07.
- Beyond File Overlap: frame built; judge run, calibration kappa, and prevalence are produced by scripts 05/06 plus two annotators on `calibration_100.csv`.
