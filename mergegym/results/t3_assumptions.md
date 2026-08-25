# T3 scheduling replay — implementation assumptions and audit trail

All numbers computed 2026-08-16 from `derived/pairs_labeled.csv.gz` (577,045 pairs),
`data/pr_files.csv.gz`, and `data/pr_texts.csv.gz`. Tie-corrected Mann-Whitney rank AUC
used everywhere. Outputs: `t3_episode_results.csv` (episode x policy x tau),
`t3_summary.json`, `fig_t3_pareto.pdf`, `fig_t3_pareto_nomega.pdf`.

## Episodes

- One episode per repo with **>=100 co-active pairs AND >=20 conflicting pairs** in
  `pairs_labeled`: **97 repos** (matches the audited count), covering **563,658 pairs**,
  **38,526 conflicting pairs**, and **19,397 distinct PRs** (all three match the audited
  reference numbers).
- Episode PRs = all distinct PRs of the repo appearing in `pairs_labeled` (as `pr_a` or
  `pr_b`); arrival = `opened`, observed close = `closed`, duration = `closed - opened`.
  Every (repo, pr) had a unique, internally consistent (opened, closed); zero PRs had
  zero or negative duration; all 19,397 PRs have file rows in `pr_files` and texts in
  `pr_texts`.

## Right-censoring

- Global max close timestamp (the censoring fill value) = **2025-07-30 23:20:55+00:00**.
  PRs with `closed` equal to it are treated as right-censored: **778 PRs** (of 19,397).
- Censored durations were capped at the repo's **median duration over non-censored
  PRs** (cap applied as `min(duration, repo median)`); the global median over
  non-censored episode PRs (0.0397 h) was the defined fallback for repos with no
  non-censored PR, but **no repo needed it**. **759 of the 778** censored PRs were
  actually shortened (the other 19 opened so late that their filled duration was
  already below the repo median).
- **Capped durations are used consistently in ALL policies, including the `observed`
  baseline**, so policy comparisons are apples-to-apples. Consequence: the capped
  observed baseline materializes **18,071** conflicting overlapping pairs vs the 38,526
  labeled ones (32/97 repos match exactly; censored-heavy repos lose conflicts —
  extreme case `MontrealAI/AGI-Alpha-Agent-v0`: 6 capped vs 13,644 labeled; its 54
  censored PRs pseudo-overlapped nearly everything under the fill value). All
  "% conflicts avoided" numbers are relative to the capped observed baseline.
- **Machinery verification (uncapped)**: with uncapped (as-filled) durations, the
  materialized conflicting-overlap count equals the repo's labeled conflicting-pair
  count in **97/97 repos** (totals 38,526 = 38,526), confirming both the overlap
  convention and the conflict oracle reproduce the labeled data exactly.

## Conflict oracle and overlap convention

- conflict(A,B) = changed-file sets (from `pr_files`, exact filepath equality)
  intersect. Per-repo boolean conflict matrix precomputed over ALL PR pairs via an
  inverted file->PR index, so pairs newly co-scheduled by a policy (absent from
  `pairs_labeled`) are also labeled.
- Two scheduled PRs "overlap" iff their half-open intervals `[start, start+dur)`
  intersect with positive length: `max(start_i, start_j) < min(finish_i, finish_j)`.
  This exactly reproduces the labeled co-activity definition (see verification above).
- Ties: finish events are processed before arrival events at identical timestamps
  (a PR finishing at t does not block, and does not overlap, one starting at t).

## Policies (event-driven replay; a PR keeps its duration; delaying shifts its whole interval)

1. **observed** — start = arrival (capped durations; defines the baseline; 0 added
   delay, 0% makespan inflation by construction).
2. **fifo-serialize** — concurrency 1; PRs start in arrival order at
   `max(arrival, previous finish)`.
3. **oracle-gated** — a PR starts only if NO currently-running PR truly conflicts with
   it (file-set oracle); otherwise it joins a FIFO queue re-checked whenever any PR
   finishes. **Skip-ahead is allowed**: an arriving or queued PR may start ahead of
   earlier blocked PRs if it passes the gate (queue is scanned in FIFO order; PRs
   started during a scan immediately count as running for later PRs in the same scan).
   The gate is checked only against RUNNING PRs, never queued ones; because the
   running set is therefore always mutually conflict-free, oracle materializes 0
   conflicts — including newly created overlaps — netting to exactly 100% avoided.
4. **predictor-gated(tau)** — same mechanics, gate = predicted conflict probability
   > tau against every running PR. tau swept over {0.10, 0.15, 0.20, 0.25, 0.30,
   0.40, 0.50, 0.65, 0.80} (9 values spanning the 1st–99th percentile of training-pool
   predicted probabilities, 0.101–0.801).

## Predictor (trained ONCE, on non-episode repos only)

- Training pool = all `pairs_labeled` rows in repos NOT among the 97 episode repos:
  **13,387 pairs, 30.43% conflicting** (matches the audited 13,387 / 30.4%).
- Features (all computable at decision time; NO overlap-duration features):
  1. **TF-IDF cosine** of title+body texts: tokens `[a-z0-9_]+` lowercased; log-tf
     `(1+ln c)` x smoothed idf `ln((1+N)/(1+df))+1`; **IDF fit only on the 6,911
     distinct training-pool PR documents**; unseen tokens get `ln((1+N)/1)+1`;
     L2-normalized vectors.
  2. **Title token-set Jaccard** (same tokenizer).
  3. **Same-agent indicator** (agent from `pr_texts`).
  4. **Trailing repo conflict rate**: fraction conflicting among the repo's
     `pairs_labeled` pairs whose BOTH PRs' **observed** closes precede the decision
     time, smoothed toward the training-pool global rate g=0.3043 with pseudocount 10:
     `(k + 10g)/(n + 10)`. Pairs containing a censored PR are excluded from this
     record (they never truly closed). Decision time in training = `opened_b`; in
     simulation = the current event time (arrival or queue re-check), looked up
     against the repo's observed historical record (not the simulated schedule).
- Model: logistic regression via IRLS/Newton, L2 = 1e-4 on non-intercept weights,
  features standardized by training mean/sd (mu = [0.160, 0.088, 0.956, 0.292],
  sd = [0.145, 0.151, 0.204, 0.120]).
- **Held-out fifth AUC = 0.6615** (n = 2,677, 29.4% positive; RandomState(0)
  permutation split; single-feature held-out AUCs: trailing rate 0.621, title Jaccard
  0.565, TF-IDF cosine 0.558, same-agent 0.510). Final model refit on all 13,387
  pairs (in-sample AUC 0.668); standardized weights [intercept, cos, jac, same,
  trail] = [-0.889, 0.232, 0.213, 0.004, 0.546].
- Gate implementation detail: prob > tau is evaluated as
  `max_static_logit(candidate, running) + w_trail * z(trail(t)) > logit(tau)` —
  mathematically identical to thresholding the sigmoid; static pairwise logit parts
  are precomputed per repo in float32 (worst-case rounding ~1e-3 in logit space,
  negligible vs the tau grid).

## Metrics (per episode x policy)

- `conflicts` = materialized conflicting overlapping pairs under the policy schedule.
- `avoided_pct` = 100 x (observed_capped - policy)/observed_capped — this **nets**
  newly created conflicts.
- `new_conflicts` = conflicting pairs overlapping under the policy but NOT overlapping
  in the (capped) observed schedule — reported separately.
- `added_delay_days` = sum over PRs of (start - arrival), in PR-days; also
  `delay_days_per_pr` = that / n PRs.
- `makespan_inflation_pct` vs observed, makespan = max finish - min arrival.

## Aggregation

- **Median and IQR (q25, q75) across the 97 episodes** for every metric.
- **"Pair-count-weighted totals"** = pooled sums across episodes, with percentages
  recomputed from the sums (avoided% from summed conflict counts; makespan inflation
  from summed makespans; delay/PR from summed delay over summed PRs) — i.e., every
  conflict pair / PR / day counts once regardless of repo.
- **Sensitivity row**: same aggregates excluding `mochilang/mochi` and
  `MontrealAI/AGI-Alpha-Agent-v0` (95 episodes).

## Sanity checks (all passed before writing outputs)

- fifo: 0 materialized conflicts in every episode (100% avoided) at the worst cost
  (median makespan inflation 167.3%, median added delay 60.3 PR-days/PR).
- oracle: 0 materialized and 0 newly created conflicts in every episode — nets to
  exactly 100% avoided — at ~5x lower median cost (33.4% inflation, 6.9 days/PR).
- predictor: interpolates monotonically from do-nothing (tau=0.80: median 0% avoided,
  0% inflation) to near-FIFO (tau=0.10: median 100% avoided at 160.9% inflation),
  with the useful knee around tau=0.20-0.25 (median 91.7% avoided at 65.0% inflation;
  62.1% avoided at 17.8% inflation).

## Figure

- `fig_t3_pareto.pdf` (and `_nomega` variant): x = **makespan inflation %** (chosen
  over added PR-days per PR for readability — both are in `t3_episode_results.csv`),
  y = % conflicts avoided; marks = medians over episodes, gray bars = IQR; observed
  black diamond at origin, FIFO vermillion square, oracle green triangle, predictor
  blue curve with tau labels at 0.2 / 0.3 / 0.4. Vector PDF, 3.4 x 2.55 in, 8pt.
