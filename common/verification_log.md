
## anatomy (ok=True)

[MATCH] Label integrity pre-check: pairs_labeled.csv.gz conflict column is trustworthy
  -> 0/2000 mismatches on a fresh random sample (seed 20260816), conflict re-derived from raw pr_files.csv.gz as >=1 shared filepath; additionally all 577,045 labeled pairs have file lists for both PRs (consistent with both_files=True everywhere; 580,908 coactive minus 577,045 labeled = 3,863 pairs dropped for missing file data)
  NOTE: Label file confirmed reliable and used for all subsequent claims.

[MATCH] 577,045 labeled pairs; 42,600 conflicting = 7.38%; same-agent 7.30%; cross-agent 24.42% (cross-agent n=2,858)
  -> 577,045 pairs; 42,600 conflicting = 7.3824%; same-agent 7.2976% (n=574,187); cross-agent 24.4227% (n=2,858)

[MATCH] Shared-files-per-conflicting-pair: 65.04% share exactly 1 file; median 1; mean 10.53; p90 13; p99 250; max 904
  -> 65.0446% share exactly 1; median 1.0; mean 10.5272; p90 13.0; p99 250.0; max 904 (over 42,600 conflicting pairs)

[MATCH] Pure-config conflicts = 33.88% of conflicting pairs; strict source-code conflicts = 58.76%
  -> Under the original classification spec (lockfiles incl. *.lock/pnpm-lock.yaml/bun.lockb, manifests package.json/pom.xml/Cargo.toml/Gemfile/composer.json/build.gradle/setup.py/setup.cfg/go.mod/go.sum, requirements*/pyproject.toml, Dockerfile*/docker-compose, .github/workflows/*, README*): pure-config 14,433 = 33.8803%; strict source (>=1 file that is not config, not .md/.rst/.txt/CHANGELOG docs, not .gitignore/.env/other yaml-json-toml-ini-cfg) 25,032 = 58.7606%
  NOTE: Definitional sensitivity as the claim warns: my own independent boundary (adding .csproj and treating pyproject*/requirements* by prefix) gives 33.91%/58.73%; dropping pnpm-lock.yaml+bun.lockb from config gives 33.28%; counting ALL misc yaml/json/toml/ini/cfg as config gives 35.44%. Pure-config moves roughly 33.3-35.4% across reasonable boundaries; strict-source is stable at 58.7-58.9%. The printed 33.88%/58.76% are exact under the stated file lists.

[MATCH] Same-agent diagonal: Codex-Codex 5.60% (n=532,296); Devin-Devin 36.38% (n=19,449); Copilot-Copilot 21.68% (n=19,360); Cursor-Cursor 24.64% (n=2,890); Claude_Code-Claude_Code 63.54% (n=192)
  -> OpenAI_Codex 5.5975% (n=532,296); Devin 36.3823% (n=19,449); Copilot 21.6787% (n=19,360); Cursor 24.6367% (n=2,890); Claude_Code 63.5417% (n=192)

[MATCH] Codex-Codex pairs = 92.2% of all pair volume
  -> 532,296 / 577,045 = 92.2451%

[MATCH] Top-10 repos hold 67.62% of conflicting pairs; top-50 hold 85.37%; mochilang/mochi = 369,597 pairs; MontrealAI/AGI-Alpha-Agent-v0 = 134,132; together 87.3% of all pairs
  -> Top-10 by conflicting-pair count: 67.6150%; top-50: 85.3662%; mochilang/mochi 369,597 pairs; MontrealAI/AGI-Alpha-Agent-v0 134,132 pairs; together 87.2946% of 577,045
  NOTE: Definition used: repos ranked BY number of conflicting pairs, share measured among conflicting pairs. If instead ranked by total pair volume the top-10 share would be 65.52% (top-50: 81.83%), so the ranking criterion matters and should be stated in the PDF as 'top repos by conflict count'.

[MATCH] Concurrency dose-response: conflict rate 36.90% at concurrency 2 -> 3.32% at >50; excluding the 2 mega-repos: 38.93% -> 11.97%
  -> All pairs: 36.90% (n=2,957) at concurrency 2, monotone down to 3.32% (n=302,602) at >50 (intermediate: 3-5 34.15%, 6-10 29.06%, 11-20 25.11%, 21-50 6.98%); excluding mochilang/mochi and MontrealAI/AGI-Alpha-Agent-v0: 38.93% at 2 -> 11.97% at >50
  NOTE: Definition used (same as prior pipeline): concurrency = number of distinct agent PRs from the labeled pairs file whose [opened, closed] interval contains the pair's overlap midpoint (opened <= t <= closed); this count includes the pair's own two PRs, so minimum is 2. It counts only PRs that appear in coactive pairs, not every repo PR.

[MATCH] Overlap-duration buckets (<1h/1-6h/6-24h/1-7d/>7d) raw conflict rates 5.13/20.65/21.11/21.89/23.72%; excluding 2 mega-repos roughly flat 25.8-32.7%
  -> All pairs: 5.13% (n=499,538) / 20.65% / 21.11% / 21.89% / 23.72%; excluding the 2 mega-repos: 25.82 / 29.79 / 32.72 / 28.25 / 27.80% (range 25.8-32.7%, non-monotone/roughly flat)
  NOTE: Bucket edges treated as left-closed ([0,1h), [1,6h), etc.); no negative overlaps exist.

[MATCH] Hot files: README.md top basename in conflicting pairs' shared sets (~13,803 occurrences), package.json second (~9,369)
  -> README.md 13,803 (rank 1); package.json 9,369 (rank 2); next: page.tsx 8,668, index.ts 5,390
  NOTE: Counted as pair x basename occurrences across each conflicting pair's shared-file set.

## outcomes-pools (ok=True)

[MATCH] Later-opened PR (pr_b) merge rate: 73.33% in conflicting pairs vs 85.96% in non-conflicting (-12.63pp); pair-level odds ratio 0.449.
  -> 73.3333% (n=42,600 conflicting) vs 85.9645% (n=534,445 non-conflicting); diff -12.6312pp; OR 0.4490
  NOTE: pr_b is the later-opened PR in 100% of pairs (opened_b >= opened_a always), so 'pr_b' and 'later-opened' definitions coincide. Merged flag taken from pr_texts.csv.gz (0 missing). Conflict label independently re-derived from pr_files.csv.gz with 0 disagreements over all 577,045 pairs.

[MATCH] PR-level dedup (each distinct later-role PR counted once, ever-in-conflicting-pair vs never): 67.89% (n=11,431) vs 78.24% (n=12,823), OR 0.588.
  -> 67.8856% merged (n=11,431 ever-in-conflicting) vs 78.2422% (n=12,823 never), OR 0.5878
  NOTE: Dedup key = (repo, later PR number); ever_conf = max(conflict) over that PR's pairs. Merged flag is internally consistent per PR (0 PRs with contradictory flags), so 'first' vs 'any' aggregation makes no difference.

[MATCH] Repo-stratified Mantel-Haenszel OR = 0.851 for the later PR's merge odds (conflicting vs not, stratified by repo).
  -> MH OR = 0.8512 (num=3547.3, den=4167.6) over 534 repo strata having both conflicting and non-conflicting pairs
  NOTE: Recomputed from scratch as sum(a*d/N)/sum(b*c/N) per repo stratum with a=merged&conflict, b=unmerged&conflict, c=merged&non-conflict, d=unmerged&non-conflict; repos lacking either arm contribute nothing (standard MH behavior).

[CLOSE] Duplicate-title pool: 6,945 pairs (with min-3-token guard); 332 byte-identical normalized titles; 7 cross-agent; both merged 46.8%, exactly one 38.7%, neither 14.5%; 5,525 of 6,945 (79.6%) share no file.
  -> Pool: 6,945 strict (6,965 without guard); cross-agent 7; both merged 3,251 (46.8%), exactly one 2,685 (38.7%), neither 1,009 (14.5%); no shared file 5,525 of 6,945 (79.6%). Byte-identical normalized titles: 330 (claimed 332).
  NOTE: My rules: lowercase; delete \[...\] bracketed tags; replace [^a-z0-9\s] with space; collapse whitespace; match if token-set Jaccard >= 0.5 OR one normalized string contains the other with both >= 15 chars; guard = both titles >= 3 tokens. Every headline number matches exactly at that definition, including the 6,965 pre-guard count, so the pool itself is verified. Only the byte-identical sub-count differs: I get 330 vs claimed 332 (range 325-330 across normalization variants tested: punctuation deleted vs spaced, bracket handling, casefold, NFKC; 346 if empty normalized titles count as identical). Recommend printing 330 or dropping the identical-title figure; it does not affect the 6,945 pool.

[MATCH] Revert-title pairs (>=1 title contains revert/undo/rollback): 278; of these 35 share a file.
  -> 278 pairs; 35 with a shared file (also 2 cross-agent, matching the plan doc)
  NOTE: Exact match using word-boundary regex \b(revert|undo|rollback)\b on lowercased raw titles. Sensitive to boundary handling: plain substring matching gives 342 pairs / 45 shared-file (e.g. 'undo' inside 'undocumented'), so the printed number should state word-boundary matching.

[CLOSE] Fix-in-flight: conflicting (shared-file) pairs where the later PR's title contains fix/bug/patch keywords and it opened while the earlier PR was still open: 7,875 pairs.
  -> 7,872 pairs / 2,327 distinct later fix-PRs (best reproduction); range 7,791-8,281 across keyword-rule variants
  NOTE: The temporal condition is vacuous: opened_b < closed_a holds for all 577,045 co-active pairs by construction, so the count is just conflicting pairs whose later title matches the keywords. Best-matching rule: normalized-title token in {fix,fixes,fixed,fixing,bug,bugs,bugfix,hotfix,patch,patches,patched} gives 7,872 pairs and exactly 2,327 distinct fix PRs (matching the plan's companion figure); strict \b(fix|bug|patch)\b gives 7,795/2,304; prefix \b(fix|bug|patch) gives 7,928/2,338; substring gives 8,281. Claimed 7,875 sits inside the word-boundary band (7,795-7,928) but no tested rule hits it exactly; discrepancy <=0.1%. Suggest printing '~7,900' or stating the exact keyword regex alongside the number.

[MATCH] Both-merged pairs: 30,031 (5.2% of 577,045). Both-merged AND >=1 shared file: 6,409 (90 cross-agent); merge-gap |closed_a - closed_b| < 24h for 47.7%; median gap 27.0h.
  -> Both merged: 30,031 (5.20%); both merged & shared file: 6,409 (1.11% of all, 21.34% of both-merged), 90 cross-agent; gap<24h: 3,059 (47.7%); median gap 27.02h; supporting cuts also match (<1h: 1,556; <72h: 4,128; <7d: 5,000)
  NOTE: Gap computed as |closed_b - closed_a| in hours on the 6,409 both-merged shared-file pairs; closed_* used as merge-time proxy per the plan doc (no NaNs).

[MATCH] Near-miss pool: no shared file but shared depth-2 directory excluding repo root: 176,756 pairs.
  -> 176,756 pairs
  NOTE: Definition used: for each changed file, directory path truncated to its first 2 components (files with a 1-level directory keep that single component; repo-root files map to '<root>', which is excluded from the intersection); pool = pairs with conflict=False whose depth-2 dir sets intersect. Cross-checks also reproduce the plan doc exactly: depth-1 354,622 (noroot 344,574) and depth-2 with root 187,650, confirming the same truncation convention.

## prediction (ok=True)

[MATCH] Pre-check: pairs_labeled conflict column re-derived from pr_files (shared changed filepath) on random sample of 2,000 pairs
  -> 2000/2000 agree (0 disagreements; seed 42 sample)
  NOTE: Label = |files(pr_a) ∩ files(pr_b)| >= 1 within repo, from data/pr_files.csv.gz. pairs_labeled has 577,045 rows, 42,600 conflicts, 0 duplicate keys. Column is trustworthy.

[MATCH] 1. Inner join test_pairs_for_llm + llm_scores + pairs_labeled on (repo, pr_a, pr_b): n = 5,387, no duplicates, positive rate 39.93% (2,151 conflicts)
  -> n = 5,387; 0 duplicate keys (also 0 dups within each input); 2,151 conflicts; positive rate 39.9295%
  NOTE: All 5,387 test rows matched llm_scores and pairs_labeled directly (no swapped-key fallback needed).

[MATCH] 2. LLM zero-shot ROC-AUC = 0.7044 (tie-corrected rank AUC)
  -> 0.7044496 (tie-corrected Mann-Whitney rank AUC, own implementation)
  NOTE: Agrees at stated 4-dp precision.

[MATCH] 3. Precision@k by prob: P@50 = 0.94, P@100 = 0.96, P@270 = 0.948, P@539 = 0.905; random-ranking P@100 ~ 0.40
  -> P@50 = 0.9400, P@100 = 0.9600, P@270 = 0.9481, P@539 = 0.9054 (deterministic descending stable sort); random P@100 = positive rate = 0.3993
  NOTE: P@50 is tie-sensitive: the tied group at prob=0.95 straddles rank 50; expected P@50 under random tie-breaking is 0.958 (other k's are tie-stable within 0.002). The printed 0.94 corresponds to the deterministic-order convention.

[MATCH] 4. Calibration by prob bin [0,.2/.2,.4/.4,.6/.6,.8/.8,1]: actual conflict rates ~ 0.266/0.388/0.456/0.748/0.939
  -> 0.2661 / 0.3884 / 0.4563 / 0.7482 / 0.9394 (bin sizes 2788/1290/504/409/396)
  NOTE: Definition used: left-closed bins [0,.2),[.2,.4),[.4,.6),[.6,.8),[.8,1]. Top-bin rate is 0.9389 if the 3 prob=1.0 pairs (all label 1) are excluded, 0.9394 if included — both print as 0.939. Bin convention matters: right-closed bins give 0.282/.387/.502/.793/.941, so the PDF should state bins are left-closed.

[MATCH] 5. Single-feature AUCs: repo base rate (non-test pairs) = 0.601 with 2,787 test pairs filled at global ~0.0708; overlap-hours = 0.570; TF-IDF cosine = 0.644; title Jaccard = 0.629
  -> repo base rate AUC = 0.6010 (571,658 non-test pairs; 2,787 test pairs in repos with zero non-test pairs, filled with global rate 0.070757); overlap-hours AUC = 0.5699 (overlap = max(0, min(closed)-max(opened)), no missing closed timestamps in test set); TF-IDF cosine AUC = 0.6436 (log-tf (1+ln c) x smoothed idf ln((1+N)/(1+df))+1, [a-z0-9_]+ tokens, idf over all 10,774 test task texts); title Jaccard AUC = 0.6291 (titles from pr_texts, token-set Jaccard, 0 missing titles)
  NOTE: TF-IDF sensitivity: a raw-tf x ln(N/df) variant gives 0.6356, so 0.644 is within the requested +-0.02 defensible range but the PDF should note it is implementation-specific (log-tf x smoothed-idf).

[MATCH] 6. 5-fold-CV pooled OOF logistic regression (L2, standardized): (a)-(d) = 0.676; (a)+(c)+LLM = 0.728; all four + LLM = 0.740
  -> seed 0: (a)-(d) = 0.6759; (a)+(c)+LLM = 0.7280; all4+LLM = 0.7404 (own Newton-method logistic regression, per-fold standardization, L2 = 1e-4 on non-intercept weights)
  NOTE: Definition: overlap enters as log1p(hours) (raw hours gives 0.6615 for (a)-(d), so the log transform is load-bearing); fold split = RandomState(0).permutation + array_split(5). Seed sensitivity is small: seeds 1/2/7 give 0.678-0.679, 0.728-0.729, 0.7415-0.7417 respectively — the printed values hold to +-0.003. Result is insensitive to L2 (l2=1.0 also gives 0.7404).

[MATCH] 7. Within-repo AUC: ~64 repos with >=10 test pairs and both classes covering ~4,773 pairs; mean per-repo LLM AUC = 0.7425, >0.5 in 58/64
  -> 64 repos, 4,773 pairs covered, mean per-repo LLM AUC = 0.7425, AUC > 0.5 in 58/64 repos

[MATCH] 8. Only 7 label-0 pairs have prob >= 0.9; 468 label-1 pairs have prob <= 0.1 (21.8% of positives)
  -> label-0 with prob>=0.9: 7; label-1 with prob<=0.1: 468 = 21.76% of 2,151 positives
  NOTE: 21.76% rounds to 21.8% as stated.

[CLOSE] 9. 27.2% of test-set task texts contain an explicit filename of a file that PR actually changed
  -> Per-text denominator (10,774 = 5,387 pairs x 2 sides, with multiplicity): 31.21% (3,363/10,774) under my primary definition; 27.87% (3,003/10,774) restricting to a curated common-extension list; 18.78% under full-path-only matching. Per-pair denominator (>=1 of the two texts): 48.47% primary / 43.60% curated (both texts: 13.96%; unique PRs: 34.90%)
  NOTE: Definition used: candidate tokens = regex [A-Za-z0-9_./-]*[A-Za-z0-9_-]\.[A-Za-z][A-Za-z0-9]{0,5} (dot + alphabetic-initial extension, optional path prefix, leading ./ stripped) plus slash-paths; a token counts if it equals a changed filepath, is a '/'-suffix of one, or (no-slash tokens) equals a changed file's basename, case-sensitive. The claimed 27.2% clearly uses the per-text denominator and sits inside the defensible 19-31% range (my closest reasonable variant gives 27.9%), but the number is strongly definition-sensitive: the PDF must state the exact token regex and match rule, and must not present it as a per-pair rate (which is ~44-48%).

[MATCH] 10. Bodies exactly at the 1,200-char cap: 7,251/33,596 = 21.58% in pr_texts
  -> 7,251 bodies with len == 1200 out of 33,596 pr_texts rows = 21.58% (denominator includes 360 null bodies; of 33,236 non-null bodies it is 21.82%). Max body length is exactly 1,200 and none exceed it, confirming the cap
  NOTE: Stated fraction uses all rows as denominator; fine as printed.

## sizes (ok=True)

[MATCH] Precondition: conflict label in pairs_labeled.csv.gz is trustworthy (re-derived from pr_files for a random sample of 2,000 pairs)
  -> 0/2000 conflict-label mismatches; 0/2000 both_files mismatches (label re-derived as >=1 shared changed filepath between pr_a and pr_b from pr_files.csv.gz; seed 12345)
  NOTE: Label column confirmed trustworthy; used for all subsequent claims. pairs_labeled has exactly 577,045 rows; pr_files has 575,913 rows.

[MATCH] T1 strata over 577,045 pairs: no-conflict/same 532,285; no-conflict/cross 2,160; conflict/same 41,902; conflict/cross 698; 1,100 distinct repos overall; conflict/cross spans 81 repos
  -> 532,285 / 2,160 / 41,902 / 698 (total 577,045); 1,100 distinct repos; conflict-and-cross-agent stratum spans 81 repos

[MATCH] 100% of the 577,045 pairs have both PRs present in pr_texts
  -> 577,045 / 577,045 = 100.0000% (keyed join on (repo, pr) for both pr_a and pr_b)

[MISMATCH] Per-repo cap of 200 pairs, stratified, mega-repos EXCLUDED pool: ~29,870 pairs
  -> sum of min(pairs_per_repo, 200): mega-repos INCLUDED = 29,870 pairs / 1,100 repos; mega-repos EXCLUDED = 29,470 pairs / 1,098 repos
  NOTE: The number 29,870 is exactly reproduced only with the 2 mega-repos (mochilang/mochi 369,597 pairs; MontrealAI/AGI-Alpha-Agent-v0 134,132 pairs) INCLUDED and capped at 200 each — consistent with ev_mergegym.md, which reports '29,870 pairs, 1,100 repos'. The 'mega-repos EXCLUDED' descriptor is wrong. Corrected text to print: either '29,870 pairs / 1,100 repos (per-repo cap 200; mega-repos included but capped)' or '29,470 pairs / 1,098 repos' if mega-repos are truly excluded. Count is insensitive to the stratified-sampling seed since each repo contributes min(n,200).

[MATCH] Balanced 4-stratum ceiling = 4 x 698 = 2,792
  -> rarest stratum (conflict x cross-agent) = 698 pairs; 4 x 698 = 2,792

[MATCH] Mega-repo-excluded full pool = 73,316 pairs / 1,098 repos
  -> 73,316 pairs / 1,098 repos (52,535 negative, 20,781 positive) after dropping mochilang/mochi and MontrealAI/AGI-Alpha-Agent-v0

[CLOSE] T2 source-file filter removed 16,545/575,913 file rows = 2.87%
  -> 16,493 / 575,913 = 2.86% removed
  NOTE: Definition used: lowercase basename ends with .lock or is one of {package-lock.json, go.sum, yarn.lock, pnpm-lock.yaml, cargo.lock, poetry.lock}, ends with .svg, 'dist' appears as a directory segment, or lowercase path contains 'generated'. Discrepancy is 52 rows (0.009 pp). Tested variants ('dist/' substring anywhere = 16,569; case-sensitive 'generated' = 12,830; +bun.lockb = 16,568) — none reproduces 16,545 exactly, so the original run used a slightly different dist/lock rule. Immaterial: all downstream T2 numbers match exactly under my filter.

[MATCH] T2 candidates (conflict pairs, both PRs merged, >=1 shared SOURCE file): 6,387 pairs / 391 repos / 88 cross-agent / 3,779 distinct PRs
  -> 6,387 pairs / 391 repos / 88 cross-agent / 3,779 distinct PRs
  NOTE: merged taken from pr_texts.merged for both PRs; shared source files recomputed after the exclusion filter above.

[MATCH] T2 median shared source files per pair = 4
  -> median 4.0 (mean 31.1, max 458)

[MATCH] T2 median PR size (additions+deletions over all its files) = 207
  -> median 207.0 over the 3,779 distinct T2 PRs (p25 40, p75 1,323.5, mean 5,109)
  NOTE: Definition used: per-PR sum of additions+deletions over ALL file rows in pr_files (not just source files), median across the 3,779 distinct T2 PRs. Sensitive to that choice: source-files-only gives median 196.

[MATCH] T2-core (both PRs <= 1,000 changed lines) = 1,614 pairs / 264 repos
  -> 1,614 pairs / 264 repos (11 cross-agent), using all-files PR size

[MATCH] T3 pool: repos with >=100 co-active pairs AND >=20 conflicting pairs = 97 repos, 19,397 distinct PRs, 38,526 conflicting pairs
  -> 97 repos; 19,397 distinct PRs; 38,526 conflicting pairs (563,658 total pairs in pool)
  NOTE: Co-active pairs counted from pairs_labeled (577,045), not pairs_coactive (580,908) — confirmed the right base because top-50 covers 556,179 = 96.4% of 577,045 as stated in ev_mergegym.md. Cross-checks: >=50&>=10 gives 146 repos; >=200&>=50 gives 52 repos, both matching the plan doc.

[MATCH] T3 top-50 repos by co-active pairs: median pairs 542.5, median conflicting 160, median agent PRs 95.5, median span 72.1 days, median time-weighted mean concurrency 6.9
  -> median pairs 542.5; median conflicting 160.0; median n_agent_prs (repo_stats) 95.5; median span 72.13 days; median time-weighted mean concurrency 6.94
  NOTE: Definitions used: PR open/close intervals reconstructed from pair rows (distinct (repo,pr)); span = max(closed) - min(opened) per repo; mean concurrency = integral of concurrently-open PR count over the span divided by span length (event sweep). All five medians reproduce to stated precision.

[MATCH] repo_stats: minimum stars among the 1,129 repos in pairs_coactive = 101; median 644; 41.1% >= 1,000 stars
  -> 1,129 distinct repos, all present in repo_stats; min stars 101; median 644.0; 464/1,129 = 41.10% >= 1,000 stars

[MATCH] Non-episode training pool (pairs in repos NOT among the 97 episode repos): 13,387 pairs at 30.4% conflict rate
  -> 13,387 pairs; 4,074 conflicts = 30.43% conflict rate
  NOTE: Computed on pairs_labeled minus the 97 repos meeting >=100 co-active pairs and >=20 conflicting pairs.

[MATCH] Overall merged rate in pr_texts = 71.5%; agent counts Codex 21,799 / Copilot 4,970 / Devin 4,827 / Cursor 1,541 / Claude_Code 459
  -> merged 24,014/33,596 = 71.48% -> 71.5%; OpenAI_Codex 21,799, Copilot 4,970, Devin 4,827, Cursor 1,541, Claude_Code 459 (sums to 33,596; no missing merged values)
  NOTE: The agent label in the data is 'OpenAI_Codex' (printed as 'Codex' in the claim).

## external (ok=True)

[MATCH] CooperBench = arXiv 2601.13295, Jan 2026: benchmark of two coding agents working simultaneously with intentionally embedded merge conflicts; 652 tasks across 12 repos in 4 languages; execution-based; reports ~30% success drop under cooperation ('curse of coordination').
  -> arXiv 2601.13295, submitted Jan 19 2026 ('CooperBench: Why Coding Agents Cannot be Your Teammates Yet', Khatua et al.): 652 collaborative tasks, 12 open-source libraries, 4 languages (Python/TypeScript/Go/Rust), execution-based (runs both unit-test sets on merged code), avg ~30% lower success when cooperating, framed as 'curse of coordination'; 77.3% of tasks have conflicting ground-truth solutions (conflicts by design).
  NOTE: All figures confirmed at https://arxiv.org/abs/2601.13295 and https://arxiv.org/html/2601.13295v2. Minor wording nit: paper says '12 libraries' grounded in real repos rather than '12 repos'.

[MATCH] Merge-Bench (Schesch & Ernst) = arXiv 2605.25890: 7,938 real merge-conflict hunks from 1,439 repos, 11 languages; best models resolve <60%.
  -> arXiv 2605.25890 'Merge-Bench: Resolve Merge Conflicts with Large Language Models' by Benedikt Schesch and Michael D. Ernst (May 25, 2026): 7,938 real-world merge conflict hunks from 1,439 GitHub repos, 11 languages tested, best models correctly resolve <60% of conflicts.
  NOTE: Confirmed at https://arxiv.org/abs/2605.25890.

[MATCH] SWE-Gym (Pan et al.) = arXiv 2412.21139, ICML 2025: 2,438 real Python task instances.
  -> arXiv 2412.21139 'Training Software Engineering Agents and Verifiers with SWE-Gym' (Jiayi Pan et al.), accepted at ICML 2025: 2,438 real-world Python task instances.
  NOTE: Confirmed at https://arxiv.org/abs/2412.21139.

[MATCH] AgenticFlict = arXiv 2604.03551, AIware 2026: 142,652+ agentic PRs, 27.67% merge-conflict rate, 336k+ conflict regions; per-agent Codex 31.85%, Copilot 15.24%.
  -> arXiv 2604.03551 (Ogenrwot & Businge, accepted AIware 2026): 142,652 agentic PRs from 59K+ repos; 107K+ PRs processed via merge simulation, 29K+ conflicted = 27.67% conflict rate; 336K+ conflict regions; Table 2 per-agent rates: OpenAI Codex 31.85%, Claude Code 25.93%, Devin 22.85%, Cursor 19.75%, GitHub Copilot 15.24%.
  NOTE: Confirmed at https://arxiv.org/abs/2604.03551 and full text https://arxiv.org/html/2604.03551v2. Note the 27.67% rate is over the ~107K successfully processed PRs, not all 142,652.

[MATCH] AIDev = arXiv 2602.09185: 932,791 agent PRs, 116,211 repos, 72,189 developers; curated AIDev-pop subset 33,596 PRs / 2,807 repos (100+ stars). Earlier version arXiv 2507.15003: 456K+ PRs.
  -> arXiv 2602.09185 'AIDev: Studying AI Coding Agents on GitHub' (Li, Zhang, Hassan): 932,791 Agentic-PRs, 116,211 repos, 72,189 developers; AIDev-pop: 33,596 PRs from 2,807 repos with >100 stars. arXiv 2507.15003 (same authors, 'The Rise of AI Teammates in SE 3.0'): 456,000+ PRs across five agents in ~61,000 repos.
  NOTE: Confirmed at https://arxiv.org/abs/2602.09185 and https://arxiv.org/abs/2507.15003.

[MATCH] Xu, Subramanian & Karthik = arXiv 2607.04697: 79.4% of agent PRs co-active; 580,913 co-active pairs at k=0; 0.50% cross-agent; 1,129 repos with pairs; replayed textual conflict rates 19.8% intra-agent vs 41.7% cross-agent (747-pair stratified sample, 716 evaluable = 95.8%); conflicted files 84.4% source code.
  -> arXiv 2607.04697 'AI Agent Pull Requests on GitHub: Frequency, Structure, and Merge Conflict Rates' (George Xu, Arjun Subramanian, Nithilan Karthik): 79.4% of agent PRs in co-active pairs; 580,913 co-active pairs at k=0; 0.50% cross-agent (in 122 repos); 1,129 repos (40.2% of 2,807) with >=1 pair; replay on 747 stratified pairs, 716 evaluable (95.8%); conflict rates 19.8% intra-agent vs 41.7% cross-agent; 84.4% of conflicted files are source code.
  NOTE: All figures confirmed at https://arxiv.org/abs/2607.04697 (full text https://arxiv.org/html/2607.04697v1).

[MATCH] NeurIPS 2026: workshop list announced Aug 10, 2026; 102 workshops across Sydney/Paris/Atlanta; common suggested deadline Aug 29, 2026 AoE; Sydney workshops Dec 11-12. 'Who Verifies the Agents?' (Sydney) deadline Aug 29, 4-9 pages; 'Meta Agents' (Sydney) deadline Aug 29, full <=9 / short <=4; VeriCodeGen (Atlanta) abstract Sept 11 / paper Sept 13.
  -> NeurIPS blog Aug 10, 2026: 102 workshops (Sydney 48, Paris 28, Atlanta 26); suggested submission date Aug 29, 2026 (many workshops adopt it, each sets its own CFP); Sydney workshops Fri Dec 11 & Sat Dec 12, Paris/Atlanta Dec 12-13. 'Who Verifies the Agents?' (Sydney): deadline Aug 29, 2026 23:59 AoE, papers 4-9 pages excl. refs/appendix (demos <=4). 'Managing Agents that Manage Agents' / Meta Agents (Sydney): deadline Aug 29, 2026 AoE, full <=9 / short <=4 pages main text. VeriCodeGen (Atlanta, Dec 12): abstract Sept 11, 2026 / paper Sept 13, 2026 per its tentative schedule.
  NOTE: Sources: https://blog.neurips.cc/2026/08/10/announcing-the-neurips-2026-workshops/ , https://verify-agents-workshop.github.io/ , https://meta-agents-workshop.github.io/ , https://vericodegen.github.io/ . Caveat: VeriCodeGen dates are labeled tentative on the site (a search result also noted a site/OpenReview discrepancy), so recheck before printing them as firm deadlines; the Aug 29 date is a suggested common date, not mandated.

[CLOSE] Owhadi-Kareshk et al. MSR 2019: ~267k merge scenarios, 28 features. ConflictBench (Shen & Meng, JSS 2024): 180 Java merge scenarios. DupPR (Yu et al., MSR 2018): 2,323 verified duplicate PR pairs.
  -> Owhadi-Kareshk, Nadi & Rubin, 'Predicting Merge Conflicts in Collaborative Software Development', ESEM 2019 (not MSR): 267,657 merge scenarios from 744 repos in 7 languages; 9 feature sets totaling 28 features per merge scenario. ConflictBench (Shen & Meng, JSS 2024): 180 merging scenarios from 180 Java projects. DupPR (Yu et al., MSR 2018 Data Showcase): 2,323 manually-verified duplicate PR pairs from 26 projects.
  NOTE: All numbers correct, but the venue is wrong: Owhadi-Kareshk et al. appeared at ESEM 2019, not MSR 2019 (confirmed from the paper PDF https://sanadlab.org/assets/pdf/OwhadiKareshkESEM19.pdf and https://arxiv.org/abs/1907.06274, which states 28 total features from 9 feature sets). ConflictBench: https://people.cs.vt.edu/nm8247/publications/bowen-jss-2024-preprint.pdf . DupPR: https://2018.msrconf.org/details/msr-2018-Data-Showcase-Papers/11/A-Dataset-of-Duplicate-Pull-requests-in-GitHub . Fix 'MSR 2019' to 'ESEM 2019' before printing.

[MATCH] Kasi & Sarma Cassandra ICSE 2013: textual conflict rates 7.6%-19.3% across projects. Brindescu et al. EMSE 2020: merge-conflict code ~2x as likely to be buggy. Sousa et al. SafeMerge OOPSLA 2018: 52 merges, 11 violations.
  -> Kasi & Sarma (Cassandra, ICSE 2013): conflicting merges were 7.6%-19.3% of all merges across four OSS projects. Brindescu et al., EMSE vol 25 (2020): code involved in a merge conflict is 2x as likely to have a bug (26x when manual resolution was required). Sousa, Dillig & Lahiri (SafeMerge, OOPSLA 2018): evaluated on 52 real-world GitHub merge scenarios; SafeMerge failed to verify 13, of which 2 were false positives, leaving 11 genuine semantic conflict-freedom violations.
  NOTE: Sources: https://epiclab.github.io/publications/icse13-kasi.pdf (7.6%-19.3% figure cited in follow-up literature), https://link.springer.com/article/10.1007/s10664-019-09735-4 (2x bug likelihood), https://arxiv.org/abs/1802.06551 (Tables 1-2: 52 benchmarks; 33 verified textual, 6 verified manual, 5 fail-textual incl. 2 false positives, 8 fail-manual => 11 true violations). The '11 violations' reading (13 unverified minus 2 false positives) is supported but worth phrasing carefully.

[MATCH] Brun et al. Crystal ESEC/FSE 2011: speculative merging; studied ~9 OSS systems.
  -> Brun, Holmes, Ernst & Notkin, 'Proactive Detection of Collaboration Conflicts', ESEC/FSE 2011 (Crystal tool): uses speculative analysis (speculative merging of version-control operations); historical study of nine open-source systems totaling 3.4M lines of code and 550,000 development versions.
  NOTE: Confirmed from the paper PDF https://www.cs.ubc.ca/~rtholmes/papers/fse_2011_brun.pdf ('We analyze nine open-source systems...'); ACM record https://dl.acm.org/doi/10.1145/2025113.2025139 . The exact count is 9, so '~9' can be stated as exactly nine.
