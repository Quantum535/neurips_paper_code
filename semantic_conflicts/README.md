# Semantic Conflicts research artifact

Code and data for **Beyond File Overlap: Semantic Conflicts Between Concurrent AI Coding Agent Pull Requests**.

This package is a research-grade, reproducible artifact: canonical pool definitions, probability sampling, human annotation, a versioned LLM-judge runner, a static dependency detector, and a gold-only benchmark scaffold.

Gold = human-adjudicated labels. Silver = LLM judge. Heuristic = deterministic candidate pools. LLM labels are never gold.

## Install

From the repository root (Python 3.10–3.12):

```bash
pip install -e ".[dev]"
```

Optional extras: `.[static]` (tree-sitter), `.[github]` (requests).

## One-command deterministic rebuild

```bash
python -m semantic_conflicts.pipeline deterministic
# or: make semantic-conflicts
```

Writes versioned artifacts to `semantic_conflicts/results/v1/`, including:

- `pool_flags.csv.gz` and `pool_counts.json`
- `near_miss_reconciliation.json` (176,756 vs 171,067)
- `p1_outcomes.json`, `p1_headline_cis.json` (Wilson + repo-cluster bootstrap)
- `judging_frame.csv.gz` (Option A disjoint strata + Option B prevalence sample)
- `sampling_design.json`, `sampling_audit.csv`
- `calibration_sheet.csv` / `validation_sheet.csv` (blinded)
- `manifest.json` (git commit, hashes, seeds, row counts)
- `WAITING_FOR_HUMAN_LABELS` until gold exists

Compare a regeneration to a frozen manifest:

```bash
python -m semantic_conflicts.pipeline check
python -m semantic_conflicts.audit --strict
```

## Near-miss definition (canonical)

Parent directory of each changed file, truncated to the first two **directory** components. The filename is not a component. Repo-root files are excluded. A pair is a near-miss iff it has no shared file and those key sets intersect.

The 171,067 figure is a **bug**: treating `src/foo.py` as key `src/foo.py` instead of `src`. See `near_miss_reconciliation.json`. The membership test never hardcodes 176,756.

## Human annotation

```bash
python -m semantic_conflicts.annotation export --round calibration
python -m semantic_conflicts.annotation import-labels --round calibration --annotator a1 --path ...
python -m semantic_conflicts.annotation agreement --round calibration
python -m semantic_conflicts.annotation adjudicate --round calibration
```

Blinded sheets omit merge status, close time, pool labels, inclusion probabilities, and LLM output.

## LLM judge (silver)

```bash
python -m semantic_conflicts.judges --provider echo --model dummy --run-id smoke --only calibration --max-items 2
python -m semantic_conflicts.judges --provider claude_code --model sonnet --run-id claude-v1 --resume
```

Requires `unset ANTHROPIC_API_KEY` for the Claude Code subscription path. Raw JSONL is never overwritten in place; predictions are rebuilt from the log.

## GitHub evidence (network)

```bash
python -m semantic_conflicts.github.fetch_pr_evidence \
  --manifest semantic_conflicts/results/v1/judging_frame.csv.gz \
  --cache semantic_conflicts/cache
```

Cache is gitignored. `evidence_coverage.json` is the redistributable coverage manifest.

## Tests

```bash
pytest semantic_conflicts/tests -q
```

CI runs lint, unit tests, and a tiny-fixture pipeline. It does not call GitHub or LLMs.

## What still needs humans / network

- Calibration and validation gold labels
- Judge runs against a live model
- GitHub patch download for H/B/I and the static detector on real PRs
- Category-I repository deep-dive verdicts

Depth-2 co-location remains a heuristic baseline. Only static/repo evidence can upgrade a Category I *claim*.
