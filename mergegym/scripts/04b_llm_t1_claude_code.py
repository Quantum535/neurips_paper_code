#!/usr/bin/env python3
"""04b_llm_t1_claude_code.py -- LLM zero-shot T1 baseline via YOUR Claude
subscription (no API key, $0 extra). Uses Claude Code's official headless
print mode (`claude -p`).

Requirements:
  1. Claude Code installed and logged in on this machine
     (if you've used `claude` in a terminal before, you're set;
      otherwise: npm install -g @anthropic-ai/claude-code && claude)
  2. test_pairs_for_llm.csv.gz (from the chat) in this folder.
  3. pip install pandas

Run:
    python3 04b_llm_t1_claude_code.py

Resumable: rerun the same command any time -- finished pairs are skipped.
If you hit your plan's usage window ("limit reached"), just rerun later;
it picks up where it stopped. Output: llm_scores.csv -- upload to the chat.

Notes:
  - Batches 10 pairs per call (~540 calls total) to be gentle on your
    usage window. Reduce with --batch-size 5 if outputs get truncated.
  - Uses the `haiku` model alias by default (lightest on your quota);
    --model sonnet for the stronger variant.
  - On a headless server instead of your laptop: run `claude setup-token`
    once, then export CLAUDE_CODE_OAUTH_TOKEN=<token> before running.
"""
import argparse, csv, json, os, re, shutil, subprocess, sys
import pandas as pd

PROMPT = """You are scoring pairs of coding tasks. Each pair below will be
implemented as two separate pull requests in the same repository,
concurrently, by AI coding agents. For each pair, estimate the probability
(0-100) that the two resulting PRs will modify at least one common file.
Consider whether the tasks target the same feature area, module, or
configuration surface, and whether both would plausibly touch shared entry
points, manifests, or hot files.

{pairs}

Respond with ONLY a JSON array, one object per pair, in the same order,
no other text: [{{"id": <id>, "prob": <0-100>}}, ...]"""


def render_pair(i, row):
    return (f"PAIR id={i} (repo: {row.repo})\n"
            f"TASK A: {str(row.task_a)[:900]}\n"
            f"TASK B: {str(row.task_b)[:900]}\n")


def call_claude(prompt, model):
    r = subprocess.run(
        ["claude", "-p", prompt, "--model", model],
        capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout, r.stderr


def parse_batch(text, ids):
    """Extract {id: prob} from the model output; tolerant of prose."""
    out = {}
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            for obj in json.loads(m.group(0)):
                if isinstance(obj, dict) and "id" in obj and "prob" in obj:
                    p = max(0.0, min(100.0, float(obj["prob"]))) / 100.0
                    out[int(obj["id"])] = p
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    if not out:  # fallback: id/prob pairs anywhere in the text
        for mm in re.finditer(
                r'"?id"?\s*[:=]\s*(\d+)\D+?"?prob"?\s*[:=]\s*(\d+(?:\.\d+)?)',
                text):
            out[int(mm.group(1))] = max(
                0.0, min(100.0, float(mm.group(2)))) / 100.0
    return {i: out[i] for i in ids if i in out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="test_pairs_for_llm.csv.gz")
    ap.add_argument("--out", default="llm_scores.csv")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--max-calls", type=int, default=1200)
    a = ap.parse_args()

    if shutil.which("claude") is None:
        sys.exit("`claude` CLI not found. Install Claude Code first:\n"
                 "  npm install -g @anthropic-ai/claude-code\n"
                 "then run `claude` once to log in.")
    if os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is set -- it would shadow your "
                 "subscription login and bill the API instead.\n"
                 "Run:  unset ANTHROPIC_API_KEY   and retry.")

    df = pd.read_csv(a.pairs).reset_index(drop=True)
    df["id"] = df.index
    done = set()
    if os.path.exists(a.out):
        prev = pd.read_csv(a.out)
        done = set(zip(prev.repo, prev.pr_a, prev.pr_b))
        print(f"[resume] {len(done)} pairs already scored")
    todo = df[[ (r.repo, r.pr_a, r.pr_b) not in done
                for r in df.itertuples() ]]
    print(f"[plan] {len(todo)} pairs, batches of {a.batch_size}, "
          f"model={a.model}")

    new_file = not os.path.exists(a.out)
    calls = unparsed = 0
    with open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["repo", "pr_a", "pr_b", "prob", "raw"])
        batch_rows = [todo.iloc[i:i + a.batch_size]
                      for i in range(0, len(todo), a.batch_size)]
        for bi, batch in enumerate(batch_rows, 1):
            if calls >= a.max_calls:
                print("[stop] --max-calls guard reached"); break
            ids = list(batch.id)
            prompt = PROMPT.format(pairs="\n".join(
                render_pair(r.id, r) for r in batch.itertuples()))
            rc, out, err = call_claude(prompt, a.model)
            calls += 1
            low = (out + err).lower()
            if rc != 0 and ("limit" in low or "rate" in low):
                print("[pause] usage limit reached -- rerun this script "
                      "after your window resets; progress is saved.")
                break
            scores = parse_batch(out, ids)
            for r in batch.itertuples():
                if r.id in scores:
                    w.writerow([r.repo, r.pr_a, r.pr_b,
                                scores[r.id], ""])
                else:
                    unparsed += 1
            f.flush()
            if bi % 10 == 0:
                print(f"[{bi}/{len(batch_rows)}] calls={calls} "
                      f"unparsed={unparsed}")
    print(f"[done for now] {a.out}; unparsed={unparsed} "
          f"(rerun to retry unparsed/remaining pairs) -> upload "
          f"{a.out} to the chat when the plan count reaches 0")


if __name__ == "__main__":
    main()
