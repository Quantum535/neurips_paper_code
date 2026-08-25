#!/usr/bin/env python3
"""05_judge_p2_claude_code.py -- rubric-guided LLM judge for Paper 2's judging frame,
run through Claude Code's headless mode (`claude -p`) on your subscription, exactly
like 04b_llm_t1_claude_code.py.

Inputs : derived/judge/judging_frame.csv.gz   (from build_judging_frame.py)
Output : derived/judge/judge_labels.csv        (resumable; rerun to continue)

Leakage scrub: the judge sees ONLY title, opening body, and changed-file lists.
It never sees PR state, merged flags, dates, or agent names.
Demotion rule (pre-registered): pairs without diff content on both sides can only
receive D / C / none as the OFFICIAL label. The judge's H/B/I opinion is still
recorded as `provisional_category` for the camera-ready patch-excerpt run.

Usage:
    python3 05_judge_p2_claude_code.py --model sonnet          # full frame
    python3 05_judge_p2_claude_code.py --only calibration      # just the 100 calibration pairs
    python3 05_judge_p2_claude_code.py --dry-run               # print one prompt and exit
"""
import argparse, csv, json, os, re, shutil, subprocess, sys
import pandas as pd

HERE = os.environ.get('MG_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
FRAME = os.path.join(HERE, 'semantic_conflicts', 'results', 'judging_frame.csv.gz')
CAL = os.path.join(HERE, 'semantic_conflicts', 'results', 'calibration_100.csv')
OUT = os.path.join(HERE, 'semantic_conflicts', 'results', 'judge_labels.csv')

RUBRIC = """You are an expert software engineer judging whether two pull requests (PRs) that
were OPEN AT THE SAME TIME in the same repository SEMANTICALLY CONFLICT. Both PRs were
written by AI coding agents. You see each PR's title, opening description, and the
list of files it changed, plus the files both PRs changed. You do NOT see diffs.

Definition. Two co-active PRs semantically conflict if merging both would change the
meaning or value of either contribution, regardless of whether they collide textually.

Assign exactly ONE category:
  D  = DUPLICATE WORK. Both PRs implement the same task, fix the same bug, or add the
       same feature; at most one of them is needed. (Complementary sub-tasks of one
       larger job -- "part 1 / part 2", different components of the same migration --
       are NOT duplicates.)
  C  = CONTRADICTORY. One PR undoes, negates, or reverses the other, OR one PR fixes
       a problem that the other PR's still-open change introduces or addresses.
  H  = HARMFUL CO-EDIT. They change at least one common file and their intents collide
       in that file (same function/section/config key edited for different purposes),
       so a merge would need a real reconciliation decision.
  B  = BENIGN CO-EDIT. They change at least one common file but the overlap is
       mechanical: lockfile / manifest / changelog / README / CI-config lines that
       merge or rebase trivially, or unrelated regions of a large file.
  I  = DEPENDENCY INTERFERENCE. No common file, but one PR changes a definition
       (function signature, config key, schema, exported symbol) that the other
       PR's changed files plausibly use. Only choose I when you can name the
       specific symbol/key; otherwise choose none.
  none = No semantic conflict evident. Independent work.

Rules:
- Judge from the evidence in front of you. Do not assume conflict from mere topical
  similarity; do not assume independence just because file lists are disjoint.
- If the two descriptions are near-identical but the file lists are disjoint, that is
  still D if they target the same task.
- Confidence 1 = weak, 2 = moderate, 3 = strong.
- "evidence" must quote or point to the specific words/files that decided it (<= 200 chars).

Respond with ONLY a JSON object, no other text:
{"category": "D|C|H|B|I|none", "confidence": 1|2|3, "evidence": "..."}
"""

def render(r):
    def side(s):
        return (f"--- PR {s} ---\nTITLE: {r[f'title_{s}']}\n"
                f"DESCRIPTION: {str(r[f'body_{s}'])[:1200]}\n"
                f"CHANGED FILES ({r[f'n_files_{s}']}):\n{r[f'files_{s}']}\n")
    shared = r['files_shared'] if isinstance(r['files_shared'], str) and r['files_shared'] else '(none)'
    return RUBRIC + f"\nREPOSITORY: {r['repo']}\n\n" + side('a') + "\n" + side('b') + f"\nFILES CHANGED BY BOTH:\n{shared}\n"

def call_claude(prompt, model):
    p = subprocess.run(['claude', '-p', prompt, '--model', model], capture_output=True, text=True, timeout=300)
    return p.returncode, p.stdout, p.stderr

def parse(text):
    m = re.search(r'\{.*\}', text, re.S)
    if not m: return None
    try:
        o = json.loads(m.group(0))
        cat = str(o.get('category', '')).strip()
        if cat not in {'D', 'C', 'H', 'B', 'I', 'none'}: return None
        return dict(category=cat, confidence=int(o.get('confidence', 0)), evidence=str(o.get('evidence', ''))[:300])
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='sonnet')
    ap.add_argument('--only', choices=['all', 'calibration'], default='all')
    ap.add_argument('--max-calls', type=int, default=20000)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    frame = pd.read_csv(FRAME)
    if a.only == 'calibration':
        ids = set(pd.read_csv(CAL).frame_id); frame = frame[frame.frame_id.isin(ids)]
    if a.dry_run:
        print(render(frame.iloc[0])); return
    if shutil.which('claude') is None:
        sys.exit('`claude` CLI not found: npm install -g @anthropic-ai/claude-code, then run `claude` once to log in.')
    if os.environ.get('ANTHROPIC_API_KEY'):
        sys.exit('ANTHROPIC_API_KEY is set -- it would bill the API instead of your subscription. `unset ANTHROPIC_API_KEY` and retry.')
    done = set()
    new = not os.path.exists(OUT)
    if not new:
        done = set(pd.read_csv(OUT).frame_id)
    todo = frame[~frame.frame_id.isin(done)]
    print(f'[plan] {len(todo)} pairs to judge (model={a.model}); {len(done)} already done')
    calls = unparsed = 0
    with open(OUT, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['frame_id', 'repo', 'pr_a', 'pr_b', 'pools', 'model', 'category', 'provisional_category', 'confidence', 'evidence', 'has_diff'])
        for i, r in enumerate(todo.itertuples(index=False), 1):
            if calls >= a.max_calls: print('[stop] --max-calls reached'); break
            rc, out, err = call_claude(render(r._asdict()), a.model); calls += 1
            low = (out + err).lower()
            if rc != 0 and ('limit' in low or 'rate' in low):
                print('[pause] usage limit reached -- rerun later; progress is saved.'); break
            lab = parse(out)
            if lab is None: unparsed += 1; continue
            prov = lab['category']
            official = prov if (r.has_diff or prov in {'D', 'C', 'none'}) else 'none'   # demotion rule
            w.writerow([r.frame_id, r.repo, r.pr_a, r.pr_b, r.pools, a.model, official, prov, lab['confidence'], lab['evidence'], r.has_diff])
            f.flush()
            if i % 25 == 0: print(f'[{i}/{len(todo)}] calls={calls} unparsed={unparsed}')
    print(f'[done] calls={calls} unparsed={unparsed} -> {OUT}')

if __name__ == '__main__':
    main()
