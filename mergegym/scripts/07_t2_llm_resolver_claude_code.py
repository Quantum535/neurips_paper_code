#!/usr/bin/env python3
"""07_t2_llm_resolver_claude_code.py -- zero-shot LLM conflict resolution pilot for T2,
run through Claude Code headless mode (`claude -p`) like 04b/05.

Input : t2_hunk_bundle.jsonl   (from t2_groundtruth.py; one locatable hunk per line with
                                 pre-context, ours=earlier PR, theirs=later PR, post-context, truth)
Output: t2_llm_resolutions.csv (resumable) and a printed summary matching Table tab:t2pilot:
        hunk exact %, hunk sim>=0.9 %, per-pair macro exact %.

Sampling: by default caps hunks per pair (--per-pair-cap 10) so the seven mega-hunk pairs
do not dominate the pilot, and takes --max-hunks 300 total. Report the cap you used.

Usage:
    python3 07_t2_llm_resolver_claude_code.py --model sonnet
    python3 07_t2_llm_resolver_claude_code.py --dry-run
"""
import argparse, csv, json, os, re, shutil, subprocess, sys, difflib, random
import pandas as pd

HERE = os.environ.get('MG_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
BUNDLE = os.path.join(HERE, 'mergegym', 'results', 't2_hunk_bundle.jsonl.gz')
OUT = os.path.join(HERE, 'mergegym', 'results', 't2_llm_resolutions.csv')

PROMPT = """You are resolving a textual merge conflict between two pull requests that were
open at the same time in the same repository, both written by AI coding agents. Both PRs
were eventually merged, so the repository ended up with SOME resolution of this region.
Your job: produce the resolved content for the conflict region only -- the lines that
should replace the whole <<<<<<< ... >>>>>>> block -- so that both PRs' intents survive
where they are compatible, and the later PR wins where they are not.

File: {path}   (repository: {repo})

Context before the conflict (unchanged lines):
```
{pre}
```
Side A -- the EARLIER pull request:
```
{ours}
```
Side B -- the LATER pull request:
```
{theirs}
```
Context after the conflict (unchanged lines):
```
{post}
```

Rules:
- Output ONLY the resolved lines for the conflict region, inside a single fenced block.
- Do not repeat the context lines. Do not add commentary, markers, or explanations.
- If one side should simply win, output that side verbatim. If both belong, order them sensibly.
- An empty region is a valid answer: output an empty fenced block.
"""

def norm(s): return re.sub(r'\s+', ' ', s).strip()

def score(cand_lines, truth_lines):
    c, t = norm('\n'.join(cand_lines)), norm('\n'.join(truth_lines))
    exact = int(c == t)
    if not (c or t): return exact, 1.0
    if len(c) > 20000 or len(t) > 20000:
        sim = difflib.SequenceMatcher(None, c[:20000], t[:20000]).quick_ratio()
    else:
        sim = difflib.SequenceMatcher(None, c, t).ratio()
    return exact, round(sim, 4)

def call_claude(prompt, model):
    p = subprocess.run(['claude', '-p', prompt, '--model', model], capture_output=True, text=True, timeout=300)
    return p.returncode, p.stdout, p.stderr

def parse(text):
    m = re.search(r'```[a-zA-Z0-9_+-]*\n(.*?)```', text, re.S)
    if m: return m.group(1).rstrip('\n').split('\n') if m.group(1).strip('\n') else []
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='sonnet')
    ap.add_argument('--per-pair-cap', type=int, default=10)
    ap.add_argument('--max-hunks', type=int, default=300)
    ap.add_argument('--seed', type=int, default=20260824)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--bundle', default=BUNDLE)
    a = ap.parse_args()
    import gzip
    opener = gzip.open if a.bundle.endswith('.gz') else open
    hunks = [json.loads(l) for l in opener(a.bundle, 'rt')]
    random.Random(a.seed).shuffle(hunks)
    per = {}; sel = []
    for h in hunks:
        k = (h['repo'], h['pr_a'], h['pr_b'])
        if per.get(k, 0) >= a.per_pair_cap: continue
        per[k] = per.get(k, 0) + 1; sel.append(h)
    sel = sel[:a.max_hunks]
    if a.dry_run:
        h = sel[0]; print(PROMPT.format(path=h['path'], repo=h['repo'], pre='\n'.join(h['pre']), ours='\n'.join(h['ours']), theirs='\n'.join(h['theirs']), post='\n'.join(h['post']))); return
    if shutil.which('claude') is None: sys.exit('`claude` CLI not found.')
    if os.environ.get('ANTHROPIC_API_KEY'): sys.exit('unset ANTHROPIC_API_KEY to use your subscription login.')
    done = set()
    new = not os.path.exists(OUT)
    if not new:
        prev = pd.read_csv(OUT); done = set(zip(prev.repo, prev.pr_a, prev.pr_b, prev.path, prev.hunk))
    with open(OUT, 'a', newline='') as f:
        w = csv.writer(f)
        if new: w.writerow(['repo', 'pr_a', 'pr_b', 'path', 'hunk', 'model', 'parsed', 'llm_exact', 'llm_sim', 'ours_exact', 'theirs_exact', 'n_llm', 'n_truth'])
        n = 0
        for h in sel:
            key = (h['repo'], h['pr_a'], h['pr_b'], h['path'], h['hunk'])
            if key in done: continue
            prompt = PROMPT.format(path=h['path'], repo=h['repo'], pre='\n'.join(h['pre']), ours='\n'.join(h['ours']), theirs='\n'.join(h['theirs']), post='\n'.join(h['post']))
            rc, out, err = call_claude(prompt, a.model)
            low = (out + err).lower()
            if rc != 0 and ('limit' in low or 'rate' in low):
                print('[pause] usage limit reached -- rerun later; progress is saved.'); break
            res = parse(out)
            oe, _ = score(h['ours'], h['truth']); te, _ = score(h['theirs'], h['truth'])
            if res is None:
                w.writerow([*key, a.model, 0, 0, 0.0, oe, te, '', len(h['truth'])])
            else:
                e, s = score(res, h['truth'])
                w.writerow([*key, a.model, 1, e, s, oe, te, len(res), len(h['truth'])])
            f.flush(); n += 1
            if n % 25 == 0: print(f'[{n}/{len(sel)}]')
    d = pd.read_csv(OUT)
    print(f'\nhunks scored: {len(d)} (parsed {d.parsed.mean():.3f}); per-pair cap {a.per_pair_cap}; model {a.model}')
    print(f'  LLM      exact {d.llm_exact.mean():.3f}   sim>=0.9 {(d.llm_sim>=0.9).mean():.3f}')
    print(f'  earlier  exact {d.ours_exact.mean():.3f}   later exact {d.theirs_exact.mean():.3f}   (same hunks)')
    pp = d.groupby(['repo', 'pr_a', 'pr_b'])[['llm_exact', 'ours_exact', 'theirs_exact']].mean()
    print(f'  per-pair macro exact: LLM {pp.llm_exact.mean():.3f}  earlier {pp.ours_exact.mean():.3f}  later {pp.theirs_exact.mean():.3f}  (pairs {len(pp)})')

if __name__ == '__main__':
    main()
