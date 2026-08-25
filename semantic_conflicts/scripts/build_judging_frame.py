#!/usr/bin/env python3
"""Build the judging frame described in Paper 2 §4.3 from derived/pool_flags.csv.gz.

Outputs (in derived/judge/):
  judging_frame.csv.gz      one row per pair: ids, pools, stratum, inclusion prob, judge inputs
  calibration_100.csv       100-pair human calibration sheet (blank label columns)
  frame_summary.json        counts per pool / stratum
Seeds are fixed so the frame is reproducible.
"""
import os, json, re
import numpy as np, pandas as pd
from collections import defaultdict

SEED = 20260824
D = os.environ.get('MG_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
O = f'{D}/semantic_conflicts/results'; os.makedirs(O, exist_ok=True)
MEGA = {'mochilang/mochi', 'MontrealAI/AGI-Alpha-Agent-v0'}

p = pd.read_csv(f'{D}/semantic_conflicts/results/pool_flags.csv.gz')
txt = pd.read_csv(f'{D}/data/pr_texts.csv.gz')
pf = pd.read_csv(f'{D}/data/pr_files.csv.gz')
p['mega'] = p.repo.isin(MEGA)
p['gap_days'] = (pd.to_datetime(p.opened_b) - pd.to_datetime(p.opened_a)).dt.total_seconds() / 86400
rng = np.random.default_rng(SEED)

sel = {}  # (repo,pr_a,pr_b) -> dict(pools=set, incl=prob)
def add(df, pool, prob):
    for k, pr in zip(zip(df.repo, df.pr_a, df.pr_b), prob):
        e = sel.setdefault(k, {'pools': set(), 'incl': {}})
        e['pools'].add(pool); e['incl'][pool] = float(pr)

def strat_sample(df, n, strata_cols, pool):
    """Proportional stratified sample; returns df with inclusion probabilities."""
    if len(df) <= n:
        add(df, pool, np.ones(len(df))); return df
    g = df.groupby(strata_cols, dropna=False)
    sizes = g.size()
    alloc = (sizes / sizes.sum() * n).round().astype(int).clip(lower=1)
    alloc = alloc.clip(upper=sizes)
    out = []
    for key, sub in g:
        k = int(alloc.loc[key])
        pick = sub.iloc[rng.permutation(len(sub))[:k]]
        add(pick, pool, np.full(k, k / len(sub)))
        out.append(pick)
    return pd.concat(out)

# Frame A: both-merged + shared file, opened <= 7 days apart, cap 50/repo
A = p[p.bm_shared & (p.gap_days.abs() <= 7)]
parts = []
for repo, sub in A.groupby('repo'):
    if len(sub) > 50:
        pick = sub.iloc[rng.permutation(len(sub))[:50]]; add(pick, 'frameA', np.full(50, 50 / len(sub)))
    else:
        pick = sub; add(pick, 'frameA', np.ones(len(sub)))
    parts.append(pick)
frameA = pd.concat(parts)

dup = strat_sample(p[p.dup_title], 1500, ['both_merged', 'merged_b', 'mega', 'cross_agent'], 'dup')
rev = p[p.revert]; add(rev, 'revert', np.ones(len(rev)))
cross = p[p.cross_agent & (p.bm_shared | p.dup_title | p.revert)]; add(cross, 'cross_agent', np.ones(len(cross)))
nm = strat_sample(p[p.near_miss], 2000, ['both_merged', 'mega', 'cross_agent'], 'near_miss')
ctrl_pool = p[~p.conflict & ~p.dup_title & ~p.revert & ~p.near_miss]
ctrl = strat_sample(ctrl_pool, 2000, ['both_merged', 'mega', 'cross_agent'], 'control')

# assemble
keys = list(sel.keys())
frame = p.set_index(['repo', 'pr_a', 'pr_b']).loc[keys].reset_index()
frame['pools'] = ['|'.join(sorted(sel[k]['pools'])) for k in keys]
# union inclusion probability across pools (independent draws)
frame['incl_prob'] = [1 - np.prod([1 - v for v in sel[k]['incl'].values()]) for k in keys]

# judge inputs
title = dict(zip(zip(txt.repo, txt.pr), txt.title.fillna('')))
body = dict(zip(zip(txt.repo, txt.pr), txt.body.fillna('')))
files = defaultdict(list)
for r, pr, f in zip(pf.repo.values, pf.pr.values, pf.filepath.values): files[(r, pr)].append(f)
for side in ['a', 'b']:
    frame[f'title_{side}'] = [title.get((r, x), '') for r, x in zip(frame.repo, frame[f'pr_{side}'])]
    frame[f'body_{side}'] = [body.get((r, x), '') for r, x in zip(frame.repo, frame[f'pr_{side}'])]
    frame[f'files_{side}'] = ['\n'.join(sorted(files[(r, x)])[:80]) for r, x in zip(frame.repo, frame[f'pr_{side}'])]
    frame[f'n_files_{side}'] = [len(files[(r, x)]) for r, x in zip(frame.repo, frame[f'pr_{side}'])]
shared = []
for r, a, b in zip(frame.repo, frame.pr_a, frame.pr_b):
    s = sorted(set(files[(r, a)]) & set(files[(r, b)])); shared.append('\n'.join(s[:80]))
frame['files_shared'] = shared
frame['has_diff'] = False   # no per-PR patch table locally; H/B and I labels are provisional (demotion rule)
frame = frame.sample(frac=1, random_state=SEED).reset_index(drop=True)
frame['frame_id'] = range(len(frame))
frame.to_csv(f'{O}/judging_frame.csv.gz', index=False)

# calibration sheet: 100 pairs, roughly 30 frameA / 30 dup / 10 revert / 15 near-miss / 15 control, cross-agent over-sampled
def take(mask, n):
    sub = frame[mask]; return sub.iloc[rng.permutation(len(sub))[:n]]
cal = pd.concat([take(frame.pools.str.contains('frameA'), 30), take(frame.pools.str.contains('dup'), 30),
                 take(frame.pools.str.contains('revert'), 10), take(frame.pools.str.contains('near_miss'), 15),
                 take(frame.pools.str.contains('control'), 15)]).drop_duplicates('frame_id').head(100)
cols = ['frame_id', 'repo', 'pr_a', 'pr_b', 'agent_a', 'agent_b', 'title_a', 'body_a', 'files_a', 'title_b', 'body_b', 'files_b', 'files_shared']
cal = cal[cols].copy()
for c in ['label_category (D/C/H/B/I/none)', 'confidence (1-3)', 'evidence', 'annotator']: cal[c] = ''
cal.to_csv(f'{O}/calibration_100.csv', index=False)

summary = {'frame_size': len(frame), 'by_pool': {k: int(frame.pools.str.contains(k).sum()) for k in ['frameA', 'dup', 'revert', 'cross_agent', 'near_miss', 'control']},
           'frameA_before_cap': int(len(A)), 'frameA_after_cap': int(len(frameA)), 'dup_pool': int(p.dup_title.sum()), 'near_miss_pool': int(p.near_miss.sum()),
           'control_pool': int(len(ctrl_pool)), 'cross_agent_candidates': int(len(cross)), 'seed': SEED}
json.dump(summary, open(f'{O}/frame_summary.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))
