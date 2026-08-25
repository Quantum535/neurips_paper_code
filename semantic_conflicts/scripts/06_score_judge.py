#!/usr/bin/env python3
"""06_score_judge.py -- turn judge labels + human calibration into the numbers for
Paper 2's pending slots.

Inputs : derived/judge/judge_labels.csv          (from 05_judge_p2_claude_code.py)
         derived/judge/calibration_100.csv       (filled in by annotators; one row per pair
                                                  per annotator -> duplicate rows with different
                                                  'annotator' values, or a single adjudicated row)
Output : derived/judge/judge_report.json and a printed summary.
"""
import os, json, math
import numpy as np, pandas as pd

HERE = os.environ.get('MG_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
J = os.path.join(HERE, 'semantic_conflicts', 'results')
MEGA = {'mochilang/mochi', 'MontrealAI/AGI-Alpha-Agent-v0'}
TARGET = {'frameA': {'H', 'B'}, 'dup': {'D'}, 'revert': {'C'}, 'near_miss': {'I'}, 'control': set(), 'cross_agent': {'D', 'C', 'H', 'I'}}

def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'), float('nan'), float('nan'))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, c - h, c + h)

def kappa(a, b):
    a, b = pd.Series(a), pd.Series(b); cats = sorted(set(a) | set(b))
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float('nan')

lab = pd.read_csv(f'{J}/judge_labels.csv')
frame = pd.read_csv(f'{J}/judging_frame.csv.gz')
lab = lab.merge(frame[['frame_id', 'incl_prob', 'conflict', 'cross_agent', 'both_merged']], on='frame_id')
lab['mega'] = lab.repo.isin(MEGA)
rep = {'n_judged': int(len(lab)), 'label_counts': lab.category.value_counts().to_dict(),
       'provisional_counts': lab.provisional_category.value_counts().to_dict()}

# --- per-pool judge label distribution (candidate precision under the judge) ---
pool_prec = {}
for pool, tgt in TARGET.items():
    sub = lab[lab.pools.str.contains(pool)]
    if not len(sub): continue
    col = 'provisional_category' if pool in ('frameA', 'near_miss') else 'category'
    hit = sub[col].isin(tgt).sum() if tgt else (sub['category'] != 'none').sum()
    p, lo, hi = wilson(int(hit), len(sub))
    pool_prec[pool] = dict(n=int(len(sub)), hits=int(hit), precision=round(p, 4), ci=[round(lo, 4), round(hi, 4)],
                           note='provisional (no diffs)' if pool in ('frameA', 'near_miss') else ('any non-none' if not tgt else 'official'))
rep['pool_precision'] = pool_prec
dbm = lab[lab.pools.str.contains('dup') & lab.both_merged]
if len(dbm):
    k = int((dbm.category == 'D').sum()); p_, lo, hi = wilson(k, len(dbm))
    rep['dup_both_merged_D_rate'] = dict(n=int(len(dbm)), D=k, rate=round(p_, 4), ci=[round(lo, 4), round(hi, 4)],
                                          implied_landed_redundancy_pct=round(46.81 * p_, 2))

# --- human calibration: kappa and gold precision ---
calp = f'{J}/calibration_100.csv'
cal = pd.read_csv(calp)
lc = [c for c in cal.columns if c.startswith('label_category')][0]
cal = cal.rename(columns={lc: 'gold'})
cal = cal[cal.gold.astype(str).str.strip().ne('') & cal.gold.notna()]
if len(cal):
    cal['gold'] = cal.gold.astype(str).str.strip()
    ann = cal.annotator.astype(str).str.strip().replace('nan', '')
    if ann.nunique() >= 2:      # two annotators: inter-annotator kappa on shared pairs
        piv = cal.pivot_table(index='frame_id', columns='annotator', values='gold', aggfunc='first').dropna()
        c1, c2 = piv.columns[:2]
        rep['human_kappa'] = dict(annotators=[str(c1), str(c2)], n=int(len(piv)), kappa=round(kappa(piv[c1], piv[c2]), 4),
                                  agreement=round(float((piv[c1] == piv[c2]).mean()), 4))
        gold = piv[c1].where(piv[c1] == piv[c2])          # adjudicate later; agreed pairs only for now
        gold = gold.dropna().rename('gold').reset_index()
    else:
        gold = cal[['frame_id', 'gold']].drop_duplicates('frame_id')
    m = gold.merge(lab[['frame_id', 'category', 'provisional_category', 'pools']], on='frame_id')
    if len(m):
        rep['judge_vs_gold'] = dict(n=int(len(m)), kappa_official=round(kappa(m.gold.where(m.gold.isin(['D', 'C', 'none']), 'none'), m.category), 4),
                                    kappa_provisional=round(kappa(m.gold, m.provisional_category), 4),
                                    agreement_provisional=round(float((m.gold == m.provisional_category).mean()), 4))
        gp = {}
        for pool, tgt in TARGET.items():
            sub = m[m.pools.str.contains(pool)]
            if not len(sub) or not tgt: continue
            k = int(sub.gold.isin(tgt).sum()); p, lo, hi = wilson(k, len(sub))
            gp[pool] = dict(n=int(len(sub)), gold_precision=round(p, 4), ci=[round(lo, 4), round(hi, 4)])
        rep['gold_pool_precision'] = gp
else:
    rep['human_kappa'] = 'calibration_100.csv has no labels yet'

# --- inverse-inclusion-probability prevalence over the 577,045-pair population ---
def ipw(df, cat_set, col='category'):
    w = 1.0 / df.incl_prob
    return float((w * df[col].isin(cat_set)).sum() / w.sum())
prev = {}
for name, df in [('all', lab), ('excl_mega', lab[~lab.mega])]:
    prev[name] = {'D': round(ipw(df, {'D'}), 4), 'C': round(ipw(df, {'C'}), 4),
                  'H_provisional_among_overlap': round(ipw(df[df.conflict], {'H'}, 'provisional_category'), 4) if df.conflict.any() else None,
                  'I_provisional_among_disjoint': round(ipw(df[~df.conflict], {'I'}, 'provisional_category'), 4) if (~df.conflict).any() else None,
                  'invisible_lower_bound': round(ipw(df[~df.conflict], {'D', 'C'}), 4)}
rep['ipw_prevalence'] = prev
rep['note'] = ('Prevalence is judge-weighted (silver). Replace with gold-corrected estimates once the 300-pair validation lands. '
               'H/B/I are provisional in this run because no per-PR diff content was available (demotion rule).')
json.dump(rep, open(f'{J}/judge_report.json', 'w'), indent=1)
print(json.dumps(rep, indent=1))
