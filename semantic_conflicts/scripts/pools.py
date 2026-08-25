import os
#!/usr/bin/env python3
"""Recompute the five deterministic candidate pools (Paper 2 §4.2) and write
pool membership flags for all 577,045 labeled pairs -> derived/pool_flags.csv.gz
Prints each pool size next to the number printed in the paper.
"""
import re, os
import pandas as pd, numpy as np
from collections import defaultdict

D = os.environ.get('MG_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
pairs = pd.read_csv(f'{D}/derived/pairs_labeled.csv.gz')
txt = pd.read_csv(f'{D}/data/pr_texts.csv.gz')
pf = pd.read_csv(f'{D}/data/pr_files.csv.gz')

title = {(r, p): (t if isinstance(t, str) else '') for r, p, t in zip(txt.repo, txt.pr, txt.title)}
merged = {(r, p): bool(m) for r, p, m in zip(txt.repo, txt.pr, txt.merged)}
files = defaultdict(set)
for r, p, f in zip(pf.repo.values, pf.pr.values, pf.filepath.values):
    files[(r, p)].add(f)

# ---- duplicate-title (D) ----
def norm(t):
    t = t.lower()
    t = re.sub(r'\[[^\]]*\]', ' ', t)
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()
def is_dup(ta, tb):
    na, nb = norm(ta), norm(tb)
    A, B = set(na.split()), set(nb.split())
    if len(A) < 3 or len(B) < 3: return False
    j = len(A & B) / len(A | B) if (A | B) else 0.0
    if j >= 0.5: return True
    if len(na) >= 15 and len(nb) >= 15 and (na in nb or nb in na): return True
    return False

# ---- revert (C) ----
REV = re.compile(r'\b(revert|undo|rollback)\b')
FIX = {'fix','fixes','fixed','fixing','bug','bugs','bugfix','hotfix','patch','patches','patched'}

def d2set(fs, variant):
    out = set()
    for f in fs:
        parts = f.split('/')
        if len(parts) < 2: continue            # repo-root files excluded
        if variant == 'dir2':                   # first two path components (dir or file)
            out.add('/'.join(parts[:2]))
        elif variant == 'dirs_only':            # first two *directory* components; needs >=2 dirs
            if len(parts) >= 3: out.add('/'.join(parts[:2]))
    return out

ta = [title.get((r, p), '') for r, p in zip(pairs.repo, pairs.pr_a)]
tb = [title.get((r, p), '') for r, p in zip(pairs.repo, pairs.pr_b)]
pairs['dup_title'] = [is_dup(a, b) for a, b in zip(ta, tb)]
pairs['revert'] = [bool(REV.search(a.lower()) or REV.search(b.lower())) for a, b in zip(ta, tb)]
pairs['fix_in_flight'] = [c and bool(set(norm(b).split()) & FIX) for c, b in zip(pairs.conflict, tb)]
pairs['merged_a'] = [merged.get((r, p), False) for r, p in zip(pairs.repo, pairs.pr_a)]
pairs['merged_b'] = [merged.get((r, p), False) for r, p in zip(pairs.repo, pairs.pr_b)]
pairs['both_merged'] = pairs.merged_a & pairs.merged_b
pairs['bm_shared'] = pairs.both_merged & pairs.conflict
for v in ['dir2', 'dirs_only']:
    pairs[f'near_miss_{v}'] = [(not c) and bool(d2set(files[(r, a)], v) & d2set(files[(r, b)], v))
                               for r, a, b, c in zip(pairs.repo, pairs.pr_a, pairs.pr_b, pairs.conflict)]

print('pool                     mine      paper')
print(f"dup_title             {pairs.dup_title.sum():>8}    6,945   cross-agent {int((pairs.dup_title&pairs.cross_agent).sum())} (7)")
print(f"revert                {pairs.revert.sum():>8}      278   shared-file {int((pairs.revert&pairs.conflict).sum())} (35)  cross {int((pairs.revert&pairs.cross_agent).sum())} (2)")
print(f"fix_in_flight         {pairs.fix_in_flight.sum():>8}    7,872")
for v in ['dir2', 'dirs_only']:
    print(f"near_miss[{v:9}]  {pairs[f'near_miss_{v}'].sum():>8}  176,756")
print(f"both_merged+shared    {pairs.bm_shared.sum():>8}    6,409   cross {int((pairs.bm_shared&pairs.cross_agent).sum())} (90)")
d = pairs[pairs.dup_title]
print(f"dup outcomes: both {d.both_merged.mean():.4f} (0.4681)  one {((d.merged_a^d.merged_b)).mean():.4f} (0.3866)  none {((~d.merged_a)&(~d.merged_b)).mean():.4f} (0.1453)  no-shared-file {(~d.conflict).mean():.4f} (0.7955)")
pairs.to_csv(f'{D}/semantic_conflicts/results/pool_flags.csv.gz', index=False)
print('wrote derived/pool_flags.csv.gz')
