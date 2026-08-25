#!/usr/bin/env python3
"""t3_simulator.py -- MergeGym T3 scheduling replay (paper §4.3, Appendix D).

Episode = one repository's PRs that appear in >=1 labeled co-active pair; arrival = opened,
duration = closed - opened (right-censored PRs capped at the repo median over non-censored
PRs); overlap = half-open intervals [s, s+d) intersect with positive length; finishes are
processed before arrivals at identical timestamps; conflict oracle = exact changed-file-set
intersection over ALL PR pairs (including pairs never co-active in reality).

Policies: observed | fifo | oracle | gated(score_fn, tau)
  gated(...) is the mechanism behind the paper's predictor gate: a PR starts unless a running
  PR has score >= tau against it; blocked PRs join a FIFO queue re-scanned on every finish,
  skip-ahead allowed, PRs started during a scan count as running for later PRs in the scan.
  Pass score_fn=(lambda pr_x, pr_y: 1.0 if files overlap else 0.0) and tau=1.0 to get the oracle.

Usage:
  python3 t3_simulator.py --check      # rerun observed/fifo/oracle on all 97 episodes and
                                       # compare with derived/paper_outputs/t3_episode_results.csv
"""
import os, sys, json, argparse
from collections import defaultdict
import numpy as np, pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
CENSOR_TS = pd.Timestamp('2025-07-30 23:20:55+00:00')

def load_episodes():
    pairs = pd.read_csv(f'{D}/derived/pairs_labeled.csv.gz')
    g = pairs.groupby('repo').agg(n=('conflict', 'size'), k=('conflict', 'sum'))
    repos = g[(g.n >= 100) & (g.k >= 20)].index.tolist()
    pf = pd.read_csv(f'{D}/data/pr_files.csv.gz')
    files = defaultdict(set)
    for r, p, f in zip(pf.repo.values, pf.pr.values, pf.filepath.values): files[(r, p)].add(f)
    eps = {}
    for repo in repos:
        sub = pairs[pairs.repo == repo]
        a = sub[['pr_a', 'opened_a', 'closed_a']].rename(columns={'pr_a': 'pr', 'opened_a': 'opened', 'closed_a': 'closed'})
        b = sub[['pr_b', 'opened_b', 'closed_b']].rename(columns={'pr_b': 'pr', 'opened_b': 'opened', 'closed_b': 'closed'})
        prs = pd.concat([a, b]).drop_duplicates('pr').copy()
        prs['opened'] = pd.to_datetime(prs.opened); prs['closed'] = pd.to_datetime(prs.closed)
        prs['dur'] = (prs.closed - prs.opened).dt.total_seconds()
        prs['censored'] = prs.closed == CENSOR_TS
        med = prs.loc[~prs.censored, 'dur'].median()
        prs.loc[prs.censored, 'dur'] = np.minimum(prs.loc[prs.censored, 'dur'], med)
        prs['arr'] = (prs.opened - prs.opened.min()).dt.total_seconds()
        prs = prs.sort_values(['arr', 'pr']).reset_index(drop=True)
        eps[repo] = dict(prs=prs, files={p: files[(repo, p)] for p in prs.pr}, labeled_conf=int(sub.conflict.sum()))
    return eps

def overlaps(s1, e1, s2, e2):
    return max(s1, s2) < min(e1, e2)

def count_conflicts(starts, durs, prs, files):
    """Number of file-overlapping PR pairs whose scheduled intervals overlap; also the set of pairs."""
    order = np.argsort(starts, kind='stable'); active = []; conf = set()
    for i in order:
        s, e = starts[i], starts[i] + durs[i]
        active = [j for j in active if starts[j] + durs[j] > s]
        fi = files[prs[i]]
        for j in active:
            if fi & files[prs[j]]: conf.add((min(i, j), max(i, j)))
        active.append(i)
    return conf

def schedule(ep, policy, score_fn=None, tau=None):
    prs = ep['prs']; n = len(prs); arr = prs.arr.values; dur = prs.dur.values; ids = prs.pr.values; files = ep['files']
    starts = np.full(n, np.nan)
    if policy == 'observed':
        starts[:] = arr
    elif policy == 'fifo':
        t = -np.inf
        for i in range(n):
            starts[i] = max(arr[i], t); t = starts[i] + dur[i]
    else:  # gated
        if policy == 'oracle':
            score_fn = lambda x, y: 1.0 if files[x] & files[y] else 0.0; tau = 1.0
        blocked = lambda i, running: any(score_fn(ids[i], ids[j]) >= tau for j in running)
        running = []; queue = []; i = 0
        import heapq
        finish = []  # (end_time, idx)
        def scan(t):
            nonlocal queue
            remaining = []
            for q in queue:
                if not blocked(q, running):
                    starts[q] = t; running.append(q); heapq.heappush(finish, (t + dur[q], q))
                else:
                    remaining.append(q)
            queue = remaining
        while i < n or finish:
            next_arr = arr[i] if i < n else np.inf
            next_fin = finish[0][0] if finish else np.inf
            if next_fin <= next_arr:                       # finishes before arrivals at ties
                t, j = heapq.heappop(finish); running.remove(j); scan(t)
            else:
                t = arr[i]
                if not blocked(i, running):
                    starts[i] = t; running.append(i); heapq.heappush(finish, (t + dur[i], i))
                else:
                    queue.append(i)
                i += 1
    return starts

def evaluate(ep, policy, obs=None, **kw):
    prs = ep['prs']; ids = prs.pr.values; files = ep['files']; dur = prs.dur.values; arr = prs.arr.values
    starts = schedule(ep, policy, **kw)
    conf = count_conflicts(starts, dur, ids, files)
    if obs is None: obs = count_conflicts(arr, dur, ids, files)
    new = conf - obs
    mk = (starts + dur).max() - arr.min(); mk_obs = (arr + dur).max() - arr.min()
    return dict(n_prs=len(prs), conflicts=len(conf), obs_conflicts=len(obs),
                avoided_pct=100 * (len(obs) - len(conf)) / len(obs) if obs else np.nan,
                new_conflicts=len(new), added_delay_days=(starts - arr).sum() / 86400,
                delay_days_per_pr=(starts - arr).sum() / 86400 / len(prs),
                makespan_days=mk / 86400, makespan_inflation_pct=100 * (mk / mk_obs - 1)), obs

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true'); a = ap.parse_args()
    eps = load_episodes(); print(f'{len(eps)} episodes, {sum(len(e["prs"]) for e in eps.values())} PRs')
    ref = pd.read_csv(f'{D}/derived/paper_outputs/t3_episode_results.csv') if a.check else None
    rows = []; uncapped_ok = 0
    for repo, ep in eps.items():
        obs = None
        for pol in ['observed', 'fifo', 'oracle']:
            m, obs = evaluate(ep, pol, obs); m.update(repo=repo, policy=pol); rows.append(m)
        # machinery check: uncapped durations reproduce the labeled conflicting-pair count
        prs = ep['prs']; d0 = (prs.closed - prs.opened).dt.total_seconds().values
        uncapped_ok += len(count_conflicts(prs.arr.values, d0, prs.pr.values, ep['files'])) == ep['labeled_conf']
    out = pd.DataFrame(rows); out.to_csv('t3_sim_results.csv', index=False)
    print('uncapped == labeled in', uncapped_ok, '/', len(eps), 'episodes')
    for pol in ['fifo', 'oracle']:
        s = out[out.policy == pol]
        print(f'{pol}: median avoided {s.avoided_pct.median():.1f}%  median inflation {s.makespan_inflation_pct.median():.1f}%  median delay/PR {s.delay_days_per_pr.median():.1f}  pooled new {s.new_conflicts.sum()}')
    print('observed capped total conflicts', int(out[out.policy == "observed"].conflicts.sum()))
    if ref is not None:
        m = out.merge(ref[ref.policy.isin(['observed', 'fifo', 'oracle'])], on=['repo', 'policy'], suffixes=('', '_ref'))
        for c in ['conflicts', 'obs_conflicts', 'new_conflicts', 'makespan_inflation_pct', 'delay_days_per_pr']:
            diff = (m[c] - m[c + '_ref']).abs()
            print(f'  {c:24} max|diff| = {diff.max():.4f}   rows mismatching (>1e-3): {(diff > 1e-3).sum()}/{len(m)}')

if __name__ == '__main__':
    main()
