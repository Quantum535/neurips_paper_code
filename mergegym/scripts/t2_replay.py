#!/usr/bin/env python3
"""T2 replay: re-acquire PR head SHAs via git (refs/pull/N/head) and replay a
three-way merge with `git merge-tree --write-tree` to detect textual conflicts.

Resumable: per-pair results are appended to OUT; repos already fully done are skipped.
Usage: python3 t2_replay.py [--limit-repos N] [--subset core|full]
"""
import argparse, csv, os, subprocess, sys, time, json
import pandas as pd

ROOT = os.environ.get('MG_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
CAND = os.path.join(ROOT, 'mergegym/results/t2_candidates.csv.gz')
WORK = os.path.join(ROOT, 'mergegym', 't2_work')
OUT = os.path.join(ROOT, 'mergegym', 'results', 't2_replay_results.csv')
LOG = os.path.join(ROOT, 'mergegym', 't2_replay.log')
FIELDS = ['repo','pr_a','pr_b','status','sha_a','sha_b','merge_base','conflict',
          'n_conflict_files','conflict_files','n_shared_src','cross_agent','secs','note']

def log(msg):
    with open(LOG,'a') as f: f.write(time.strftime('%H:%M:%S ')+msg+'\n')

def run(cmd, cwd=None, timeout=600):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr

def ensure_clone(repo):
    d = os.path.join(WORK, repo.replace('/','__')+'.git')
    if os.path.isdir(d) and os.path.exists(os.path.join(d,'HEAD')):
        return d, 'cached'
    os.makedirs(WORK, exist_ok=True)
    rc, out, err = run(['git','clone','--bare','--quiet','--filter=blob:none',
                        f'https://github.com/{repo}', d], timeout=1800)
    if rc != 0:
        return None, err.strip().splitlines()[-1][:200] if err.strip() else 'clone failed'
    return d, 'cloned'

def fetch_pulls(d, prs):
    specs = [f'+refs/pull/{n}/head:refs/pull/{n}/head' for n in prs]
    rc, out, err = run(['git','fetch','--quiet','origin']+specs, cwd=d, timeout=1800)
    got = {}
    for n in prs:
        rc2, sha, _ = run(['git','rev-parse','--verify','--quiet',f'refs/pull/{n}/head'], cwd=d)
        got[n] = sha.strip() if rc2 == 0 else None
    # if batch fetch failed because one ref is missing, retry individually for the missing ones
    missing = [n for n in prs if got[n] is None]
    for n in missing:
        run(['git','fetch','--quiet','origin',f'+refs/pull/{n}/head:refs/pull/{n}/head'], cwd=d, timeout=600)
        rc2, sha, _ = run(['git','rev-parse','--verify','--quiet',f'refs/pull/{n}/head'], cwd=d)
        got[n] = sha.strip() if rc2 == 0 else None
    return got

def merge_tree(d, a, b):
    rc, base, _ = run(['git','merge-base',a,b], cwd=d)
    base = base.strip() if rc == 0 else ''
    if not base:
        return 'no_merge_base', base, None, []
    rc, out, err = run(['git','merge-tree','--write-tree','--name-only',a,b], cwd=d, timeout=1200)
    if rc == 0:
        return 'clean', base, False, []
    if rc == 1:
        lines = out.split('\n')
        files = []
        for ln in lines[1:]:
            if ln == '': break
            files.append(ln)
        return 'conflict', base, True, files
    return 'merge_error:'+(err.strip().splitlines()[-1][:120] if err.strip() else str(rc)), base, None, []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit-repos', type=int, default=0)
    ap.add_argument('--subset', default='core')
    args = ap.parse_args()
    t2 = pd.read_csv(CAND)
    if args.subset == 'core':
        t2 = t2[(t2.sa <= 1000) & (t2.sb <= 1000)]
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.repo, prev.pr_a, prev.pr_b))
    else:
        with open(OUT,'w',newline='') as f: csv.DictWriter(f, FIELDS).writeheader()
    t2 = t2[[ (r,a,b) not in done for r,a,b in zip(t2.repo,t2.pr_a,t2.pr_b)]]
    repos = t2.groupby('repo').size().sort_values(ascending=False).index.tolist()
    if args.limit_repos: repos = repos[:args.limit_repos]
    log(f'start: {len(t2)} pairs over {len(repos)} repos to do')
    for i, repo in enumerate(repos):
        sub = t2[t2.repo == repo]
        t0 = time.time()
        d, how = ensure_clone(repo)
        rows = []
        if d is None:
            for _, r in sub.iterrows():
                rows.append(dict(repo=repo, pr_a=r.pr_a, pr_b=r.pr_b, status='repo_unavailable',
                                 n_shared_src=r.n_shared_src, cross_agent=r.cross_agent, note=how))
        else:
            prs = sorted(set(sub.pr_a) | set(sub.pr_b))
            shas = fetch_pulls(d, prs)
            for _, r in sub.iterrows():
                t1 = time.time()
                a, b = shas.get(r.pr_a), shas.get(r.pr_b)
                row = dict(repo=repo, pr_a=r.pr_a, pr_b=r.pr_b, sha_a=a or '', sha_b=b or '',
                           n_shared_src=r.n_shared_src, cross_agent=r.cross_agent, note='')
                if not a or not b:
                    row.update(status='ref_missing', conflict='', n_conflict_files='', conflict_files='', merge_base='')
                else:
                    try:
                        st, base, conf, files = merge_tree(d, a, b)
                    except subprocess.TimeoutExpired:
                        st, base, conf, files = 'merge_timeout', '', None, []
                    row.update(status=st, merge_base=base, conflict='' if conf is None else conf,
                               n_conflict_files=len(files) if conf else (0 if conf is False else ''),
                               conflict_files='|'.join(files))
                row['secs'] = round(time.time()-t1, 2)
                rows.append(row)
        with open(OUT,'a',newline='') as f:
            w = csv.DictWriter(f, FIELDS); [w.writerow({k: r.get(k,'') for k in FIELDS}) for r in rows]
        st = pd.Series([r['status'] for r in rows]).value_counts().to_dict()
        log(f'[{i+1}/{len(repos)}] {repo} ({how}) pairs={len(rows)} {st} {time.time()-t0:.0f}s')
    log('done')

if __name__ == '__main__':
    main()
