#!/usr/bin/env python3
"""T2 ground truth and trivial resolvers (Paper 1 §4.2).

For every pair with status=='conflict' in t2_replay_results.csv:
  1. re-run `git merge-tree --write-tree A B` to get the conflicted tree; read each
     conflicted file with markers; parse hunks (<<<<<<< ours / ======= / >>>>>>> theirs,
     optional |||||||| base with diff3 style not used here).
  2. reconciliation = first commit on the default branch (first-parent) whose committer
     date >= max(closed_a, closed_b); read the file at that commit.
  3. locate each hunk's region in the reconciled file via context anchors (up to 3 lines
     before/after); compare take-ours / take-theirs / union(ours+theirs) / union(theirs+ours)
     against that region: exact match and normalized similarity (difflib ratio on
     whitespace-normalized text).
Outputs:
  t2_hunks.csv.gz        one row per hunk with scores per resolver and 'locatable' flag
  t2_hunk_bundle.jsonl   ours/theirs/context per hunk for an LLM resolver pilot
"""
import os, re, json, subprocess, difflib, csv, gzip
import pandas as pd

HERE = os.environ.get('MG_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
WORK = os.path.join(HERE, 'mergegym', 't2_work')
RES = os.path.join(HERE, 'mergegym', 'results', 't2_replay_results.csv')
CAND = os.path.join(HERE, 'mergegym/results/t2_candidates.csv.gz')
OUT = os.path.join(HERE, 'mergegym', 'results', 't2_hunks.csv.gz')
BUNDLE = os.path.join(HERE, 'mergegym', 'results', 't2_hunk_bundle.jsonl.gz')
CTX = 3
FIELDS = ['repo','pr_a','pr_b','path','hunk','kind','cross_agent','recon_commit','n_ours','n_theirs','recon_file_exists','locatable','truth_n',
          'ours_exact','ours_sim','theirs_exact','theirs_sim','union_ot_exact','union_ot_sim','union_to_exact','union_to_sim']

def git(d, *args, timeout=300):
    p = subprocess.run(['git'] + list(args), cwd=d, capture_output=True, text=True, timeout=timeout, errors='replace')
    return p.returncode, p.stdout

def default_branch(d):
    rc, out = git(d, 'symbolic-ref', 'HEAD')
    return out.strip() if rc == 0 else 'HEAD'

def recon_commit(d, ts):
    rc, out = git(d, 'rev-list', '--first-parent', '--reverse', f'--after={ts}', default_branch(d))
    lines = out.split()
    return lines[0] if lines else ''

def parse_hunks(text):
    """Return list of (pre_ctx, ours, theirs, post_ctx) from a file with conflict markers."""
    lines = text.split('\n'); hunks = []; i = 0; clean_so_far = []
    while i < len(lines):
        if lines[i].startswith('<<<<<<< '):
            j = i + 1; ours = []
            while j < len(lines) and not lines[j].startswith('=======') and not lines[j].startswith('||||||| '):
                ours.append(lines[j]); j += 1
            if j < len(lines) and lines[j].startswith('||||||| '):     # skip base section if present
                while j < len(lines) and not lines[j].startswith('======='): j += 1
            j += 1; theirs = []
            while j < len(lines) and not lines[j].startswith('>>>>>>> '):
                theirs.append(lines[j]); j += 1
            # post context: next CTX non-marker lines
            post = []; k = j + 1
            while k < len(lines) and len(post) < CTX and not lines[k].startswith('<<<<<<< '):
                post.append(lines[k]); k += 1
            hunks.append((clean_so_far[-CTX:], ours, theirs, post))
            i = j + 1
        else:
            clean_so_far.append(lines[i]); i += 1
    return hunks

def norm(s): return re.sub(r'\s+', ' ', s).strip()

def find_seq(hay, needle, start=0):
    if not needle: return -1
    n = len(needle)
    for i in range(start, len(hay) - n + 1):
        if hay[i:i + n] == needle: return i
    return -1

def locate(recon_lines, pre, post):
    """Return (lo, hi) slice of recon_lines between the anchors, or None."""
    pre = [l for l in pre]; post = [l for l in post]
    if pre:
        a = find_seq(recon_lines, pre)
        if a < 0: return None
        lo = a + len(pre)
    else:
        lo = 0
    if post:
        b = find_seq(recon_lines, post, lo)
        if b < 0: return None
        hi = b
    else:
        hi = len(recon_lines)
    return lo, hi

def score(cand, truth):
    c, t = norm('\n'.join(cand)), norm('\n'.join(truth))
    exact = int(c == t)
    if not (c or t): return exact, 1.0
    if len(c) > 20000 or len(t) > 20000:      # cap quadratic cost on huge generated files
        sim = difflib.SequenceMatcher(None, c[:20000], t[:20000]).quick_ratio()
    else:
        sim = difflib.SequenceMatcher(None, c, t).ratio()
    return exact, round(sim, 4)

def main():
    import sys
    r = pd.read_csv(RES); r = r[r.status == 'conflict']
    cand = pd.read_csv(CAND)[['repo', 'pr_a', 'pr_b', 'closed_a', 'closed_b', 'cross_agent', 'sa', 'sb']]
    r = r.merge(cand, on=['repo', 'pr_a', 'pr_b'], suffixes=('', '_c'))
    if '--core' in sys.argv: r = r[(r.sa <= 1000) & (r.sb <= 1000)]
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT); done = set(zip(prev.repo, prev.pr_a, prev.pr_b))
    r = r[[(a, b, c) not in done for a, b, c in zip(r.repo, r.pr_a, r.pr_b)]]
    rows = []; bundle = open(BUNDLE, 'a')
    for _, x in r.iterrows():
        pair_rows = []
        d = os.path.join(WORK, x.repo.replace('/', '__') + '.git')
        rc, out = git(d, 'merge-tree', '--write-tree', x.sha_a, x.sha_b, timeout=1200)
        tree = out.split('\n', 1)[0].strip()
        ts = max(str(x.closed_a), str(x.closed_b))
        rc_commit = recon_commit(d, ts)
        for path in str(x.conflict_files).split('|'):
            if not path: continue
            rc1, merged = git(d, 'show', f'{tree}:{path}')
            if rc1 != 0:                     # e.g. modify/delete conflict, binary
                row = dict(repo=x.repo, pr_a=x.pr_a, pr_b=x.pr_b, path=path, hunk=-1, kind='non_content_conflict',
                                 locatable=0, cross_agent=x.cross_agent); rows.append(row); pair_rows.append(row); continue
            hunks = parse_hunks(merged)
            recon = None
            if rc_commit:
                rc2, rtxt = git(d, 'show', f'{rc_commit}:{path}')
                recon = rtxt.split('\n') if rc2 == 0 else None
            for hi, (pre, ours, theirs, post) in enumerate(hunks):
                row = dict(repo=x.repo, pr_a=x.pr_a, pr_b=x.pr_b, path=path, hunk=hi, kind='content',
                           cross_agent=x.cross_agent, recon_commit=rc_commit, n_ours=len(ours), n_theirs=len(theirs),
                           recon_file_exists=int(recon is not None), locatable=0)
                loc = locate(recon, pre, post) if recon is not None else None
                if loc is not None:
                    truth = recon[loc[0]:loc[1]]; row['locatable'] = 1
                    for name, candl in [('ours', ours), ('theirs', theirs), ('union_ot', ours + theirs), ('union_to', theirs + ours)]:
                        e, s = score(candl, truth); row[f'{name}_exact'] = e; row[f'{name}_sim'] = s
                    row['truth_n'] = len(truth)
                    bundle.write(json.dumps(dict(repo=x.repo, pr_a=int(x.pr_a), pr_b=int(x.pr_b), path=path, hunk=hi,
                                                 pre=pre, ours=ours, theirs=theirs, post=post, truth=truth)) + '\n')
                rows.append(row); pair_rows.append(row)
        if pair_rows or True:
            pdf = pd.DataFrame(pair_rows if pair_rows else [dict(repo=x.repo, pr_a=x.pr_a, pr_b=x.pr_b, path='', hunk=-1, kind='no_hunks_parsed', locatable=0, cross_agent=x.cross_agent)])
            pdf.reindex(columns=FIELDS).to_csv(OUT, mode='a', index=False, header=not os.path.exists(OUT), compression='gzip')
            bundle.flush()
    bundle.close()
    df = pd.read_csv(OUT)
    c = df[df.kind == 'content']; L = c[c.locatable == 1]
    print(f'conflicting pairs: {r[["repo","pr_a","pr_b"]].drop_duplicates().shape[0]}; files: {df[["repo","pr_a","pr_b","path"]].drop_duplicates().shape[0]}; '
          f'content hunks: {len(c)}; non-content conflicts: {(df.kind!="content").sum()}; recon file exists: {c.recon_file_exists.mean():.3f}; locatable hunks: {len(L)} ({len(L)/max(len(c),1):.3f})')
    for name in ['ours', 'theirs', 'union_ot', 'union_to']:
        print(f'  {name:9} exact {L[name+"_exact"].mean():.3f}  sim>=0.9 {(L[name+"_sim"]>=0.9).mean():.3f}  mean sim {L[name+"_sim"].mean():.3f}')
    best = L[['ours_exact', 'theirs_exact', 'union_ot_exact', 'union_to_exact']].max(axis=1)
    print(f'  best-of-trivial exact {best.mean():.3f}')

if __name__ == '__main__':
    main()
