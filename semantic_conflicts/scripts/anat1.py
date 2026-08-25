import pandas as pd, numpy as np, re, json, os
from collections import defaultdict, Counter

SP = os.environ.get('MG_DERIVED', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'derived'))
D = os.environ.get('MG_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

pairs = pd.read_csv(f'{SP}/pairs_labeled.csv.gz')
pf = pd.read_csv(f'{D}/data/pr_files.csv.gz')

# file sets per (repo, pr)
fs = defaultdict(set)
for r, p, f in zip(pf.repo.values, pf.pr.values, pf.filepath.values):
    fs[(r, p)].add(f)

conf = pairs[pairs.conflict].copy()
print('conflicting pairs:', len(conf))

# ---- classify shared files ----
def classify(path):
    b = path.rsplit('/',1)[-1].lower()
    p = path.lower()
    if '.github/workflows' in p: return 'ci_workflow'
    if b == 'package-lock.json' or b.endswith('.lock') or b in ('go.sum','go.mod','yarn.lock','gemfile.lock','cargo.lock','poetry.lock','composer.lock','pnpm-lock.yaml','uv.lock','bun.lockb'): return 'lockfile'
    if b.startswith('requirements') or b == 'pyproject.toml' or b in ('package.json','setup.py','setup.cfg','cargo.toml','gemfile','composer.json','build.gradle','pom.xml'): return 'manifest'
    if b.startswith('dockerfile') or b in ('docker-compose.yml','docker-compose.yaml'): return 'docker'
    if b.startswith('readme'): return 'readme'
    if b.startswith('changelog') or b.endswith('.md') or b.endswith('.rst') or b.endswith('.txt'): return 'other_docs'
    if b.startswith('.gitignore') or b.startswith('.env') or b.endswith('.yml') or b.endswith('.yaml') or b.endswith('.toml') or b.endswith('.ini') or b.endswith('.cfg') or b.endswith('.json'): return 'other_config'
    return 'source'

# strict category set from the task spec (lockfiles/manifests/CI/README/Dockerfile)
SPEC_CONFIG = {'ci_workflow','lockfile','manifest','docker','readme'}

n_shared = []
pure_spec_config = 0     # ALL shared files in spec config set
has_source_strict = 0    # >=1 shared file classified 'source' (excl docs & other config too)
has_nonspec = 0          # >=1 shared file NOT in spec config set (task's "source code" complement)
file_cat_counter = Counter()   # file-level (pair x file occurrences)
basename_counter = Counter()
fullpath_counter = Counter()
rows_extra = []
for r, a, b in zip(conf.repo.values, conf.pr_a.values, conf.pr_b.values):
    shared = fs[(r,a)] & fs[(r,b)]
    k = len(shared)
    n_shared.append(k)
    cats = [classify(x) for x in shared]
    for c in cats: file_cat_counter[c] += 1
    for x in shared:
        basename_counter[x.rsplit('/',1)[-1]] += 1
        fullpath_counter[x] += 1
    allspec = all(c in SPEC_CONFIG for c in cats)
    anysrc  = any(c=='source' for c in cats)
    anynon  = any(c not in SPEC_CONFIG for c in cats)
    pure_spec_config += allspec
    has_source_strict += anysrc
    has_nonspec += anynon
    rows_extra.append((k, anysrc, anynon, allspec))

conf['n_shared'] = [x[0] for x in rows_extra]
conf['has_source'] = [x[1] for x in rows_extra]

ns = np.array(n_shared)
sev = {
  'n_conflicting_pairs': len(conf),
  'shared_files_dist': {'1': int((ns==1).sum()), '2': int((ns==2).sum()), '3': int((ns==3).sum()),
                        '4-5': int(((ns>=4)&(ns<=5)).sum()), '6-10': int(((ns>=6)&(ns<=10)).sum()),
                        '11-50': int(((ns>=11)&(ns<=50)).sum()), '>50': int((ns>50).sum())},
  'mean': float(ns.mean()), 'median': float(np.median(ns)), 'p90': float(np.percentile(ns,90)),
  'p99': float(np.percentile(ns,99)), 'max': int(ns.max()),
  'pure_spec_config_pairs': pure_spec_config,
  'has_source_strict_pairs': has_source_strict,
  'has_nonspec_pairs': has_nonspec,
  'file_level_categories': dict(file_cat_counter),
  'total_shared_file_occurrences': int(sum(file_cat_counter.values())),
}
print(json.dumps(sev, indent=1))

# hot files
hot_base = basename_counter.most_common(25)
hot_full = fullpath_counter.most_common(25)
print('HOT BASENAMES', hot_base)
print('HOT FULLPATHS', hot_full)

conf[['repo','pr_a','pr_b','n_shared','has_source']].to_csv(f'{SP}/conf_extra.csv.gz', index=False)
json.dump({'sev':sev,'hot_base':hot_base,'hot_full':hot_full}, open(f'{SP}/part1.json','w'))
