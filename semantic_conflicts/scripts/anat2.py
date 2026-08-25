import os
import pandas as pd, numpy as np, json
SP = os.environ.get('MG_DERIVED', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'derived'))
D = os.environ.get('MG_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
pairs = pd.read_csv(f'{SP}/pairs_labeled.csv.gz')
for c in ['opened_a','closed_a','opened_b','closed_b']:
    pairs[c] = pd.to_datetime(pairs[c], utc=True)

# ---- 2. agent x agent matrix (unordered) ----
aa = pairs.agent_a.astype(str); ab = pairs.agent_b.astype(str)
pairs['ag1'] = aa.where(aa<=ab, ab); pairs['ag2'] = ab.where(aa<=ab, aa)
mat = pairs.groupby(['ag1','ag2']).agg(n=('conflict','size'), conf=('conflict','sum')).reset_index()
mat['rate'] = mat.conf/mat.n
print('AGENT MATRIX')
print(mat.to_string())

# ---- 3a. overlap duration ----
ov_start = np.maximum(pairs.opened_a.values, pairs.opened_b.values)
ov_end = np.minimum(pairs.closed_a.values, pairs.closed_b.values)
ov_h = (ov_end - ov_start) / np.timedelta64(1,'h')
pairs['ov_h'] = ov_h
print('negative overlap rows:', (ov_h<0).sum())
bins = [-1e9, 1, 6, 24, 168, 1e12]
labels = ['<1h','1-6h','6-24h','1-7d','>7d']
pairs['ovbin'] = pd.cut(pairs.ov_h, bins=bins, labels=labels)
dur = pairs.groupby('ovbin', observed=True).agg(n=('conflict','size'), conf=('conflict','sum'))
dur['rate']=dur.conf/dur.n
print('OVERLAP DURATION'); print(dur.to_string())

# ---- 3b. concurrency at overlap midpoint ----
# PR intervals from pairs file
ints = pd.concat([
    pairs[['repo','pr_a','opened_a','closed_a']].rename(columns={'pr_a':'pr','opened_a':'op','closed_a':'cl'}),
    pairs[['repo','pr_b','opened_b','closed_b']].rename(columns={'pr_b':'pr','opened_b':'op','closed_b':'cl'})
]).drop_duplicates(['repo','pr'])
print('distinct PRs in pairs:', len(ints))
mid = ov_start + (ov_end - ov_start)/2
pairs['mid'] = mid
concur = np.zeros(len(pairs), dtype=np.int64)
pairs['_idx'] = np.arange(len(pairs))
for repo, g in pairs.groupby('repo', sort=False):
    gi = ints[ints.repo==repo]
    ops = np.sort(gi.op.values.astype('datetime64[ns]'))
    cls = np.sort(gi.cl.values.astype('datetime64[ns]'))
    t = g['mid'].values.astype('datetime64[ns]')
    cnt = np.searchsorted(ops, t, side='right') - np.searchsorted(cls, t, side='left')
    concur[g._idx.values] = cnt
pairs['concur'] = concur
print('concurrency min/max:', concur.min(), concur.max())
cbins = [1.5, 2.5, 5.5, 10.5, 20.5, 50.5, 1e9]
clabels = ['2','3-5','6-10','11-20','21-50','>50']
pairs['cbin'] = pd.cut(pairs.concur, bins=cbins, labels=clabels)
cc = pairs.groupby('cbin', observed=True).agg(n=('conflict','size'), conf=('conflict','sum'))
cc['rate']=cc.conf/cc.n
print('CONCURRENCY'); print(cc.to_string())

# ---- 4. repo concentration ----
rc = pairs.groupby('repo').agg(n=('conflict','size'), conf=('conflict','sum')).sort_values('conf', ascending=False)
tot_conf = rc.conf.sum(); tot_pairs = rc.n.sum()
top10 = rc.head(10); top50 = rc.head(50)
print('total conflicts', tot_conf, 'repos with >=1 conflict:', (rc.conf>0).sum(), 'of', len(rc))
print('top10 conflicts', top10.conf.sum(), 'frac', top10.conf.sum()/tot_conf, 'their pairs frac', top10.n.sum()/tot_pairs)
print('top50 conflicts', top50.conf.sum(), 'frac', top50.conf.sum()/tot_conf, 'their pairs frac', top50.n.sum()/tot_pairs)
print(top10.assign(rate=top10.conf/top10.n).to_string())

rs = pd.read_csv(f'{D}/data/repo_stats.csv')
rc2 = rc.reset_index().merge(rs, on='repo', how='left')
print('repos missing stats:', rc2.stars.isna().sum())
sbins = [-1, 10, 100, 1000, 10000, 1e9]
slabels = ['0-10','11-100','101-1k','1k-10k','>10k']
rc2['sbin'] = pd.cut(rc2.stars, bins=sbins, labels=slabels)
sb = rc2.groupby('sbin', observed=True).agg(n_repos=('repo','size'), pairs=('n','sum'), conf=('conf','sum'))
sb['rate']=sb.conf/sb.pairs
print('STARS'); print(sb.to_string())
lb = rc2.groupby('language').agg(n_repos=('repo','size'), pairs=('n','sum'), conf=('conf','sum'))
lb['rate']=lb.conf/lb.pairs
lb = lb.sort_values('pairs', ascending=False).head(12)
print('LANGUAGE'); print(lb.to_string())

out = {'mat': mat.to_dict('records'), 'dur': dur.reset_index().to_dict('records'),
       'concur': cc.reset_index().to_dict('records'),
       'top10': top10.reset_index().to_dict('records'), 'top50_confsum': int(top50.conf.sum()),
       'top10_confsum': int(top10.conf.sum()), 'tot_conf': int(tot_conf), 'tot_pairs': int(tot_pairs),
       'repos_with_conflict': int((rc.conf>0).sum()), 'n_repos_pairs': len(rc),
       'top10_pairsfrac': float(top10.n.sum()/tot_pairs), 'top50_pairsfrac': float(top50.n.sum()/tot_pairs),
       'stars': sb.reset_index().astype(str).to_dict('records'), 'lang': lb.reset_index().astype(str).to_dict('records')}
json.dump(out, open(f'{SP}/part2.json','w'), default=str)
pairs[['repo','pr_a','pr_b','conflict','ovbin','cbin','ov_h','concur']].to_csv(f'{SP}/pairs_temporal.csv.gz', index=False)
