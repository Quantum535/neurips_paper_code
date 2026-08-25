import os
import pandas as pd, numpy as np, json
SP = os.environ.get('MG_DERIVED', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'derived'))
D = os.environ.get('MG_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
pairs = pd.read_csv(f'{SP}/pairs_labeled.csv.gz')
tmp = pd.read_csv(f'{SP}/pairs_temporal.csv.gz')
pairs = pairs.merge(tmp[['repo','pr_a','pr_b','ovbin','cbin','ov_h','concur']], on=['repo','pr_a','pr_b'])
tex = pd.read_csv(f'{D}/data/pr_texts.csv.gz')
m = tex.set_index(['repo','pr'])['merged'].to_dict()

pairs['merged_a'] = [m.get((r,p)) for r,p in zip(pairs.repo, pairs.pr_a)]
pairs['merged_b'] = [m.get((r,p)) for r,p in zip(pairs.repo, pairs.pr_b)]
print('missing merged_a:', pairs.merged_a.isna().sum(), 'missing merged_b:', pairs.merged_b.isna().sum())

# later-opened PR (pairs are ordered by construction? check)
later_is_b = (pairs.opened_b >= pairs.opened_a)
print('later is b frac:', later_is_b.mean())
pairs['merged_later'] = np.where(later_is_b, pairs.merged_b, pairs.merged_a)
pairs['merged_earlier'] = np.where(later_is_b, pairs.merged_a, pairs.merged_b)
pairs['pr_later'] = np.where(later_is_b, pairs.pr_b, pairs.pr_a)

g = pairs.groupby('conflict')[['merged_a','merged_b','merged_later','merged_earlier']].agg(['mean','size'])
print(g.to_string())

def orat(df, col):
    a = df[df.conflict][col].astype(float); b = df[~df.conflict][col].astype(float)
    p1, p0 = a.mean(), b.mean()
    orr = (p1/(1-p1))/(p0/(1-p0))
    return p1, p0, orr, len(a), len(b)
p1,p0,orr,n1,n0 = orat(pairs,'merged_later')
print(f'PAIR-LEVEL later PR: conflict merge {p1:.4f} (n={n1}) vs non {p0:.4f} (n={n0}), diff {p1-p0:.4f}, OR {orr:.3f}')
p1e,p0e,orre,_,_ = orat(pairs,'merged_earlier')
print(f'PAIR-LEVEL earlier PR: conflict merge {p1e:.4f} vs non {p0e:.4f}, diff {p1e-p0e:.4f}, OR {orre:.3f}')

# PR-level: for each PR, was it ever the later PR in a conflicting pair vs only in non-conflicting pairs
pl = pairs.groupby(['repo','pr_later']).agg(ever_conf=('conflict','max'), merged=('merged_later','first')).reset_index()
gg = pl.groupby('ever_conf').merged.agg(['mean','size'])
print('PR-LEVEL (later role):'); print(gg.to_string())
pm1 = gg.loc[True,'mean']; pm0 = gg.loc[False,'mean']
orp = (pm1/(1-pm1))/(pm0/(1-pm0))
print(f'PR-level OR: {orp:.3f}, diff {pm1-pm0:.4f}')

# logistic with repo fixed effect proxy: control repo merge rate — simple: within-repo comparison
# stratified: only repos having both conflicting & non-conflicting pairs; Mantel-Haenszel OR
mh_num, mh_den = 0.0, 0.0
for repo, gr in pairs.dropna(subset=['merged_later']).groupby('repo'):
    c = gr[gr.conflict]; nc = gr[~gr.conflict]
    if len(c)==0 or len(nc)==0: continue
    a_ = c.merged_later.sum(); b_ = len(c)-a_
    c_ = nc.merged_later.sum(); d_ = len(nc)-c_
    N = len(gr)
    mh_num += a_*d_/N; mh_den += b_*c_/N
print(f'Mantel-Haenszel OR (repo-stratified, later PR): {mh_num/mh_den:.3f}')

# ---- robustness: temporal curves excluding 2 mega repos ----
excl = ~pairs.repo.isin(['MontrealAI/AGI-Alpha-Agent-v0','mochilang/mochi'])
sub = pairs[excl]
print(f'\nEXCLUDING 2 mega-repos: {len(sub)} pairs, conflict rate {sub.conflict.mean():.4f}')
d2 = sub.groupby('ovbin', observed=True).agg(n=('conflict','size'), conf=('conflict','sum')); d2['rate']=d2.conf/d2.n
print('DURATION (excl):'); print(d2.to_string())
c2 = sub.groupby('cbin', observed=True).agg(n=('conflict','size'), conf=('conflict','sum')); c2['rate']=c2.conf/c2.n
print('CONCURRENCY (excl):'); print(c2.to_string())
json.dump({'pair_later':[p1,p0,orr,int(n1),int(n0)],'pair_earlier':[p1e,p0e,orre],
 'pr_level':[float(pm1),float(pm0),float(orp),int(gg.loc[True,'size']),int(gg.loc[False,'size'])],
 'mh_or': mh_num/mh_den,
 'dur_excl': d2.reset_index().astype(str).to_dict('records'),
 'con_excl': c2.reset_index().astype(str).to_dict('records'),
 'sub_n': len(sub), 'sub_rate': float(sub.conflict.mean())}, open(f'{SP}/part3.json','w'))
