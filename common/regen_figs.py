import json, pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'axes.linewidth': 0.7})
BLUE, VERM, GREEN, BLACK, GRAY = '#0072B2', '#D55E00', '#009E73', '#000000', '#8a8a8a'

# ---------- Forest plot (paper 1) ----------
d = json.load(open('p1_outcomes.json'))
rows = [  # (label, pair-key, dedup-key or None)
    ('File-overlap conflict\nvs. no shared file', 'a_conflict_vs_none', 'a_dedup_pr_level'),
    ('Duplicate title\nvs. all other pairs', 'b_dup_title_vs_rest', 'b_dedup_pr_level'),
    ('Fix-in-flight\nvs. other conflicting', 'c_fif_vs_other_conflict', None),
    ('Near-miss (shared dir)\nvs. neither', 'd_nearmiss_vs_nothing', None),
]
fig, ax = plt.subplots(figsize=(3.4, 2.5))
ys = np.arange(len(rows))[::-1]
for y, (lab, k, dk) in zip(ys, rows):
    e = d[k]
    ax.errorbar(e['or_mh'], y + 0.12, xerr=[[e['or_mh']-e['ci_lo']],[e['ci_hi']-e['or_mh']]],
                fmt='o', color=BLUE, ms=4.5, capsize=2, lw=1.1, zorder=3)
    ax.annotate(f"{e['or_mh']:.2f}", (e['or_mh'], y + 0.32), ha='center', fontsize=7, color=BLUE)
    if dk:
        e2 = d[dk]
        ax.errorbar(e2['or_mh'], y - 0.18, xerr=[[e2['or_mh']-e2['ci_lo']],[e2['ci_hi']-e2['or_mh']]],
                    fmt='s', mfc='none', color=VERM, ms=4.5, capsize=2, lw=1.1, ls='--', zorder=3)
ax.axvline(1.0, color=BLACK, lw=0.8, zorder=1)
ax.set_xscale('log'); ax.set_xlim(0.55, 2.05)
ax.set_xticks([0.6, 0.8, 1, 1.25, 1.5, 2]); ax.set_xticklabels(['0.6','0.8','1','1.25','1.5','2'])
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
ax.set_xlabel('Mantel–Haenszel OR, later-PR merge\n(repo-stratified, 95% CI)', fontsize=7.5)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],marker='o',color=BLUE,ls='-',ms=4.5,label='Pair-level'),
                   Line2D([],[],marker='s',mfc='none',color=VERM,ls='--',ms=4.5,label='PR-level dedup')],
          frameon=False, fontsize=7, loc='lower right')
ax.grid(axis='x', alpha=0.2, lw=0.5)
fig.savefig('fig_p1_forest.pdf', bbox_inches='tight'); plt.close(fig)

# ---------- Pareto (paper 2) ----------
df = pd.read_csv('t3_episode_results.csv')
MEGA = {'mochilang/mochi', 'MontrealAI/AGI-Alpha-Agent-v0'}
def pareto(sub, fname):
    fig, ax = plt.subplots(figsize=(3.9, 2.7))
    def med_iqr(g):
        return (g['makespan_inflation_pct'].median(), g['avoided_pct'].median(),
                g['makespan_inflation_pct'].quantile(.25), g['makespan_inflation_pct'].quantile(.75),
                g['avoided_pct'].quantile(.25), g['avoided_pct'].quantile(.75))
    # predictor curve
    taus = sorted(sub[sub.policy=='predictor'].tau.unique())
    px, py = [], []
    for t in taus:
        x, y, xl, xh, yl, yh = med_iqr(sub[(sub.policy=='predictor') & (sub.tau==t)])
        px.append(x); py.append(y)
        ax.errorbar(x, y, xerr=[[x-xl],[xh-x]], yerr=[[y-yl],[yh-y]], fmt='o', mfc='white',
                    color=BLUE, ms=4, capsize=1.5, lw=0.7, elinewidth=0.6, ecolor='#9db9d1', zorder=3)
        if t in (0.2, 0.3, 0.4):
            ax.annotate(f'τ={t:.1f}', (x, y), xytext=(8, -10) if t!=0.2 else (7, -2),
                        textcoords='offset points', fontsize=7, color=BLUE)
    order = np.argsort(px)
    ax.plot(np.array(px)[order], np.array(py)[order], color=BLUE, lw=1.2, zorder=2)
    ax.annotate('predictor gate\n(τ sweep)', (78, 62), fontsize=7.5, color=BLUE)
    for pol, col, mk, lab, dx, dy in [('fifo', VERM, 's', 'FIFO serialize', -8, -11),
                                      ('oracle', GREEN, '^', 'oracle gate', 6, -3)]:
        x, y, xl, xh, yl, yh = med_iqr(sub[sub.policy==pol])
        ax.errorbar(x, y, xerr=[[x-xl],[xh-x]], fmt=mk, color=col, ms=5.5, capsize=1.5,
                    lw=0.7, elinewidth=0.6, ecolor=GRAY, zorder=4)
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords='offset points', fontsize=7.5, color=col)
    ax.plot(0, 0, 'D', color=BLACK, ms=5, zorder=4)
    ax.annotate('observed', (0, 0), xytext=(7, 3), textcoords='offset points', fontsize=7.5)
    ax.annotate('n = %d episodes; marks = medians, bars = IQR' % sub.repo.nunique(),
                (0.98, 0.06), xycoords='axes fraction', ha='right', fontsize=6.5, color=GRAY)
    ax.set_xlabel('Makespan inflation vs. observed (%)', fontsize=8)
    ax.set_ylabel('Conflicts avoided (%)', fontsize=8)
    ax.set_xlim(-12, 305); ax.set_ylim(-5, 106)
    ax.grid(alpha=0.2, lw=0.5)
    fig.savefig(fname, bbox_inches='tight'); plt.close(fig)
pareto(df, 'fig_t3_pareto.pdf')
pareto(df[~df.repo.isin(MEGA)], 'fig_t3_pareto_nomega.pdf')
print('done')
