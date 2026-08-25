#!/usr/bin/env python3
"""figs_extra_p1.py -- six extra figures for Paper 1 (semantic-conflicts measurement).

Recomputes every plotted quantity fresh from:
  derived/pairs_labeled.csv.gz, data/pr_files.csv.gz, data/pr_texts.csv.gz,
  llm_scores.csv, test_pairs_for_llm.csv.gz

Outputs (PDF, vector) into derived/paper_outputs/ and copies into paper_plans/paper1/:
  fig_p1_agentmatrix.pdf  fig_p1_dose.pdf  fig_p1_hotfiles.pdf
  fig_p1_concentration.pdf  fig_p1_dupgap.pdf  fig_p1_llmerr.pdf

Rerunning this script regenerates all six figures and prints every cross-check number.
"""
import os, re, json, shutil, collections
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get('MG_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OUT = os.path.join(BASE, "derived", "paper_outputs")
PAPER1 = os.path.join(BASE, "paper_plans", "paper1")
os.makedirs(PAPER1, exist_ok=True)

plt.rcParams.update({
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "axes.linewidth": 0.7,
})
BLUE, VERM, GREEN, BLACK, GRAY = "#0072B2", "#D55E00", "#009E73", "#000000", "#8a8a8a"
LGRAY = "#c4c4c4"
MEGA = {"mochilang/mochi", "MontrealAI/AGI-Alpha-Agent-v0"}

def wilson(k, n, z=1.959963984540054):
    """Wilson 95% CI for a binomial proportion."""
    k, n = float(k), float(n)
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)

def savefig(fig, name):
    for d in (OUT, PAPER1):
        fig.savefig(os.path.join(d, name), bbox_inches="tight")
    plt.close(fig)
    sz = os.path.getsize(os.path.join(OUT, name))
    print(f"[saved] {name}  {sz} bytes")
    assert sz > 3072, f"{name} suspiciously small"

print("== loading pairs_labeled ==")
pairs = pd.read_csv(os.path.join(BASE, "derived", "pairs_labeled.csv.gz"))
assert len(pairs) == 577045, len(pairs)
for c in ("opened_a", "closed_a", "opened_b", "closed_b"):
    pairs[c] = pd.to_datetime(pairs[c], utc=True)
pairs["conflict"] = pairs["conflict"].astype(bool)
n_conf = int(pairs["conflict"].sum())
print("pairs:", len(pairs), "conflicting:", n_conf)

# ============================================================ FIG A: agent matrix
print("\n== FIG A: agent-pair conflict matrix ==")
disp = {"OpenAI_Codex": "Codex", "Copilot": "Copilot", "Cursor": "Cursor",
        "Devin": "Devin", "Claude_Code": "Claude Code"}
order = ["OpenAI_Codex", "Devin", "Copilot", "Cursor", "Claude_Code"]  # by diagonal volume
key = pairs.apply(lambda r: tuple(sorted((r.name,))), axis=0)  # placeholder no-op
a = pairs["agent_a"].to_numpy(); b = pairs["agent_b"].to_numpy()
lo = np.minimum(a, b); hi = np.maximum(a, b)
gp = pd.DataFrame({"lo": lo, "hi": hi, "conflict": pairs["conflict"].to_numpy()})
agg = gp.groupby(["lo", "hi"])["conflict"].agg(["sum", "size"]).reset_index()
cell = {}
for _, r in agg.iterrows():
    cell[frozenset((r["lo"], r["hi"])) if r["lo"] != r["hi"] else frozenset((r["lo"],))] = (
        int(r["sum"]), int(r["size"]))
for k, (s, n) in sorted(cell.items(), key=lambda kv: -kv[1][1]):
    print("  ", "&".join(sorted(k)), f"{100*s/n:.2f}% (n={n})")

N = len(order)
rate = np.full((N, N), np.nan)
nmat = np.zeros((N, N), dtype=int)
for i in range(N):
    for j in range(i + 1):
        k = frozenset((order[i], order[j])) if i != j else frozenset((order[i],))
        if k in cell:
            s, n = cell[k]
            nmat[i, j] = n
            rate[i, j] = 100.0 * s / n

fig, ax = plt.subplots(figsize=(3.4, 3.1))
cmap = plt.get_cmap("Blues")
valid = [(i, j) for i in range(N) for j in range(i + 1) if nmat[i, j] >= 100]
vmax = max(rate[i, j] for i, j in valid)
norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
for i in range(N):
    for j in range(i + 1):
        n = nmat[i, j]
        if n == 0:
            continue
        if n < 100:
            ax.add_patch(plt.Rectangle((j, N - 1 - i), 1, 1, facecolor="#dddddd",
                                       edgecolor="white", hatch="///", lw=1.0))
            txt = f"n={n}" if n > 1 else "n=1"
            ax.text(j + 0.5, N - 1 - i + 0.5, txt, ha="center", va="center",
                    fontsize=6.5, color="#444444")
        else:
            c = cmap(norm(rate[i, j]))
            ax.add_patch(plt.Rectangle((j, N - 1 - i), 1, 1, facecolor=c,
                                       edgecolor="white", lw=1.0))
            lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
            tc = "white" if lum < 0.55 else "#1a1a1a"
            ax.text(j + 0.5, N - 1 - i + 0.55, f"{rate[i, j]:.1f}%", ha="center",
                    va="center", fontsize=7, fontweight="bold", color=tc)
            ax.text(j + 0.5, N - 1 - i + 0.28, f"n={n:,}", ha="center", va="center",
                    fontsize=5.5, color=tc)
ax.set_xlim(0, N); ax.set_ylim(0, N)
ax.set_xticks([j + 0.5 for j in range(N)])
ax.set_xticklabels([disp[o] for o in order], rotation=30, ha="right")
ax.set_yticks([N - 1 - i + 0.5 for i in range(N)])
ax.set_yticklabels([disp[o] for o in order])
ax.set_aspect("equal")
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.03)
cb.set_label("file-overlap conflict rate (%)")
cb.outline.set_linewidth(0.7)
ax.text(0.98, 0.97, "hatched: n < 100", transform=ax.transAxes, ha="right",
        va="top", fontsize=6.5, color="#444444")
savefig(fig, "fig_p1_agentmatrix.pdf")

# ============================================================ FIG B: dose-response
print("\n== FIG B: dose-response ==")
op_a = pairs["opened_a"].astype("int64").to_numpy()
cl_a = pairs["closed_a"].astype("int64").to_numpy()
op_b = pairs["opened_b"].astype("int64").to_numpy()
cl_b = pairs["closed_b"].astype("int64").to_numpy()
mid = (np.maximum(op_a, op_b) + np.minimum(cl_a, cl_b)) // 2
overlap_h = (np.minimum(cl_a, cl_b) - np.maximum(op_a, op_b)) / 3.6e12
assert (overlap_h >= 0).all()

# distinct PR intervals per repo, from both roles of the labeled pairs file
pr_iv = pd.concat([
    pairs[["repo", "pr_a", "opened_a", "closed_a"]].rename(
        columns={"pr_a": "pr", "opened_a": "op", "closed_a": "cl"}),
    pairs[["repo", "pr_b", "opened_b", "closed_b"]].rename(
        columns={"pr_b": "pr", "opened_b": "op", "closed_b": "cl"}),
]).drop_duplicates(subset=["repo", "pr"])
pr_iv["op"] = pr_iv["op"].astype("int64"); pr_iv["cl"] = pr_iv["cl"].astype("int64")

conc = np.zeros(len(pairs), dtype=np.int64)
pairs_idx = np.arange(len(pairs))
repo_arr = pairs["repo"].to_numpy()
for repo, grp in pr_iv.groupby("repo", sort=False):
    opens = np.sort(grp["op"].to_numpy())
    closes = np.sort(grp["cl"].to_numpy())
    m = repo_arr == repo
    t = mid[m]
    conc[m] = (np.searchsorted(opens, t, side="right")
               - np.searchsorted(closes, t, side="left"))
assert conc.min() >= 2, conc.min()

conc_edges = [(2, 2, "2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, 20, "11-20"),
              (21, 50, "21-50"), (51, 10**9, ">50")]
dur_edges = [(0, 1, "<1h"), (1, 6, "1-6h"), (6, 24, "6-24h"),
             (24, 168, "1-7d"), (168, 1e18, ">7d")]
is_mega = pairs["repo"].isin(MEGA).to_numpy()
conf = pairs["conflict"].to_numpy()

def bucket_stats(values, edges, mask, left_closed=True):
    out = []
    for lo_, hi_, lab in edges:
        if isinstance(lo_, int) and edges is conc_edges:
            m = mask & (values >= lo_) & (values <= hi_)
        else:
            m = mask & (values >= lo_) & (values < hi_)
        n = int(m.sum()); k = int(conf[m].sum())
        lo_ci, hi_ci = wilson(k, n)
        out.append((lab, n, k, 100 * k / n if n else np.nan,
                    100 * lo_ci, 100 * hi_ci))
    return out

allm = np.ones(len(pairs), dtype=bool)
sA_all = bucket_stats(conc, conc_edges, allm)
sA_nom = bucket_stats(conc, conc_edges, ~is_mega)
sB_all = bucket_stats(overlap_h, dur_edges, allm)
sB_nom = bucket_stats(overlap_h, dur_edges, ~is_mega)
for name, s in [("conc all", sA_all), ("conc no-mega", sA_nom),
                ("dur all", sB_all), ("dur no-mega", sB_nom)]:
    print(" ", name, [(lab, n, f"{r:.2f}%") for lab, n, k, r, *_ in s])

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
for ax, sall, snom, labs, xlab in [
        (axes[0], sA_all, sA_nom, [e[2] for e in conc_edges],
         "co-active PRs at pair overlap midpoint"),
        (axes[1], sB_all, sB_nom, [e[2] for e in dur_edges],
         "pair overlap duration")]:
    x = np.arange(len(labs))
    for s, color, mk, ls, lab in [(sall, BLUE, "o", "-", "all pairs"),
                                  (snom, VERM, "s", "--", "excl. 2 mega-repos")]:
        r = np.array([v[3] for v in s]); lo_ = np.array([v[4] for v in s])
        hi_ = np.array([v[5] for v in s])
        ax.plot(x, r, marker=mk, ls=ls, color=color, ms=4, lw=1.2, label=lab)
        ax.fill_between(x, lo_, hi_, color=color, alpha=0.15, lw=0)
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_xlabel(xlab)
    ax.set_ylim(0, 45)
    ax.grid(axis="y", alpha=0.2, lw=0.5)
axes[0].set_ylabel("conflict rate (%)")
axes[0].text(0.62, 38.5, "excl. 2 mega-repos", color=VERM, fontsize=7)
axes[0].text(2.8, 21.5, "all pairs", color=BLUE, fontsize=7)
axes[1].text(1.1, 35.0, "excl. 2 mega-repos", color=VERM, fontsize=7)
axes[1].text(1.9, 16.5, "all pairs", color=BLUE, fontsize=7)
axes[0].text(-0.13, 1.02, "(a)", transform=axes[0].transAxes, fontweight="bold")
axes[1].text(-0.13, 1.02, "(b)", transform=axes[1].transAxes, fontweight="bold")
savefig(fig, "fig_p1_dose.pdf")

# ============================================================ FIG C: hot files
print("\n== FIG C: hot shared basenames ==")
pf = pd.read_csv(os.path.join(BASE, "data", "pr_files.csv.gz"))
files_by_pr = {k: set(v) for k, v in pf.groupby(["repo", "pr"])["filepath"].apply(list).items()}
cpairs = pairs[pairs["conflict"]]
cnt = collections.Counter()
for repo, pa, pb in zip(cpairs["repo"], cpairs["pr_a"], cpairs["pr_b"]):
    shared = files_by_pr.get((repo, pa), set()) & files_by_pr.get((repo, pb), set())
    for f in shared:
        cnt[f.rsplit("/", 1)[-1]] += 1
top12 = cnt.most_common(12)
print("  top-12:", top12)

CLASSMAP = {
    "README.md": ("docs", GRAY), "package.json": ("manifest/config", VERM),
    "page.tsx": ("source code", BLUE), "index.ts": ("source code", BLUE),
    "connect.mdx": ("docs", GRAY), "promptfooconfig.yaml": ("manifest/config", VERM),
    "vm.go": ("source code", BLUE), "form.png": ("asset", LGRAY),
    "ci.yml": ("CI", GREEN), "CHANGELOG.md": ("docs", GRAY),
    "common.json": ("manifest/config", VERM), "pyproject.toml": ("manifest/config", VERM),
}
fig, ax = plt.subplots(figsize=(3.4, 2.9))
names = [t[0] for t in top12][::-1]
vals = [t[1] for t in top12][::-1]
cols = [CLASSMAP.get(n, ("source code", BLUE))[1] for n in names]
y = np.arange(len(names))
ax.barh(y, vals, color=cols, height=0.68)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
for yi, v in zip(y, vals):
    ax.text(v + 150, yi, f"{v:,}", va="center", fontsize=6.5)
ax.set_xlabel("shared-file occurrences in conflicting pairs")
ax.set_xlim(0, max(vals) * 1.16)
ax.grid(axis="x", alpha=0.2, lw=0.5)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in [BLUE, VERM, GRAY, GREEN, LGRAY]]
ax.legend(handles, ["source code", "manifest/config", "docs", "CI", "asset"],
          frameon=False, fontsize=6.5, loc="lower right", handlelength=1.0,
          handleheight=1.0, borderaxespad=0.2)
savefig(fig, "fig_p1_hotfiles.pdf")

# ============================================================ FIG D: concentration
print("\n== FIG D: repo concentration ==")
by_conf = pairs[pairs["conflict"]].groupby("repo").size().sort_values(ascending=False)
by_all = pairs.groupby("repo").size().sort_values(ascending=False)
cum_conf = 100 * by_conf.cumsum().to_numpy() / n_conf
cum_all = 100 * by_all.cumsum().to_numpy() / len(pairs)
top10 = cum_conf[9]; top50 = cum_conf[49]
mega2 = 100 * by_all.loc[list(MEGA)].sum() / len(pairs)
mega_rank_top2 = set(by_all.index[:2]) == MEGA
print(f"  top-10 conflict share {top10:.2f}%  top-50 {top50:.2f}%  "
      f"mega-2 total-volume share {mega2:.2f}% (mega are top-2 by volume: {mega_rank_top2})")

fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.plot(np.arange(1, len(cum_conf) + 1), cum_conf, color=BLUE, lw=1.4,
        solid_capstyle="round")
ax.plot(np.arange(1, len(cum_all) + 1), cum_all, color=GRAY, lw=1.2, ls="--")
ax.set_xscale("log")
ax.plot([10], [top10], marker="o", color=BLUE, ms=4)
ax.plot([50], [top50], marker="o", color=BLUE, ms=4)
ax.annotate(f"top 10: {top10:.1f}%", xy=(10, top10), xytext=(13, 52),
            fontsize=6.5, color=BLUE,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=BLUE))
ax.annotate(f"top 50: {top50:.1f}%", xy=(50, top50), xytext=(65, 72),
            fontsize=6.5, color=BLUE,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=BLUE))
ax.plot([2], [cum_all[1]], marker="D", color=GRAY, ms=4)
ax.annotate(f"2 mega-repos: {cum_all[1]:.1f}% of all pairs", xy=(2, cum_all[1]),
            xytext=(1.02, 96.5), fontsize=6.5, color="#555555",
            arrowprops=dict(arrowstyle="-", lw=0.6, color=GRAY))
ax.text(20, 30, "solid: repos ranked by conflicting-pair\ncount (share of 42,600 conflicting pairs)",
        fontsize=6.5, color=BLUE)
ax.text(20, 12, "dashed: repos ranked by total pair\nvolume (share of all 577,045 pairs)",
        fontsize=6.5, color="#555555")
ax.set_xlabel("repo rank (log scale)")
ax.set_ylabel("cumulative share (%)")
ax.set_ylim(0, 103)
ax.grid(alpha=0.2, lw=0.5)
savefig(fig, "fig_p1_concentration.pdf")

# ============================================================ FIG E: dup close-gap ECDF
print("\n== FIG E: duplicate-title close-time gap ==")
pt = pd.read_csv(os.path.join(BASE, "data", "pr_texts.csv.gz"))
tmap = {(r, p): (t if isinstance(t, str) else "") for r, p, t in
        zip(pt["repo"], pt["pr"], pt["title"])}
mmap = {(r, p): bool(m) for r, p, m in zip(pt["repo"], pt["pr"], pt["merged"])}

BRACKET = re.compile(r"\[[^\]]*\]")
NONALNUM = re.compile(r"[^a-z0-9\s]")
def norm_title(t):
    t = t.lower()
    t = BRACKET.sub(" ", t)
    t = NONALNUM.sub(" ", t)
    return " ".join(t.split())

na = cpairs  # not used; dup pool spans ALL pairs
ta = [norm_title(tmap.get((r, p), "")) for r, p in zip(pairs["repo"], pairs["pr_a"])]
tb = [norm_title(tmap.get((r, p), "")) for r, p in zip(pairs["repo"], pairs["pr_b"])]
dup = np.zeros(len(pairs), dtype=bool)
for i, (x, y_) in enumerate(zip(ta, tb)):
    sx, sy = x.split(), y_.split()
    if len(sx) < 3 or len(sy) < 3:
        continue
    A, B = set(sx), set(sy)
    j = len(A & B) / len(A | B)
    if j >= 0.5 or (len(x) >= 15 and len(y_) >= 15 and (x in y_ or y_ in x)):
        dup[i] = True
n_dup = int(dup.sum())
merged_a = np.array([mmap[(r, p)] for r, p in zip(pairs["repo"], pairs["pr_a"])])
merged_b = np.array([mmap[(r, p)] for r, p in zip(pairs["repo"], pairs["pr_b"])])
bm = dup & merged_a & merged_b
gap_h = np.abs(cl_a - cl_b) / 3.6e12
g_all = np.sort(gap_h[bm])
g_nom = np.sort(gap_h[bm & ~is_mega])
med_all, med_nom = np.median(g_all), np.median(g_nom)
p24_all = 100 * (g_all < 24).mean()
p24_nom = 100 * (g_nom < 24).mean()
print(f"  dup pool {n_dup}; both-merged {len(g_all)} (median {med_all:.3f}h, "
      f"{p24_all:.1f}%<24h); excl-mega {len(g_nom)} (median {med_nom:.3f}h, "
      f"{p24_nom:.1f}%<24h)")

fig, ax = plt.subplots(figsize=(3.4, 2.6))
CLIP = 1e-3  # zero gaps clipped to 0.001 h for the log axis
for g, color, ls, lab in [(g_all, BLUE, "-", None), (g_nom, VERM, "--", None)]:
    gx = np.clip(g, CLIP, None)
    ax.step(np.concatenate([[CLIP], gx]),
            np.concatenate([[0], 100 * np.arange(1, len(g) + 1) / len(g)]),
            where="post", color=color, ls=ls, lw=1.3)
ax.set_xscale("log")
ax.axvline(24, color=BLACK, lw=0.8, ls=":")
ax.text(28, 40, "24 h", fontsize=6.5)
ax.annotate(f"{p24_all:.1f}% < 24h", xy=(24, p24_all), xytext=(0.9, 97),
            fontsize=6.5, color=BLUE,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=BLUE))
ax.annotate(f"{p24_nom:.1f}% < 24h", xy=(24, p24_nom), xytext=(60, 60),
            fontsize=6.5, color=VERM,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=VERM))
ax.text(0.0016, 76, f"all repos\n(n={len(g_all):,},\nmedian {med_all:.3f} h)",
        fontsize=6.5, color=BLUE)
ax.text(1.3, 18, f"excl. 2 mega-repos\n(n={len(g_nom)}, median {med_nom:.3f} h)",
        fontsize=6.5, color=VERM)
ax.set_xlabel("|close-time gap| between duplicate-title PRs (hours)")
ax.set_ylabel("cumulative share of pairs (%)")
ax.set_ylim(0, 104)
ax.grid(alpha=0.2, lw=0.5)
savefig(fig, "fig_p1_dupgap.pdf")

# ============================================================ FIG F: LLM error anatomy
print("\n== FIG F: LLM error anatomy ==")
tp = pd.read_csv(os.path.join(BASE, "test_pairs_for_llm.csv.gz"))
ls_ = pd.read_csv(os.path.join(BASE, "llm_scores.csv"))
m = tp.merge(ls_, on=["repo", "pr_a", "pr_b"], validate="one_to_one")
m = m.merge(pairs[["repo", "pr_a", "pr_b", "conflict"]], on=["repo", "pr_a", "pr_b"],
            validate="one_to_one")
assert len(m) == 5387, len(m)
pos = m[m["conflict"]]["prob"].to_numpy()
neg = m[~m["conflict"]]["prob"].to_numpy()
fn = int((pos <= 0.1).sum()); fp = int((neg >= 0.9).sum())
fn_pct = 100 * fn / len(pos)
print(f"  n={len(m)}, positives={len(pos)} ({100*len(pos)/len(m):.2f}%); "
      f"FN(prob<=0.1)={fn} ({fn_pct:.1f}% of positives); FP(prob>=0.9)={fp}")

fig, ax = plt.subplots(figsize=(3.4, 2.6))
bins = np.linspace(0, 1, 21)
ax.hist(pos, bins=bins, color=BLUE, alpha=0.45)
ax.hist(neg, bins=bins, histtype="step", color="#555555", lw=1.2)
ax.set_yscale("log")
ax.axvspan(0, 0.1, color=VERM, alpha=0.10, lw=0)
ax.axvspan(0.9, 1.0, color=VERM, alpha=0.10, lw=0)
ax.text(0.02, 1900, f"{fn} confident FNs\n({fn_pct:.1f}% of positives)",
        fontsize=6.5, ha="left", color="#8c3800")
ax.text(0.98, 1900, f"{fp} confident\nFPs", fontsize=6.5, ha="right", color="#8c3800")
ax.text(0.50, 650, "file overlap (label 1)\nfilled blue", fontsize=6.5,
        ha="center", color=BLUE)
ax.text(0.50, 210, "no overlap (label 0)\noutline", fontsize=6.5,
        ha="center", color="#555555")
ax.set_xlabel("zero-shot LLM conflict probability")
ax.set_ylabel("pairs (log scale)")
ax.set_xlim(0, 1)
ax.set_ylim(0.7, 5000)
ax.grid(axis="y", alpha=0.2, lw=0.5)
savefig(fig, "fig_p1_llmerr.pdf")

print("\nAll six figures written to", OUT, "and", PAPER1)
