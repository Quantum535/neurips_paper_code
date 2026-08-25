#!/usr/bin/env python3
"""
figs_extra_p2.py -- six additional Paper-2 (MergeGym) figures.

Recomputes every plotted quantity from the underlying data and cross-checks
against paper_plans/verification_log.md / t1_leaderboard.csv / t3_summary.json.

Outputs (written to derived/paper_outputs/ and copied to paper_plans/paper2/):
  fig_t1_roc.pdf        (A) ROC curves, 5,387-pair T1 test set
  fig_t1_quality.pdf    (B) LLM reliability diagram + precision@k
  fig_t1_withinrepo.pdf (C) per-repo LLM AUC dot plot (64 repos)
  fig_t3_episodes.pdf   (D) episode diversity + tau=0.20 knee heterogeneity
  fig_t2_composition.pdf(E) T2 candidate languages + per-PR size histogram
  fig_t3_delaycost.pdf  (F) delay-cost (PR-days/PR) Pareto view
Also writes t1_score_vectors.csv.gz (one column per baseline score).
"""
import gzip, json, math, os, re, shutil
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.environ.get('MG_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OUT = f"{D}/derived/paper_outputs"
PAPER = f"{D}/paper_plans/paper2"

BLUE, VERM, GREEN, BLACK, GRAY = "#0072B2", "#D55E00", "#009E73", "#000000", "#8a8a8a"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "pdf.fonttype": 42, "font.family": "sans-serif",
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})

CHECKS = []
def check(name, got, expect, tol=0.0):
    ok = (abs(got - expect) <= tol) if tol else (got == expect)
    CHECKS.append((name, got, expect, ok))
    print(("[MATCH]" if ok else "[MISMATCH]"), name, "got", got, "expect", expect)
    return ok

# ---------------------------------------------------------------- utilities
def auc(s, l):
    """Tie-corrected Mann-Whitney rank AUC."""
    s = np.asarray(s, float); l = np.asarray(l)
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0; i = j + 1
    n1 = l.sum(); n0 = len(l) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (ranks[l == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)

def roc_curve(s, l):
    """FPR/TPR at every distinct threshold (descending), starting at (0,0)."""
    s = np.asarray(s, float); l = np.asarray(l, int)
    order = np.argsort(-s, kind="mergesort")
    y = l[order]; ss = s[order]
    tps = np.cumsum(y); fps = np.cumsum(1 - y)
    # keep last index of each tie block
    keep = np.r_[ss[1:] != ss[:-1], True]
    P = l.sum(); N = len(l) - P
    return np.r_[0, fps[keep] / N], np.r_[0, tps[keep] / P]

def prec_at_k(s, l, ks):
    """Deterministic descending stable sort (audited convention)."""
    order = np.argsort(-np.asarray(s, float), kind="mergesort")
    y = np.asarray(l, int)[order]
    c = np.cumsum(y)
    return np.array([c[k - 1] / k for k in ks])

def wilson(p, n, z=1.96):
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h

def logreg_fit(X, y, l2=1e-4, iters=200):
    Xb = np.hstack([np.ones((len(X), 1)), X]); w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = Xb.T @ (p - y) + l2 * np.r_[0, w[1:]]
        W = p * (1 - p) + 1e-9
        H = (Xb * W[:, None]).T @ Xb + l2 * np.eye(Xb.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w

def logreg_pred(w, X):
    Xb = np.hstack([np.ones((len(X), 1)), X])
    return 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))

tok_re = re.compile(r"[a-z0-9_]+")
def toks(t):
    return tok_re.findall(str(t).lower())

# ================================================================ score vectors
print("=== building T1 score vectors ===")
df = pd.read_pickle(f"{D}/derived/joined_feats.pkl")
assert len(df) == 5387
y = df["conflict"].astype(int).values
check("test n", len(df), 5387)
check("positive rate", round(y.mean(), 4), 0.3993, 0.0001)

auc_llm = auc(df["prob"].values, y)
check("LLM AUC", round(auc_llm, 4), 0.7044, 0.0002)
auc_tfidf = auc(df["f_tfidf_cos"].values, y)
check("TF-IDF cosine AUC", round(auc_tfidf, 4), 0.6436, 0.0002)

# ---- fused 4-feature LR, 5-fold pooled OOF (seed 0, per-fold standardization)
df["f_log_ov"] = np.log1p(df["f_overlap_h"])
base4 = ["f_repo_rate", "f_log_ov", "f_tfidf_cos", "f_title_jac"]
X4 = df[base4].values.astype(float)
rng = np.random.RandomState(0)
idx = rng.permutation(len(df)); folds = np.array_split(idx, 5)
oof = np.full(len(df), np.nan)
for f in range(5):
    te = folds[f]; tr = np.concatenate([folds[g] for g in range(5) if g != f])
    mu, sd = X4[tr].mean(0), X4[tr].std(0) + 1e-9
    w = logreg_fit((X4[tr] - mu) / sd, y[tr])
    oof[te] = logreg_pred(w, (X4[te] - mu) / sd)
auc_fused = auc(oof, y)
check("fused LR OOF AUC", round(auc_fused, 4), 0.6759, 0.0002)

# ---- filename-token overlap count (audited leakage baseline)
ext_re = re.compile(r"[A-Za-z0-9_./-]*[A-Za-z0-9_-]\.[A-Za-z][A-Za-z0-9]{0,5}")
slash_re = re.compile(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+")
def file_tokens(text):
    text = str(text)
    tt = set(ext_re.findall(text)) | set(slash_re.findall(text))
    out = set()
    for t in tt:
        while t.startswith("./"):
            t = t[2:]
        t = t.strip("/").strip(".")
        if t and "." in t or "/" in t:
            if t:
                out.add(t)
    return out
toks_a = [file_tokens(t) for t in df["task_a"]]
toks_b = [file_tokens(t) for t in df["task_b"]]
fname_count = np.array([len(a & b) for a, b in zip(toks_a, toks_b)], float)
auc_fname = auc(fname_count, y)
check("filename overlap AUC", round(auc_fname, 4), 0.5744, 0.0002)

# ---- trained repo-held-out LR, retrained from the frozen manifest
print("--- retraining repo-held-out LR from t1_train_manifest ---")
train = pd.read_csv(f"{OUT}/t1_train_manifest.csv.gz")
check("train n", len(train), 24283)
check("train repos", train["repo"].nunique(), 870)
check("train conflict rate", round(train["conflict"].mean(), 4), 0.3240, 0.0001)
lab = pd.read_csv(f"{D}/derived/pairs_labeled.csv.gz")
train = train.merge(
    lab[["repo", "pr_a", "pr_b", "agent_a", "agent_b",
         "opened_a", "closed_a", "opened_b", "closed_b"]],
    on=["repo", "pr_a", "pr_b"], how="left", validate="1:1")
assert train["agent_a"].notna().all()

texts = pd.read_csv(f"{D}/data/pr_texts.csv.gz")
texts["body"] = texts["body"].fillna(""); texts["title"] = texts["title"].fillna("")
texts["task"] = texts["title"] + " || " + texts["body"]
tmap_task = texts.set_index(["repo", "pr"])["task"].to_dict()
tmap_title = texts.set_index(["repo", "pr"])["title"].to_dict()

tr_task_a = [tmap_task.get((r, p), "") for r, p in zip(train["repo"], train["pr_a"])]
tr_task_b = [tmap_task.get((r, p), "") for r, p in zip(train["repo"], train["pr_b"])]
docs_a = [toks(t) for t in tr_task_a]; docs_b = [toks(t) for t in tr_task_b]
dfreq = Counter()
for dd in docs_a + docs_b:
    dfreq.update(set(dd))
N_docs = len(docs_a) + len(docs_b)
def get_idf(wd):
    return math.log((1 + N_docs) / (1 + dfreq.get(wd, 0))) + 1.0
def tfidf_cos(da, db):
    ca, cb = Counter(da), Counter(db)
    va = {w_: (1 + math.log(c)) * get_idf(w_) for w_, c in ca.items()}
    vb = {w_: (1 + math.log(c)) * get_idf(w_) for w_, c in cb.items()}
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return sum(v * vb[w_] for w_, v in va.items() if w_ in vb) / (na * nb)
def jac(a, b):
    sa, sb = set(toks(a)), set(toks(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def build_feats(dfr, task_a_list, task_b_list):
    F = {}
    da = [toks(t) for t in task_a_list]; db = [toks(t) for t in task_b_list]
    F["tfidf_cos"] = np.array([tfidf_cos(x, z) for x, z in zip(da, db)])
    ta = [tmap_title.get((r, p), "") for r, p in zip(dfr["repo"], dfr["pr_a"])]
    tb = [tmap_title.get((r, p), "") for r, p in zip(dfr["repo"], dfr["pr_b"])]
    F["title_jac"] = np.array([jac(a, b) for a, b in zip(ta, tb)])
    o_a = pd.to_datetime(dfr["opened_a"], utc=True, format="mixed")
    o_b = pd.to_datetime(dfr["opened_b"], utc=True, format="mixed")
    c_a = pd.to_datetime(dfr["closed_a"], utc=True, format="mixed")
    c_b = pd.to_datetime(dfr["closed_b"], utc=True, format="mixed")
    start = np.maximum(o_a.values, o_b.values); end = np.minimum(c_a.values, c_b.values)
    ov = np.maximum(((end - start) / np.timedelta64(1, "h")).astype(float), 0.0)
    F["log_ov"] = np.log1p(ov)
    F["same_agent"] = (dfr["agent_a"].values == dfr["agent_b"].values).astype(float)
    F["agent_pair"] = [tuple(sorted([a, b])) for a, b in zip(dfr["agent_a"], dfr["agent_b"])]
    return F

Ftr = build_feats(train, tr_task_a, tr_task_b)
tst = df.merge(lab[["repo", "pr_a", "pr_b", "agent_a", "agent_b"]],
               on=["repo", "pr_a", "pr_b"], how="left", validate="1:1")
assert tst["agent_a"].notna().all()
Fte = build_feats(tst, tst["task_a"].tolist(), tst["task_b"].tolist())

ap_counts = Counter(Ftr["agent_pair"])
ap_cols = [ap for ap, _ in ap_counts.most_common()[1:]]  # drop most frequent (reference)
def onehot(apl):
    M = np.zeros((len(apl), len(ap_cols)))
    for i, ap in enumerate(apl):
        if ap in ap_cols:
            M[i, ap_cols.index(ap)] = 1.0
    return M
rr = train.groupby("repo")["conflict"].mean()
grate = train["conflict"].mean()
Xtr = np.column_stack([train["repo"].map(rr).values, Ftr["log_ov"], Ftr["tfidf_cos"],
                       Ftr["title_jac"], Ftr["same_agent"], onehot(Ftr["agent_pair"])])
Xte = np.column_stack([tst["repo"].map(rr).fillna(grate).values, Fte["log_ov"],
                       Fte["tfidf_cos"], Fte["title_jac"], Fte["same_agent"],
                       onehot(Fte["agent_pair"])])
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
w = logreg_fit((Xtr - mu) / sd, train["conflict"].astype(int).values)
trained_scores = logreg_pred(w, (Xte - mu) / sd)
auc_trained = auc(trained_scores, y)
trained_ok = check("trained repo-held-out LR AUC", round(auc_trained, 4), 0.6444, 0.003)

# ---- persist score vectors for reuse
sv = df[["repo", "pr_a", "pr_b"]].copy()
sv["label"] = y
sv["llm_prob"] = df["prob"].values
sv["fused_lr_oof"] = oof
sv["trained_lr"] = trained_scores
sv["tfidf_cos"] = df["f_tfidf_cos"].values
sv["title_jaccard"] = df["f_title_jac"].values
sv["filename_overlap_count"] = fname_count
sv.to_csv(f"{OUT}/t1_score_vectors.csv.gz", index=False, compression="gzip")
print("wrote t1_score_vectors.csv.gz", sv.shape)

# ================================================================ FIG A: ROC
print("=== FIG A: ROC ===")
fig, ax = plt.subplots(figsize=(3.4, 3.1))
series = [
    ("Zero-shot LLM",        df["prob"].values, BLUE,  "-",  1.4, auc_llm),
    ("Fused LR (4 feat., OOF)", oof,             VERM,  "--", 1.2, auc_fused),
    ("Trained LR (repo-held-out)", trained_scores, GREEN, "-.", 1.2, auc_trained),
    ("TF-IDF cosine",        df["f_tfidf_cos"].values, GRAY, "-", 0.8, auc_tfidf),
    ("Filename-token overlap", fname_count,      GRAY,  ":",  1.0, auc_fname),
]
if not trained_ok:
    series = [s for s in series if not s[0].startswith("Trained")]
for name, s, col, ls, lw, a in series:
    fpr, tpr = roc_curve(s, y)
    ax.plot(fpr, tpr, ls, color=col, lw=lw, label=f"{name}  ({a:.3f})")
ax.plot([0, 1], [0, 1], "-", color=BLACK, lw=0.7, label="chance  (0.500)")
ax.set_xlabel("False-positive rate")
ax.set_ylabel("True-positive rate")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.grid(True, color=GRAY, alpha=0.18, lw=0.5)
ax.legend(loc="lower right", fontsize=6.6, title="AUC", title_fontsize=6.6,
          handlelength=2.4, borderaxespad=0.2)
fig.savefig(f"{OUT}/fig_t1_roc.pdf", bbox_inches="tight")
plt.close(fig)

# ================================================================ FIG B: quality
print("=== FIG B: reliability + precision@k ===")
edges_inner = [0.2, 0.4, 0.6, 0.8]
b = np.digitize(df["prob"].values, edges_inner)  # left-closed bins, [.8,1] closed
bin_pred, bin_act, bin_n, bin_lo, bin_hi = [], [], [], [], []
for k in range(5):
    m = b == k
    n = int(m.sum()); p_act = y[m].mean()
    bin_pred.append(df["prob"].values[m].mean()); bin_act.append(p_act); bin_n.append(n)
    lo, hi = wilson(p_act, n)
    bin_lo.append(lo); bin_hi.append(hi)
exp_pred = [0.092, 0.263, 0.463, 0.696, 0.878]
exp_act = [0.266, 0.388, 0.456, 0.748, 0.939]
exp_n = [2788, 1290, 504, 409, 396]
for k in range(5):
    check(f"reliability bin{k} pred", round(bin_pred[k], 3), exp_pred[k], 0.001)
    check(f"reliability bin{k} actual", round(bin_act[k], 3), exp_act[k], 0.001)
    check(f"reliability bin{k} n", bin_n[k], exp_n[k])
pk_llm50 = prec_at_k(df["prob"].values, y, [50, 100])
check("LLM P@50", round(pk_llm50[0], 2), 0.94, 0.001)
check("LLM P@100", round(pk_llm50[1], 2), 0.96, 0.001)

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7))
axa, axb = axes
axa.plot([0, 1], [0, 1], "-", color=BLACK, lw=0.7, zorder=1)
sizes = 12 + 60 * np.array(bin_n) / max(bin_n)
axa.errorbar(bin_pred, bin_act,
             yerr=[np.array(bin_act) - np.array(bin_lo),
                   np.array(bin_hi) - np.array(bin_act)],
             fmt="none", ecolor=GRAY, elinewidth=0.8, capsize=1.5, zorder=2)
axa.scatter(bin_pred, bin_act, s=sizes, color=BLUE, zorder=3, edgecolor="white",
            linewidth=0.5)
offs = [(-2, 9, "center"), (-2, 9, "center"), (2, -13, "center"),
        (-2, 9, "center"), (-2, 9, "center")]
for k in range(5):
    xo, yo, ha = offs[k]
    axa.annotate(f"n={bin_n[k]:,}", (bin_pred[k], bin_act[k]),
                 textcoords="offset points", xytext=(xo, yo), ha=ha,
                 fontsize=6.3, color=GRAY)
axa.annotate("perfect calibration", (0.33, 0.255), rotation=38, fontsize=6.3,
             color=BLACK, ha="center")
axa.set_xlabel("Mean predicted conflict probability (bin)")
axa.set_ylabel("Actual conflict rate (bin)")
axa.set_xlim(0, 1); axa.set_ylim(0, 1)
axa.grid(True, color=GRAY, alpha=0.18, lw=0.5)
axa.set_title("")

ks = np.unique(np.round(np.logspace(np.log10(10), np.log10(2000), 40)).astype(int))
pk_series = [
    ("Zero-shot LLM", df["prob"].values, BLUE, "-", "o"),
    ("Fused LR", oof, VERM, "--", "s"),
    ("Trained LR", trained_scores, GREEN, "-.", "^"),
    ("Filename overlap", fname_count, GRAY, ":", "x"),
]
for name, s, col, ls, mk in pk_series:
    pk = prec_at_k(s, y, ks)
    axb.plot(ks, pk, ls, color=col, lw=1.1, marker=mk, ms=2.4,
             markevery=6, mfc="white" if mk != "x" else col, mew=0.8, label=name)
axb.axhline(0.3993, color=BLACK, ls="--", lw=0.8)
axb.annotate("random (enriched set) = 0.399", (2000, 0.3993),
             textcoords="offset points", xytext=(-2, -9), ha="right",
             fontsize=6.3, color=BLACK)
axb.set_xscale("log")
axb.set_xlabel("k (pairs flagged)")
axb.set_ylabel("Precision@k")
axb.set_xlim(10, 2000); axb.set_ylim(0.05, 1.02)
axb.grid(True, color=GRAY, alpha=0.18, lw=0.5)
axb.legend(loc="center left", bbox_to_anchor=(0.0, 0.52), fontsize=6.6,
           handlelength=2.6)
for axx, lab_ in [(axa, "(a)"), (axb, "(b)")]:
    axx.annotate(lab_, (-0.16, 1.02), xycoords="axes fraction", fontsize=8.5,
                 fontweight="bold")
fig.tight_layout(w_pad=2.0)
fig.savefig(f"{OUT}/fig_t1_quality.pdf", bbox_inches="tight")
plt.close(fig)

# ================================================================ FIG C: within-repo
print("=== FIG C: within-repo AUC ===")
rows = []
for repo, g in df.groupby("repo"):
    if len(g) >= 10 and g["conflict"].nunique() == 2:
        rows.append((repo, len(g), auc(g["prob"].values, g["conflict"].astype(int).values)))
pr = pd.DataFrame(rows, columns=["repo", "n", "auc_llm"]).sort_values("auc_llm")
check("within-repo repo count", len(pr), 64)
check("within-repo pairs covered", int(pr["n"].sum()), 4773)
check("within-repo mean LLM AUC", round(pr["auc_llm"].mean(), 4), 0.7425, 0.0002)
n_gt = int((pr["auc_llm"] > 0.5).sum())
check("repos with AUC>0.5", n_gt, 58)

fig, ax = plt.subplots(figsize=(3.4, 3.0))
yy = np.arange(len(pr))
ax.axvline(0.5, color=GRAY, lw=0.7, ls="-")
ax.axvline(auc_llm, color=BLACK, lw=0.9, ls="--")
ax.scatter(pr["auc_llm"], yy, s=8 + 0.35 * pr["n"], color=BLUE, alpha=0.75,
           edgecolor="white", linewidth=0.4, zorder=3)
mean_auc = pr["auc_llm"].mean()
ax.scatter([mean_auc], [len(pr) * 0.5], marker="D", s=32, color=VERM, zorder=4,
           edgecolor="white", linewidth=0.6)
ax.annotate(f"mean = {mean_auc:.4f}", (mean_auc, len(pr) * 0.5),
            textcoords="offset points", xytext=(8, -3), fontsize=7, color=VERM)
ax.annotate(f"pooled = {auc_llm:.3f}", (auc_llm, 2.0),
            textcoords="offset points", xytext=(5, 0), ha="left", fontsize=7,
            color=BLACK)
ax.annotate("chance", (0.5, len(pr) * 0.80), textcoords="offset points",
            xytext=(-3, 0), ha="right", fontsize=6.5, color=GRAY, rotation=90,
            va="center")
ax.annotate(f"{n_gt}/{len(pr)} repos > 0.5", (0.03, 0.95), xycoords="axes fraction",
            fontsize=7.5)
ax.set_xlabel("Within-repo zero-shot LLM AUC")
ax.set_ylabel("Repos (sorted by AUC)")
ax.set_yticks([])
ax.set_xlim(0, 1.02)
ax.grid(True, axis="x", color=GRAY, alpha=0.18, lw=0.5)
fig.savefig(f"{OUT}/fig_t1_withinrepo.pdf", bbox_inches="tight")
plt.close(fig)

# ================================================================ FIG D: episodes
print("=== FIG D: episode diversity + knee ===")
per_repo = lab.groupby("repo").agg(pairs=("conflict", "size"),
                                   conf=("conflict", "sum"))
pool = per_repo[(per_repo["pairs"] >= 100) & (per_repo["conf"] >= 20)].copy()
check("T3 episode count", len(pool), 97)
check("T3 conflicting pairs", int(pool["conf"].sum()), 38526)
lab_pool = lab[lab["repo"].isin(pool.index)]
nprs = lab_pool.groupby("repo").apply(
    lambda g: len(set(g["pr_a"]).union(set(g["pr_b"]))), include_groups=False)
check("T3 distinct PRs", int(nprs.sum()), 19397)
pool["n_prs"] = nprs
pool["conf_per_pr"] = pool["conf"] / pool["n_prs"]
MEGA = {"mochilang/mochi", "MontrealAI/AGI-Alpha-Agent-v0"}

er = pd.read_csv(f"{OUT}/t3_episode_results.csv")
p20 = er[(er["policy"] == "predictor") & (er["tau"] == 0.2)].copy()
assert len(p20) == 97
# net avoided: CSV avoided_pct already nets out newly created conflicts
net_chk = (p20["obs_conflicts"] - p20["conflicts"]) / p20["obs_conflicts"] * 100
assert np.allclose(net_chk.values, p20["avoided_pct"].values)
med_x = p20["makespan_inflation_pct"].median()
med_y = p20["avoided_pct"].median()
check("tau=0.2 median makespan inflation", round(med_x, 1), 65.0, 0.05)
check("tau=0.2 median net avoided", round(med_y, 1), 91.7, 0.05)

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7))
axa, axb = axes
sz = 4 + 1.1 * np.sqrt(pool["pairs"].values)
is_mega = pool.index.isin(MEGA)
axa.scatter(pool["n_prs"][~is_mega], pool["conf_per_pr"][~is_mega],
            s=sz[~is_mega], color=BLUE, alpha=0.55, edgecolor="white", lw=0.3)
axa.scatter(pool["n_prs"][is_mega], pool["conf_per_pr"][is_mega],
            s=sz[is_mega], color=BLUE, alpha=0.55, edgecolor=BLACK, lw=0.6)
mega_offs = {"mochilang/mochi": (-18, -16, "right"),
             "MontrealAI/AGI-Alpha-Agent-v0": (0, -24, "center")}
for repo in MEGA:
    r = pool.loc[repo]
    short = repo.split("/")[1]
    xo, yo, ha = mega_offs[repo]
    axa.annotate(short, (r["n_prs"], r["conf_per_pr"]),
                 textcoords="offset points", xytext=(xo, yo), ha=ha,
                 fontsize=6.3, color=GRAY)
axa.set_xscale("log"); axa.set_yscale("log")
axa.set_xlim(right=pool["n_prs"].max() * 2.2)
axa.set_xlabel("Agent PRs in episode")
axa.set_ylabel("Conflicting pairs per PR")
axa.grid(True, color=GRAY, alpha=0.18, lw=0.5)
axa.annotate("point area scales with co-active\npairs (sqrt); 97 episodes",
             (0.03, 0.9), xycoords="axes fraction", fontsize=6.5, color=GRAY,
             va="top")

mega_mask = p20["repo"].isin(MEGA)
axb.scatter(p20["makespan_inflation_pct"][~mega_mask],
            p20["avoided_pct"][~mega_mask], s=12, color=GRAY, alpha=0.5,
            edgecolor="none", zorder=2)
axb.scatter(p20["makespan_inflation_pct"][mega_mask],
            p20["avoided_pct"][mega_mask], s=18, color=VERM, zorder=4,
            edgecolor="white", lw=0.4)
pb_offs = {"mochilang/mochi": (8, -11), "MontrealAI/AGI-Alpha-Agent-v0": (10, -22)}
for _, r in p20[mega_mask].iterrows():
    axb.annotate(r["repo"].split("/")[1],
                 (r["makespan_inflation_pct"], r["avoided_pct"]),
                 textcoords="offset points", xytext=pb_offs[r["repo"]],
                 fontsize=6.3, color=VERM, ha="left")
axb.axvline(med_x, color=BLUE, lw=0.9, ls="--", zorder=3)
axb.axhline(med_y, color=BLUE, lw=0.9, ls="--", zorder=3)
axb.annotate(f"median: {med_y:.1f}% avoided\nat +{med_x:.1f}% makespan",
             (med_x, 8), textcoords="offset points", xytext=(6, 0), fontsize=6.5,
             color=BLUE)
axb.set_xlabel("Makespan inflation vs. observed (%)")
axb.set_ylabel("Conflicts avoided, net (%)")
axb.set_ylim(-12, 104)
axb.grid(True, color=GRAY, alpha=0.18, lw=0.5)
axb.annotate("predictor gate, $\\tau$ = 0.20", (0.97, 0.05),
             xycoords="axes fraction", ha="right", fontsize=6.5, color=GRAY)
for axx, lab_ in [(axa, "(a)"), (axb, "(b)")]:
    axx.annotate(lab_, (-0.16, 1.02), xycoords="axes fraction", fontsize=8.5,
                 fontweight="bold")
fig.tight_layout(w_pad=2.0)
fig.savefig(f"{OUT}/fig_t3_episodes.pdf", bbox_inches="tight")
plt.close(fig)

# ================================================================ FIG E: T2 composition
print("=== FIG E: T2 composition ===")
t2 = pd.read_csv(f"{D}/derived/t2_candidates.csv.gz")
check("T2 pairs", len(t2), 6387)
check("T2 repos", t2["repo"].nunique(), 391)
rs = pd.read_csv(f"{D}/data/repo_stats.csv")
langmap = rs.set_index("repo")["language"].to_dict()
t2["language"] = t2["repo"].map(langmap).fillna("Unknown")
lc = t2["language"].value_counts()
exp_lang = {"TypeScript": 2030, "C#": 1108, "JavaScript": 645, "Python": 612,
            "Go": 473, "Ruby": 438, "Rust": 323, "Dart": 228, "Java": 207}
for k_, v_ in exp_lang.items():
    check(f"T2 lang {k_}", int(lc.get(k_, 0)), v_)
top = list(exp_lang.keys())
other_n = len(t2) - sum(int(lc.get(k_, 0)) for k_ in top)
plot_langs = top + ["Other"]
plot_counts = [int(lc.get(k_, 0)) for k_ in top] + [other_n]

prs = pd.concat([t2[["repo", "pr_a"]].rename(columns={"pr_a": "pr"}),
                 t2[["repo", "pr_b"]].rename(columns={"pr_b": "pr"})]).drop_duplicates()
check("T2 distinct PRs", len(prs), 3779)
pf = pd.read_csv(f"{D}/data/pr_files.csv.gz")
pf["_key"] = list(zip(pf["repo"], pf["pr"]))
need = set(zip(prs["repo"], prs["pr"]))
sizes_pr = pf[pf["_key"].isin(need)].groupby("_key").apply(
    lambda g: (g["additions"] + g["deletions"]).sum(), include_groups=False)
sizes_pr = prs.set_index(prs.apply(lambda r: (r["repo"], r["pr"]), axis=1)).join(
    sizes_pr.rename("size"))["size"].fillna(0)
med_size = sizes_pr.median()
check("T2 median PR size", med_size, 207.0)
szmap = sizes_pr.to_dict()
sa = np.array([szmap[(r, p)] for r, p in zip(t2["repo"], t2["pr_a"])])
sb = np.array([szmap[(r, p)] for r, p in zip(t2["repo"], t2["pr_b"])])
n_core = int(((sa <= 1000) & (sb <= 1000)).sum())
check("T2-core pairs", n_core, 1614)

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7),
                         gridspec_kw={"width_ratios": [1, 1.15]})
axa, axb = axes
yy = np.arange(len(plot_langs))[::-1]
cols = [BLUE] * len(top) + [GRAY]
axa.barh(yy, plot_counts, height=0.62, color=cols)
axa.set_yticks(yy); axa.set_yticklabels(plot_langs)
for yi, c in zip(yy, plot_counts):
    axa.annotate(f"{c:,}", (c, yi), textcoords="offset points", xytext=(3, 0),
                 va="center", fontsize=6.6)
axa.set_xlabel("T2 candidate pairs")
axa.set_xlim(0, max(plot_counts) * 1.14)
axa.grid(True, axis="x", color=GRAY, alpha=0.18, lw=0.5)

vals = np.clip(sizes_pr.values.astype(float), 1, None)
bins = np.logspace(0, np.log10(max(vals.max(), 10)), 36)
axb.hist(vals, bins=bins, color=BLUE, alpha=0.85, edgecolor="white", lw=0.3)
axb.set_xscale("log")
axb.set_ylim(0, axb.get_ylim()[1] * 1.12)
axb.axvline(med_size, color=VERM, lw=1.0)
axb.annotate(f"median = {int(med_size)}", (med_size, axb.get_ylim()[1] * 0.95),
             textcoords="offset points", xytext=(-4, 0), ha="right",
             fontsize=6.6, color=VERM)
axb.axvline(1000, color=GREEN, lw=1.0, ls="--")
axb.annotate(f"core cutoff (1,000 lines)\n{n_core:,} core pairs",
             (1000, axb.get_ylim()[1] * 0.78), textcoords="offset points",
             xytext=(4, 0), fontsize=6.6, color=GREEN)
axb.set_xlabel("PR size, changed lines (additions + deletions)")
axb.set_ylabel(f"PRs (of {len(prs):,})")
axb.grid(True, color=GRAY, alpha=0.18, lw=0.5)
for axx, lab_ in [(axa, "(a)"), (axb, "(b)")]:
    axx.annotate(lab_, (-0.22 if axx is axa else -0.13, 1.02),
                 xycoords="axes fraction", fontsize=8.5, fontweight="bold")
fig.tight_layout(w_pad=2.2)
fig.savefig(f"{OUT}/fig_t2_composition.pdf", bbox_inches="tight")
plt.close(fig)

# ================================================================ FIG F: delay cost
print("=== FIG F: delay-cost Pareto view ===")
def med(policy, tau=None):
    m = er["policy"] == policy
    if tau is not None:
        m &= er["tau"] == tau
    g = er[m]
    assert len(g) == 97
    return g["delay_days_per_pr"].median(), g["avoided_pct"].median()

fifo_x, fifo_y = med("fifo")
ora_x, ora_y = med("oracle")
check("FIFO median delay/PR", round(fifo_x, 1), 60.3, 0.05)
check("oracle median delay/PR", round(ora_x, 1), 6.9, 0.05)
TAUS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8]
pred = [med("predictor", t) for t in TAUS]
p_by_tau = dict(zip(TAUS, pred))
check("tau=0.20 median delay/PR", round(p_by_tau[0.2][0], 1), 16.5, 0.05)
check("tau=0.25 median delay/PR", round(p_by_tau[0.25][0], 1), 7.6, 0.05)

fig, ax = plt.subplots(figsize=(3.4, 2.55))
px = np.array([p[0] for p in pred]); py = np.array([p[1] for p in pred])
o = np.argsort(px)
ax.plot(px[o], py[o], "-", color=BLUE, lw=1.3, marker="o", ms=3.2, mfc="white",
        mew=0.9, mec=BLUE, zorder=3, clip_on=False)
for t, (dx, dy) in {0.2: (5, -10), 0.25: (7, -4), 0.3: (7, -3)}.items():
    ax.annotate(rf"$\tau$={t:g}", p_by_tau[t], textcoords="offset points",
                xytext=(dx, dy), fontsize=7, color=BLUE)
ax.annotate("predictor gate\n" + r"($\tau$ sweep)", p_by_tau[0.25],
            textcoords="offset points", xytext=(-9, 2), ha="right", fontsize=7,
            color=BLUE)
ax.plot(fifo_x, fifo_y, "s", color=VERM, ms=4.5, zorder=4, clip_on=False)
ax.annotate("FIFO serialize", (fifo_x, fifo_y), textcoords="offset points",
            xytext=(2, 6), ha="right", fontsize=7, color=VERM,
            annotation_clip=False)
ax.plot(ora_x, ora_y, "^", color=GREEN, ms=5, zorder=4, clip_on=False)
ax.annotate("oracle gate", (ora_x, ora_y), textcoords="offset points",
            xytext=(-6, -2), ha="right", fontsize=7, color=GREEN)
ax.plot(0, 0, "D", color=BLACK, ms=4, zorder=5, clip_on=False)
ax.annotate("observed", (0, 0), textcoords="offset points", xytext=(6, 3),
            fontsize=7, color=BLACK)
ax.set_xscale("symlog", linthresh=1.0, linscale=0.45)
ax.set_xlim(-0.05, 75)
ax.set_xticks([0, 1, 3, 10, 30, 60])
ax.set_xticklabels(["0", "1", "3", "10", "30", "60"])
ax.set_ylim(-4, 104)
ax.set_xlabel("Added delay (PR-days per PR, median; symlog)")
ax.set_ylabel("Conflicts avoided (%)")
ax.grid(True, color=GRAY, alpha=0.18, lw=0.5)
ax.annotate("n = 97 episodes; marks = medians", (0.98, 0.05),
            xycoords="axes fraction", ha="right", fontsize=6.5, color=GRAY)
fig.savefig(f"{OUT}/fig_t3_delaycost.pdf", bbox_inches="tight")
plt.close(fig)

# ================================================================ copy + report
FIGS = ["fig_t1_roc.pdf", "fig_t1_quality.pdf", "fig_t1_withinrepo.pdf",
        "fig_t3_episodes.pdf", "fig_t2_composition.pdf", "fig_t3_delaycost.pdf"]
for f in FIGS:
    shutil.copy(f"{OUT}/{f}", f"{PAPER}/{f}")
    print(f, os.path.getsize(f"{OUT}/{f}"), "bytes")

print("\n=== CROSSCHECK SUMMARY ===")
bad = [c for c in CHECKS if not c[3]]
print(f"{len(CHECKS) - len(bad)}/{len(CHECKS)} checks passed")
for c in bad:
    print("FAILED:", c)
with open(f"{OUT}/figs_extra_p2_checks.json", "w") as fh:
    json.dump([{"name": n, "got": float(g) if isinstance(g, (int, float, np.floating)) else g,
                "expect": e, "ok": bool(o)} for n, g, e, o in CHECKS], fh, indent=1, default=str)
