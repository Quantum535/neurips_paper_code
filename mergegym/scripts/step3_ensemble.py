import os
import pandas as pd, numpy as np

SP = os.environ.get('MG_DERIVED', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'derived'))
df = pd.read_pickle(f"{SP}/joined_feats.pkl")
y = df["conflict"].astype(int).values
n = len(df)

def auc(scores, labels):
    s = np.asarray(scores, dtype=float); l = np.asarray(labels)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float); ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j+1] == ss[i]:
            j += 1
        ranks[order[i:j+1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = l.sum(); n0 = len(l) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (ranks[l == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)

def logreg_fit(X, y, l2=1e-4, iters=200):
    """IRLS/Newton logistic regression with intercept."""
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
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

def cv_auc(feat_cols, seed=0, k=5):
    X = df[feat_cols].values.astype(float).copy()
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    oof = np.full(n, np.nan)
    fold_aucs = []
    for f in range(k):
        te = folds[f]; tr = np.concatenate([folds[g] for g in range(k) if g != f])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        w = logreg_fit((X[tr]-mu)/sd, y[tr])
        oof[te] = logreg_pred(w, (X[te]-mu)/sd)
        fold_aucs.append(auc(oof[te], y[te]))
    return auc(oof, y), np.mean(fold_aucs)

df["f_log_ov"] = np.log1p(df["f_overlap_h"])

base4 = ["f_repo_rate", "f_log_ov", "f_tfidf_cos", "f_title_jac"]
combos = {
    "(e) LR: repo_rate+overlap+tfidf+jaccard": base4,
    "LR: repo_rate + tfidf  [a+c]": ["f_repo_rate", "f_tfidf_cos"],
    "LR: repo_rate + tfidf + LLM prob": ["f_repo_rate", "f_tfidf_cos", "prob"],
    "LR: all4 + LLM prob": base4 + ["prob"],
    "LR: LLM prob only (calibration check)": ["prob"],
}
for name, cols in combos.items():
    pooled, meanf = cv_auc(cols)
    print(f"{name}: pooled OOF AUC={pooled:.4f}  mean fold AUC={meanf:.4f}")

# rank-average ensemble (no fitting)
def ranks(v):
    return pd.Series(v).rank().values / len(v)
ra4 = np.mean([ranks(df[c].values) for c in base4], axis=0)
print("rank-avg (a-d):", round(auc(ra4, y), 4))
ra5 = np.mean([ranks(df[c].values) for c in base4 + ["prob"]], axis=0)
print("rank-avg (a-d)+LLM:", round(auc(ra5, y), 4))
ra_ac = np.mean([ranks(df[c].values) for c in ["f_repo_rate","f_tfidf_cos"]], axis=0)
print("rank-avg (a+c):", round(auc(ra_ac, y), 4))
ra_acp = np.mean([ranks(df[c].values) for c in ["f_repo_rate","f_tfidf_cos","prob"]], axis=0)
print("rank-avg (a+c)+LLM:", round(auc(ra_acp, y), 4))

# ---------- per-repo generalization ----------
rows = []
for repo, g in df.groupby("repo"):
    if len(g) >= 10 and g["conflict"].nunique() == 2:
        rows.append({
            "repo": repo, "n": len(g), "pos": g["conflict"].mean(),
            "auc_llm": auc(g["prob"].values, g["conflict"].astype(int).values),
            "auc_tfidf": auc(g["f_tfidf_cos"].values, g["conflict"].astype(int).values),
            "auc_jac": auc(g["f_title_jac"].values, g["conflict"].astype(int).values),
            "auc_ov": auc(g["f_log_ov"].values, g["conflict"].astype(int).values),
        })
pr = pd.DataFrame(rows)
print("\nrepos with >=10 test pairs & both classes:", len(pr), " pairs covered:", pr["n"].sum())
print("mean within-repo AUC  LLM: %.4f  tfidf: %.4f  jaccard: %.4f  overlap: %.4f"
      % (pr["auc_llm"].mean(), pr["auc_tfidf"].mean(), pr["auc_jac"].mean(), pr["auc_ov"].mean()))
print("median within-repo LLM AUC: %.4f ; repos where LLM AUC>0.5: %d/%d"
      % (pr["auc_llm"].median(), (pr["auc_llm"] > 0.5).sum(), len(pr)))
# weighted by n
print("n-weighted mean within-repo LLM AUC: %.4f" % np.average(pr["auc_llm"], weights=pr["n"]))
# pooled AUC restricted to same-repo comparisons == mean is fine; also report between-repo:
# between-repo signal proxy: AUC of repo mean prob predicting repo mean conflict
rm = df.groupby("repo").agg(p=("prob","mean"), c=("conflict","mean"), n=("prob","size"))
print("repo-level: corr(mean prob, mean conflict) = %.3f over %d repos" % (rm["p"].corr(rm["c"]), len(rm)))
pr.sort_values("auc_llm").to_csv(f"{SP}/per_repo_auc.csv", index=False)

# ---------- error analysis ----------
fp = df[df["conflict"] == False].nlargest(5, "prob")
fn = df[df["conflict"] == True].nsmallest(5, "prob")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
print("\n=== top-5 FALSE POSITIVES (label=0, highest prob) ===")
for _, r in fp.iterrows():
    print(f"prob={r['prob']:.3f} repo={r['repo']} PRs #{r['pr_a']} / #{r['pr_b']}")
    print(f"   A: {r['title_a'][:120]}")
    print(f"   B: {r['title_b'][:120]}")
print("\n=== top-5 FALSE NEGATIVES (label=1, lowest prob) ===")
for _, r in fn.iterrows():
    print(f"prob={r['prob']:.3f} repo={r['repo']} PRs #{r['pr_a']} / #{r['pr_b']}")
    print(f"   A: {r['title_a'][:120]}")
    print(f"   B: {r['title_b'][:120]}")

# how many label-0 pairs have prob >= 0.9, and label-1 with prob <= 0.1
print("\nlabel=0 with prob>=0.9:", ((df['conflict']==False)&(df['prob']>=0.9)).sum())
print("label=1 with prob<=0.1:", ((df['conflict']==True)&(df['prob']<=0.1)).sum())
df.to_pickle(f"{SP}/joined_feats.pkl")
