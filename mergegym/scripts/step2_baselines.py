import os
import pandas as pd, numpy as np, re, math
from collections import Counter

SP = os.environ.get('MG_DERIVED', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'derived'))
D = os.environ.get('MG_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

df = pd.read_pickle(f"{SP}/joined.pkl")
print("columns:", list(df.columns))
y = df["conflict"].astype(int).values

def auc(scores, labels):
    """Rank-based (Mann-Whitney) AUC with tie correction."""
    s = np.asarray(scores, dtype=float)
    l = np.asarray(labels)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j+1] == sorted_s[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j+1]] = avg_rank
        i = j + 1
    n1 = l.sum(); n0 = len(l) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (ranks[l == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)

# sanity: LLM prob AUC should be ~0.704
print("LLM prob AUC (sanity):", round(auc(df["prob"].values, y), 4))

# ---------- (a) repo conflict base rate, from full pairs_labeled EXCLUDING test pairs ----------
lab = pd.read_csv(f"{SP}/pairs_labeled.csv.gz")
testkeys = set(zip(df["repo"], df["pr_a"], df["pr_b"]))
lab["_key"] = list(zip(lab["repo"], lab["pr_a"], lab["pr_b"]))
lab_ex = lab[~lab["_key"].isin(testkeys)]
print("pairs_labeled excl test:", len(lab_ex), "(removed", len(lab)-len(lab_ex), ")")
repo_rate = lab_ex.groupby("repo")["conflict"].mean()
global_rate = lab_ex["conflict"].mean()
df["f_repo_rate"] = df["repo"].map(repo_rate).fillna(global_rate).values
n_missing = (~df["repo"].isin(repo_rate.index)).sum()
print("test pairs whose repo has no non-test pairs (filled w/ global %.4f):" % global_rate, n_missing)
print("(a) repo base-rate AUC:", round(auc(df["f_repo_rate"].values, y), 4))

# ---------- (b) overlap duration in hours ----------
for c in ["opened_a","closed_a","opened_b","closed_b"]:
    df[c] = pd.to_datetime(df[c], utc=True, format="mixed")
print("missing closed_a:", df["closed_a"].isna().sum(), "closed_b:", df["closed_b"].isna().sum())
end_cap = max(df["closed_a"].max(), df["closed_b"].max())
ca = df["closed_a"].fillna(end_cap); cb = df["closed_b"].fillna(end_cap)
start = np.maximum(df["opened_a"].values, df["opened_b"].values)
end = np.minimum(ca.values, cb.values)
ov_h = (end - start) / np.timedelta64(1, "h")
ov_h = np.maximum(ov_h.astype(float), 0.0)
df["f_overlap_h"] = ov_h
print("overlap hours: median %.2f mean %.2f max %.1f" % (np.median(ov_h), ov_h.mean(), ov_h.max()))
print("(b) overlap-hours AUC:", round(auc(ov_h, y), 4))

# ---------- (c) TF-IDF cosine between task_a and task_b ----------
tok_re = re.compile(r"[a-z0-9_]+")
def toks(t):
    return tok_re.findall(str(t).lower())

docs_a = [toks(t) for t in df["task_a"]]
docs_b = [toks(t) for t in df["task_b"]]
# IDF over all task texts in test set (each side counted as a doc)
dfreq = Counter()
all_docs = docs_a + docs_b
for d in all_docs:
    dfreq.update(set(d))
N = len(all_docs)
idf = {w: math.log((1 + N) / (1 + c)) + 1.0 for w, c in dfreq.items()}

def tfidf_cos(da, db):
    ca, cb = Counter(da), Counter(db)
    va = {w: (1 + math.log(c)) * idf[w] for w, c in ca.items()}
    vb = {w: (1 + math.log(c)) * idf[w] for w, c in cb.items()}
    na = math.sqrt(sum(v*v for v in va.values())); nb = math.sqrt(sum(v*v for v in vb.values()))
    if na == 0 or nb == 0:
        return 0.0
    dot = sum(v * vb[w] for w, v in va.items() if w in vb)
    return dot / (na * nb)

df["f_tfidf_cos"] = [tfidf_cos(a, b) for a, b in zip(docs_a, docs_b)]
print("tfidf cos: median %.3f mean %.3f" % (df["f_tfidf_cos"].median(), df["f_tfidf_cos"].mean()))
print("(c) TF-IDF cosine AUC:", round(auc(df["f_tfidf_cos"].values, y), 4))

# ---------- (d) title Jaccard ----------
texts = pd.read_csv(f"{D}/data/pr_texts.csv.gz")
tmap = texts.set_index(["repo","pr"])["title"].to_dict()
df["title_a"] = [tmap.get((r,p), "") for r,p in zip(df["repo"], df["pr_a"])]
df["title_b"] = [tmap.get((r,p), "") for r,p in zip(df["repo"], df["pr_b"])]
print("missing titles a/b:", (df["title_a"]=="").sum(), (df["title_b"]=="").sum())
def jac(a, b):
    sa, sb = set(toks(a)), set(toks(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
df["f_title_jac"] = [jac(a,b) for a,b in zip(df["title_a"], df["title_b"])]
print("(d) title Jaccard AUC:", round(auc(df["f_title_jac"].values, y), 4))

df.to_pickle(f"{SP}/joined_feats.pkl")
