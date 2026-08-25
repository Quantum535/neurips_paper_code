import os
import pandas as pd, numpy as np

SP = os.environ.get('MG_DERIVED', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'derived'))
D = os.environ.get('MG_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

test = pd.read_csv(f"{D}/test_pairs_for_llm.csv.gz")
llm = pd.read_csv(f"{D}/llm_scores.csv")
lab = pd.read_csv(f"{SP}/pairs_labeled.csv.gz")

print("test", test.shape, "llm", llm.shape, "lab", lab.shape)

# join test + llm
m = test.merge(llm, on=["repo", "pr_a", "pr_b"], how="inner")
print("test+llm:", m.shape)

# join labels
lab_small = lab[["repo","pr_a","pr_b","conflict","opened_a","closed_a","opened_b","closed_b"]]
m2 = m.merge(lab_small, on=["repo","pr_a","pr_b"], how="inner")
print("test+llm+labels (direct):", m2.shape)

if len(m2) < len(m):
    # try swapped key
    lab_sw = lab_small.rename(columns={"pr_a":"pr_b","pr_b":"pr_a","opened_a":"opened_b","opened_b":"opened_a","closed_a":"closed_b","closed_b":"closed_a"})
    m3 = m.merge(lab_sw, on=["repo","pr_a","pr_b"], how="inner")
    print("swapped match:", m3.shape)

print("n =", len(m2), "positive rate =", m2["conflict"].mean())
print("dup keys:", m2.duplicated(["repo","pr_a","pr_b"]).sum())
m2.to_pickle(f"{SP}/joined.pkl")
