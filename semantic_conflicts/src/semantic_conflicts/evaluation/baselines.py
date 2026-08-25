"""Baselines for semantic-conflict prediction. No outcome leakage."""

from __future__ import annotations


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from semantic_conflicts.file_classes import classify_path, is_pure_config
from semantic_conflicts.leakage import assert_no_leakage
from semantic_conflicts.pools import is_duplicate_title, parent_dir_prefix, title_jaccard


def always_safe(n: int) -> np.ndarray:
    return np.zeros(n, dtype=float)


def file_overlap_score(conflict: pd.Series) -> np.ndarray:
    return conflict.astype(float).to_numpy()


def pure_config_aware_overlap(pairs: pd.DataFrame, files_idx: dict) -> np.ndarray:
    scores = np.zeros(len(pairs), dtype=float)
    for i, (r, a, b, c) in enumerate(zip(pairs.repo, pairs.pr_a, pairs.pr_b, pairs.conflict)):
        if not bool(c):
            continue
        shared = files_idx.get((str(r), int(a)), set()) & files_idx.get((str(r), int(b)), set())
        classes = [classify_path(p) for p in shared]
        scores[i] = 0.0 if is_pure_config(classes) else 1.0
    return scores


def depth2_near_miss_score(pairs: pd.DataFrame, files_idx: dict, max_dir_components: int = 2) -> np.ndarray:
    scores = np.zeros(len(pairs), dtype=float)
    for i, (r, a, b, c) in enumerate(zip(pairs.repo, pairs.pr_a, pairs.pr_b, pairs.conflict)):
        if bool(c):
            continue
        ka = {parent_dir_prefix(f, max_dir_components=max_dir_components) for f in files_idx.get((str(r), int(a)), set())}
        kb = {parent_dir_prefix(f, max_dir_components=max_dir_components) for f in files_idx.get((str(r), int(b)), set())}
        ka.discard(None)
        kb.discard(None)
        scores[i] = float(bool(ka & kb))
    return scores


def duplicate_title_score(title_a: list[str], title_b: list[str]) -> np.ndarray:
    return np.array([float(is_duplicate_title(a, b)) for a, b in zip(title_a, title_b)])


def tfidf_text_similarity(text_a: list[str], text_b: list[str]) -> np.ndarray:
    docs = [*(str(x) for x in text_a), *(str(x) for x in text_b)]
    vec = TfidfVectorizer(min_df=1, token_pattern=r"[a-zA-Z0-9_]+")
    try:
        X = vec.fit_transform(docs)
    except ValueError:
        return np.zeros(len(text_a), dtype=float)
    n = len(text_a)
    sim = np.array([cosine_similarity(X[i], X[n + i])[0, 0] for i in range(n)])
    return sim


def path_token_similarity(files_a: list[list[str]], files_b: list[list[str]]) -> np.ndarray:
    out = np.zeros(len(files_a), dtype=float)
    for i, (fa, fb) in enumerate(zip(files_a, files_b)):
        ta = set("/".join(p.split("/")[:2]) for p in fa)
        tb = set("/".join(p.split("/")[:2]) for p in fb)
        union = ta | tb
        out[i] = (len(ta & tb) / len(union)) if union else 0.0
    return out


def hybrid_features(frame: pd.DataFrame, files_idx: dict, title_a, title_b, body_a, body_b) -> pd.DataFrame:
    """Legal prediction-time features only."""
    cols = {
        "title_jaccard": [title_jaccard(a, b) for a, b in zip(title_a, title_b)],
        "dup_title": duplicate_title_score(title_a, title_b),
        "conflict": frame["conflict"].astype(float).to_numpy(),
        "cross_agent": frame["cross_agent"].astype(float).to_numpy() if "cross_agent" in frame.columns else 0,
        "near_miss": depth2_near_miss_score(frame, files_idx),
        "tfidf": tfidf_text_similarity(
            [f"{a} {b}" for a, b in zip(title_a, body_a)],
            [f"{a} {b}" for a, b in zip(title_b, body_b)],
        ),
    }
    feat = pd.DataFrame(cols)
    assert_no_leakage(list(feat.columns), retrospective=False)
    return feat


def fit_hybrid(X: pd.DataFrame, y: np.ndarray, seed: int = 20260824) -> LogisticRegression:
    clf = LogisticRegression(max_iter=200, random_state=seed)
    clf.fit(X.to_numpy(), y)
    return clf
