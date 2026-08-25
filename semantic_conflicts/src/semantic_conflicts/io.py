"""IO helpers: load frozen source tables, write versioned artifacts, hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from semantic_conflicts.paths import repo_root


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, df: pd.DataFrame, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz" or str(path).endswith(".csv.gz"):
        df.to_csv(path, index=index, compression="gzip")
    else:
        df.to_csv(path, index=index)


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def as_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).astype(bool)
    return s.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "t"])


def load_source_tables(
    root: Path | None = None,
    *,
    pairs_path: Path | None = None,
    files_path: Path | None = None,
    texts_path: Path | None = None,
    repos_path: Path | None = None,
    settings: Any = None,
) -> dict[str, pd.DataFrame]:
    root = root or repo_root()
    if settings is not None and getattr(settings, "data", None) is not None:
        d = settings.data

        def _p(val, default):
            if val:
                p = Path(val)
                return p if p.is_absolute() else (root / p)
            return default

        pairs_path = pairs_path or _p(d.pairs, None)
        files_path = files_path or _p(d.files, None)
        texts_path = texts_path or _p(d.texts, None)
        repos_path = repos_path or _p(d.repos, None)
    pairs = pd.read_csv(pairs_path or (root / "derived" / "pairs_labeled.csv.gz"))
    files = pd.read_csv(files_path or (root / "data" / "pr_files.csv.gz"))
    texts = pd.read_csv(texts_path or (root / "data" / "pr_texts.csv.gz"))
    repos = pd.read_csv(repos_path or (root / "data" / "repo_stats.csv"))
    for col in ("conflict", "cross_agent", "both_files"):
        if col in pairs.columns:
            pairs[col] = as_bool_series(pairs[col])
    if "merged" in texts.columns:
        texts["merged"] = as_bool_series(texts["merged"])
    return {"pairs": pairs, "files": files, "texts": texts, "repos": repos}


def source_input_paths(root: Path | None = None, settings: Any = None) -> dict[str, Path]:
    root = root or repo_root()
    defaults = {
        "pairs_labeled": root / "derived" / "pairs_labeled.csv.gz",
        "pr_files": root / "data" / "pr_files.csv.gz",
        "pr_texts": root / "data" / "pr_texts.csv.gz",
        "repo_stats": root / "data" / "repo_stats.csv",
    }
    if settings is None or getattr(settings, "data", None) is None:
        return defaults
    d = settings.data

    def _p(val, default):
        if not val:
            return default
        p = Path(val)
        return p if p.is_absolute() else (root / p)

    return {
        "pairs_labeled": _p(d.pairs, defaults["pairs_labeled"]),
        "pr_files": _p(d.files, defaults["pr_files"]),
        "pr_texts": _p(d.texts, defaults["pr_texts"]),
        "repo_stats": _p(d.repos, defaults["repo_stats"]),
    }


def hash_inputs(paths: Mapping[str, Path]) -> dict[str, str]:
    return {k: sha256_file(v) for k, v in paths.items() if v.exists()}
