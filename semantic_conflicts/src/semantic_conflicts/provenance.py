"""Run provenance: git commit, hashes, versions, row counts."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

from semantic_conflicts.config import Settings
from semantic_conflicts.io import sha256_file, sha256_text, write_json
from semantic_conflicts.paths import repo_root


def git_commit(root: Path | None = None) -> str | None:
    root = root or repo_root()
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except OSError:
        return None
    return None


def dependency_versions(pkgs: list[str] | None = None) -> dict[str, str]:
    pkgs = pkgs or [
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "matplotlib",
        "pyarrow",
        "pydantic",
        "PyYAML",
    ]
    out = {}
    for p in pkgs:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "not-installed"
    return out


def build_manifest(
    *,
    settings: Settings,
    inputs: Mapping[str, Path],
    outputs: Mapping[str, Path],
    row_counts: Mapping[str, Any],
    pool_counts: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    cfg_text = settings.source_path.read_text(encoding="utf-8") if settings.source_path else ""
    man = {
        "dataset_version": settings.version,
        "config_version": settings.version,
        "config_path": str(settings.source_path) if settings.source_path else None,
        "config_sha256": sha256_text(cfg_text) if cfg_text else None,
        "git_commit": git_commit(root),
        "utc_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "dependency_versions": dependency_versions(),
        "seed": settings.seed,
        "sampling_seed": settings.sampling.seed,
        "input_sha256": {k: sha256_file(v) for k, v in inputs.items() if v.exists()},
        "output_sha256": {k: sha256_file(v) for k, v in outputs.items() if v.exists()},
        "row_counts": dict(row_counts),
        "pool_counts": dict(pool_counts),
    }
    if extra:
        man["extra"] = dict(extra)
    return man


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    write_json(path, manifest)


def compare_manifests(current: dict, frozen: dict, *, keys: tuple[str, ...] = ("output_sha256", "pool_counts")) -> list[str]:
    diffs = []
    for key in keys:
        a, b = current.get(key), frozen.get(key)
        if a != b:
            diffs.append(f"{key} mismatch:\n  current={a}\n  frozen={b}")
    return diffs
