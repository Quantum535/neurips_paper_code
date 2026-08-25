"""Shared pytest helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from semantic_conflicts.config import load_settings
from semantic_conflicts.paths import repo_root

FIXTURE = Path(__file__).parent / "fixtures" / "tiny"


@pytest.fixture
def settings_v1():
    return load_settings()


@pytest.fixture
def settings_ci():
    root = repo_root()
    return load_settings(root / "semantic_conflicts" / "configs" / "ci.yaml")


@pytest.fixture
def tiny_tables():
    return {
        "pairs": pd.read_csv(FIXTURE / "pairs_labeled.csv"),
        "files": pd.read_csv(FIXTURE / "pr_files.csv"),
        "texts": pd.read_csv(FIXTURE / "pr_texts.csv"),
        "repos": pd.read_csv(FIXTURE / "repo_stats.csv"),
    }
