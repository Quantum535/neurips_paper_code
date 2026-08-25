#!/usr/bin/env python3
"""Backward-compatible judging-frame CLI. Uses disjoint-stratum + prevalence sampling."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from semantic_conflicts.pipeline import main

if __name__ == "__main__":
    main(["deterministic", *sys.argv[1:]])
