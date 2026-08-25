#!/usr/bin/env python3
"""Backward-compatible judge CLI."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from semantic_conflicts.judges.runner import main

if __name__ == "__main__":
    main()
