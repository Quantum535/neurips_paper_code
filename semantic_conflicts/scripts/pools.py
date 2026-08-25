#!/usr/bin/env python3
"""Backward-compatible CLI. Delegates to the canonical package."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from semantic_conflicts.pools import main

if __name__ == "__main__":
    main()
