"""Load versioned judge prompts from disk and hash them."""

from __future__ import annotations

from pathlib import Path

from semantic_conflicts.io import sha256_text
from semantic_conflicts.paths import prompts_dir

DEFAULT_PROMPT_NAME = "judge_v1.txt"


def load_prompt(version: str = "v1", root=None) -> tuple[str, str, Path]:
    path = prompts_dir(root) / f"judge_{version}.txt"
    if not path.exists():
        path = prompts_dir(root) / DEFAULT_PROMPT_NAME
    text = path.read_text(encoding="utf-8")
    return text, sha256_text(text), path


def render_item(rubric: str, row: dict, *, body_chars: int = 1200) -> str:
    def side(s: str) -> str:
        body = str(row.get(f"body_{s}", "") or "")[:body_chars]
        return (
            f"--- PR {s} ---\nTITLE: {row.get(f'title_{s}', '')}\n"
            f"DESCRIPTION: {body}\n"
            f"CHANGED FILES ({row.get(f'n_files_{s}', '')}):\n{row.get(f'files_{s}', '')}\n"
        )

    shared = row.get("files_shared") or "(none)"
    repo = row.get("repo", "")
    return rubric + f"\nREPOSITORY: {repo}\n\n" + side("a") + "\n" + side("b") + f"\nFILES CHANGED BY BOTH:\n{shared}\n"
