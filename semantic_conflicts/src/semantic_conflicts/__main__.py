"""CLI entry: python -m semantic_conflicts <cmd>"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(
            "usage: python -m semantic_conflicts {pipeline,audit,annotation,github,judge} ...\n"
            "   or: python -m semantic_conflicts.pipeline deterministic\n"
        )
        raise SystemExit(2)
    cmd, rest = argv[0], argv[1:]
    if cmd in {"pipeline", "audit"}:
        mod = __import__(f"semantic_conflicts.{cmd}", fromlist=["main"])
        sys.argv = [f"semantic_conflicts.{cmd}", *rest]
        mod.main()
        return
    if cmd in {"annotation", "annotate"}:
        from semantic_conflicts.annotation.cli import main as amain

        sys.argv = ["semantic_conflicts.annotation", *rest]
        amain()
        return
    if cmd in {"github", "fetch"}:
        from semantic_conflicts.github.fetch import main as gmain

        sys.argv = ["semantic_conflicts.github.fetch_pr_evidence", *rest]
        gmain()
        return
    if cmd in {"judge", "judges"}:
        from semantic_conflicts.judges.runner import main as jmain

        sys.argv = ["semantic_conflicts.judges", *rest]
        jmain()
        return
    print(f"unknown command {cmd}", file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
