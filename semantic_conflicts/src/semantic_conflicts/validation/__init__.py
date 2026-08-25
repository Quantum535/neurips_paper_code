"""Category I repository-level validation bundles.

Machine evidence is never gold. A human fills ``human_verdict``.
"""

from __future__ import annotations

from pathlib import Path

from semantic_conflicts.io import write_json

__all__ = ["build_bundle", "write_bundle"]


def build_bundle(
    *,
    repo: str,
    pr_a: int,
    pr_b: int,
    prediction: dict,
    definition_excerpt: str = "",
    use_excerpt: str = "",
    dependency_path: str = "",
    base_sha: str | None = None,
    head_sha_a: str | None = None,
    head_sha_b: str | None = None,
    evidence_status: str = "plausible_unverified",
) -> dict:
    """evidence_status: verified | plausible_unverified | false_positive | unavailable."""
    return {
        "repo": repo,
        "pr_a": int(pr_a),
        "pr_b": int(pr_b),
        "relevant_files": {
            "definition_file": prediction.get("definition_file"),
            "consumer_file": prediction.get("consumer_file"),
        },
        "changed_symbol_or_key": prediction.get("symbol"),
        "definition_excerpt": definition_excerpt[:2000],
        "use_site_excerpt": use_excerpt[:2000],
        "dependency_path": dependency_path,
        "machine_rationale": {
            "direction": prediction.get("direction"),
            "relation": prediction.get("relation"),
            "confidence": prediction.get("confidence"),
            "n_evidence": prediction.get("n_evidence"),
            "evidence": prediction.get("evidence", []),
        },
        "evidence_status": evidence_status,
        "shas": {"base": base_sha, "head_a": head_sha_a, "head_b": head_sha_b},
        "human_verdict": "",
        "human_notes": "",
        "label_source": "machine-proposal-not-gold",
    }


def write_bundle(path: Path, bundle: dict) -> None:
    write_json(path, bundle)
