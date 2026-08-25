"""Resumable GitHub PR evidence cache. Never commits credentials or raw licensed blobs by default."""

from semantic_conflicts.github.fetch import fetch_pr_evidence

__all__ = ["fetch_pr_evidence"]
