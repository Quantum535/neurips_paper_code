"""Pluggable model providers. Core evaluation does not require Claude."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class ProviderResult:
    text: str
    returncode: int
    stderr: str
    latency_s: float
    model: str
    provider: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    raw_meta: dict | None = None


class Provider(Protocol):
    name: str

    def complete(self, prompt: str, *, model: str, timeout: int) -> ProviderResult: ...


class ClaudeCodeProvider:
    name = "claude_code"

    def complete(self, prompt: str, *, model: str, timeout: int) -> ProviderResult:
        if shutil.which("claude") is None:
            raise RuntimeError("`claude` CLI not found")
        if os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is set; unset it to use the Claude Code subscription login"
            )
        t0 = time.monotonic()
        p = subprocess.run(
            ["claude", "-p", prompt, "--model", model],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return ProviderResult(
            text=p.stdout,
            returncode=p.returncode,
            stderr=p.stderr,
            latency_s=time.monotonic() - t0,
            model=model,
            provider=self.name,
        )


class EchoProvider:
    """Deterministic provider for tests. Returns a fixed valid JSON object."""

    name = "echo"

    def __init__(self, payload: str | None = None) -> None:
        self.payload = payload or '{"category": "none", "confidence": 1, "evidence": "echo"}'

    def complete(self, prompt: str, *, model: str, timeout: int) -> ProviderResult:
        return ProviderResult(
            text=self.payload,
            returncode=0,
            stderr="",
            latency_s=0.0,
            model=model,
            provider=self.name,
        )


def get_provider(name: str) -> Provider:
    if name in {"claude", "claude_code", "sonnet"}:
        return ClaudeCodeProvider()
    if name in {"echo", "dummy"}:
        return EchoProvider()
    raise ValueError(f"unknown provider {name}")
