"""Resumable, versioned LLM-judge experiment runner.

Writes results/v1/judges/<run_id>/{manifest.json, raw.jsonl, predictions.csv, errors.jsonl}
Never overwrites raw.jsonl rows. Silver labels only.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from semantic_conflicts.config import load_settings
from semantic_conflicts.io import write_csv, write_json
from semantic_conflicts.judges.parser import parse_judge_output
from semantic_conflicts.judges.prompts import load_prompt, render_item
from semantic_conflicts.judges.providers import get_provider
from semantic_conflicts.paths import results_dir
from semantic_conflicts.provenance import git_commit


def official_label(provisional: str, has_diff: bool, settings) -> str:
    if has_diff or provisional in set(settings.judge.official_without_diff):
        return provisional
    return "none"


def _load_done(raw_path: Path) -> set[int]:
    done: set[int] = set()
    if not raw_path.exists():
        return done
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            done.add(int(obj["frame_id"]))
    return done


def run_judge(
    *,
    model: str,
    run_id: str,
    provider_name: str,
    settings,
    subset: str = "full",
    resume: bool = True,
    max_items: int | None = None,
    out_dir: Path | None = None,
    root=None,
) -> Path:
    out_dir = out_dir or results_dir(settings.version, root)
    frame = pd.read_csv(out_dir / "judging_frame.csv.gz")
    if subset == "calibration":
        cal = pd.read_csv(out_dir / "calibration_sheet.csv")
        frame = frame[frame["frame_id"].isin(set(cal["frame_id"]))]
    elif subset == "validation":
        val = pd.read_csv(out_dir / "validation_sheet.csv")
        frame = frame[frame["frame_id"].isin(set(val["frame_id"]))]
    elif subset != "full":
        raise ValueError(subset)
    frame = frame.sort_values("frame_id")
    rubric, prompt_hash, prompt_path = load_prompt(settings.judge.prompt_version, root)
    dest = out_dir / "judges" / run_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "prompt.txt").write_text(rubric, encoding="utf-8")
    raw_path = dest / "raw.jsonl"
    err_path = dest / "errors.jsonl"
    done = _load_done(raw_path) if resume else set()
    todo = frame[~frame["frame_id"].isin(done)]
    if max_items is not None:
        todo = todo.head(int(max_items))
    provider = get_provider(provider_name)
    n_ok = n_err = 0
    with raw_path.open("a", encoding="utf-8") as raw, err_path.open("a", encoding="utf-8") as errf:
        for _, row in todo.iterrows():
            prompt = render_item(rubric, row.to_dict(), body_chars=settings.judge.body_chars)
            delay = 1.0
            last_exc = None
            result = None
            for attempt in range(settings.judge.max_retries):
                try:
                    result = provider.complete(prompt, model=model, timeout=settings.judge.timeout_seconds)
                    break
                except Exception as exc:  # noqa: BLE001 — record and backoff
                    last_exc = exc
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
            rec = {
                "frame_id": int(row["frame_id"]),
                "repo": row["repo"],
                "pr_a": int(row["pr_a"]),
                "pr_b": int(row["pr_b"]),
                "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "model_requested": model,
                "provider": provider_name,
                "prompt_sha256": prompt_hash,
                "attempt_error": str(last_exc) if result is None else None,
            }
            if result is None:
                rec["error"] = "provider_failed"
                errf.write(json.dumps(rec) + "\n")
                n_err += 1
                continue
            rec.update(
                {
                    "returncode": result.returncode,
                    "latency_s": result.latency_s,
                    "raw_text": result.text,
                    "stderr": result.stderr[-2000:],
                    "model_returned": result.model,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "cost_usd": result.cost_usd,
                }
            )
            parsed = parse_judge_output(result.text, categories=set(settings.judge.categories))
            rec["parsed"] = parsed
            if parsed is None:
                rec["error"] = "unparsed"
                errf.write(json.dumps({k: rec[k] for k in rec if k != "raw_text"}) + "\n")
                n_err += 1
            raw.write(json.dumps(rec) + "\n")
            raw.flush()
            n_ok += 1
    # Rebuild predictions from the full raw log (resume-safe, no duplicates)
    rows = []
    seen = set()
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                fid = int(obj["frame_id"])
                if fid in seen:
                    continue
                seen.add(fid)
                parsed = obj.get("parsed")
                if not parsed:
                    continue
                has_diff = bool(frame.loc[frame.frame_id == fid, "has_diff"].iloc[0]) if (frame.frame_id == fid).any() else False
                rows.append(
                    {
                        "frame_id": fid,
                        "repo": obj["repo"],
                        "pr_a": obj["pr_a"],
                        "pr_b": obj["pr_b"],
                        "provisional_category": parsed["category"],
                        "category": official_label(parsed["category"], has_diff, settings),
                        "confidence": parsed["confidence"],
                        "evidence": parsed["evidence"],
                        "has_diff": has_diff,
                        "model": obj.get("model_returned") or obj.get("model_requested"),
                        "label_source": "silver-llm",
                        "run_id": run_id,
                    }
                )
    pred = pd.DataFrame(rows)
    if len(pred) and pred["frame_id"].duplicated().any():
        raise RuntimeError("duplicate frame_id in predictions")
    write_csv(dest / "predictions.csv", pred)
    write_json(
        dest / "manifest.json",
        {
            "run_id": run_id,
            "model_requested": model,
            "provider": provider_name,
            "prompt_path": str(prompt_path),
            "prompt_sha256": prompt_hash,
            "prompt_version": settings.judge.prompt_version,
            "subset": subset,
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": git_commit(root),
            "n_predictions": int(len(pred)),
            "n_raw_ok": n_ok,
            "n_errors_this_session": n_err,
            "label_source": "silver-llm",
            "dataset_version": settings.version,
        },
    )
    return dest


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="semantic_conflicts.judges")
    p.add_argument("--model", default="sonnet")
    p.add_argument("--run-id", required=True)
    p.add_argument("--provider", default="claude_code")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--only", choices=["calibration", "validation", "full"], default="full")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    settings = load_settings(args.config)
    dest = run_judge(
        model=args.model,
        run_id=args.run_id,
        provider_name=args.provider,
        settings=settings,
        subset=args.only,
        resume=not args.no_resume,
        max_items=args.max_items,
        out_dir=args.out,
    )
    print(dest)


if __name__ == "__main__":
    main()
