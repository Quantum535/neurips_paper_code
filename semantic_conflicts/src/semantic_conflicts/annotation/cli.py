"""Export blinded sheets, import labels, agreement, adjudication."""

from __future__ import annotations

import argparse
from pathlib import Path

from semantic_conflicts.config import load_settings
from semantic_conflicts.io import read_csv, write_csv, write_json
from semantic_conflicts.paths import results_dir


def export_round(round_name: str, settings, *, root=None, out_dir: Path | None = None) -> Path:
    from semantic_conflicts.annotation.io_labels import blinded_export

    out_dir = out_dir or results_dir(settings.version, root)
    src = {
        "calibration": out_dir / "calibration_sheet.csv",
        "validation": out_dir / "validation_sheet.csv",
    }[round_name]
    if not src.exists():
        raise FileNotFoundError(f"{src} missing; run pipeline deterministic first")
    dest_dir = out_dir / "annotation" / round_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    df = read_csv(src)
    blinded = blinded_export(df, settings)
    path = dest_dir / "annotator_sheet.csv"
    write_csv(path, blinded)
    write_json(
        dest_dir / "export_meta.json",
        {
            "round": round_name,
            "n": int(len(blinded)),
            "rubric_version": settings.annotation.rubric_version,
            "annotation_version": settings.version,
            "hidden_fields": settings.annotation.hidden_fields,
            "labels": settings.annotation.labels,
        },
    )
    return path


def cmd_import(args, settings) -> None:
    from semantic_conflicts.annotation.io_labels import import_labels

    out_dir = Path(args.out) if args.out else results_dir(settings.version)
    dest = out_dir / "annotation" / args.round / f"labels_{args.annotator}.csv"
    n = import_labels(Path(args.path), dest, annotator=args.annotator, settings=settings)
    print(f"imported {n} rows -> {dest}")


def cmd_agreement(args, settings) -> None:
    from semantic_conflicts.annotation.agreement import compute_agreement, write_agreement_artifacts

    out_dir = Path(args.out) if args.out else results_dir(settings.version)
    adir = out_dir / "annotation" / args.round
    files = sorted(adir.glob("labels_*.csv"))
    if len(files) < 2:
        raise SystemExit(f"need >=2 labels_*.csv in {adir}; found {len(files)}")
    frames = [read_csv(f) for f in files]
    report, confusion = compute_agreement(frames, settings)
    write_agreement_artifacts(out_dir, report, confusion)
    print(f"wrote {out_dir / 'human_agreement.json'}")


def cmd_adjudicate(args, settings) -> None:
    from semantic_conflicts.annotation.io_labels import adjudicate, write_gold

    out_dir = Path(args.out) if args.out else results_dir(settings.version)
    adir = out_dir / "annotation" / args.round
    files = sorted(adir.glob("labels_*.csv"))
    frames = [read_csv(f) for f in files]
    adj_path = Path(args.adjudication) if args.adjudication else None
    adj = read_csv(adj_path) if adj_path and adj_path.exists() else None
    gold = adjudicate(frames, adj, settings)
    dest = out_dir / "gold_labels.csv"
    write_gold(dest, gold, settings)
    waiting = out_dir / "WAITING_FOR_HUMAN_LABELS"
    if waiting.exists() and len(gold):
        waiting.unlink()
    print(f"wrote {dest} n={len(gold)}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="semantic_conflicts.annotation")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--round", required=True, choices=["calibration", "validation"])
    e.add_argument("--config", type=Path, default=None)
    e.add_argument("--out", type=Path, default=None)
    i = sub.add_parser("import-labels")
    i.add_argument("--path", required=True)
    i.add_argument("--annotator", required=True)
    i.add_argument("--round", required=True, choices=["calibration", "validation"])
    i.add_argument("--config", type=Path, default=None)
    i.add_argument("--out", type=Path, default=None)
    a = sub.add_parser("agreement")
    a.add_argument("--round", required=True, choices=["calibration", "validation"])
    a.add_argument("--config", type=Path, default=None)
    a.add_argument("--out", type=Path, default=None)
    j = sub.add_parser("adjudicate")
    j.add_argument("--round", required=True, choices=["calibration", "validation"])
    j.add_argument("--adjudication", type=Path, default=None)
    j.add_argument("--config", type=Path, default=None)
    j.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    settings = load_settings(args.config)
    if args.cmd == "export":
        path = export_round(args.round, settings, out_dir=args.out)
        print(path)
    elif args.cmd == "import-labels":
        cmd_import(args, settings)
    elif args.cmd == "agreement":
        cmd_agreement(args, settings)
    elif args.cmd == "adjudicate":
        cmd_adjudicate(args, settings)
