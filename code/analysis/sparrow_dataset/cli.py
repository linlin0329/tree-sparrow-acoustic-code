"""Commands for dataset validation, processing and internal assessment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

STAGES = {
    "reviewed-labels": "annotation.archive",
    "cluster": "cluster",
    "mp3-assessment": "assessments.mp3_assessment",
    "inventory-audit": "processing.inventory_audit",
    "inventory": "processing.build_recording_manifest",
    "prepare-qc": "processing.prepare_release_qc",
    "audio-qc": "processing.audit_release_audio",
    "summarize-qc": "processing.summarize_release_qc",
    "prepare-corpus": "processing.prepare_corpus",
    "assemble-release": "processing.build_dataset_release_bundle",
    "recording-features": "features.recording",
    "all-recording-features": "features.full_recordings",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("inspect", "hashes", "reconstruct", "export-raven", "features"):
        sub = commands.add_parser(name)
        sub.add_argument(
            "--package", type=Path, required=True, help="Released dataset directory"
        )
        if name in ("reconstruct", "features"):
            sub.add_argument("--syllable-id")
        if name in ("reconstruct", "export-raven", "features"):
            sub.add_argument("--output-dir", type=Path, required=name != "reconstruct")
        if name == "features":
            sub.add_argument(
                "--kind", choices=("acoustic", "105d", "logmel"), required=True
            )
            sub.add_argument(
                "--run",
                action="store_true",
                help="Compute features; default only prints the plan",
            )
    from .assessment import DEFAULT_ASSETS

    history = commands.add_parser(
        "assessment", help="Validate matched inputs or plan an assessment"
    )
    history.add_argument("task", choices=("inspect", "hashes", "taxonomy", "songiness"))
    history.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    history.add_argument("--output-dir", type=Path)
    history.add_argument(
        "--run",
        action="store_true",
        help="Start the assessment computation; may refit models",
    )
    process = commands.add_parser(
        "process", help="Plan a processing stage; --run starts it"
    )
    process.add_argument("--run", action="store_true")
    process.add_argument("stage", choices=tuple(STAGES))
    process.add_argument(
        "arguments", nargs=argparse.REMAINDER, help="Pass stage arguments after --"
    )
    stage_help = commands.add_parser(
        "stage-help", help="Show a processing stage's full arguments"
    )
    stage_help.add_argument("stage", choices=tuple(STAGES))
    segments = commands.add_parser(
        "aggregate-segments", help="Aggregate archived segment CSVs without inference"
    )
    segments.add_argument("--segments", type=Path, action="append", required=True)
    segments.add_argument("--output", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> None:
    ap = parser()
    args = ap.parse_args(argv)
    if args.command == "assessment":
        from .assessment import check_assets, inspect_assessment, replay

        if args.task in ("inspect", "hashes"):
            report = (
                inspect_assessment(args.assets.resolve())
                if args.task == "inspect"
                else check_assets(args.assets.resolve())
            )
        else:
            if args.output_dir is None:
                ap.error("Historical replay requires --output-dir")
            report = replay(
                args.task,
                args.assets.resolve(),
                args.output_dir.resolve(),
                run=args.run,
            )
    elif args.command in ("stage-help", "process"):
        command = [sys.executable, "-m", "sparrow_dataset." + STAGES[args.stage]]
        if args.command == "stage-help":
            subprocess.run(command + ["--help"], check=True)
            return
        arguments = (
            args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
        )
        command.extend(arguments)
        report = {"status": "planned", "command": command, "run_requested": args.run}
        if args.run:
            subprocess.run(command, check=True)
            report["status"] = "completed"
    elif args.command == "aggregate-segments":
        import pandas as pd
        from .features.segments import per_recording_allseg
        from .io import require

        require(not args.output.exists(), "Output exists")
        frame = pd.concat(
            [per_recording_allseg(path) for path in args.segments], ignore_index=True
        )
        require(
            not frame.source_file.duplicated().any(),
            "Duplicate recording keys across segment inputs",
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
        report = {"status": "complete", "recordings": len(frame)}
    elif args.command in ("inspect", "hashes", "export-raven"):
        from .release import check_hashes, export_raven, inspect_package

        if args.command == "inspect":
            report = inspect_package(args.package.resolve())
        elif args.command == "hashes":
            report = check_hashes(args.package.resolve())
        else:
            report = export_raven(args.package.resolve(), args.output_dir.resolve())
    elif args.command == "reconstruct":
        from .audio import reconstruct

        report = reconstruct(
            args.package.resolve(),
            args.syllable_id,
            args.output_dir.resolve() if args.output_dir else None,
        )
    else:
        from .audio import extract_features

        if args.run:
            report = extract_features(
                args.package.resolve(),
                args.output_dir.resolve(),
                args.kind,
                args.syllable_id,
            )
        else:
            report = {
                "status": "planned",
                "kind": args.kind,
                "package": str(args.package.resolve()),
                "syllable_id": args.syllable_id,
                "output_dir": str(args.output_dir.resolve()),
                "boundary_basis": "Released integer frames",
            }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
