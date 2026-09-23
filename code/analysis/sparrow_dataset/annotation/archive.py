"""Apply a reviewed leaf mapping to a new CSV audit archive. Audio is copied only when an explicit matching audio root is supplied."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from ..io import digest, require, safe_path
from ..assessment import DEFAULT_ASSETS, check_assets
from .refinement import build_crosswalk, build_decision_audit, update_complete_audit


def _copy_audio(retained: pd.DataFrame, audio_root: Path, staging: Path) -> None:
    """Preserve the original 48 kHz / mono / FLOAT and byte-identity checks."""
    import soundfile as sf

    for row in retained.itertuples(index=False):
        source = safe_path(audio_root, str(row.pre_sdt_refined_output_relative_path))
        require(
            source.is_file() and not source.is_symlink(),
            f"missing or symlinked source audio: {source}",
        )
        expected_hash = str(row.output_audio_sha256)
        require(
            digest(source) == expected_hash, f"source audio hash mismatch: {source}"
        )
        info = sf.info(source)
        require(
            info.samplerate == 48000 and info.channels == 1 and info.subtype == "FLOAT",
            f"source audio format mismatch: {source}",
        )
        target = safe_path(staging, str(row.output_relative_path))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        require(
            digest(target) == expected_hash, f"copied audio hash mismatch: {target}"
        )


def materialize(
    manifest_path: Path,
    prior_audit_path: Path,
    mapping_path: Path,
    output_dir: Path,
    *,
    design_name: str = "sdt_refinement_plus_taxonomy_r2",
    audio_root: Path | None = None,
    canonical: bool = True,
    dry_run: bool = False,
) -> dict:
    """Validate and write a new final mapping archive from three explicit CSVs.

    The input manifest must already represent the manually reviewed SDT cohort.
    No labels are inferred from microtype, filenames, directories or Selection.
    The full input columns survive in the decisions and retained manifest. Only
    the final label/path fields are updated; prior deletions remain audited.
    A CSV-only output names archived WAV locations without copying audio.
    """
    inputs = {
        "manifest": Path(manifest_path),
        "prior_audit": Path(prior_audit_path),
        "mapping": Path(mapping_path),
    }
    output = Path(output_dir).absolute()
    require(
        not output.exists() and not output.is_symlink(),
        f"output must be a new directory: {output}",
    )
    require(bool(design_name.strip()), "design_name cannot be empty")
    before = {name: digest(path) for name, path in inputs.items()}
    manifest = pd.read_csv(inputs["manifest"])
    prior = pd.read_csv(inputs["prior_audit"])
    mapping = pd.read_csv(inputs["mapping"])
    decisions = build_decision_audit(
        manifest, mapping, design_name=design_name, canonical=canonical
    )
    complete = update_complete_audit(prior, decisions, canonical=canonical)
    retained = (
        decisions.loc[decisions["sdt_refined_action"].ne("delete_whole_bucket")]
        .copy()
        .sort_values(
            [
                "syllable_structure",
                "final_group_id",
                "final_bucket_code",
                "stable_id",
            ]
        )
        .reset_index(drop=True)
    )
    complete = complete.sort_values(["complete_output_status", "stable_id"])
    deleted = complete.loc[
        complete["complete_output_status"].str.startswith("deleted", na=False)
    ].copy()
    crosswalk = build_crosswalk(decisions)
    summary = {
        "profile": "canonical2556" if canonical else "custom_not_canonical2556",
        "label_version": "sdt-taxonomy-r2-2026-09-24",
        "boundary_basis": "historical_research_waveforms_before_R1",
        "design_name": design_name,
        "input_rows": len(manifest),
        "retained_rows": len(retained),
        "new_deleted_rows": len(decisions) - len(retained),
        "cumulative_audit_rows": len(complete),
        "cumulative_deleted_rows": len(deleted),
        "families": int(retained["final_group_code"].nunique()),
        "leaf_buckets": int(retained["final_bucket_code"].nunique()),
        "bucket_counts": retained["final_bucket_code"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "inputs": {
            name: {"path": str(path.absolute()), "sha256": before[name]}
            for name, path in inputs.items()
        },
        "source_raven_row_present": "source_raven_row" in manifest,
        "audio_copied": bool(audio_root is not None and not dry_run),
        "audio_root": (
            str(Path(audio_root).absolute()) if audio_root is not None else None
        ),
        "output_dir": str(output),
        "dry_run": dry_run,
        "scope": "reviewed_mapping; no boundary revision or model fitting",
    }
    require(
        before == {name: digest(path) for name, path in inputs.items()},
        "input CSV changed during mapping",
    )
    if dry_run:
        return summary
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        if audio_root is not None:
            _copy_audio(retained, Path(audio_root), staging)
        for filename, frame in (
            ("manifest.csv", retained),
            ("sdt_refinement_decision_audit.csv", decisions),
            ("complete_final_review_audit.csv", complete),
            ("deleted_syllables_audit.csv", deleted),
            ("sdt_refinement_crosswalk.csv", crosswalk),
        ):
            frame.to_csv(staging / filename, index=False)
        shutil.copy2(
            inputs["mapping"],
            staging
            / (
                "leaf_renumber_mapping.csv.gz"
                if inputs["mapping"].suffix == ".gz"
                else "leaf_renumber_mapping.csv"
            ),
        )
        require(
            before == {name: digest(path) for name, path in inputs.items()},
            "input CSV changed during materialization",
        )
        summary["output_sha256"] = {
            path.name: digest(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        require(
            not output.exists() and not output.is_symlink(),
            f"output appeared during materialization: {output}",
        )
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_ASSETS / "annotation_refinement/base_manifest.csv",
    )
    parser.add_argument(
        "--prior-audit",
        type=Path,
        default=DEFAULT_ASSETS / "annotation_refinement/prior_audit.csv",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_ASSETS / "annotation_refinement/leaf_mapping.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--design-name", default="sdt_refinement_plus_taxonomy_r2")
    parser.add_argument(
        "--profile", choices=("canonical2556", "custom"), default="canonical2556"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    default_root = DEFAULT_ASSETS / "annotation_refinement"
    if (
        args.manifest == default_root / "base_manifest.csv"
        and args.prior_audit == default_root / "prior_audit.csv"
        and args.mapping == default_root / "leaf_mapping.csv"
    ):
        check_assets(DEFAULT_ASSETS)
    result = materialize(
        args.manifest,
        args.prior_audit,
        args.mapping,
        args.output_dir,
        design_name=args.design_name,
        audio_root=args.audio_root,
        canonical=args.profile == "canonical2556",
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
