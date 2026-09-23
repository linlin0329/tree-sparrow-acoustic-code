"""Matched assessment inputs and explicit computation plans."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .io import digest, require, safe_path

DEFAULT_ASSETS = Path(__file__).resolve().parent / "assets"


def check_assets(assets: Path) -> dict:
    """Verify every distributed numeric input and identity table."""
    manifest = json.loads((assets / "manifest.json").read_text())
    entries = manifest["files"]
    require(
        len({entry["path"] for entry in entries}) == len(entries),
        "Duplicate asset path",
    )
    for entry in entries:
        path = safe_path(assets, entry["path"])
        require(
            path.stat().st_size == entry["size_bytes"]
            and digest(path) == entry["sha256"],
            f"Assessment input changed: {entry['path']}",
        )
    return {"status": "passed", "files_checked": len(entries)}


def inspect_assessment(assets: Path) -> dict:
    import pandas as pd
    from .assessments import taxonomy, songiness

    checked = check_assets(assets)
    label = assets / "label_assessment"
    frame, features, patches, names, indices = taxonomy.load_analysis_inputs(
        label / "archive",
        label / "run",
        label / "classifier",
        expected_size=2556,
        expected_buckets=41,
    )
    frame = taxonomy.add_taxonomy_columns(frame)
    song = assets / "songiness"
    full, columns = songiness.build_table(
        song, [song / "reliable_2023.csv", song / "reliable_2024.csv"]
    )
    scores = pd.read_csv(song / "scores_all.csv")
    gold = pd.read_csv(song / "song_final_list.csv")
    require(
        not full.source_file.duplicated().any()
        and not scores.source_file.duplicated().any(),
        "Duplicate songiness keys",
    )
    require(
        set(full.source_file) == set(scores.source_file),
        "Feature/score identity mismatch",
    )
    aligned = scores.set_index("source_file").loc[full.source_file]
    require(
        full.source_file.isin(set(gold.base_recording)).tolist()
        == aligned.is_gold.tolist(),
        "Recorded gold membership differs",
    )
    train = scores.is_gold | scores.is_listened
    return {
        "status": "passed",
        "hashes": checked,
        "taxonomy": {
            "syllables": len(frame),
            "families": frame.family.nunique(),
            "leaves": frame.leaf.nunique(),
            "source_excerpts": frame.source_stem.nunique(),
            "features_shape": list(features.shape),
            "patch_shape": list(patches.shape),
            "pair_evidence_rows": len(
                pd.read_csv(label / "pair_evidence/pairwise_similarity_evidence.csv")
            ),
        },
        "songiness": {
            "recordings": len(full),
            "features": len(columns),
            "training_rows": int(train.sum()),
            "positive_training_rows": int(scores.is_gold.sum()),
            "negative_training_rows": int((scores.is_listened & ~scores.is_gold).sum()),
            "deployment_rows": int((~train).sum()),
        },
        "computation": "Input validation only; no feature extraction, PCA fitting, training or inference",
    }


def replay(kind: str, assets: Path, output: Path, *, run: bool = False) -> dict:
    require(not output.exists(), f"Output must be new: {output}")
    label = assets / "label_assessment"
    if kind == "taxonomy":
        command = [
            sys.executable,
            "-m",
            "sparrow_dataset.assessments.taxonomy",
            "--taxonomy-profile",
            "2556-taxonomy-r2",
            "--split-feasibility-policy",
            "retry-seed",
            "--archive-dir",
            str(label / "archive"),
            "--run-dir",
            str(label / "run"),
            "--classifier-dir",
            str(label / "classifier"),
            "--pair-evidence-dir",
            str(label / "pair_evidence"),
            "--output-dir",
            str(output),
            "--seed",
            "42",
            "--pca-components",
            "30",
            "--cv-splits",
            "3",
            "--cv-repeats",
            "5",
            "--bootstrap",
            "200",
            "--geometry-permutations",
            "1000",
            "--classification-permutations",
            "100",
        ]
    else:
        song = assets / "songiness"
        command = [
            sys.executable,
            "-m",
            "sparrow_dataset.assessments.songiness",
            "--features-dir",
            str(song),
            "--output-dir",
            str(output),
            "--gold-list",
            str(song / "song_final_list.csv"),
            "--listened-scores",
            str(song / "scores_all.csv"),
            "--reliable-features",
            str(song / "reliable_2023.csv"),
            "--reliable-features",
            str(song / "reliable_2024.csv"),
            "--jobs",
            "1",
            "--n-trees",
            "500",
            "--folds",
            "5",
        ]
    result = {
        "status": "planned",
        "command": command,
        "assessment_boundary_basis": "canonical2556-boundary-r1-2026-09-17",
        "label_version": "sdt-taxonomy-r2-2026-09-24",
        "run_requested": run,
    }
    if run:
        inspect_assessment(assets)
        subprocess.run(command, check=True)
        result["status"] = "completed"
    return result
