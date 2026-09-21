"""Replay the actual final-song clustering aid from explicit 105D/RMS inputs."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path

from .assessment import DEFAULT_ASSETS, check_assets
from .io import digest, require


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory with feature_rows.csv, features_105d.npy, rms_energy.npy and feature_names.json; default uses bundled matched feature inputs",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("embedding_2d", "robust_hd", "both"), default="both"
    )
    parser.add_argument("--comparison-neighbors", default="15,20,30")
    parser.add_argument("--comparison-min-dist", default="0.05,0.1,0.2")
    parser.add_argument("--robust-neighbors", default="10,20,30")
    parser.add_argument("--robust-min-dist", default="0.0,0.1")
    parser.add_argument("--min-cluster-size", default="3,5,7,10")
    parser.add_argument("--min-samples", default="1,2,3,5")
    parser.add_argument("--robust-components", type=int, default=10)
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--max-noise", type=float, default=0.35)
    parser.add_argument("--min-clusters", type=int, default=20)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the exact grid without fitting UMAP or HDBSCAN",
    )
    return parser.parse_args(argv)


def load_inputs(input_dir: Path | None):
    import numpy as np
    import pandas as pd

    if input_dir is None:
        check_assets(DEFAULT_ASSETS)
        directory = DEFAULT_ASSETS / "label_assessment/run"
        rms_path = DEFAULT_ASSETS / "clustering/rms_energy.npy"
    else:
        directory = input_dir.resolve()
        rms_path = directory / "rms_energy.npy"
    paths = {
        "features": directory / "features_105d.npy",
        "rows": directory / "feature_rows.csv",
        "rms": rms_path,
        "names": directory / "feature_names.json",
    }
    features = np.load(paths["features"], allow_pickle=False)
    rows = pd.read_csv(paths["rows"])
    rms = np.load(paths["rms"], allow_pickle=False)
    names = json.loads(paths["names"].read_text())
    expected_names = json.loads(
        (DEFAULT_ASSETS / "label_assessment/run/feature_names.json").read_text()
    )
    require(
        features.ndim == 2 and features.shape == (len(rows), 105),
        "Clustering features must have 105 columns aligned with row identities",
    )
    require(rms.shape == (len(rows),), "RMS and feature-row counts differ")
    require(
        np.isfinite(features).all() and np.isfinite(rms).all() and (rms >= 0).all(),
        "Features/RMS contain invalid values",
    )
    require(names == expected_names, "Historical feature-name/order contract differs")
    required = {
        "stable_id",
        "source_stem",
        "source_file",
        "selection_id",
        "begin_time",
        "end_time",
        "low_freq",
        "high_freq",
    }
    require(
        required <= set(rows.columns),
        f"Missing row-identity fields: {sorted(required-set(rows.columns))}",
    )
    require(
        rows.stable_id.notna().all() and rows.stable_id.nunique() == len(rows),
        "Duplicate/missing stable_id",
    )
    return (
        features,
        rows,
        rms,
        names,
        {name: digest(path) for name, path in paths.items()},
    )


def execute(args) -> dict:
    from .clustering import (
        parse_ints,
        parse_floats,
        select_two_dimensional,
        select_robust,
        compare_assignments,
    )

    output = args.output_dir.resolve()
    require(not output.exists(), f"Output must be new: {output}")
    features, rows, rms, names, hashes = load_inputs(args.input_dir)
    sizes, samples, seeds = (
        parse_ints(args.min_cluster_size),
        parse_ints(args.min_samples),
        parse_ints(args.seeds),
    )
    require(
        sizes
        and samples
        and all(value >= 2 for value in sizes)
        and all(value >= 1 for value in samples),
        "Invalid HDBSCAN grid",
    )
    require(
        0 <= args.max_noise <= 1
        and args.min_clusters >= 2
        and args.robust_components >= 2,
        "Invalid clustering acceptance settings",
    )
    if args.mode in ("robust_hd", "both"):
        require(
            len(seeds) >= 2 and len(set(seeds)) == len(seeds),
            "Robust stability requires at least two distinct seeds",
        )
    grid = {
        "embedding_2d": {
            "neighbors": parse_ints(args.comparison_neighbors),
            "min_dist": parse_floats(args.comparison_min_dist),
            "components": 2,
            "seeds": [42],
            "rms_filter": "global 10th percentile; retains rms >= threshold",
        },
        "robust_hd": {
            "neighbors": parse_ints(args.robust_neighbors),
            "min_dist": parse_floats(args.robust_min_dist),
            "components": args.robust_components,
            "seeds": seeds,
            "rms_filter": "none; RMS retained for QC",
        },
        "min_cluster_size": sizes,
        "min_samples": samples,
        "selection_method": "leaf",
        "max_noise": args.max_noise,
        "min_clusters": args.min_clusters,
    }
    for mode in ("embedding_2d", "robust_hd"):
        if args.mode in (mode, "both"):
            require(
                grid[mode]["neighbors"]
                and all(2 <= value < len(rows) for value in grid[mode]["neighbors"]),
                "UMAP neighbors exceed the input cohort",
            )
            require(
                grid[mode]["min_dist"]
                and all(value >= 0 for value in grid[mode]["min_dist"]),
                "Invalid UMAP min_dist grid",
            )
    report = {
        "status": "planned",
        "input_rows": len(rows),
        "feature_shape": list(features.shape),
        "input_sha256": hashes,
        "mode": args.mode,
        "grid": grid,
        "input_basis": (
            "bundled clustering candidate feature rows"
            if args.input_dir is None
            else "explicit caller-supplied feature inputs"
        ),
        "output_role": "microtype candidates for human review; does not assign or overwrite final families",
    }
    if args.dry_run:
        return report
    output.mkdir(parents=True)
    assignments = {}
    if args.mode in ("embedding_2d", "both"):
        assignments["embedding_2d"] = select_two_dimensional(
            features, rows, rms, output / "embedding_2d", args
        )
    if args.mode in ("robust_hd", "both"):
        assignments["robust_hd"] = select_robust(
            features, rows, rms, output / "robust_hd", args
        )
    if len(assignments) == 2:
        compare_assignments(
            assignments["embedding_2d"], assignments["robust_hd"], output / "comparison"
        )
    report.update(
        status="complete",
        feature_names=names,
        versions=runtime_versions(),
    )
    (output / "clustering_run_metadata.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def runtime_versions() -> dict:
    # Runtime versions avoid ambiguous duplicate dist-info directories.
    import numpy
    import pandas
    import sklearn
    import umap

    return {
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "umap-learn": umap.__version__,
        "hdbscan": version("hdbscan"),
    }


def main(argv=None):
    print(json.dumps(execute(parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
