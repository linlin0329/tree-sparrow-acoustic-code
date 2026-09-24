"""UMAP/HDBSCAN clustering aid with parameter-grid and seed-stability selection. Outputs are candidates for human review."""

from __future__ import annotations
import argparse
import json
from itertools import combinations
from pathlib import Path
import hdbscan
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler

FEATURE_TYPES = ["mfcc", "mfcc_delta", "spectral", "temporal", "amplitude"]


COMPARISON_SEEDS = [42]


ROBUST_SEEDS = [42, 52, 62]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def fit_hdbscan(
    embedding: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> tuple[hdbscan.HDBSCAN, np.ndarray, dict]:
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="leaf",
        gen_min_span_tree=True,
        prediction_data=True,
    )
    labels = model.fit_predict(embedding)
    keep = labels >= 0
    n_clusters = len(set(labels[keep]))
    noise_ratio = float(np.mean(~keep))
    silhouette = np.nan
    if n_clusters >= 2 and keep.sum() > n_clusters:
        sample_size = min(1000, int(keep.sum()))
        silhouette = float(
            silhouette_score(
                embedding[keep],
                labels[keep],
                sample_size=sample_size,
                random_state=42,
            )
        )
    metrics = {
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "silhouette": silhouette,
        "relative_validity": float(getattr(model, "relative_validity_", np.nan)),
        "mean_cluster_persistence": (
            float(np.mean(model.cluster_persistence_))
            if len(model.cluster_persistence_)
            else np.nan
        ),
    }
    return model, labels, metrics


def build_embedding(
    scaled: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    n_components: int,
    seed: int,
) -> tuple[umap.UMAP, np.ndarray]:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric="euclidean",
        random_state=seed,
        transform_seed=seed,
        verbose=False,
    )
    return reducer, reducer.fit_transform(scaled)


def save_scatter(
    embedding: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 10))
    points = axis.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="nipy_spectral",
        s=7,
        alpha=0.7,
    )
    axis.set_title(title)
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    figure.colorbar(points, ax=axis, label="micro_type")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def assignment_frame(
    rows: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    visualization: np.ndarray,
    rms: np.ndarray,
) -> pd.DataFrame:
    result = rows.copy()
    result["micro_type"] = labels
    result["micro_confidence"] = probabilities
    result["umap_x"] = visualization[:, 0]
    result["umap_y"] = visualization[:, 1]
    result["rms_energy"] = rms
    return result


def select_two_dimensional(
    features: np.ndarray,
    rows: pd.DataFrame,
    rms: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold = float(np.percentile(rms, 10.0))
    keep = rms >= threshold
    selected_features = features[keep]
    selected_rows = rows.loc[keep].reset_index(drop=True)
    selected_rms = rms[keep]
    scaler = RobustScaler()
    scaled = scaler.fit_transform(selected_features)

    grid_rows: list[dict] = []
    candidates: list[dict] = []
    for n_neighbors in parse_ints(args.comparison_neighbors):
        for min_dist in parse_floats(args.comparison_min_dist):
            reducer, embedding = build_embedding(
                scaled, n_neighbors, min_dist, 2, COMPARISON_SEEDS[0]
            )
            for min_cluster_size in parse_ints(args.min_cluster_size):
                for min_samples in parse_ints(args.min_samples):
                    model, labels, metrics = fit_hdbscan(
                        embedding, min_cluster_size, min_samples
                    )
                    record = {
                        "n_neighbors": n_neighbors,
                        "min_dist": min_dist,
                        "min_cluster_size": min_cluster_size,
                        "min_samples": min_samples,
                        **metrics,
                    }
                    grid_rows.append(record)
                    candidates.append(
                        {
                            **record,
                            "reducer": reducer,
                            "embedding": embedding,
                            "model": model,
                            "labels": labels,
                        }
                    )
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(output_dir / "grid_search_results.csv", index=False)
    eligible = [
        item
        for item in candidates
        if item["n_clusters"] >= args.min_clusters
        and item["noise_ratio"] <= args.max_noise
        and np.isfinite(item["silhouette"])
    ]
    if not eligible:
        eligible = [
            item
            for item in candidates
            if item["n_clusters"] >= 2 and np.isfinite(item["silhouette"])
        ]
    if not eligible:
        raise RuntimeError("embedding_2d 参数网格没有可用聚类")
    best = max(eligible, key=lambda item: item["silhouette"])

    assignments = assignment_frame(
        selected_rows,
        best["labels"],
        best["model"].probabilities_,
        best["embedding"],
        selected_rms,
    )
    assignments.to_csv(output_dir / "final_syllable_assignments.csv", index=False)
    np.save(output_dir / "cluster_embedding.npy", best["embedding"])
    joblib.dump(scaler, output_dir / "robust_scaler.joblib")
    joblib.dump(best["reducer"], output_dir / "umap_model.joblib")
    joblib.dump(best["model"], output_dir / "hdbscan_model.joblib")
    save_scatter(
        best["embedding"],
        best["labels"],
        output_dir / "fine_clustering_umap.png",
        f"embedding_2d: {best['n_clusters']} micro-types",
    )
    metadata = {
        key: value
        for key, value in best.items()
        if key not in {"reducer", "embedding", "model", "labels"}
    }
    metadata.update(
        {
            "mode": "embedding_2d",
            "energy_filter": {
                "method": "global_rms_percentile",
                "percentile": 10.0,
                "threshold": threshold,
                "n_removed": int((~keep).sum()),
            },
            "cluster_selection_method": "leaf",
            "note": "使用输入中明确登记的Raven边界。",
        }
    )
    with (output_dir / "clustering_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return assignments


def robust_aggregate(run_grid: pd.DataFrame, labels_by_key: dict) -> pd.DataFrame:
    parameter_columns = [
        "n_neighbors",
        "min_dist",
        "min_cluster_size",
        "min_samples",
    ]
    rows = []
    for params, group in run_grid.groupby(parameter_columns, sort=False):
        key_prefix = tuple(params)
        seed_labels = [
            labels_by_key[(*key_prefix, int(seed))] for seed in group["seed"]
        ]
        pairwise_ari = [
            adjusted_rand_score(left, right)
            for left, right in combinations(seed_labels, 2)
        ]
        record = dict(zip(parameter_columns, params))
        record.update(
            {
                "stability_ari": float(np.mean(pairwise_ari)),
                "median_clusters": float(group["n_clusters"].median()),
                "median_noise": float(group["noise_ratio"].median()),
                "median_silhouette": float(group["silhouette"].median()),
                "median_relative_validity": float(group["relative_validity"].median()),
                "median_persistence": float(group["mean_cluster_persistence"].median()),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def rank_robust_candidates(aggregate: pd.DataFrame) -> pd.DataFrame:
    ranked = aggregate.copy()
    metrics = {
        "stability_ari": True,
        "median_relative_validity": True,
        "median_persistence": True,
        "median_silhouette": True,
        "median_noise": False,
    }
    score = np.zeros(len(ranked), dtype=float)
    for column, higher_is_better in metrics.items():
        values = ranked[column].replace([np.inf, -np.inf], np.nan)
        values = values.fillna(values.median() if values.notna().any() else 0.0)
        percentile = values.rank(pct=True).to_numpy()
        score += percentile if higher_is_better else 1.0 - percentile
    ranked["selection_score"] = score / len(metrics)
    return ranked.sort_values(
        ["selection_score", "stability_ari"], ascending=False
    ).reset_index(drop=True)


def select_robust(
    features: np.ndarray,
    rows: pd.DataFrame,
    rms: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler = RobustScaler()
    scaled = scaler.fit_transform(features)
    seeds = parse_ints(args.seeds)
    grid_rows: list[dict] = []
    labels_by_key: dict[tuple, np.ndarray] = {}
    fitted: dict[tuple, tuple] = {}

    for n_neighbors in parse_ints(args.robust_neighbors):
        for min_dist in parse_floats(args.robust_min_dist):
            for seed in seeds:
                reducer, embedding = build_embedding(
                    scaled,
                    n_neighbors,
                    min_dist,
                    args.robust_components,
                    seed,
                )
                for min_cluster_size in parse_ints(args.min_cluster_size):
                    for min_samples in parse_ints(args.min_samples):
                        model, labels, metrics = fit_hdbscan(
                            embedding, min_cluster_size, min_samples
                        )
                        key = (
                            n_neighbors,
                            min_dist,
                            min_cluster_size,
                            min_samples,
                            seed,
                        )
                        labels_by_key[key] = labels
                        fitted[key] = (reducer, embedding, model)
                        grid_rows.append(
                            {
                                "n_neighbors": n_neighbors,
                                "min_dist": min_dist,
                                "min_cluster_size": min_cluster_size,
                                "min_samples": min_samples,
                                "seed": seed,
                                **metrics,
                            }
                        )

    run_grid = pd.DataFrame(grid_rows)
    run_grid.to_csv(output_dir / "seed_grid_results.csv", index=False)
    aggregate = robust_aggregate(run_grid, labels_by_key)
    eligible = aggregate[
        (aggregate["median_clusters"] >= args.min_clusters)
        & (aggregate["median_noise"] <= args.max_noise)
    ]
    if eligible.empty:
        eligible = aggregate[aggregate["median_clusters"] >= 2]
    ranked = rank_robust_candidates(eligible)
    ranked.to_csv(output_dir / "parameter_stability_summary.csv", index=False)
    if ranked.empty:
        raise RuntimeError("robust_hd 参数网格没有可用聚类")
    best_params = ranked.iloc[0]
    subset = run_grid[
        (run_grid["n_neighbors"] == int(best_params["n_neighbors"]))
        & (run_grid["min_dist"] == float(best_params["min_dist"]))
        & (run_grid["min_cluster_size"] == int(best_params["min_cluster_size"]))
        & (run_grid["min_samples"] == int(best_params["min_samples"]))
    ].sort_values(
        ["relative_validity", "mean_cluster_persistence"],
        ascending=False,
    )
    selected_seed = int(subset.iloc[0]["seed"])
    key = (
        int(best_params["n_neighbors"]),
        float(best_params["min_dist"]),
        int(best_params["min_cluster_size"]),
        int(best_params["min_samples"]),
        selected_seed,
    )
    reducer, embedding, model = fitted[key]
    labels = labels_by_key[key]

    visualizer, visualization = build_embedding(
        scaled,
        int(best_params["n_neighbors"]),
        float(best_params["min_dist"]),
        2,
        selected_seed,
    )
    assignments = assignment_frame(
        rows,
        labels,
        model.probabilities_,
        visualization,
        rms,
    )
    assignments.to_csv(output_dir / "final_syllable_assignments.csv", index=False)
    np.save(output_dir / "cluster_embedding_10d.npy", embedding)
    np.save(output_dir / "visualization_embedding_2d.npy", visualization)
    joblib.dump(scaler, output_dir / "robust_scaler.joblib")
    joblib.dump(reducer, output_dir / "cluster_umap_model.joblib")
    joblib.dump(visualizer, output_dir / "visualization_umap_model.joblib")
    joblib.dump(model, output_dir / "hdbscan_model.joblib")
    save_scatter(
        visualization,
        labels,
        output_dir / "fine_clustering_umap.png",
        f"robust_hd: {len(set(labels)) - (-1 in labels)} micro-types",
    )
    metadata = {
        "mode": "robust_hd",
        "selected_parameters": {
            "n_neighbors": key[0],
            "min_dist": key[1],
            "min_cluster_size": key[2],
            "min_samples": key[3],
            "seed": key[4],
            "n_components": args.robust_components,
        },
        "selection_metrics": {
            column: float(best_params[column])
            for column in [
                "stability_ari",
                "median_clusters",
                "median_noise",
                "median_silhouette",
                "median_relative_validity",
                "median_persistence",
                "selection_score",
            ]
        },
        "energy_filter": "none; rms_energy retained as QC",
        "cluster_selection_method": "leaf",
    }
    with (output_dir / "clustering_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return assignments


def compare_assignments(
    embedding_2d: pd.DataFrame,
    robust: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    merged = embedding_2d[["stable_id", "micro_type", "micro_confidence"]].merge(
        robust[["stable_id", "micro_type", "micro_confidence"]],
        on="stable_id",
        suffixes=("_embedding_2d", "_robust"),
        validate="one_to_one",
    )
    comparable = merged[
        (merged["micro_type_embedding_2d"] >= 0) & (merged["micro_type_robust"] >= 0)
    ]
    metrics = {
        "n_embedding_2d": len(embedding_2d),
        "n_robust": len(robust),
        "n_common": len(merged),
        "n_common_non_noise": len(comparable),
        "ari_non_noise": (
            float(
                adjusted_rand_score(
                    comparable["micro_type_embedding_2d"],
                    comparable["micro_type_robust"],
                )
            )
            if len(comparable)
            else None
        ),
        "nmi_non_noise": (
            float(
                normalized_mutual_info_score(
                    comparable["micro_type_embedding_2d"],
                    comparable["micro_type_robust"],
                )
            )
            if len(comparable)
            else None
        ),
    }
    merged.to_csv(output_dir / "common_syllable_assignments.csv", index=False)
    pd.crosstab(
        comparable["micro_type_embedding_2d"],
        comparable["micro_type_robust"],
    ).to_csv(output_dir / "cluster_crosswalk_counts.csv")

    crosswalk = pd.crosstab(
        comparable["micro_type_embedding_2d"],
        comparable["micro_type_robust"],
    )
    correspondence_rows = []
    for source_mode, matrix in [
        ("embedding_2d_to_robust", crosswalk),
        ("robust_to_embedding_2d", crosswalk.T),
    ]:
        for source_cluster, counts in matrix.iterrows():
            nonzero = counts[counts > 0].sort_values(ascending=False)
            total = int(nonzero.sum())
            if not total:
                continue
            correspondence_rows.append(
                {
                    "direction": source_mode,
                    "source_cluster": int(source_cluster),
                    "best_target_cluster": int(nonzero.index[0]),
                    "best_overlap_n": int(nonzero.iloc[0]),
                    "source_non_noise_n": total,
                    "best_overlap_fraction": float(nonzero.iloc[0] / total),
                    "n_target_clusters": len(nonzero),
                    "relationship": (
                        "mostly_one_to_one"
                        if nonzero.iloc[0] / total >= 0.8
                        else "split_or_merged"
                    ),
                }
            )
    pd.DataFrame(correspondence_rows).to_csv(
        output_dir / "cluster_correspondence_summary.csv", index=False
    )

    side_by_side = merged.copy()
    side_by_side["embedding_2d_concat_wav"] = side_by_side["micro_type_embedding_2d"].map(
        lambda value: (
            f"../embedding_2d/micro_review/concatenated_review/"
            f"{'Noise' if value == -1 else f'Micro_{int(value):03d}'}_concat.wav"
        )
    )
    side_by_side["robust_concat_wav"] = side_by_side["micro_type_robust"].map(
        lambda value: (
            f"../robust_hd/micro_review/concatenated_review/"
            f"{'Noise' if value == -1 else f'Micro_{int(value):03d}'}_concat.wav"
        )
    )
    side_by_side.to_csv(output_dir / "side_by_side_review_index.csv", index=False)
    with (output_dir / "comparison_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
