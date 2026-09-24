"""Load aligned label-assessment arrays and form label-blind representations."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler, StandardScaler

EXPECTED_SIZE = 2556


EXPECTED_BUCKETS = 41


EXPECTED_FEATURE_ROWS = 2893


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_analysis_inputs(
    archive_dir: Path,
    run_dir: Path,
    classifier_dir: Path,
    *,
    expected_size: int = EXPECTED_SIZE,
    expected_buckets: int = EXPECTED_BUCKETS,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """按stable_id连接标签与2893行双视图特征。"""
    manifest = pd.read_csv(archive_dir / "manifest.csv")
    feature_rows = pd.read_csv(run_dir / "feature_rows.csv").reset_index(
        names="authoritative_feature_row"
    )
    features = np.load(run_dir / "features_105d.npy")
    patches = np.load(classifier_dir / "syllable_logmel_patches.npy", mmap_mode="r")
    feature_names = json.loads(
        (run_dir / "feature_names.json").read_text(encoding="utf-8")
    )
    if (
        len(manifest) != expected_size
        or manifest["stable_id"].nunique() != expected_size
        or manifest["final_bucket_code"].nunique() != expected_buckets
    ):
        raise ValueError(
            f"归档未满足{expected_size}音节/" f"{expected_buckets}个bucket契约"
        )
    if (
        len(feature_rows) != EXPECTED_FEATURE_ROWS
        or feature_rows["stable_id"].nunique() != EXPECTED_FEATURE_ROWS
        or features.shape != (EXPECTED_FEATURE_ROWS, 105)
        or patches.shape[0] != EXPECTED_FEATURE_ROWS
        or len(feature_names) != 105
    ):
        raise ValueError("上游2893行特征或log-mel契约异常")
    joined = manifest.merge(
        feature_rows[["authoritative_feature_row", "stable_id", "source_stem"]].rename(
            columns={"source_stem": "feature_source_stem"}
        ),
        on="stable_id",
        how="left",
        validate="one_to_one",
    )
    if joined["authoritative_feature_row"].isna().any():
        raise ValueError(f"部分{expected_size}音节无法连接feature_rows")
    if (
        joined["source_stem"].astype(str) != joined["feature_source_stem"].astype(str)
    ).any():
        raise ValueError("manifest与feature_rows的source_stem不一致")
    joined = joined.drop(columns=["feature_source_stem"])
    rows = joined["authoritative_feature_row"].to_numpy(dtype=int)
    if not np.isfinite(features[rows]).all() or not np.isfinite(patches[rows]).all():
        raise ValueError(f"{expected_size}音节双视图包含非有限值")
    return joined, features[rows], np.asarray(patches[rows]), feature_names, rows


def prepare_label_blind_views(
    features: np.ndarray,
    patches: np.ndarray,
    feature_names: list[str],
    *,
    seed: int,
    pca_components: int = 30,
) -> tuple[dict[str, np.ndarray], dict]:
    """在不读取bucket标签的前提下构建94D、log-mel与等权融合表示。"""
    perceptual_indices, perceptual_names = feature_subset(feature_names)
    scaler94 = RobustScaler()
    scaled94 = scaler94.fit_transform(features[:, perceptual_indices])
    pca94 = PCA(
        n_components=pca_components,
        random_state=seed,
        svd_solver="randomized",
    )
    projected94 = pca94.fit_transform(scaled94)
    projected94 = StandardScaler().fit_transform(projected94)
    patch_scaler = StandardScaler()
    scaled_patch = patch_scaler.fit_transform(patches)
    patch_pca = PCA(
        n_components=pca_components,
        random_state=seed,
        svd_solver="randomized",
    )
    projected_patch = patch_pca.fit_transform(scaled_patch)
    projected_patch = StandardScaler().fit_transform(projected_patch)
    normalized94 = projected94 / np.sqrt(projected94.shape[1])
    normalized_patch = projected_patch / np.sqrt(projected_patch.shape[1])
    fused = np.concatenate([normalized94, normalized_patch], axis=1)
    views = {
        "perceptual_94d": scaled94,
        "perceptual_pca30": normalized94,
        "logmel_pca30": normalized_patch,
        "fused_60d": fused,
    }
    metadata = {
        "label_blind": True,
        "perceptual_feature_count": len(perceptual_indices),
        "perceptual_feature_names": perceptual_names,
        "pca_components_per_view": pca_components,
        "perceptual_pca_explained_variance": float(
            pca94.explained_variance_ratio_.sum()
        ),
        "logmel_pca_explained_variance": float(
            patch_pca.explained_variance_ratio_.sum()
        ),
        "fusion": "equal_dimension_normalized_concatenation",
    }
    metadata["_models"] = {
        "scaler94": scaler94,
        "pca94": pca94,
        "patch_scaler": patch_scaler,
        "patch_pca": patch_pca,
    }
    return views, metadata


def recording_blocked_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """生成并验证录音完全互斥的分折。"""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros((len(labels), 1)), labels, groups))
    for train, test in folds:
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("录音阻断分折发生来源泄漏")
    return folds


LOUDNESS_INDICES = {0, 1, 2, 3, 4, 5, 25, 45, 65, 103, 104}


def feature_subset(feature_names: list[str]) -> tuple[list[int], list[str]]:
    indices = [
        index for index in range(len(feature_names)) if index not in LOUDNESS_INDICES
    ]
    return indices, [feature_names[index] for index in indices]


def support_tier(
    n_syllables: int,
    n_recordings: int,
    effective_recordings: float,
) -> str:
    if n_syllables >= 20 and n_recordings >= 5 and effective_recordings >= 3:
        return "high_support"
    if n_syllables >= 10 and n_recordings >= 3:
        return "moderate_support"
    return "low_support"
