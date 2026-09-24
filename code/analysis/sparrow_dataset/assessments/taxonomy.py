#!/usr/bin/env python3
"""对2556宏音节分类体系执行只读、分层、录音阻断的内部量化验证。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import binomtest, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder


from .taxonomy_inputs import (load_analysis_inputs, prepare_label_blind_views,
                              recording_blocked_splits, sha256_file, support_tier)
SITE_TO_LEVEL = {site: None for site in ("liujiaxia", "yongxing", "liangzhuang", "shuanghe", "minqin", "guanyinya")}

matplotlib.use("Agg")

EXPECTED_SIZE = 2556
EXPECTED_FAMILIES = 27
EXPECTED_LEAVES = 41
EXPECTED_VARIANT_FAMILIES = 14
EXPECTED_REUSED_PAIRS = 127
TAXONOMY_PROFILE = "canonical2556"
SPLIT_FEASIBILITY_POLICY = "retry-seed"
STRUCTURES = ("single", "double", "triple")
STEM_RE = re.compile(
    r"^(?P<site>[a-z]+)_(?P<date>\d{4}-\d{2}-\d{2})-birdnet-"
)


def file_hash(path: Path) -> str:
    """计算输入文件SHA-256。"""
    return sha256_file(path)


def parse_source_stem(stem: str) -> tuple[str, int]:
    """从冻结录音名提取样地和年份。"""
    match = STEM_RE.match(str(stem))
    if match is None:
        raise ValueError(f"非法source_stem：{stem}")
    site = match.group("site")
    year = int(match.group("date")[:4])
    if site not in SITE_TO_LEVEL:
        raise ValueError(f"未知样地：{site}")
    return site, year


def add_taxonomy_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """补充结构、family、leaf和来源字段并验证层级闭合。"""
    result = frame.copy()
    result["family"] = result["final_group_code"].astype(str)
    result["leaf"] = result["final_bucket_code"].astype(str)
    result["structure"] = result["syllable_structure"].astype(str)
    parsed = result["source_stem"].map(parse_source_stem)
    result[["site", "year"]] = pd.DataFrame(
        parsed.tolist(), index=result.index
    )
    if (
        len(result) != EXPECTED_SIZE
        or result["stable_id"].nunique() != EXPECTED_SIZE
        or result["family"].nunique() != EXPECTED_FAMILIES
        or result["leaf"].nunique() != EXPECTED_LEAVES
        or set(result["structure"]) != set(STRUCTURES)
    ):
        raise ValueError("2556/27 family/41 leaf/SDT层级契约异常")
    if not result.apply(
        lambda row: row["family"].startswith(f"{row['structure']}_"),
        axis=1,
    ).all():
        raise ValueError("family与SDT结构不一致")
    leaf_to_family = result.groupby("leaf")["family"].nunique()
    if not leaf_to_family.eq(1).all():
        raise ValueError("leaf未唯一隶属于family")
    return result


def source_balanced_units(
    representation: np.ndarray,
    frame: pd.DataFrame,
    label_column: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """以label×source_stem中位原型建立录音等权分析单元。"""
    rows: list[dict] = []
    vectors: list[np.ndarray] = []
    for (label, source), group in frame.assign(
        representation_row=np.arange(len(frame))
    ).groupby([label_column, "source_stem"], sort=True):
        indices = group["representation_row"].to_numpy(dtype=int)
        rows.append(
            {
                "label": str(label),
                "source_stem": str(source),
                "n_syllables": int(len(indices)),
            }
        )
        vectors.append(np.median(representation[indices], axis=0))
    return pd.DataFrame(rows), np.vstack(vectors)


def distance_ratio(vectors: np.ndarray, labels: np.ndarray) -> float:
    """返回组间距离中位数与组内距离中位数之比。"""
    distances = np.linalg.norm(
        vectors[:, None, :] - vectors[None, :, :], axis=2
    )
    upper = np.triu(np.ones(distances.shape, dtype=bool), k=1)
    same = labels[:, None] == labels[None, :]
    within = distances[upper & same]
    between = distances[upper & ~same]
    if len(within) == 0 or len(between) == 0:
        return float("nan")
    return float(np.median(between) / np.median(within))


def bootstrap_geometry(
    units: pd.DataFrame,
    vectors: np.ndarray,
    *,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> dict:
    """按label内录音单元重采样，并以标签置换构造ratio零分布。"""
    labels = units["label"].astype(str).to_numpy()
    unique_labels = sorted(np.unique(labels))
    rng = np.random.default_rng(seed)
    bootstrap_values: list[float] = []
    for _ in range(n_bootstrap):
        sampled_rows = np.concatenate(
            [
                rng.choice(
                    np.where(labels == label)[0],
                    size=int(np.sum(labels == label)),
                    replace=True,
                )
                for label in unique_labels
            ]
        )
        bootstrap_values.append(
            distance_ratio(vectors[sampled_rows], labels[sampled_rows])
        )
    observed = distance_ratio(vectors, labels)
    null = np.asarray(
        [
            distance_ratio(vectors, rng.permutation(labels))
            for _ in range(n_permutations)
        ],
        dtype=float,
    )
    return {
        "between_within_ratio": observed,
        "ratio_ci_low": float(np.nanquantile(bootstrap_values, 0.025)),
        "ratio_ci_high": float(np.nanquantile(bootstrap_values, 0.975)),
        "permutation_null_mean": float(np.nanmean(null)),
        "permutation_p_greater": float(
            (1 + np.sum(null >= observed)) / (n_permutations + 1)
        ),
        "bootstrap_values": bootstrap_values,
        "permutation_values": null.tolist(),
    }


def label_prototypes(
    representation: np.ndarray,
    frame: pd.DataFrame,
    label_column: str,
    order: list[str],
) -> np.ndarray:
    """用录音中位原型的中位数构建label原型。"""
    units, vectors = source_balanced_units(
        representation, frame, label_column
    )
    labels = units["label"].astype(str).to_numpy()
    return np.vstack(
        [np.median(vectors[labels == label], axis=0) for label in order]
    )


def geometry_evidence(
    frame: pd.DataFrame,
    views: dict[str, np.ndarray],
    *,
    subset: str,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """计算SDT、family和leaf三级录音等权几何证据。"""
    metric_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    null_rows: list[dict] = []
    for level, column in (
        ("structure", "structure"),
        ("family", "family"),
        ("leaf", "leaf"),
    ):
        units, vectors = source_balanced_units(
            views["fused_60d"], frame, column
        )
        labels = units["label"].astype(str).to_numpy()
        result = bootstrap_geometry(
            units,
            vectors,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            seed=seed + len(metric_rows),
        )
        order = sorted(frame[column].astype(str).unique())
        prototypes_94 = label_prototypes(
            views["perceptual_94d"], frame, column, order
        )
        prototypes_logmel = label_prototypes(
            views["logmel_pca30"], frame, column, order
        )
        rho = (
            float(spearmanr(pdist(prototypes_94), pdist(prototypes_logmel)).statistic)
            if len(order) > 2
            else float("nan")
        )
        metric_rows.append(
            {
                "subset": subset,
                "level": level,
                "n_syllables": int(len(frame)),
                "n_labels": int(len(order)),
                "n_source_balanced_units": int(len(units)),
                "silhouette_source_balanced_fused": float(
                    silhouette_score(vectors, labels)
                ),
                "between_within_ratio": result["between_within_ratio"],
                "ratio_ci_low": result["ratio_ci_low"],
                "ratio_ci_high": result["ratio_ci_high"],
                "permutation_null_mean": result["permutation_null_mean"],
                "permutation_p_greater": result["permutation_p_greater"],
                "perceptual_logmel_distance_spearman": rho,
            }
        )
        bootstrap_rows.extend(
            {
                "subset": subset,
                "level": level,
                "iteration": index,
                "between_within_ratio": value,
            }
            for index, value in enumerate(result["bootstrap_values"])
        )
        null_rows.extend(
            {
                "subset": subset,
                "level": level,
                "iteration": index,
                "between_within_ratio": value,
            }
            for index, value in enumerate(result["permutation_values"])
        )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(bootstrap_rows),
        pd.DataFrame(null_rows),
    )


def eligible_labels(
    frame: pd.DataFrame, label_column: str
) -> tuple[list[str], pd.DataFrame]:
    """按至少10音节且至少3条录音确定可稳定估计类别。"""
    support = (
        frame.groupby(label_column)
        .agg(
            n_syllables=("stable_id", "size"),
            n_source_recordings=("source_stem", "nunique"),
        )
        .reset_index()
        .rename(columns={label_column: "label"})
    )
    support["classification_estimable"] = (
        support["n_syllables"].ge(10)
        & support["n_source_recordings"].ge(3)
    )
    labels = support.loc[
        support["classification_estimable"], "label"
    ].astype(str).tolist()
    return labels, support


def classification_once(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """执行一次录音阻断OOF预测，返回类别和概率。"""
    predicted = np.full(len(labels), -1, dtype=int)
    probabilities = np.zeros((len(labels), len(np.unique(labels))), dtype=float)
    for train, test in folds:
        if len(np.unique(labels[train])) != len(np.unique(labels)):
            raise ValueError("录音阻断训练折缺少类别")
        model = LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            random_state=seed,
            solver="lbfgs",
        )
        model.fit(features[train], labels[train])
        predicted[test] = model.predict(features[test])
        probabilities[np.ix_(test, model.classes_)] = model.predict_proba(
            features[test]
        )
    if (predicted < 0).any():
        raise ValueError("录音阻断预测未覆盖全部样本")
    return predicted, probabilities


def top_k_accuracy(labels: np.ndarray, probabilities: np.ndarray, k: int) -> float:
    """计算多分类top-k准确率。"""
    top = np.argpartition(
        probabilities, -min(k, probabilities.shape[1]), axis=1
    )[:, -min(k, probabilities.shape[1]) :]
    return float(np.mean(np.any(top == labels[:, None], axis=1)))


def classification_folds(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    requested_seed: int,
    *,
    policy: str = "strict",
    used_seeds: set[int] | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int, list[dict]]:
    """只按训练类别覆盖检查分折；不拟合模型、不读取预测或成绩。"""
    if policy not in {"strict", "retry-seed"}:
        raise ValueError(f"Unknown split feasibility policy: {policy}")
    used_seeds = set() if used_seeds is None else set(used_seeds)
    classes = set(np.unique(labels))
    attempts = []
    for candidate in range(requested_seed, requested_seed + 100):
        if candidate in used_seeds:
            attempts.append({"seed": candidate, "status": "already_used", "missing_train_classes": []})
            continue
        folds = recording_blocked_splits(labels, groups, n_splits, candidate)
        missing = [sorted(int(x) for x in classes - set(labels[train])) for train, _ in folds]
        valid = not any(missing)
        attempts.append({"seed": candidate, "status": "accepted" if valid else "missing_training_class", "missing_train_classes": missing})
        if valid:
            return folds, candidate, attempts
        if policy == "strict":
            raise ValueError(f"录音阻断训练折缺少类别: seed={candidate}, missing={missing}")
    raise ValueError("100个固定顺序候选seed均无法满足训练类别覆盖；未计算任何预测成绩")


def classification_evidence(
    frame: pd.DataFrame,
    fused: np.ndarray,
    *,
    subset: str,
    repeats: int,
    n_splits: int,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """计算SDT、family和leaf的重复三折录音阻断可预测性。"""
    summaries: list[dict] = []
    recalls: list[dict] = []
    null_rows: list[dict] = []
    matrices: dict[str, pd.DataFrame] = {}
    for level, column in (
        ("structure", "structure"),
        ("family", "family"),
        ("leaf", "leaf"),
    ):
        if level == "structure":
            labels_keep = sorted(frame[column].unique())
        else:
            labels_keep, _ = eligible_labels(frame, column)
        selected = frame[column].astype(str).isin(labels_keep).to_numpy()
        current = frame.loc[selected].reset_index(drop=True)
        x = fused[selected]
        encoder = LabelEncoder()
        y = encoder.fit_transform(current[column].astype(str))
        groups = current["source_stem"].astype(str).to_numpy()
        repeat_predictions: list[np.ndarray] = []
        repeat_probabilities: list[np.ndarray] = []
        repeat_metrics: list[tuple[float, float, float]] = []
        first_folds: list[tuple[np.ndarray, np.ndarray]] | None = None
        split_audit: list[dict] = []
        used_split_seeds: set[int] = set()
        for repeat in range(repeats):
            folds, split_seed, attempts = classification_folds(
                y, groups, n_splits, seed + repeat,
                policy=SPLIT_FEASIBILITY_POLICY, used_seeds=used_split_seeds,
            )
            used_split_seeds.add(split_seed)
            split_audit.append({"repeat": repeat, "requested_seed": seed + repeat, "used_seed": split_seed, "attempts": attempts})
            if first_folds is None:
                first_folds = folds
            predicted, probabilities = classification_once(
                x, y, groups, folds, seed=split_seed
            )
            repeat_predictions.append(predicted)
            repeat_probabilities.append(probabilities)
            repeat_metrics.append(
                (
                    float(balanced_accuracy_score(y, predicted)),
                    float(f1_score(y, predicted, average="macro")),
                    top_k_accuracy(y, probabilities, 3),
                )
            )
        metrics = np.asarray(repeat_metrics)
        summaries.append(
            {
                "subset": subset,
                "level": level,
                "n_syllables": int(len(current)),
                "n_labels_total": int(frame[column].nunique()),
                "n_labels_estimable": int(len(encoder.classes_)),
                "n_source_recordings": int(current["source_stem"].nunique()),
                "cv_repeats": repeats,
                "cv_splits": n_splits,
                "split_feasibility_policy": SPLIT_FEASIBILITY_POLICY,
                "split_seed_audit": json.dumps(split_audit, ensure_ascii=False),
                "balanced_accuracy_mean": float(metrics[:, 0].mean()),
                "balanced_accuracy_sd": float(metrics[:, 0].std(ddof=1))
                if repeats > 1
                else 0.0,
                "macro_f1_mean": float(metrics[:, 1].mean()),
                "macro_f1_sd": float(metrics[:, 1].std(ddof=1))
                if repeats > 1
                else 0.0,
                "top3_accuracy_mean": float(metrics[:, 2].mean()),
                "top3_accuracy_sd": float(metrics[:, 2].std(ddof=1))
                if repeats > 1
                else 0.0,
                "chance_balanced_accuracy": float(1 / len(encoder.classes_)),
            }
        )
        all_confusion = np.zeros(
            (len(encoder.classes_), len(encoder.classes_)), dtype=float
        )
        for repeat, predicted in enumerate(repeat_predictions):
            matrix = confusion_matrix(
                y,
                predicted,
                labels=np.arange(len(encoder.classes_)),
                normalize="true",
            )
            all_confusion += matrix
            for class_index, label in enumerate(encoder.classes_):
                recalls.append(
                    {
                        "subset": subset,
                        "level": level,
                        "repeat": repeat,
                        "split_seed": split_audit[repeat]["used_seed"],
                        "label": label,
                        "recall": float(matrix[class_index, class_index]),
                    }
                )
        matrices[f"{subset}_{level}"] = pd.DataFrame(
            all_confusion / repeats,
            index=encoder.classes_,
            columns=encoder.classes_,
        )
        if first_folds is None:
            raise RuntimeError("未生成录音阻断分折")
        rng = np.random.default_rng(seed + 1000 + len(summaries))
        for permutation in range(n_permutations):
            permuted = rng.permutation(y)
            predicted, probabilities = classification_once(
                x,
                permuted,
                groups,
                first_folds,
                seed=seed + permutation,
            )
            null_rows.append(
                {
                    "subset": subset,
                    "level": level,
                    "iteration": permutation,
                    "balanced_accuracy": float(
                        balanced_accuracy_score(permuted, predicted)
                    ),
                    "macro_f1": float(
                        f1_score(permuted, predicted, average="macro")
                    ),
                    "top3_accuracy": top_k_accuracy(
                        permuted, probabilities, 3
                    ),
                }
            )
    summary = pd.DataFrame(summaries)
    null = pd.DataFrame(null_rows)
    null_summary = (
        null.groupby(["subset", "level"])
        .agg(
            null_ba_mean=("balanced_accuracy", "mean"),
            null_ba_p95=("balanced_accuracy", lambda x: x.quantile(0.95)),
        )
        .reset_index()
    )
    summary = summary.merge(null_summary, on=["subset", "level"])
    summary["permutation_p_greater"] = summary.apply(
        lambda row: (
            1
            + int(
                (
                    null.loc[
                        null["subset"].eq(row["subset"])
                        & null["level"].eq(row["level"]),
                        "balanced_accuracy",
                    ]
                    >= row["balanced_accuracy_mean"]
                ).sum()
            )
        )
        / (n_permutations + 1),
        axis=1,
    )
    return summary, pd.DataFrame(recalls), matrices, null


def source_support_table(frame: pd.DataFrame) -> pd.DataFrame:
    """汇总27个family的来源覆盖与支持等级。"""
    rows: list[dict] = []
    for family, group in frame.groupby("family", sort=True):
        recordings = group["source_stem"].value_counts()
        shares = recordings.to_numpy(dtype=float) / len(group)
        hhi = float(np.square(shares).sum())
        n = len(group)
        n_recordings = recordings.size
        effective = float(1 / hhi)
        rows.append(
            {
                "family": family,
                "structure": group["structure"].iloc[0],
                "n_syllables": int(n),
                "n_leaf_buckets": int(group["leaf"].nunique()),
                "n_source_recordings": int(n_recordings),
                "n_sites": int(group["site"].nunique()),
                "n_years": int(group["year"].nunique()),
                "max_recording_share": float(shares.max()),
                "recording_hhi": hhi,
                "effective_recordings": effective,
                "support_tier": support_tier(n, n_recordings, effective),
                "source_concentration_warning": bool(shares.max() > 0.50),
                "individual_identity_status": "not_available",
            }
        )
    return pd.DataFrame(rows)


def bootstrap_leaf_distance(
    representation: np.ndarray,
    frame: pd.DataFrame,
    left: str,
    right: str,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """按叶桶内录音重采样原型距离。"""
    values: list[float] = []
    for _ in range(n_bootstrap):
        prototypes = []
        for leaf in (left, right):
            selected = frame["leaf"].eq(leaf)
            recording_prototypes = np.vstack(
                [
                    np.median(
                        representation[
                            selected.to_numpy()
                            & frame["source_stem"].eq(source).to_numpy()
                        ],
                        axis=0,
                    )
                    for source in sorted(
                        frame.loc[selected, "source_stem"].unique()
                    )
                ]
            )
            sampled = rng.choice(
                len(recording_prototypes),
                size=len(recording_prototypes),
                replace=True,
            )
            prototypes.append(
                np.median(recording_prototypes[sampled], axis=0)
            )
        values.append(float(np.linalg.norm(prototypes[0] - prototypes[1])))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def variant_evidence(
    frame: pd.DataFrame,
    fused: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """比较14个主体—变体距离与同结构跨family叶桶距离。"""
    order = sorted(frame["leaf"].unique())
    prototypes = label_prototypes(fused, frame, "leaf", order)
    distance = np.linalg.norm(
        prototypes[:, None, :] - prototypes[None, :, :], axis=2
    )
    position = {label: index for index, label in enumerate(order)}
    family_by_leaf = (
        frame[["leaf", "family", "structure"]]
        .drop_duplicates()
        .set_index("leaf")
    )
    cross_by_structure: dict[str, list[float]] = {}
    for structure in STRUCTURES:
        leaves = family_by_leaf[
            family_by_leaf["structure"].eq(structure)
        ].index.tolist()
        cross_by_structure[structure] = [
            float(distance[position[left], position[right]])
            for left, right in combinations(leaves, 2)
            if family_by_leaf.loc[left, "family"]
            != family_by_leaf.loc[right, "family"]
        ]
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for family, group in frame.groupby("family", sort=True):
        leaves = sorted(group["leaf"].unique())
        if len(leaves) == 1:
            continue
        if family not in leaves:
            raise ValueError(f"变体family缺少主体叶桶：{family}")
        variants = [leaf for leaf in leaves if leaf != family]
        for variant in variants:
            observed = float(distance[position[family], position[variant]])
            cross = np.asarray(
                cross_by_structure[group["structure"].iloc[0]], dtype=float
            )
            low, high = bootstrap_leaf_distance(
                fused,
                frame,
                family,
                variant,
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
            rows.append(
                {
                    "family": family,
                    "structure": group["structure"].iloc[0],
                    "main_leaf": family,
                    "variant_leaf": variant,
                    "main_n": int(frame["leaf"].eq(family).sum()),
                    "variant_n": int(frame["leaf"].eq(variant).sum()),
                    "main_variant_distance": observed,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "cross_family_leaf_distance_median": float(
                        np.median(cross)
                    ),
                    "cross_family_distance_percentile": float(
                        np.mean(cross <= observed)
                    ),
                    "closer_than_cross_family_median": bool(
                        observed < np.median(cross)
                    ),
                }
            )
    evidence = pd.DataFrame(rows)
    if (
        evidence["family"].nunique() != EXPECTED_VARIANT_FAMILIES
        or len(evidence) != EXPECTED_VARIANT_FAMILIES
    ):
        raise ValueError("14个变体family契约异常")
    successes = int(evidence["closer_than_cross_family_median"].sum())
    test = binomtest(
        successes, n=len(evidence), p=0.5, alternative="greater"
    )
    summary = {
        "n_variant_families": EXPECTED_VARIANT_FAMILIES,
        "n_main_variant_pairs": int(len(evidence)),
        "n_closer_than_cross_family_median": successes,
        "median_cross_family_distance_percentile": float(
            evidence["cross_family_distance_percentile"].median()
        ),
        "sign_test_p_greater": float(test.pvalue),
    }
    return evidence, summary


def reused_pair_summary(pair_path: Path) -> tuple[pd.DataFrame, dict]:
    """读取匹配当前taxonomy版本的结构内证据并核对计数。"""
    pairs = pd.read_csv(pair_path)
    if len(pairs) != EXPECTED_REUSED_PAIRS:
        raise ValueError(f"结构内pair证据不是{EXPECTED_REUSED_PAIRS}对")
    counts = pairs["recommendation"].value_counts().to_dict()
    rows = pd.DataFrame(
        [
            {
                "recommendation": recommendation,
                "n_pairs": int(count),
                "interpretation": {
                    "keep_separate": "保持分离",
                    "insufficient_evidence": "证据不足",
                    "local_boundary_overlap_review": "边界重叠",
                    "merge_evidence_review": "低支持合并复核",
                    "high_priority_merge_candidate": "高优先级合并候选",
                }.get(recommendation, recommendation),
            }
            for recommendation, count in sorted(counts.items())
        ]
    )
    return rows, {key: int(value) for key, value in counts.items()}


def save_figures(
    output_dir: Path,
    geometry: pd.DataFrame,
    classification: pd.DataFrame,
    support: pd.DataFrame,
    variants: pd.DataFrame,
) -> None:
    """生成论文整理所需的四张诊断图。"""
    figure_dir = output_dir / "figures"
    figure_dir.mkdir()
    full_geometry = geometry[geometry["subset"].eq("all_2556")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        full_geometry["level"],
        full_geometry["between_within_ratio"],
        color="#3976a8",
    )
    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Between/within median distance ratio")
    ax.set_title("Source-balanced hierarchical geometry")
    fig.tight_layout()
    fig.savefig(figure_dir / "hierarchical_geometry.png", dpi=180)
    plt.close(fig)

    full_classification = classification[
        classification["subset"].eq("all_2556")
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(full_classification))
    ax.bar(
        x - 0.18,
        full_classification["balanced_accuracy_mean"],
        width=0.36,
        label="Observed BA",
    )
    ax.bar(
        x + 0.18,
        full_classification["null_ba_mean"],
        width=0.36,
        label="Permutation null",
    )
    ax.set_xticks(x, full_classification["level"])
    ax.set_ylabel("Recording-blocked balanced accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "recording_blocked_classification.png", dpi=180)
    plt.close(fig)

    ordered = support.sort_values(
        ["support_tier", "effective_recordings", "n_syllables"]
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(ordered["family"], ordered["n_syllables"], color="#4b8f6d")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.set_ylabel("Syllables")
    ax.set_title("Family source support")
    fig.tight_layout()
    fig.savefig(figure_dir / "family_source_support.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(
        variants["cross_family_distance_percentile"],
        variants["variant_n"],
        color="#a75b45",
    )
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Main–variant distance percentile among cross-family leaves")
    ax.set_ylabel("Variant syllables")
    ax.set_title("Variant-family consistency")
    fig.tight_layout()
    fig.savefig(figure_dir / "variant_consistency.png", dpi=180)
    plt.close(fig)


def write_report(
    output_path: Path,
    geometry: pd.DataFrame,
    classification: pd.DataFrame,
    support: pd.DataFrame,
    variant_summary: dict,
    pair_counts: dict,
) -> None:
    """写出中文、论文可引用口径的内部验证报告。"""
    full_geometry = geometry[geometry["subset"].eq("all_2556")].set_index(
        "level"
    )
    full_classification = classification[
        classification["subset"].eq("all_2556")
    ].set_index("level")
    support_counts = support["support_tier"].value_counts().to_dict()
    lines = [
        "# 2556宏音节类型体系内部量化验证",
        "",
        "## 结论边界",
        "",
        "- 本分析是标签冻结后的内部支持证据，不是生物学真值证明。",
        "- 表示构建不读取标签；分类按source_stem录音阻断，避免同一录音同时进入训练和测试。",
        "- 数据没有真实鸟个体ID、独立第二审听者和外部录音验证。",
        "",
        "## 分层声学几何",
        "",
    ]
    for level in ("structure", "family", "leaf"):
        row = full_geometry.loc[level]
        lines.append(
            f"- {level}：录音等权silhouette={row['silhouette_source_balanced_fused']:.3f}；"
            f"组间/组内距离比={row['between_within_ratio']:.3f}"
            f"（95% bootstrap CI {row['ratio_ci_low']:.3f}—{row['ratio_ci_high']:.3f}），"
            f"置换p={row['permutation_p_greater']:.4f}；"
            f"双视图距离排序rho={row['perceptual_logmel_distance_spearman']:.3f}。"
        )
    lines.extend(["", "## 录音阻断可预测性", ""])
    for level in ("structure", "family", "leaf"):
        row = full_classification.loc[level]
        lines.append(
            f"- {level}：{int(row['n_labels_estimable'])}/"
            f"{int(row['n_labels_total'])}类可稳定估计，"
            f"BA={row['balanced_accuracy_mean']:.3f}，"
            f"macro-F1={row['macro_f1_mean']:.3f}，"
            f"top-3={row['top3_accuracy_mean']:.3f}，"
            f"置换null BA={row['null_ba_mean']:.3f}，"
            f"p={row['permutation_p_greater']:.4f}。"
        )
    lines.extend(
        [
            "",
            "## 变体、边界与来源支持",
            "",
            f"- 14个带变体family中，"
            f"{variant_summary['n_closer_than_cross_family_median']}/14个主体—变体距离"
            "小于同结构跨family叶桶距离中位数；"
            f"距离百分位中位数={variant_summary['median_cross_family_distance_percentile']:.3f}，"
            f"符号检验p={variant_summary['sign_test_p_greater']:.4f}。",
            f"- 匹配版本的{EXPECTED_REUSED_PAIRS}对结构内证据：保持分离{pair_counts.get('keep_separate', 0)}对，"
            f"证据不足{pair_counts.get('insufficient_evidence', 0)}对，"
            f"边界重叠{pair_counts.get('local_boundary_overlap_review', 0)}对，"
            f"低支持合并复核{pair_counts.get('merge_evidence_review', 0)}对。",
            f"- family来源支持：高支持{support_counts.get('high_support', 0)}个，"
            f"中支持{support_counts.get('moderate_support', 0)}个，"
            f"低支持{support_counts.get('low_support', 0)}个。",
            "",
            "## 综合判读",
            "",
            "- SDT结构、family和leaf层级若同时表现为显著高于置换null的距离比与录音阻断预测，"
            "可作为内部声学一致性与可区分性的互补支持。",
            "- silhouette接近零或为负表示局部边界重叠客观存在，不能用总体显著性掩盖。",
            "- 小样本、来源集中或pair证据不足的类别应标记为“证据不足”，"
            "不据此自动删除、合并或改写人工标签。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    """执行分析并以同级staging原子安装输出。"""
    global SPLIT_FEASIBILITY_POLICY
    SPLIT_FEASIBILITY_POLICY = getattr(args, "split_feasibility_policy", "retry-seed")
    if getattr(args, "taxonomy_profile", TAXONOMY_PROFILE) != TAXONOMY_PROFILE:
        raise ValueError(f"Only the released taxonomy profile is supported: {TAXONOMY_PROFILE}")
    archive = args.archive_dir.resolve()
    run_dir = args.run_dir.resolve()
    classifier = args.classifier_dir.resolve()
    pair_dir = args.pair_evidence_dir.resolve()
    output = args.output_dir.resolve()
    protected_paths = [
        archive / "manifest.csv",
        archive / "summary.json",
        run_dir / "feature_rows.csv",
        run_dir / "features_105d.npy",
        run_dir / "feature_names.json",
        classifier / "syllable_logmel_patches.npy",
        pair_dir / "pairwise_similarity_evidence.csv",
        pair_dir / "analysis_metadata.json",
    ]
    input_hashes = {str(path): file_hash(path) for path in protected_paths}
    assignments, features, patches, feature_names, feature_rows = (
        load_analysis_inputs(
            archive,
            run_dir,
            classifier,
            expected_size=EXPECTED_SIZE,
            expected_buckets=EXPECTED_LEAVES,
        )
    )
    assignments = add_taxonomy_columns(assignments)
    observed = assignments.groupby("structure").size().to_dict()
    if observed != {"single": 1153, "double": 768, "triple": 635}:
        raise ValueError(f"Released SDT counts incorrect: {observed}")
    print(f"taxonomy={TAXONOMY_PROFILE}: inputs validated, fitting views", flush=True)
    views, view_metadata = prepare_label_blind_views(
        features,
        patches,
        feature_names,
        seed=args.seed,
        pca_components=args.pca_components,
    )
    view_metadata.pop("_models", None)

    print("Computing all-cohort geometry", flush=True)
    all_geometry, all_bootstrap, all_geometry_null = geometry_evidence(
        assignments,
        views,
        subset="all_2556",
        n_bootstrap=args.bootstrap,
        n_permutations=args.geometry_permutations,
        seed=args.seed,
    )
    print("Computing all-cohort repeated recording-blocked CV", flush=True)
    all_classification, all_recall, all_confusion, all_class_null = (
        classification_evidence(
            assignments,
            views["fused_60d"],
            subset="all_2556",
            repeats=args.cv_repeats,
            n_splits=args.cv_splits,
            n_permutations=args.classification_permutations,
            seed=args.seed,
        )
    )

    print("Computing nonnoise sensitivity", flush=True)
    nonnoise_mask = assignments["micro_type"].ge(0).to_numpy()
    nonnoise = assignments.loc[nonnoise_mask].reset_index(drop=True)
    nonnoise_features = features[nonnoise_mask]
    nonnoise_patches = patches[nonnoise_mask]
    nonnoise_views, nonnoise_view_metadata = prepare_label_blind_views(
        nonnoise_features,
        nonnoise_patches,
        feature_names,
        seed=args.seed,
        pca_components=args.pca_components,
    )
    nonnoise_view_metadata.pop("_models", None)
    nonnoise_geometry, nonnoise_bootstrap, nonnoise_geometry_null = (
        geometry_evidence(
            nonnoise,
            nonnoise_views,
            subset="micro_type_ge_0",
            n_bootstrap=args.bootstrap,
            n_permutations=args.geometry_permutations,
            seed=args.seed + 100,
        )
    )
    (
        nonnoise_classification,
        nonnoise_recall,
        nonnoise_confusion,
        nonnoise_class_null,
    ) = classification_evidence(
        nonnoise,
        nonnoise_views["fused_60d"],
        subset="micro_type_ge_0",
        repeats=args.cv_repeats,
        n_splits=args.cv_splits,
        n_permutations=args.classification_permutations,
        seed=args.seed + 100,
    )

    geometry = pd.concat(
        [all_geometry, nonnoise_geometry], ignore_index=True
    )
    bootstrap = pd.concat(
        [all_bootstrap, nonnoise_bootstrap], ignore_index=True
    )
    geometry_null = pd.concat(
        [all_geometry_null, nonnoise_geometry_null], ignore_index=True
    )
    classification = pd.concat(
        [all_classification, nonnoise_classification], ignore_index=True
    )
    recall = pd.concat([all_recall, nonnoise_recall], ignore_index=True)
    classification_null = pd.concat(
        [all_class_null, nonnoise_class_null], ignore_index=True
    )
    confusion = {**all_confusion, **nonnoise_confusion}
    sensitivity = all_geometry.merge(
        nonnoise_geometry,
        on="level",
        suffixes=("_all", "_nonnoise"),
    )[
        [
            "level",
            "n_syllables_all",
            "n_syllables_nonnoise",
            "silhouette_source_balanced_fused_all",
            "silhouette_source_balanced_fused_nonnoise",
            "between_within_ratio_all",
            "between_within_ratio_nonnoise",
            "perceptual_logmel_distance_spearman_all",
            "perceptual_logmel_distance_spearman_nonnoise",
        ]
    ]
    sensitivity = sensitivity.merge(
        all_classification[
            ["level", "balanced_accuracy_mean", "macro_f1_mean"]
        ],
        on="level",
    ).merge(
        nonnoise_classification[
            ["level", "balanced_accuracy_mean", "macro_f1_mean"]
        ],
        on="level",
        suffixes=("_all", "_nonnoise"),
    )
    support = source_support_table(assignments)
    family_labels, family_estimability = eligible_labels(
        assignments, "family"
    )
    leaf_labels, leaf_estimability = eligible_labels(assignments, "leaf")
    if len(family_labels) != 26 or len(leaf_labels) != 36:
        raise ValueError("26/27 family或36/41 leaf可估计子集契约异常")
    variant_table, variant_summary = variant_evidence(
        assignments,
        views["fused_60d"],
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    pair_table, pair_counts = reused_pair_summary(
        pair_dir / "pairwise_similarity_evidence.csv"
    )

    staging = output.parent / f".{output.name}.tmp"
    backup = output.parent / f".{output.name}.backup"
    for path in (staging, backup):
        if path.exists():
            shutil.rmtree(path)
    staging.mkdir(parents=True)
    try:
        geometry.to_csv(staging / "hierarchical_geometry.csv", index=False)
        bootstrap.to_csv(
            staging / "geometry_bootstrap_intervals.csv", index=False
        )
        geometry_null.to_csv(
            staging / "geometry_permutation_null.csv", index=False
        )
        classification.to_csv(
            staging / "classification_summary.csv", index=False
        )
        recall.to_csv(
            staging / "classification_per_class_recall.csv", index=False
        )
        classification_null.to_csv(
            staging / "classification_permutation_null.csv", index=False
        )
        confusion_dir = staging / "confusion_matrices"
        confusion_dir.mkdir()
        for name, matrix in confusion.items():
            matrix.to_csv(confusion_dir / f"{name}.csv")
        variant_table.to_csv(staging / "variant_evidence.csv", index=False)
        support.to_csv(staging / "family_source_support.csv", index=False)
        family_estimability.to_csv(
            staging / "family_classification_estimability.csv", index=False
        )
        leaf_estimability.to_csv(
            staging / "leaf_classification_estimability.csv", index=False
        )
        sensitivity.to_csv(
            staging / "nonnoise_sensitivity.csv", index=False
        )
        pair_table.to_csv(
            staging / f"reused_{EXPECTED_REUSED_PAIRS}_pair_summary.csv", index=False
        )
        save_figures(
            staging, geometry, classification, support, variant_table
        )
        write_report(
            staging / "report.md",
            geometry,
            classification,
            support,
            variant_summary,
            pair_counts,
        )
        support_counts = {
            key: int(value)
            for key, value in support["support_tier"].value_counts().items()
        }
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "analysis_type": "internal_only_hierarchical_validation",
            "taxonomy_profile": TAXONOMY_PROFILE,
            "split_feasibility_policy": SPLIT_FEASIBILITY_POLICY,
            "read_only": True,
            "archive": archive.name,
            "n_syllables": EXPECTED_SIZE,
            "n_structures": 3,
            "n_families": EXPECTED_FAMILIES,
            "n_leaf_buckets": EXPECTED_LEAVES,
            "classification_estimable": {
                "families": len(family_labels),
                "leaf_buckets": len(leaf_labels),
            },
            "n_variant_families": EXPECTED_VARIANT_FAMILIES,
            "reused_pair_count": EXPECTED_REUSED_PAIRS,
            "reused_pair_recommendation_counts": pair_counts,
            "family_support_counts": support_counts,
            "variant_summary": variant_summary,
            "parameters": {
                "seed": args.seed,
                "pca_components": args.pca_components,
                "cv_splits": args.cv_splits,
                "cv_repeats": args.cv_repeats,
                "bootstrap": args.bootstrap,
                "geometry_permutations": args.geometry_permutations,
                "classification_permutations": (
                    args.classification_permutations
                ),
            },
            "view_metadata": {
                "all_2556": view_metadata,
                "micro_type_ge_0": nonnoise_view_metadata,
            },
            "feature_row_indices_sha256": hashlib.sha256(
                feature_rows.astype(np.int64).tobytes()
            ).hexdigest(),
            "input_sha256": input_hashes,
            "limitations": [
                "no_independent_second_annotator",
                "no_external_recording_validation",
                "no_true_individual_bird_ids",
                "source_stem_blocking_only",
                "internal_support_not_biological_truth",
            ],
        }
        (staging / "analysis_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for path in protected_paths:
            if file_hash(path) != input_hashes[str(path)]:
                raise ValueError(f"只读输入发生变化：{path}")
        if output.exists():
            output.replace(backup)
        try:
            staging.replace(output)
        except Exception:
            if backup.exists() and not output.exists():
                backup.replace(output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and output.exists():
            shutil.rmtree(backup)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-feasibility-policy", choices=["strict", "retry-seed"], default="retry-seed")
    parser.add_argument("--taxonomy-profile", choices=[TAXONOMY_PROFILE], default=TAXONOMY_PROFILE)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--classifier-dir", type=Path, required=True)
    parser.add_argument("--pair-evidence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--cv-repeats", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--geometry-permutations", type=int, default=1000)
    parser.add_argument("--classification-permutations", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    metadata = run(parse_args())
    print(
        "2556内部量化验证完成："
        f"{metadata['n_families']}个family、"
        f"{metadata['n_leaf_buckets']}个leaf、"
        f"{metadata['reused_pair_count']}对复用证据"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
