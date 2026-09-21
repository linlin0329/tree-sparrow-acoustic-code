#!/usr/bin/env python3
"""训练 songiness RandomForest 打分器并对未复听录音排序。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


def build_table(
    output_dir: Path,
    reliable_feature_csvs: list[Path],
) -> tuple[pd.DataFrame, list[str]]:
    full = pd.read_csv(output_dir / "features_full.csv")
    boundary_free_columns = [
        column for column in full.columns if column != "source_file"
    ]

    all_segments = pd.read_csv(output_dir / "recording_features_allseg.csv")
    full = full.merge(all_segments, on="source_file", how="left")
    all_segment_columns = [
        "n_seg_all",
        "max_bout_all",
        "bout_dur_all",
        "active_dur_all",
    ]
    for column in all_segment_columns:
        full[column] = full[column].fillna(0)

    reliable = pd.concat(
        [pd.read_csv(path) for path in reliable_feature_csvs],
        ignore_index=True,
    )
    reliable_columns = [
        "max_bout",
        "bout_dur",
        "bout_rate",
        "isi_cv",
        "n_song_total",
        "n_nontype1_total",
        "frac_nontype1",
        "n_distinct_nontype1",
    ]
    reliable = reliable[["source_file"] + reliable_columns].drop_duplicates(
        "source_file"
    )
    full = full.merge(reliable, on="source_file", how="left")
    for column in [
        "max_bout",
        "bout_dur",
        "bout_rate",
        "n_song_total",
        "n_nontype1_total",
        "frac_nontype1",
        "n_distinct_nontype1",
    ]:
        full[column] = full[column].fillna(0)
    full["isi_cv"] = full["isi_cv"].fillna(-1)
    return (
        full,
        boundary_free_columns + all_segment_columns + reliable_columns,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--features-dir",
        type=Path,
        required=True,
        help="Directory containing features_full.csv and recording_features_allseg.csv",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--gold-list", type=Path, required=True)
    ap.add_argument("--reliable-features", action="append", type=Path, required=True)
    ap.add_argument(
        "--listened-scores",
        type=Path,
        required=True,
        help="Matched scores table containing is_listened flags",
    )
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--n-trees", type=int, default=500)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Output must be new: " + str(args.output_dir))
    full, feature_columns = build_table(args.features_dir, args.reliable_features)
    song_final = pd.read_csv(args.gold_list)
    gold_all = set(song_final.base_recording)
    gold_manual = set(song_final[song_final.kind == "manual"].base_recording)
    reviewed_scores = pd.read_csv(args.listened_scores)
    listened = set(reviewed_scores.loc[reviewed_scores.is_listened, "source_file"])

    full["is_gold"] = full.source_file.isin(gold_all)
    full["is_gold_manual"] = full.source_file.isin(gold_manual)
    full["is_listened"] = full.source_file.isin(listened)
    full["site"] = full.source_file.str.split("_").str[0]

    print(f"全集(已提特征) {len(full)}")
    print(
        f"金标准可及: all={int(full.is_gold.sum())}/{len(gold_all)}  "
        f"manual={int(full.is_gold_manual.sum())}/{len(gold_manual)}"
    )
    print(f"已复听录音: {int(full.is_listened.sum())}")

    train = full[full.is_gold | (full.is_listened & ~full.is_gold)].copy()
    train["y"] = train.is_gold.astype(int)
    n_positive = int(train.y.sum())
    n_negative = int((train.y == 0).sum())
    print(f"训练集 {len(train)}: 正={n_positive} 负={n_negative}")

    if args.dry_run:
        print(
            {
                "status": "planned",
                "feature_count": len(feature_columns),
                "training_rows": len(train),
            }
        )
        return
    args.output_dir.mkdir(parents=True)

    features = train[feature_columns].values
    labels = train.y.values
    pipeline = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=args.n_trees,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    n_jobs=args.jobs,
                    random_state=0,
                ),
            ),
        ]
    )

    folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0)
    oof = np.zeros(len(train))
    for fold, (train_indices, test_indices) in enumerate(folds.split(features, labels)):
        pipeline.fit(features[train_indices], labels[train_indices])
        oof[test_indices] = pipeline.predict_proba(features[test_indices])[:, 1]
        print(f"  fold {fold + 1}/{args.folds} done")
    train["oof"] = oof

    is_manual = train.is_gold_manual.values
    positive_oof = oof[labels == 1]
    manual_oof = oof[(labels == 1) & is_manual]
    negative_oof = oof[labels == 0]
    grid = np.concatenate(
        [
            np.round(np.arange(0.002, 0.02, 0.002), 4),
            np.round(np.arange(0.02, 0.9, 0.02), 3),
        ]
    )
    curve = pd.DataFrame(
        [
            {
                "threshold": threshold,
                "recall_all": float((positive_oof >= threshold).mean()),
                "recall_manual": float((manual_oof >= threshold).mean()),
                "neg_kept_frac": float((negative_oof >= threshold).mean()),
            }
            for threshold in grid
        ]
    )
    curve_path = args.output_dir / "scorer_recall_retention.csv"
    curve.to_csv(curve_path, index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n=== 召回-留存曲线 (OOF) ===")
    print(curve.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    def threshold_for_recall(target, probabilities):
        acceptable = [
            threshold
            for threshold in curve.threshold
            if (probabilities >= threshold).mean() >= target
        ]
        return max(acceptable) if acceptable else 0.0

    threshold_95 = threshold_for_recall(0.95, positive_oof)
    threshold_95_manual = threshold_for_recall(0.95, manual_oof)
    threshold_98 = threshold_for_recall(0.98, positive_oof)
    for label, threshold in [
        ("召回0.95(all)", threshold_95),
        ("召回0.95(manual)", threshold_95_manual),
        ("召回0.98(all)", threshold_98),
    ]:
        print(
            f"\n阈值@{label} = {threshold}  "
            f"留存(neg)={(negative_oof >= threshold).mean():.1%}"
        )

    lowest = train[train.y == 1].sort_values("oof").head(20)
    print("\n=== OOF 最低的 20 条金标准 (最难召回) ===")
    print(
        lowest[
            [
                "source_file",
                "oof",
                "is_gold_manual",
                "max_bout_all",
                "bout_dur_all",
                "max_bout",
                "n_nontype1_total",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )

    pipeline.fit(features, labels)
    deploy = full[~full.is_listened & ~full.is_gold].copy()
    deploy["prob"] = pipeline.predict_proba(deploy[feature_columns].values)[:, 1]
    print(f"\n部署集(未听) {len(deploy)}")
    for name, threshold in [
        ("0.95all", threshold_95),
        ("0.95manual", threshold_95_manual),
        ("0.98all", threshold_98),
    ]:
        count = int((deploy.prob >= threshold).sum())
        print(
            f"  @{name} (thr={threshold}): 候选池 {count} "
            f"({count / len(deploy):.1%})"
        )

    importance = pd.Series(
        pipeline.named_steps["rf"].feature_importances_,
        index=feature_columns,
    ).sort_values(ascending=False)
    print("\n=== Top15 特征重要度 ===")
    print(importance.head(15).to_string(float_format=lambda value: f"{value:.3f}"))

    full["prob"] = np.nan
    full.loc[train.index, "prob"] = train["oof"].values
    full.loc[deploy.index, "prob"] = deploy["prob"].values
    keep_columns = [
        "source_file",
        "site",
        "prob",
        "is_gold",
        "is_gold_manual",
        "is_listened",
        "max_bout_all",
        "bout_dur_all",
        "n_seg_all",
        "max_bout",
        "n_nontype1_total",
        "n_distinct_nontype1",
        "aci",
        "duty_cycle",
        "peak_rate",
    ]
    scores_path = args.output_dir / "scores_all.csv"
    full[keep_columns].to_csv(scores_path, index=False)
    print(f"\n写出 {scores_path}")
    print(f"写出 {curve_path}")


if __name__ == "__main__":
    main()
