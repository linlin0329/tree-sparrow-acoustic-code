"""Apply the final reviewed leaf mapping with per-syllable identities and cumulative deletion records. The input is the already reviewed 2560-syllable cohort."""

from __future__ import annotations

import re

import pandas as pd

from ..io import require

# Original pattern from materialize_sdt_reorganized_confirmed_macro_syllables_2560.py.
SDT_BUCKET_PATTERN = re.compile(
    r"(?P<structure>single|double|triple)_"
    r"group_(?P<number>0[1-9]|[12]\d|3[0-2])"
    r"(?P<variant>_v[1-9]\d*)?"
)

EXPECTED_INPUT = 2560

EXPECTED_OUTPUT = 2556

EXPECTED_AUDIT = 2591


EXPECTED_NEW_DELETED = 4

EXPECTED_DELETED = 35

EXPECTED_INPUT_LEAVES = 43

EXPECTED_OUTPUT_LEAVES = 41

EXPECTED_FAMILIES = 27

LABEL_VERSION = 'sdt-taxonomy-r2-2026-09-24'
EXPECTED_TARGET_BUCKET_COUNTS = {'double_group_01': 314, 'double_group_01_v1': 28, 'double_group_02': 153, 'double_group_02_v1': 19, 'double_group_03': 69, 'double_group_03_v1': 19, 'double_group_04': 69, 'double_group_06': 23, 'single_group_01': 375, 'single_group_01_v1': 95, 'single_group_02': 40, 'single_group_02_v1': 13, 'single_group_03': 125, 'single_group_04': 83, 'single_group_04_v1': 21, 'single_group_05': 66, 'single_group_05_v1': 17, 'single_group_06': 29, 'single_group_07': 31, 'single_group_08': 18, 'single_group_08_v1': 23, 'single_group_09': 32, 'single_group_10': 59, 'single_group_11': 55, 'single_group_12': 15, 'single_group_12_v1': 17, 'single_group_13': 8, 'single_group_14': 31, 'triple_group_01': 239, 'triple_group_01_v1': 130, 'triple_group_02': 77, 'triple_group_03': 35, 'triple_group_03_v1': 7, 'triple_group_05': 66, 'triple_group_05_v1': 12, 'double_group_05': 74, 'triple_group_04': 12, 'triple_group_04_v1': 8, 'triple_group_06': 39, 'triple_group_07': 3, 'triple_group_07_v1': 7}


MANIFEST_COLUMNS = (
    "stable_id",
    "micro_type",
    "source_stem",
    "final_group_id",
    "final_group_code",
    "final_bucket_code",
    "final_variant_code",
    "storage_layout",
    "storage_bucket_code",
    "output_relative_path",
    "output_audio_sha256",
    "syllable_structure",
)
MAPPING_COLUMNS = (
    "source_bucket_code",
    "action",
    "target_family_code",
    "target_bucket_code",
)
IDENTITY_COLUMNS = (
    "source_stem",
    "source_raven_row",
    "source_row",
    "selection_id",
    "View",
    "Channel",
    "begin_time",
    "end_time",
    "source_frame_start",
    "source_frame_stop",
    "sample_rate",
    "channels",
    "wav_subtype",
)
ACTIONS = frozenset(
    (
        "delete_whole_bucket",
        "merge_into_main_bucket",
        "family_renumber",
        "structure_reclassification",
        "retain_code",
    )
)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    require(not frame.columns.duplicated().any(), f"{name}: duplicate columns")
    require(
        set(columns) <= set(frame.columns),
        f"{name}: missing columns {sorted(set(columns) - set(frame.columns))}",
    )


def _unique_ids(frame: pd.DataFrame, column: str, name: str) -> None:
    values = frame[column]
    require(
        not values.isna().any() and values.astype(str).str.strip().ne("").all(),
        f"{name}: empty {column}",
    )
    require(not values.duplicated().any(), f"{name}: duplicate {column}")


def validate_mapping_inputs(
    manifest: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    canonical: bool = True,
) -> None:
    """Reject ambiguous identity or incomplete maps before the original mapping."""
    _require_columns(manifest, MANIFEST_COLUMNS, "manifest")
    _require_columns(mapping, MAPPING_COLUMNS, "mapping")
    require(len(manifest) > 0, "manifest is empty")
    _unique_ids(manifest, "stable_id", "manifest")
    _unique_ids(mapping, "source_bucket_code", "mapping")
    require(
        manifest["stable_id"].astype(str).str.fullmatch(r"[A-Za-z0-9_-]+").all(),
        "stable_id is not a safe filename",
    )
    require(
        set(manifest["final_bucket_code"]) == set(mapping["source_bucket_code"]),
        "mapping must cover exactly all source buckets",
    )
    require(set(mapping["action"]) <= ACTIONS, "mapping has an unknown action")
    for row in mapping.itertuples(index=False):
        parse_target_bucket(row.source_bucket_code)
        if row.action == "delete_whole_bucket":
            require(
                pd.isna(row.target_bucket_code) or row.target_bucket_code == "",
                "deleted bucket must have an empty target",
            )
            require(
                pd.isna(row.target_family_code) or row.target_family_code == "",
                "deleted bucket must have an empty target family",
            )
            continue
        structure, _, family, variant = parse_target_bucket(row.target_bucket_code)
        require(family == row.target_family_code, "target family and bucket disagree")
        source_structure = parse_target_bucket(row.source_bucket_code)[0]
        require((structure != source_structure) == (row.action == "structure_reclassification"),
                "cross-structure correction requires an explicit structure_reclassification action")
        if row.action == "retain_code":
            require(
                row.source_bucket_code == row.target_bucket_code,
                "retain_code cannot change the bucket",
            )
        if row.action == "merge_into_main_bucket":
            require(variant is None, "merge_into_main_bucket cannot target a variant")
    if "n_syllables" in mapping:
        observed = manifest["final_bucket_code"].value_counts()
        supplied = pd.to_numeric(
            mapping.set_index("source_bucket_code")["n_syllables"], errors="raise"
        )
        require(
            observed.reindex(supplied.index).eq(supplied).all(),
            "mapping source bucket counts disagree with the manifest",
        )
    if canonical:
        require(len(manifest) == EXPECTED_INPUT, "canonical input requires 2560 IDs")
        require(
            len(mapping) == EXPECTED_INPUT_LEAVES,
            "canonical mapping requires 43 source buckets",
        )


def validate_decision_integrity(decisions: pd.DataFrame) -> None:
    _unique_ids(decisions, "stable_id", "decisions")
    require(not decisions["sdt_refined_decision"].isna().any(), "unresolved decision")
    retained = decisions.loc[decisions["sdt_refined_action"].ne("delete_whole_bucket")]
    require(
        not retained["output_relative_path"].duplicated().any(),
        "duplicate output relative path",
    )
    deleted = decisions.loc[decisions["sdt_refined_action"].eq("delete_whole_bucket")]
    require(deleted["final_bucket_code"].isna().all(), "deleted labels were retained")
    if "source_raven_row" in decisions:
        present = decisions["source_raven_row"].notna() & decisions[
            "source_raven_row"
        ].ne("")
        rows = pd.to_numeric(decisions.loc[present, "source_raven_row"], errors="raise")
        require(
            ((rows > 0) & (rows % 1 == 0)).all(),
            "source_raven_row must be a positive integer when provided",
        )
        pairs = decisions.loc[present, ["source_stem"]].copy()
        pairs["row"] = rows
        require(not pairs.duplicated().any(), "duplicate source recording / Raven row")


def validate_prior_audit(prior: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """Keep prior deletions and require exact input membership in the live cohort."""
    _require_columns(prior, ("stable_id", "complete_output_status"), "prior audit")
    _unique_ids(prior, "stable_id", "prior audit")
    require(prior["complete_output_status"].notna().all(), "prior audit has no status")
    deleted = prior["complete_output_status"].str.startswith("deleted", na=False)
    require(
        set(prior.loc[~deleted, "stable_id"]) == set(decisions["stable_id"]),
        "prior audit live IDs disagree with input manifest",
    )
    result = prior.copy()
    lookup = decisions.set_index("stable_id")
    present = result["stable_id"].isin(lookup.index)
    for column in IDENTITY_COLUMNS:
        if column not in decisions:
            continue
        incoming = result.loc[present, "stable_id"].map(lookup[column])
        if column in result:
            old = result.loc[present, column]
            equal = old.eq(incoming) | (old.isna() & incoming.isna())
            require(equal.all(), f"prior audit identity mismatch: {column}")
        else:
            result[column] = result["stable_id"].map(lookup[column])
    return result


def parse_target_bucket(
    bucket: str,
) -> tuple[str, int, str, str | None]:
    match = SDT_BUCKET_PATTERN.fullmatch(str(bucket))
    if match is None:
        raise ValueError(f"非法精炼目标bucket：{bucket}")
    structure = match.group("structure")
    group_id = int(match.group("number"))
    group_code = f"{structure}_group_{group_id:02d}"
    variant = str(bucket) if match.group("variant") else None
    return structure, group_id, group_code, variant


def build_decision_audit(
    manifest: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    design_name: str,
    canonical: bool = True,
) -> pd.DataFrame:
    validate_mapping_inputs(manifest, mapping, canonical=canonical)
    result = manifest.copy()
    source_fields = {
        "group_id": "final_group_id",
        "group_code": "final_group_code",
        "bucket_code": "final_bucket_code",
        "variant_code": "final_variant_code",
        "output_relative_path": "output_relative_path",
        "storage_bucket_code": "storage_bucket_code",
    }
    for suffix, column in source_fields.items():
        result[f"pre_sdt_refined_{suffix}"] = result[column]
    lookup = mapping.set_index("source_bucket_code")
    source_bucket = result["final_bucket_code"].astype(str)
    result["sdt_refined_action"] = source_bucket.map(lookup["action"])
    result["sdt_refined_target_family_code"] = source_bucket.map(
        lookup["target_family_code"]
    )
    result["sdt_refined_target_bucket_code"] = source_bucket.map(
        lookup["target_bucket_code"]
    )
    if result["sdt_refined_action"].isna().any():
        raise ValueError("精炼映射未覆盖全部2560个输入音节")
    result["sdt_refined_decision"] = result["sdt_refined_action"].map(
        {
            "delete_whole_bucket": "manual_deleted_sdt_refined",
            "merge_into_main_bucket": "merged_into_main_bucket",
            "family_renumber": "family_renumbered",
            "structure_reclassification": "author_corrected_structure",
            "retain_code": "retained_unchanged",
        }
    )
    result["sdt_refined_design"] = design_name
    deleted = result["sdt_refined_action"].eq("delete_whole_bucket")
    retained = result[~deleted].copy()
    parts = retained["sdt_refined_target_bucket_code"].map(parse_target_bucket)
    retained["syllable_structure"] = parts.map(lambda item: item[0])
    retained["final_group_id"] = parts.map(lambda item: item[1])
    retained["final_group_code"] = parts.map(lambda item: item[2])
    retained["final_bucket_code"] = retained["sdt_refined_target_bucket_code"]
    retained["final_variant_code"] = parts.map(lambda item: item[3])
    retained["storage_layout"] = "flat_sdt_prefix_top_level"
    retained["storage_bucket_code"] = retained["final_bucket_code"]
    retained["output_relative_path"] = retained.apply(
        lambda row: f"{row['final_bucket_code']}/{row['stable_id']}.wav",
        axis=1,
    )
    for column in [
        "syllable_structure",
        "final_group_id",
        "final_group_code",
        "final_bucket_code",
        "final_variant_code",
        "storage_bucket_code",
        "output_relative_path",
    ]:
        result.loc[retained.index, column] = retained[column]
        result.loc[deleted, column] = pd.NA
    result["label_version"] = LABEL_VERSION
    validate_decisions(result, canonical=canonical)
    return result


def validate_decisions(decisions: pd.DataFrame, *, canonical: bool = True) -> None:
    validate_decision_integrity(decisions)
    if not canonical:
        return
    if (
        len(decisions) != EXPECTED_INPUT
        or decisions["stable_id"].nunique() != EXPECTED_INPUT
    ):
        raise ValueError("SDT精炼决策未覆盖2560个stable ID")
    counts = decisions["sdt_refined_decision"].value_counts().to_dict()
    if counts.get("manual_deleted_sdt_refined") != EXPECTED_NEW_DELETED:
        raise ValueError(f"SDT精炼删除计数异常：{counts}")
    if counts.get("merged_into_main_bucket") != 62:
        raise ValueError(f"SDT精炼合并计数异常：{counts}")
    if canonical and int(decisions["sdt_refined_action"].eq("structure_reclassification").sum()) != 74:
        raise ValueError("taxonomy R2 requires the confirmed74-syllable structural correction")
    retained = decisions[
        decisions["sdt_refined_decision"].ne("manual_deleted_sdt_refined")
    ]
    target_counts = retained["final_bucket_code"].value_counts().sort_index().to_dict()
    if target_counts != dict(sorted(EXPECTED_TARGET_BUCKET_COUNTS.items())):
        raise ValueError(f"SDT精炼目标叶桶计数异常：{target_counts}")
    if (
        len(retained) != EXPECTED_OUTPUT
        or retained["final_bucket_code"].nunique() != EXPECTED_OUTPUT_LEAVES
        or retained["final_group_code"].nunique() != EXPECTED_FAMILIES
    ):
        raise ValueError("SDT精炼目标未闭合2556/41/27")
    if retained["output_relative_path"].duplicated().any():
        raise ValueError("SDT精炼输出路径冲突")


def update_complete_audit(
    prior: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    canonical: bool = True,
) -> pd.DataFrame:
    prior = validate_prior_audit(prior, decisions)
    decision_columns = [
        "pre_sdt_refined_group_id",
        "pre_sdt_refined_group_code",
        "pre_sdt_refined_bucket_code",
        "pre_sdt_refined_variant_code",
        "pre_sdt_refined_output_relative_path",
        "pre_sdt_refined_storage_bucket_code",
        "sdt_refined_action",
        "sdt_refined_target_family_code",
        "sdt_refined_target_bucket_code",
        "sdt_refined_decision",
        "sdt_refined_design",
    ]
    final_columns = [
        "final_group_id",
        "final_group_code",
        "final_bucket_code",
        "final_variant_code",
        "storage_layout",
        "storage_bucket_code",
        "output_relative_path",
        "output_audio_sha256",
        "syllable_structure",
    ]
    drop_columns = [
        column for column in [*decision_columns, *final_columns] if column in prior
    ]
    base = prior.drop(columns=drop_columns, errors="ignore")
    result = base.merge(
        decisions[["stable_id", *decision_columns]],
        on="stable_id",
        how="left",
        validate="one_to_one",
    )
    if canonical and int(decisions["sdt_refined_action"].eq("structure_reclassification").sum()) != 74:
        raise ValueError("taxonomy R2 requires the confirmed74-syllable structural correction")
    retained = decisions[
        decisions["sdt_refined_decision"].ne("manual_deleted_sdt_refined")
    ]
    result = result.merge(
        retained[["stable_id", *final_columns]],
        on="stable_id",
        how="left",
        validate="one_to_one",
    )
    new_deleted = set(
        decisions.loc[
            decisions["sdt_refined_decision"].eq("manual_deleted_sdt_refined"),
            "stable_id",
        ]
    )
    new_deleted_mask = result["stable_id"].isin(new_deleted)
    prior_deleted_mask = result["sdt_refined_decision"].isna()
    result.loc[new_deleted_mask, "complete_output_status"] = "deleted_sdt_refined"
    result.loc[prior_deleted_mask, "sdt_refined_decision"] = (
        "prior_deleted_not_in_output"
    )
    result.loc[prior_deleted_mask, "sdt_refined_design"] = decisions[
        "sdt_refined_design"
    ].iloc[0]
    deleted_mask = result["complete_output_status"].str.startswith("deleted", na=False)
    if canonical and (
        len(result) != EXPECTED_AUDIT
        or result["stable_id"].nunique() != EXPECTED_AUDIT
        or int(deleted_mask.sum()) != EXPECTED_DELETED
        or int(new_deleted_mask.sum()) != EXPECTED_NEW_DELETED
    ):
        raise ValueError("SDT精炼累计审计未闭合2591/35/4")
    return result


def build_crosswalk(decisions: pd.DataFrame) -> pd.DataFrame:
    frame = decisions.copy()
    frame["target_bucket_or_deleted"] = frame["final_bucket_code"].fillna("deleted")
    return (
        frame.groupby(
            [
                "syllable_structure",
                "pre_sdt_refined_bucket_code",
                "sdt_refined_action",
                "target_bucket_or_deleted",
            ],
            dropna=False,
        )
        .agg(
            n_syllables=("stable_id", "size"),
            n_source_recordings=("source_stem", "nunique"),
            n_microtypes=("micro_type", "nunique"),
        )
        .reset_index()
    )
