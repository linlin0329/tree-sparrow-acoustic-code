"""Summarize complete decode checks, duplicate identities, clock flags and the declared publication policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from . import prepare_release_qc as prep
from .release_score_policy import ScoreRule


def truth(value):
    return str(value).lower() == "true"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def aggregate(records):
    return {
        "row_count": len(records),
        "status_counts": dict(Counter(row["qc_status"] for row in records)),
        "role_counts": dict(Counter(row["role"] for row in records)),
        "role_status_counts": dict(
            Counter(f"{row['role']}:{row['qc_status']}" for row in records)
        ),
        "strict_decode_pass_count": sum(
            truth(row["strict_decode_pass"]) for row in records
        ),
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_complete(preparation: Path, audio: Path):
    summary = load_json(audio / "summary.json")
    config = load_json(audio / "run_config.json")
    contract = config["contract"]
    require(
        summary.get("state") == "complete" and summary.get("mode") == "full",
        "必须为已完成的 full 音频执行；不能以 pilot 或未完成任务生成最终清单",
    )
    require(
        contract.get("mode") == "full" and contract.get("limit") is None,
        "full 契约不能设 limit",
    )
    contract_sha = canonical_hash(contract)
    require(
        config["contract_sha256"] == contract_sha == summary["contract_sha256"],
        "执行契约 SHA-256 不一致",
    )
    manifest = preparation / "audio_jobs_full.csv.gz"
    require(
        prep.sha256(manifest) == contract["manifest_sha256"],
        "执行清单 SHA-256 与 full 契约不一致",
    )
    require(
        prep.sha256(audio / "source_snapshot.csv.gz")
        == config["source_snapshot_sha256"],
        "源 stat 快照 SHA-256 不一致",
    )
    require(
        prep.sha256(audio / "results.csv.gz") == summary["results_sha256"],
        "合并结果 SHA-256 不一致",
    )
    jobs = list(prep.rows(manifest))
    results = list(prep.rows(audio / "results.csv.gz"))
    require(
        len(jobs) == contract["row_count"] == len(results), "输入/契约/结果行数不符"
    )
    ids = [row["physical_id"] for row in jobs]
    require(len(ids) == len(set(ids)), "full 输入 physical_id 不唯一")
    require(
        ids == [row["physical_id"] for row in results],
        "full 结果不是输入 physical_id 全集的一一同序对应",
    )
    snapshot_ids = [
        row["physical_id"] for row in prep.rows(audio / "source_snapshot.csv.gz")
    ]
    require(snapshot_ids == ids, "源 stat 快照 ID 不对应输入全集")
    for job, result in zip(jobs, results):
        require(
            all(result.get(key) == value for key, value in job.items()),
            f"结果擅改输入字段：{job['physical_id']}",
        )
    totals = aggregate(results)
    for field, value in totals.items():
        require(summary.get(field) == value, f"summary 与结果重算的 {field} 不符")
    batch_size = contract["batch_size"]
    batch_count = (len(jobs) + batch_size - 1) // batch_size
    require(summary["batch_count"] == batch_count, "批次数不符")
    require(
        len(list((audio / "batches").glob("batch_*.json"))) == batch_count,
        "批次 checkpoint 有额外或缺失项",
    )
    require(
        len(list((audio / "batches").glob("batch_*.csv.gz"))) == batch_count,
        "批次数据有额外或缺失项",
    )
    checkpoints = []
    for number in range(batch_count):
        stem = f"batch_{number:05d}"
        checkpoint = load_json(audio / "batches" / f"{stem}.json")
        path = audio / "batches" / f"{stem}.csv.gz"
        start, end = number * batch_size, min((number + 1) * batch_size, len(jobs))
        require(checkpoint["contract_sha256"] == contract_sha, f"{stem} 契约不符")
        require(
            checkpoint["start"] == start and checkpoint["end"] == end,
            f"{stem} 行范围不符",
        )
        require(
            checkpoint["results_sha256"] == prep.sha256(path), f"{stem} SHA-256 不符"
        )
        block = list(prep.rows(path))
        require(block == results[start:end], f"{stem} 与合并结果逐字段不一致")
        require(checkpoint["summary"] == aggregate(block), f"{stem} 统计不一致")
        checkpoints.append(
            {"batch": stem, "sha256": checkpoint["results_sha256"], "rows": len(block)}
        )
    return jobs, results, summary, checkpoints


def technical_reasons(record, qc, duplicate_status):
    reasons = []
    if not qc:
        return ["missing_audio_qc"]
    if not truth(qc.get("strict_decode_pass")) or not truth(qc.get("decode_complete")):
        reasons.append("strict_decode_not_passed")
    if str(qc.get("has_decoder_error", "")).lower() != "false":
        reasons.append("decoder_error_or_error_check_unverified")
    frames = prep.decimal_value(qc.get("decoded_frames", ""))
    nonfinite = prep.decimal_value(qc.get("nonfinite_samples", ""))
    if frames is None or frames <= 0:
        reasons.append("no_positive_decoded_frames")
    if nonfinite is None or nonfinite != 0:
        reasons.append("nonfinite_or_unverified_samples")
    if not truth(qc.get("source_stat_matches")):
        reasons.append("source_stat_not_stable")
    if not truth(qc.get("expected_size_matches")):
        reasons.append("size_differs_from_inventory")
    digest = qc.get("sha256", "")
    if not len(digest) == 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        reasons.append("missing_or_invalid_payload_sha256")
    if truth(record.get("duplicate_scope_conflict")):
        reasons.append("duplicate_scope_conflict")
    if duplicate_status in {"conflicting_payload", "verification_incomplete"}:
        reasons.append(f"duplicate_{duplicate_status}")
    return reasons


def duplicate_status_for(group):
    if len(group) <= 1:
        return "no_checked_alias"
    if any(
        not row.get("sha256")
        or not truth(row.get("source_stat_matches"))
        or not truth(row.get("expected_size_matches"))
        for row in group
    ):
        return "verification_incomplete"
    return (
        "identical_payload"
        if len({row["sha256"] for row in group}) == 1
        else "conflicting_payload"
    )


def score_stratum(value):
    score = prep.decimal_value(value)
    if score is None or score <= Decimal("0.5") or score > 1:
        return "outside_selected_score_range"
    return "gt_0p5_lt_0p7" if score < Decimal("0.7") else "ge_0p7_le_1"


def payload_kind(group):
    """空文件共同 SHA 不称为重复音频；非空字节也不保证已成功解码。"""
    sizes = [prep.decimal_value(row.get("actual_size_bytes", "")) for row in group]
    if sizes and all(size == 0 for size in sizes):
        return "empty_payload"
    if sizes and all(size is not None and size > 0 for size in sizes):
        return "nonempty_payload"
    return "payload_size_unverified_or_inconsistent"


def publication_policy_reasons(
    qc, decode_integrity_eligible, known_clock_error, exclude_known_clock_errors
):
    """保留真正技术失败与通过基础门后被保守发布政策排除的区别。"""
    reasons = []
    if decode_integrity_eligible:
        rate = prep.decimal_value(qc.get("native_sample_rate", ""))
        duration = prep.decimal_value(qc.get("duration_seconds", ""))
        if rate is None or rate <= 0 or duration is None:
            reasons.append("nominal_duration_not_verifiable")
        elif abs(duration - Decimal(30)) > Decimal(1) / rate:
            reasons.append("non_nominal_duration")
        if qc.get("qc_status") == "decoded_with_warnings":
            reasons.append("decoder_warning")
        if truth(qc.get("all_zero", "")):
            reasons.append("all_zero_signal")
    # 时间错误不是音频损坏，也不能因基础解码失败而丢掉这条独立证据。
    if known_clock_error and exclude_known_clock_errors:
        reasons.append("known_clock_error")
    return reasons


def resolve_policy(args):
    """显式决定文件只控制时钟纳入；既有音频技术门不随决定文件放宽。"""
    path = getattr(args, "publication_policy", None)
    if path is None:
        exclude = bool(getattr(args, "exclude_known_clock_errors", False))
        return exclude, {
            "decision_date": None,
            "decision_source": "researcher instruction in current Audio QC task",
            "user_decision": "有问题或已损坏音频不纳入开源范围；"
            + ("已知时间戳错误录音也排除" if exclude else "已知时间戳错误录音保留标记")
            + "；原始档案保留追溯",
            "eligibility_scope_id": "decode_integrity_and_publication_policy",
        }
    decision = load_json(Path(path))
    required = ("decision_date", "decision_source", "user_decision", "policy_id")
    require(
        all(isinstance(decision.get(k), str) and decision[k] for k in required),
        "发布决定缺少日期、来源、原话或政策ID",
    )
    exclude = decision.get("exclude_known_clock_errors")
    require(isinstance(exclude, bool), "时钟排除政策必须是明确布尔值")
    require(
        not getattr(args, "exclude_known_clock_errors", False) or exclude,
        "命令行时钟排除与决定文件矛盾",
    )
    require(
        decision.get("change_scope")
        in {"known_clock_error_only", "raw_score_threshold_only"},
        "本入口只接受时钟或原始分数条件变更，其他技术门不可改变",
    )
    rule = ScoreRule.from_policy(decision)
    require(
        rule.threshold > Decimal("0.5") or rule.working_pool,
        "不能将>0.5解码框扩展到未全检的低分域",
    )
    return exclude, {
        **{k: decision[k] for k in required},
        "score_selection": rule.as_dict(),
        "eligibility_scope_id": decision["policy_id"],
        "supersedes_policy": decision.get("supersedes_policy", ""),
    }


def run(args):
    preparation, audio, output = (
        args.preparation_dir.resolve(),
        args.audio_dir.resolve(),
        args.output_dir.resolve(),
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"拒绝覆盖已有汇总：{output}")
    exclude_known_clock_errors, decision = resolve_policy(args)
    rule = ScoreRule.from_policy(decision)
    initial = load_json(preparation / "summary.json")
    input_provenance = load_json(preparation / "provenance.json")
    for item in input_provenance["outputs"]:
        require(
            prep.sha256(preparation / item["path"]) == item["sha256"],
            f"准备产物被更改：{item['path']}",
        )
    require(not initial["stat_errors"], "准备阶段存在 stat 错误，不能将未扫全称为完成")
    require(
        initial["physical_inventory_changes"] == 0
        and not initial["missing_old_logical_ids"],
        "目录库存有变化，需明确处置后另建版本",
    )
    jobs, results, audio_summary, checkpoints = verify_complete(preparation, audio)
    result_by_id = {row["physical_id"]: row for row in results}
    result_groups = defaultdict(list)
    for row in results:
        result_groups[row["logical_id"]].append(row)
    duplicate_status = {
        key: duplicate_status_for(group) for key, group in result_groups.items()
    }
    logical = list(prep.rows(preparation / "logical_scope.csv.gz"))
    require(len(logical) == initial["logical_mp3"], "逻辑库存行数不符")
    logical_index = {row["logical_id"]: row for row in logical}
    require(len(logical_index) == len(logical), "逻辑 ID 不唯一")
    known_path = args.timestamp_audit_dir.resolve() / "known_clock_errors.csv.gz"
    clock = list(prep.rows(known_path))
    clock_index = {
        (row["year"], row["site"], row["filename_timestamp"]): row for row in clock
    }
    require(len(clock_index) == len(clock), "已知时钟错误键不唯一")
    logical_time_keys = {
        (row["year"], row["site"], row["timestamp"]) for row in logical
    }
    require(
        set(clock_index).issubset(logical_time_keys),
        "已知错误时钟清单有未关联的旧逻辑键",
    )
    physical_csv = {}
    representative_ids = {row["physical_id"] for row in logical}
    for row in prep.rows(preparation / "physical_detection_audit.csv.gz"):
        if row["physical_id"] in representative_ids:
            physical_csv[row["physical_id"]] = {
                field: row[field]
                for field in (
                    "csv_sha256",
                    "csv_relative_path",
                    "target_min_start_s",
                    "target_max_end_s",
                    "all_min_start_s",
                    "all_max_end_s",
                    "nominal_end_after_30_rows",
                    "window_parse_error_rows",
                    "nonpositive_window_rows",
                    "negative_window_rows",
                )
            }
    require(len(physical_csv) == len(logical), "代表录音检测表审核关联不完整")
    final = []
    for record in logical:
        physical_id, key = record["physical_id"], record["logical_id"]
        qc = result_by_id.get(physical_id)
        status = duplicate_status.get(key, "not_checked_outside_scope")
        if (
            int(record["physical_count"]) > 1
            and record["scope_status"] == prep.SCORE_SELECTED
        ):
            require(
                len(result_groups[key]) == int(record["physical_count"]),
                f"候选同键副本缺项：{key}",
            )
        require(
            qc is None or truth(qc["is_inventory_representative"]),
            "逻辑代表误关联到副本",
        )
        time_key = (record["year"], record["site"], record["timestamp"])
        known = clock_index.get(time_key)
        score_status = (
            "unknown"
            if record["scope_status"] == prep.SCORE_UNKNOWN
            else (
                "valid_target_maximum"
                if record["target_max_confidence"]
                else "no_target_detection_row"
            )
        )
        scope = rule.scope(score_status, record["target_max_confidence"])
        selected = scope == rule.selected_scope
        if selected:
            require(
                record["scope_status"] == prep.SCORE_SELECTED and qc is not None,
                "新分数候选不在已有完整>0.5 QC范围内",
            )
        scope_reason = (
            record["scope_reason"]
            if rule.working_pool
            or scope == "unknown"
            or score_status == "no_target_detection_row"
            else ("target_max_meets_threshold" if selected else rule.exclusion_reason)
        )
        basic_reasons = technical_reasons(record, qc, status) if selected else []
        decode_eligible = selected and not basic_reasons
        policy_reasons = (
            publication_policy_reasons(
                qc, decode_eligible, known is not None, exclude_known_clock_errors
            )
            if selected
            else []
        )
        reasons = basic_reasons + policy_reasons
        technical_status = (
            ("technical_eligible" if not reasons else "held_for_technical_issue")
            if selected
            else (
                "unknown_score"
                if record["scope_status"] == prep.SCORE_UNKNOWN
                else "outside_score_scope"
            )
        )
        row = {
            **record,
            **(qc or {}),
            **physical_csv[physical_id],
            "scope_status": scope,
            "scope_reason": scope_reason,
            "qc_execution_scope_status": record["scope_status"],
            "score_comparison": rule.comparison,
            "score_threshold": str(rule.threshold),
            "audio_examined": qc is not None,
            "technical_status": technical_status,
            "technical_eligible": technical_status == "technical_eligible",
            "decode_integrity_eligible": decode_eligible,
            "decode_integrity_hold_reasons": ";".join(basic_reasons),
            "release_policy_exclusion_reasons": ";".join(policy_reasons),
            "eligibility_scope": decision["eligibility_scope_id"],
            "hold_reasons": ";".join(reasons),
            "duplicate_content_status": status,
            "score_stratum": (
                score_stratum(record["target_max_confidence"])
                if selected
                else "outside_selected_score_range"
            ),
            "filename_timestamp": record["timestamp"],
            "nominal_timezone": "Asia/Shanghai",
            "time_status": (
                "known_clock_error" if known else "not_independently_verified"
            ),
            "error_evidence": known["error_evidence"] if known else "",
            "corrected_recorded_at": "",
            "calendar_date_status": (
                "unverified_due_to_clock_error"
                if known
                else "filename_date_not_independently_verified"
            ),
            "target_window_extends_decoded_audio": "",
            "target_window_excess_seconds": "",
        }
        if qc is None:
            row["qc_status"] = "not_decoded_outside_scope"
        elif truth(qc["strict_decode_pass"]):
            duration = prep.decimal_value(qc.get("duration_seconds", ""))
            window_end = prep.decimal_value(row["target_max_end_s"])
            if duration is not None and window_end is not None:
                rate = prep.decimal_value(qc.get("native_sample_rate", ""))
                tolerance = Decimal(1) / rate if rate and rate > 0 else Decimal(0)
                row["target_window_extends_decoded_audio"] = (
                    window_end > duration + tolerance
                )
                row["target_window_excess_seconds"] = str(
                    max(Decimal(0), window_end - duration)
                )
        final.append(row)
    eligible = [row for row in final if row["technical_eligible"]]
    selected = [row for row in final if row["scope_status"] == rule.selected_scope]
    decode_eligible = [row for row in selected if row["decode_integrity_eligible"]]
    policy_only_excluded = [
        row for row in decode_eligible if not row["technical_eligible"]
    ]
    require(
        len(selected)
        == sum(
            rule.scope(
                (
                    "unknown"
                    if r["scope_status"] == prep.SCORE_UNKNOWN
                    else (
                        "valid_target_maximum"
                        if r["target_max_confidence"]
                        else "no_target_detection_row"
                    )
                ),
                r["target_max_confidence"],
            )
            == rule.selected_scope
            for r in logical
        ),
        "当前分数候选总数未闭合",
    )
    require(
        sum(row["audio_examined"] for row in final)
        == sum(truth(row["is_inventory_representative"]) for row in results),
        "逻辑代表解码覆盖未闭合",
    )
    final_index = {row["logical_id"]: row for row in final}
    payload_groups = defaultdict(list)
    for row in final:
        if row.get("sha256"):
            payload_groups[row["sha256"]].append(row)
    cross_payloads = {
        digest: group for digest, group in payload_groups.items() if len(group) > 1
    }
    cross_payload_kind_counts = Counter(
        payload_kind(group) for group in cross_payloads.values()
    )
    duplicate_mapping = []
    for key, group in result_groups.items():
        if len(group) > 1:
            representative = final_index[key]
            for row in group:
                if not truth(row["is_inventory_representative"]):
                    duplicate_mapping.append(
                        {
                            "relation": "same_logical_key_alias",
                            "logical_id": key,
                            "physical_id": row["physical_id"],
                            "relative_path": row["relative_path"],
                            "representative_physical_id": representative["physical_id"],
                            "sha256": row["sha256"],
                            "representative_sha256": representative.get("sha256", ""),
                            "duplicate_content_status": duplicate_status[key],
                            "payload_kind": payload_kind(group),
                            "logical_technical_eligible": representative[
                                "technical_eligible"
                            ],
                        }
                    )
    for digest, group in cross_payloads.items():
        for row in group:
            duplicate_mapping.append(
                {
                    "relation": "different_logical_keys_identical_payload",
                    "logical_id": row["logical_id"],
                    "physical_id": row["physical_id"],
                    "relative_path": row["relative_path"],
                    "representative_physical_id": "",
                    "sha256": digest,
                    "representative_sha256": "",
                    "duplicate_content_status": (
                        "same_empty_payload_not_audio_duplicate"
                        if payload_kind(group) == "empty_payload"
                        else "same_bytes_across_distinct_logical_keys_not_identity_proof"
                    ),
                    "payload_kind": payload_kind(group),
                    "logical_technical_eligible": row["technical_eligible"],
                }
            )
    strata, year_site = [], []
    for year in ("2023", "2024"):
        for site in prep.SITES:
            group = [
                row for row in final if row["year"] == year and row["site"] == site
            ]
            valid = [row for row in group if row["technical_eligible"]]
            year_site.append(
                {
                    "year": year,
                    "site": site,
                    "logical_archive": len(group),
                    "selected_candidates": sum(
                        row["scope_status"] == rule.selected_scope for row in group
                    ),
                    "unknown_score": sum(
                        row["scope_status"] == prep.SCORE_UNKNOWN for row in group
                    ),
                    "technical_eligible": len(valid),
                    "held_candidates": sum(
                        row["technical_status"] == "held_for_technical_issue"
                        for row in group
                    ),
                    "eligible_bytes": sum(
                        int(row["actual_size_bytes"]) for row in valid
                    ),
                    "eligible_duration_seconds": sum(
                        float(row["duration_seconds"]) for row in valid
                    ),
                    "eligible_known_clock_errors": sum(
                        row["time_status"] == "known_clock_error" for row in valid
                    ),
                    "eligible_distinct_payloads_within_stratum": len(
                        {row["sha256"] for row in valid}
                    ),
                }
            )
            score_bands = (
                ("gt_0p5_lt_0p7", "ge_0p7_le_1")
                if rule.threshold < Decimal("0.7")
                else ("ge_0p7_le_1",)
            )
            for score_band in score_bands:
                candidates = [
                    row
                    for row in group
                    if row["scope_status"] == rule.selected_scope
                    and row["score_stratum"] == score_band
                ]
                pool = [row for row in candidates if row["technical_eligible"]]
                strata.append(
                    {
                        "year": year,
                        "site": site,
                        "score_stratum": score_band,
                        "selected_candidates": len(candidates),
                        "held_candidates": len(candidates) - len(pool),
                        "technical_eligible_logical_files": len(pool),
                        "eligible_bytes": sum(
                            int(row["actual_size_bytes"]) for row in pool
                        ),
                        "eligible_duration_seconds": sum(
                            float(row["duration_seconds"]) for row in pool
                        ),
                        "known_clock_error_files": sum(
                            row["time_status"] == "known_clock_error" for row in pool
                        ),
                        "distinct_payloads_within_stratum": len(
                            {row["sha256"] for row in pool}
                        ),
                    }
                )
    dispositions = {}
    for source_name, output_name, expected in (
        (
            "old_anomaly_scope.csv",
            "known_anomaly_disposition.csv",
            initial["old_anomaly_logical"],
        ),
        (
            "candidate_list_reconciliation.csv",
            "candidate_list_disposition.csv",
            initial["candidate_list_rows"],
        ),
    ):
        originals = list(prep.rows(preparation / source_name))
        require(len(originals) == expected, f"{source_name} 行数未闭合")
        require(
            len({row["logical_id"] for row in originals}) == expected,
            f"{source_name} ID 重复",
        )
        require(
            all(row["logical_id"] in final_index for row in originals),
            f"{source_name} 有未关联去向",
        )
        dispositions[output_name] = [
            final_index[row["logical_id"]] for row in originals
        ]
    eligible_hashes = {row["sha256"] for row in eligible}
    eligible_duplicate_groups = Counter(row["sha256"] for row in eligible)
    eligible_durations = [Decimal(row["duration_seconds"]) for row in eligible]
    eligible_formats = Counter(
        (int(row["native_sample_rate"]), int(row["native_channels"]))
        for row in eligible
    )
    policy = {
        **decision,
        "eligibility_scope": "publication admission after decode/integrity gates and conservative user exclusions; not proof of species/song quality",
        "exclude_non_nominal_duration_after_successful_decode": True,
        "nominal_duration_seconds": 30,
        "absolute_duration_tolerance_native_frames": 1,
        "exclude_ordinary_decoder_warning": True,
        "exclude_all_zero_signal": True,
        "exclude_known_clock_errors": exclude_known_clock_errors,
        "amplitude_abs_gt_or_eq_1_exclusion": False,
        "other_timestamps_independently_verified": False,
        "delete_original_source_files": False,
        "incomplete_decode_duration_is_publication_duration_test": False,
    }
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state": "complete",
        "definition": "proposed publication admission, including user conservative exclusions; no species/song truth validation or public upload",
        "eligibility_scope": "technical_eligible includes user publication policy, not only decoding success; see decode_integrity_eligible for baseline gate",
        "publication_policy": policy,
        "score_selection": rule.as_dict(),
        "exact_threshold_logical_maxima": sum(
            prep.decimal_value(r["target_max_confidence"]) == rule.threshold
            for r in logical
            if r["scope_status"] != prep.SCORE_UNKNOWN
        ),
        "source_qc_candidate_logical_files": initial["current_scope_counts"].get(
            prep.SCORE_SELECTED, 0
        ),
        "qc_coverage": {
            "existing_logical_representatives_checked": sum(
                r["audio_examined"] for r in final
            ),
            "current_candidates_checked": sum(r["audio_examined"] for r in selected),
            "outside_current_candidates_checked": sum(
                r["audio_examined"] and r["scope_status"] != rule.selected_scope
                for r in final
            ),
            "archive_representatives_not_checked": sum(
                not r["audio_examined"] for r in final
            ),
            "new_audio_decodes": 0,
        },
        "logical_archive": len(final),
        "selected_candidates": len(selected),
        "technical_eligible_logical_files": len(eligible),
        "held_candidates": len(selected) - len(eligible),
        "decode_integrity_eligible_logical_files": len(decode_eligible),
        "decode_integrity_held_candidates": len(selected) - len(decode_eligible),
        "policy_excluded_after_decode_integrity": len(policy_only_excluded),
        "publication_eligible_logical_files": len(eligible),
        "decode_integrity_qc_status_counts": dict(
            Counter(row["qc_status"] for row in decode_eligible)
        ),
        "policy_only_excluded_qc_status_counts": dict(
            Counter(row["qc_status"] for row in policy_only_excluded)
        ),
        "release_policy_reason_counts_all_candidates": dict(
            Counter(
                reason
                for row in selected
                for reason in row["release_policy_exclusion_reasons"].split(";")
                if reason
            )
        ),
        "release_policy_reason_counts_after_decode_integrity": dict(
            Counter(
                reason
                for row in policy_only_excluded
                for reason in row["release_policy_exclusion_reasons"].split(";")
                if reason
            )
        ),
        "release_policy_reason_counts_note": "reason counts can overlap; policy_excluded_after_decode_integrity is the unique additional excluded count; failed partial decode duration is not tested against nominal duration",
        "technical_eligible_bytes": sum(
            int(row["actual_size_bytes"]) for row in eligible
        ),
        "technical_eligible_duration_seconds": sum(
            float(row["duration_seconds"]) for row in eligible
        ),
        "technical_eligible_distinct_payloads": len(eligible_hashes),
        "technical_eligible_cross_logical_duplicate_payload_groups": sum(
            count > 1 for count in eligible_duplicate_groups.values()
        ),
        "technical_eligible_extra_logical_ids_with_identical_payload": len(eligible)
        - len(eligible_hashes),
        "all_examined_representatives_cross_logical_duplicate_payload_groups": len(
            cross_payloads
        ),
        "all_examined_representatives_cross_logical_nonempty_payload_groups": cross_payload_kind_counts[
            "nonempty_payload"
        ],
        "all_examined_representatives_cross_logical_empty_payload_groups": cross_payload_kind_counts[
            "empty_payload"
        ],
        "all_examined_representatives_cross_logical_payload_size_unverified_groups": cross_payload_kind_counts[
            "payload_size_unverified_or_inconsistent"
        ],
        "all_examined_representatives_in_cross_logical_empty_payload_groups": sum(
            len(group)
            for group in cross_payloads.values()
            if payload_kind(group) == "empty_payload"
        ),
        "cross_logical_payload_interpretation": "empty-payload groups are zero-byte files, not duplicated audio; nonempty same-byte groups may still contain decoding failures; eligible duplicate counts are reported separately",
        "scope_counts": dict(Counter(row["scope_status"] for row in final)),
        "technical_status_counts": dict(
            Counter(row["technical_status"] for row in final)
        ),
        "selected_qc_status_counts": dict(
            Counter(row["qc_status"] for row in selected)
        ),
        "hold_reason_counts": dict(
            Counter(
                reason
                for row in selected
                for reason in row["hold_reasons"].split(";")
                if reason
            )
        ),
        "all_logical_known_clock_errors": sum(
            row["time_status"] == "known_clock_error" for row in final
        ),
        "eligible_known_clock_errors": sum(
            row["time_status"] == "known_clock_error" for row in eligible
        ),
        "eligible_other_time_unverified": sum(
            row["time_status"] != "known_clock_error" for row in eligible
        ),
        "eligible_target_window_extends_decoded_audio": sum(
            truth(row["target_window_extends_decoded_audio"]) for row in eligible
        ),
        "eligible_all_zero_files": sum(
            truth(row.get("all_zero", "")) for row in eligible
        ),
        "eligible_audio_format_counts": [
            {"sample_rate_hz": rate, "channels": channels, "logical_files": count}
            for (rate, channels), count in sorted(eligible_formats.items())
        ],
        "eligible_duration_seconds_min": (
            float(min(eligible_durations)) if eligible_durations else None
        ),
        "eligible_duration_seconds_max": (
            float(max(eligible_durations)) if eligible_durations else None
        ),
        "eligible_duration_exactly_30s": sum(
            duration == Decimal(30) for duration in eligible_durations
        ),
        "eligible_duration_not_exactly_30s": sum(
            Decimal(row["duration_seconds"]) != Decimal(30) for row in eligible
        ),
        "eligible_finite_amplitude_ge_1_files": sum(
            int(row.get("samples_abs_ge_1", "0") or 0) > 0 for row in eligible
        ),
        "eligible_finite_amplitude_gt_1_files": sum(
            int(row.get("samples_abs_gt_1", "0") or 0) > 0 for row in eligible
        ),
        "eligible_finite_amplitude_eq_1_files": sum(
            int(row.get("samples_abs_eq_1", "0") or 0) > 0 for row in eligible
        ),
        "same_key_alias_files_checked": sum(
            row["role"] == "selected_duplicate" for row in results
        ),
        "same_key_duplicate_content_status_counts": dict(
            Counter(
                duplicate_status[key]
                for key, group in result_groups.items()
                if len(group) > 1
            )
        ),
        "audio_job_count": len(results),
        "audio_qc_summary": audio_summary,
        "known_anomaly_status_counts": dict(
            Counter(
                row["technical_status"]
                for row in dispositions["known_anomaly_disposition.csv"]
            )
        ),
        "candidate_list_status_counts": dict(
            Counter(
                row["technical_status"]
                for row in dispositions["candidate_list_disposition.csv"]
            )
        ),
        "known_anomaly_qc_status_counts": dict(
            Counter(
                row["qc_status"]
                for row in dispositions["known_anomaly_disposition.csv"]
            )
        ),
        "candidate_list_qc_status_counts": dict(
            Counter(
                row["qc_status"]
                for row in dispositions["candidate_list_disposition.csv"]
            )
        ),
        "sampling_frame_definition": (
            "2 model-score strata (>0.5,<0.7 and >=0.7,<=1) x 12 year-site strata"
            if len(score_bands) == 2
            else "current admitted score range x 12 year-site strata"
        )
        + "; current score candidates only; original QC coverage reported separately; no human review sample drawn here",
        "sampling_frame_strata": len(strata),
        "limitations": [
            "只对候选及其同键副本、188条已知异常做音频检查；范围外其余音频未被全量解码/哈希。",
            "技术准入不证明目标种、鸣唱或完整自然声景；模型分数不是校准概率。",
            "基础技术判定与用户发布排除分开保存；异常时长(偏差>1原生采样帧)、普通解码警告、全零按用户决定不公开；已知错误时钟是否排除由本次明确开关记录。",
            "幅值越界和CSV窗超出解码时长仍仅诊断，不据此断言损坏；已知错误时钟属于元数据限制，不称为音频损坏。",
            "同字节跨逻辑键保持全部逻辑ID并另报不同payload数，不推断独立个体/独立观测，也不偷偷缩减后续复核总体。",
            "已知时钟错误只标记，无法矫正；其他时间未独立验证。",
        ],
    }
    require(
        sum(row["technical_eligible_logical_files"] for row in strata) == len(eligible),
        "当前分层框未完整覆盖合格逻辑行",
    )
    require(
        sum(row["selected_candidates"] for row in strata) == len(selected),
        "当前分层框未完整覆盖候选行",
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / ".gitignore").write_text(
        "# 可重建逐文件清单\n*.csv.gz\n", encoding="utf-8"
    )
    fields = list(dict.fromkeys(field for row in final for field in row))
    prep.write_rows(output / "final_logical_inventory.csv.gz", final, fields)
    prep.write_rows(output / "technical_eligible.csv.gz", eligible, fields)
    prep.write_rows(
        output / "held_candidates.csv.gz",
        (row for row in selected if not row["technical_eligible"]),
        fields,
    )
    prep.write_rows(output / "year_site.csv", year_site, list(year_site[0]))
    prep.write_rows(output / "sampling_frame.csv", strata, list(strata[0]))
    duplicate_fields = [
        "relation",
        "logical_id",
        "physical_id",
        "relative_path",
        "representative_physical_id",
        "sha256",
        "representative_sha256",
        "duplicate_content_status",
        "payload_kind",
        "logical_technical_eligible",
    ]
    prep.write_rows(
        output / "duplicate_mapping.csv.gz", duplicate_mapping, duplicate_fields
    )
    for name, records in dispositions.items():
        prep.write_rows(output / name, records, fields)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validation = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "checks": {
            "preparation_output_hashes": True,
            "preparation_no_stat_errors_or_inventory_changes": True,
            "full_mode_complete_no_limit": True,
            "manifest_results_snapshot_bijection": True,
            "results_and_batch_hashes": True,
            "batch_rows_equal_merged_rows": True,
            "summary_counters_recomputed": True,
            "same_key_selected_aliases_complete": True,
            "clock_error_join_complete": True,
            "old_anomaly_disposition_complete": True,
            "candidate_list_disposition_complete": True,
            "sampling_strata_partition": True,
            "current_candidates_all_in_existing_qc": True,
        },
        "batches": checkpoints,
        "audio_source_re_read_in_summary": False,
        "validation_scope": "checks frozen execution evidence and joins; source stability was observed by engine during its reads, not resampled here",
    }
    (output / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sources = [
        preparation / "summary.json",
        preparation / "provenance.json",
        preparation / "audio_jobs_full.csv.gz",
        audio / "summary.json",
        audio / "run_config.json",
        audio / "results.csv.gz",
        audio / "source_snapshot.csv.gz",
        known_path,
        Path(__file__),
        Path(prep.__file__),
        Path(__file__).with_name("release_score_policy.py"),
    ]
    if getattr(args, "publication_policy", None):
        sources.append(Path(args.publication_policy).resolve())
    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": platform.python_version(),
        "input_files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": prep.sha256(path),
            }
            for path in sources
        ],
        "outputs": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": prep.sha256(path),
            }
            for path in sorted(output.iterdir())
            if path.is_file()
        ],
        "eligibility_policy": "strict selected threshold + positive full finite decode + source stat and expected size match + no duplicate scope/payload uncertainty, followed by recorded researcher publication exclusions",
        "publication_policy": policy,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "logical_archive",
                    "selected_candidates",
                    "technical_eligible_logical_files",
                    "held_candidates",
                    "technical_eligible_distinct_payloads",
                    "technical_eligible_duration_seconds",
                )
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-dir", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timestamp-audit-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-known-clock-errors",
        action="store_true",
        help="按研究者决定将已知错误时钟录音排除出拟发布范围；其余时间仍未独立验证",
    )
    parser.add_argument(
        "--publication-policy",
        type=Path,
        help="明确时钟处理方式的发布策略JSON；不改变其他技术门",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
