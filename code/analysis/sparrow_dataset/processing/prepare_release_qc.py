"""Prepare candidate audio QC jobs from archive inventory and detection CSVs. The working candidate gate is >0.5; final eligibility uses the declared publication policy."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path


SITES = ("liujiaxia", "yongxing", "liangzhuang", "shuanghe", "minqin", "guanyinya")
TARGET = "Passer montanus"
NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-birdnet-(\d{2})_(\d{2})_(\d{2})$")
SCORE_SELECTED = "selected_above_05"
SCORE_EXCLUDED = "excluded_at_or_below"
SCORE_UNKNOWN = "unknown"
JOB_FIELDS = [
    "physical_id",
    "logical_id",
    "relative_path",
    "year",
    "site",
    "role",
    "expected_size_bytes",
    "scope_status",
    "is_inventory_representative",
    "old_anomaly",
    "duplicate_scope_conflict",
]
DETECTION_FIELDS = [
    "csv_relative_path",
    "csv_sha256",
    "csv_bytes",
    "csv_status",
    "csv_changed_during_read",
    "detection_rows",
    "valid_score_rows",
    "target_detection_rows",
    "target_window_count",
    "target_duplicate_window_rows",
    "target_max_confidence",
    "exact_threshold_rows",
    "invalid_confidence_rows",
    "nonfinite_confidence_rows",
    "malformed_rows",
    "window_parse_error_rows",
    "nonpositive_window_rows",
    "negative_window_rows",
    "nominal_end_after_30_rows",
    "target_min_start_s",
    "target_max_end_s",
    "all_min_start_s",
    "all_max_end_s",
    "scientific_name_trimmed_rows",
    "other_scientific_names",
    "error_details",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def write_rows(path: Path, records, fields):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def decimal_value(value: str) -> Decimal | None:
    try:
        result = Decimal(str(value).strip())
        return result if result.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def scope_for(status: str, score: str, threshold: Decimal) -> tuple[str, str]:
    if status not in {"ok", "empty"}:
        return SCORE_UNKNOWN, f"detection_csv_{status}"
    if score == "":
        # 未检出目标行不意味着真实无目标鸟，也不造一个分数 0。
        return SCORE_EXCLUDED, "no_target_detection_row_not_verified_absence"
    number = decimal_value(score)
    if number is None or not Decimal(0) <= number <= Decimal(1):
        return SCORE_UNKNOWN, "invalid_target_maximum"
    if number > threshold:
        return SCORE_SELECTED, "target_max_strictly_above_threshold"
    return SCORE_EXCLUDED, "target_max_at_or_below_threshold"


def identity(year: int, site: str, relative: str) -> tuple[str, str]:
    stem = Path(relative).stem
    match = NAME_RE.fullmatch(stem)
    timestamp = ""
    if match:
        value = f"{match[1]}T{match[2]}:{match[3]}:{match[4]}"
        try:
            timestamp = datetime.fromisoformat(value).isoformat()
        except ValueError:
            pass
    # 不合法名称不能在同一站点误并为同一空时间戳。
    key = f"{year}|{site}|{timestamp or ('unparsed:' + relative)}"
    return hashlib.sha256(key.encode()).hexdigest()[:24], timestamp


def physical_identity(relative: str) -> str:
    return hashlib.sha256(relative.encode()).hexdigest()[:24]


def detection_path(relative: str) -> str:
    path = Path(relative)
    return path.with_name(path.stem + ".wav.csv").as_posix()


def empty_detection(relative: str, status: str) -> dict:
    result = dict.fromkeys(DETECTION_FIELDS, "")
    result.update(
        csv_relative_path=relative, csv_status=status, csv_changed_during_read=False
    )
    for field in DETECTION_FIELDS:
        if field.endswith("_rows") or field in {"target_window_count", "csv_bytes"}:
            result[field] = 0
    return result


def parse_detection_csv(path: Path, relative: str, threshold: Decimal) -> dict:
    result = empty_detection(relative, "missing")
    errors = []

    def error(kind, line="", detail=""):
        if len(errors) < 10:
            errors.append({"kind": kind, "line": line, "detail": detail[:240]})

    try:
        before = path.stat()
        content = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        result["csv_status"] = "missing" if not path.exists() else "read_error"
        result["error_details"] = str(exc)
        return result
    result.update(
        csv_sha256=hashlib.sha256(content).hexdigest(), csv_bytes=len(content)
    )
    result["csv_changed_during_read"] = (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(content) != before.st_size
    )
    target_windows = set()
    target_scores = []
    other_names = Counter()
    starts, ends, target_starts, target_ends = [], [], [], []
    try:
        text = content.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=";", strict=True)
        header = next(reader, None)
        required = {"Start (s)", "End (s)", "Scientific name", "Confidence"}
        if (
            not header
            or not required.issubset(header)
            or len(set(header)) != len(header)
        ):
            raise ValueError(f"missing/duplicate required header: {header!r}")
        indices = {name: header.index(name) for name in required}
        for line_number, values in enumerate(reader, 2):
            if not values:  # csv.reader 与旧 DictReader 一致，忽略全空行。
                continue
            result["detection_rows"] += 1
            if len(values) != len(header):
                result["malformed_rows"] += 1
                error("column_count", line_number, repr(values))
                continue
            name_raw = values[indices["Scientific name"]]
            name = name_raw.strip()
            result["scientific_name_trimmed_rows"] += name_raw != name
            if not name:
                result["malformed_rows"] += 1
                error("empty_scientific_name", line_number)
            raw_score = values[indices["Confidence"]].strip()
            score = decimal_value(raw_score)
            if score is None:
                result["invalid_confidence_rows"] += 1
                try:
                    result["nonfinite_confidence_rows"] += not Decimal(
                        raw_score
                    ).is_finite()
                except InvalidOperation:
                    pass
                error("nonfinite_or_unparseable_confidence", line_number, raw_score)
            elif not Decimal(0) <= score <= Decimal(1):
                result["invalid_confidence_rows"] += 1
                error("confidence_outside_0_1", line_number, raw_score)
            else:
                result["valid_score_rows"] += 1
                result["exact_threshold_rows"] += score == threshold
                if name == TARGET:
                    result["target_detection_rows"] += 1
                    target_scores.append((score, raw_score))
            if name and name != TARGET:
                other_names[name] += 1
            begin = decimal_value(values[indices["Start (s)"]])
            end = decimal_value(values[indices["End (s)"]])
            if begin is None or end is None:
                result["window_parse_error_rows"] += 1
                error("window_not_finite_number", line_number)
                continue
            starts.append(begin)
            ends.append(end)
            result["nonpositive_window_rows"] += end <= begin
            result["negative_window_rows"] += begin < 0 or end < 0
            result["nominal_end_after_30_rows"] += end > 30
            if name == TARGET:
                target_starts.append(begin)
                target_ends.append(end)
                if (begin, end) in target_windows:
                    result["target_duplicate_window_rows"] += 1
                target_windows.add((begin, end))
        score_errors = result["malformed_rows"] + result["invalid_confidence_rows"]
        result["csv_status"] = (
            "parse_error"
            if score_errors
            else ("ok" if result["detection_rows"] else "empty")
        )
    except (UnicodeError, ValueError, csv.Error) as exc:
        result["csv_status"] = "parse_error"
        error("parser_exception", "", str(exc))
    if result["csv_changed_during_read"]:
        result["csv_status"] = "changed_during_read"
    # 错表内已解析行保留诊断；scope_for 不允许使用它们宣称阈值合格。
    if target_scores:
        result["target_max_confidence"] = max(target_scores, key=lambda pair: pair[0])[
            1
        ]
    result["target_window_count"] = len(target_windows)
    for field, values, func in (
        ("all_min_start_s", starts, min),
        ("all_max_end_s", ends, max),
        ("target_min_start_s", target_starts, min),
        ("target_max_end_s", target_ends, max),
    ):
        if values:
            result[field] = str(func(values))
    result["other_scientific_names"] = json.dumps(
        other_names, ensure_ascii=False, sort_keys=True
    )
    result["error_details"] = json.dumps(errors, ensure_ascii=False)
    return result


def old_anomaly(row):
    if row["header_ok"] != "True":
        return "zero_byte" if row["bytes"] == "0" else "header_error_nonzero"
    try:
        if abs(float(row["duration_s"]) - 30) > 1 / int(row["sample_rate"]):
            return "non_30s_header_duration"
    except (ValueError, ZeroDivisionError):
        return "invalid_old_header_duration"
    return ""


def choose_pilot(jobs: list[dict], per_stratum: int) -> tuple[list[dict], list[dict]]:
    by_stratum = defaultdict(list)
    for job in jobs:
        if job["role"] == "selected_representative":
            by_stratum[(job["year"], job["site"])].append(job)
    chosen, strata = set(), []
    for year in (2023, 2024):
        for site in SITES:
            candidates = sorted(
                by_stratum[(year, site)], key=lambda row: row["physical_id"]
            )
            sample = candidates[:per_stratum]
            chosen.update(row["logical_id"] for row in sample)
            strata.append(
                {
                    "year": year,
                    "site": site,
                    "available": len(candidates),
                    "requested": per_stratum,
                    "chosen": len(sample),
                }
            )
    pilot = [row for row in jobs if row["logical_id"] in chosen or row["old_anomaly"]]
    return pilot, strata


def run(args):
    started = time.monotonic()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"拒绝覆盖已有输出：{output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / ".gitignore").write_text(
        "# 可重建逐文件清单；小型证据与摘要可跟踪。\n*.csv.gz\n", encoding="utf-8"
    )
    threshold = Decimal(args.threshold)
    if threshold != Decimal("0.5"):
        raise ValueError("工作QC候选池要求严格 >0.5；最终准入使用单列发布策略")
    root = args.data_root.resolve()
    inventory = args.inventory_dir.resolve()
    old_metadata_path = inventory / "recording_metadata.csv.gz"
    old_physical_path = (
        inventory / "soundscape_inventory/recording_manifest_physical.csv"
    )
    high_path = inventory / "high_confidence_reconciliation.csv"
    duplicate_path = inventory / "duplicate_content_check.csv"
    old_meta, old_logical, anomaly = {}, {}, {}
    for row in rows(old_metadata_path):
        relative = f"birdsongs_{row['year']}/data/wav/raw/{row['relative_path']}"
        row["logical_id"], _ = identity(int(row["year"]), row["site"], relative)
        row["scope_status"], _ = scope_for(
            row["detection_csv_status"], row["target_max_confidence"], threshold
        )
        old_meta[relative] = row
        if row["logical_id"] in old_logical:
            raise ValueError("历史逻辑键冲突")
        old_logical[row["logical_id"]] = row
        if kind := old_anomaly(row):
            anomaly[row["logical_id"]] = kind
    old_physical = {}
    for row in rows(old_physical_path):
        relative = f"birdsongs_{row['year']}/data/wav/raw/{row['relative_path']}"
        old_physical[relative] = {
            "expected_size_bytes": int(row["file_size_bytes"]),
            "is_canonical": row["is_canonical"] == "True",
        }
    print(
        f"输入清单：{len(old_meta)} 逻辑 / {len(old_physical)} 物理 / {len(anomaly)} 异常",
        flush=True,
    )
    physical, csv_files, stat_errors = {}, {}, []
    for year in (2023, 2024):
        raw = root / f"birdsongs_{year}/data/wav/raw"
        if not raw.is_dir():
            raise FileNotFoundError(raw)
        for parent, directories, names in os.walk(raw):
            directories.sort()
            for name in sorted(names):
                path = Path(parent) / name
                relative = path.relative_to(root).as_posix()
                if name.endswith(".wav.csv"):
                    csv_files[relative] = path
                if path.suffix.lower() != ".mp3":
                    continue
                try:
                    stat = path.stat()
                except OSError as exc:
                    stat_errors.append({"relative_path": relative, "error": str(exc)})
                    continue
                site = path.relative_to(raw).parts[0]
                logical_id, stamp = identity(year, site, relative)
                physical[relative] = {
                    "physical_id": physical_identity(relative),
                    "logical_id": logical_id,
                    "relative_path": relative,
                    "year": year,
                    "site": site,
                    "timestamp": stamp,
                    "expected_size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "site_known": site in SITES,
                    "filename_valid": bool(stamp),
                    "old_anomaly": bool(anomaly.get(logical_id)),
                }
    changes = []
    for relative in sorted(set(old_physical) | set(physical)):
        before, after = old_physical.get(relative), physical.get(relative)
        kind = (
            "new_physical_file"
            if before is None
            else (
                "missing_physical_file"
                if after is None
                else (
                    "size_changed"
                    if before["expected_size_bytes"] != after["expected_size_bytes"]
                    else ""
                )
            )
        )
        if kind:
            changes.append(
                {
                    "relative_path": relative,
                    "change": kind,
                    "old_bytes": before["expected_size_bytes"] if before else "",
                    "current_bytes": after["expected_size_bytes"] if after else "",
                }
            )
    all_csv_paths = sorted(set(csv_files) | {detection_path(path) for path in physical})
    parsed = {}
    print(
        f"当前目录：{len(physical)} MP3；存在 {len(csv_files)} 检测 CSV；开始独立解析",
        flush=True,
    )
    # 分批 map 避免为 34 万文件同时创建 Future。
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for offset in range(0, len(all_csv_paths), 2000):
            batch = all_csv_paths[offset : offset + 2000]
            results = pool.map(
                lambda relative: parse_detection_csv(
                    root / relative, relative, threshold
                ),
                batch,
            )
            for result in results:
                parsed[result["csv_relative_path"]] = result
            if (offset // 2000) % 25 == 0:
                print(
                    f"CSV 已检查 {min(offset + 2000, len(all_csv_paths))}/{len(all_csv_paths)}；{time.monotonic() - started:.1f}s",
                    flush=True,
                )
    groups = defaultdict(list)
    for item in physical.values():
        groups[item["logical_id"]].append(item)
    logical, conflict_rows = [], []
    for logical_id, group in sorted(groups.items()):
        group.sort(
            key=lambda item: (
                not old_physical.get(item["relative_path"], {}).get(
                    "is_canonical", False
                ),
                item["relative_path"],
            )
        )
        representative = group[0]
        detection = parsed[detection_path(representative["relative_path"])]
        scope, reason = scope_for(
            detection["csv_status"], detection["target_max_confidence"], threshold
        )
        member_scopes = [
            scope_for(
                parsed[detection_path(item["relative_path"])]["csv_status"],
                parsed[detection_path(item["relative_path"])]["target_max_confidence"],
                threshold,
            )[0]
            for item in group
        ]
        score_values = {
            parsed[detection_path(item["relative_path"])]["target_max_confidence"]
            for item in group
        }
        csv_hashes = {
            parsed[detection_path(item["relative_path"])]["csv_sha256"]
            for item in group
        }
        scope_conflict = len(set(member_scopes)) > 1
        if scope_conflict or len(score_values) > 1:
            conflict_rows.append(
                {
                    "logical_id": logical_id,
                    "representative_relative_path": representative["relative_path"],
                    "scope_conflict": scope_conflict,
                    "member_scopes": json.dumps(member_scopes),
                    "member_scores": json.dumps(sorted(score_values)),
                    "member_relative_paths": json.dumps(
                        [item["relative_path"] for item in group]
                    ),
                }
            )
        previous = old_logical.get(logical_id)
        for item in group:
            item.update(
                is_inventory_representative=item is representative,
                scope_status=scope,
                duplicate_scope_conflict=scope_conflict,
            )
        record = {
            **representative,
            "representative_relative_path": representative["relative_path"],
            "physical_count": len(group),
            "scope_reason": reason,
            "csv_status": detection["csv_status"],
            "target_max_confidence": detection["target_max_confidence"],
            "target_detection_rows": detection["target_detection_rows"],
            "target_window_count": detection["target_window_count"],
            "detection_rows": detection["detection_rows"],
            "duplicate_csv_bytes_differ": len(csv_hashes) > 1,
            "duplicate_target_score_differ": len(score_values) > 1,
            "old_scope_status": (
                previous["scope_status"] if previous else "not_in_old_logical"
            ),
            "old_detection_csv_status": (
                previous["detection_csv_status"] if previous else ""
            ),
            "old_target_max_confidence": (
                previous["target_max_confidence"] if previous else ""
            ),
            "old_anomaly_kind": anomaly.get(logical_id, ""),
        }
        difference = []
        if previous is None:
            difference.append("new_logical_key")
        else:
            if previous["scope_status"] != scope:
                difference.append("scope_changed")
            if previous["detection_csv_status"] != detection["csv_status"]:
                difference.append("csv_status_changed")
            if decimal_value(previous["target_max_confidence"]) != decimal_value(
                detection["target_max_confidence"]
            ):
                difference.append("maximum_score_changed")
            old_relative = (
                f"birdsongs_{previous['year']}/data/wav/raw/{previous['relative_path']}"
            )
            if old_relative != representative["relative_path"]:
                difference.append("representative_changed")
        record["comparison_change"] = ";".join(difference)
        logical.append(record)
    logical_index = {record["logical_id"]: record for record in logical}
    missing_logical = sorted(set(old_logical) - set(logical_index))
    jobs = []
    for item in sorted(
        physical.values(),
        key=lambda item: (
            item["year"],
            SITES.index(item["site"]) if item["site"] in SITES else 99,
            item["relative_path"],
        ),
    ):
        if item["scope_status"] == SCORE_SELECTED:
            role = (
                "selected_representative"
                if item["is_inventory_representative"]
                else "selected_duplicate"
            )
        elif item["old_anomaly"]:
            role = "diagnostic_anomaly"
        else:
            continue
        jobs.append({**item, "role": role})
    pilot, strata = choose_pilot(jobs, args.pilot_per_stratum)
    write_rows(output / "audio_jobs_full.csv.gz", jobs, JOB_FIELDS)
    write_rows(output / "audio_jobs_pilot.csv.gz", pilot, JOB_FIELDS)
    logical_fields = list(logical[0]) if logical else ["logical_id", "scope_status"]
    write_rows(output / "logical_scope.csv.gz", logical, logical_fields)
    write_rows(
        output / "logical_scope_differences.csv.gz",
        (record for record in logical if record["comparison_change"]),
        logical_fields,
    )
    audit_fields = list(next(iter(physical.values()))) + DETECTION_FIELDS
    write_rows(
        output / "physical_detection_audit.csv.gz",
        (
            {**item, **parsed[detection_path(item["relative_path"])]}
            for item in sorted(
                physical.values(), key=lambda item: item["relative_path"]
            )
        ),
        audit_fields,
    )
    companion_paths = {detection_path(relative) for relative in physical}
    orphan_paths = sorted(set(csv_files) - companion_paths)
    write_rows(
        output / "csv_orphans.csv.gz",
        (parsed[path] for path in orphan_paths),
        DETECTION_FIELDS,
    )
    write_rows(
        output / "inventory_changes.csv.gz",
        changes,
        ["relative_path", "change", "old_bytes", "current_bytes"],
    )
    write_rows(
        output / "duplicate_detection_conflicts.csv",
        conflict_rows,
        [
            "logical_id",
            "representative_relative_path",
            "scope_conflict",
            "member_scopes",
            "member_scores",
            "member_relative_paths",
        ],
    )
    anomaly_rows = [
        {
            "logical_id": key,
            "old_anomaly_kind": kind,
            "current_exists": key in logical_index,
            **{
                field: logical_index.get(key, {}).get(field, "")
                for field in (
                    "year",
                    "site",
                    "representative_relative_path",
                    "scope_status",
                    "target_max_confidence",
                    "expected_size_bytes",
                )
            },
        }
        for key, kind in sorted(anomaly.items())
    ]
    write_rows(
        output / "old_anomaly_scope.csv",
        anomaly_rows,
        [
            "logical_id",
            "old_anomaly_kind",
            "current_exists",
            "year",
            "site",
            "representative_relative_path",
            "scope_status",
            "target_max_confidence",
            "expected_size_bytes",
        ],
    )
    reconciliation = []
    for previous in rows(high_path):
        relative = (
            f"birdsongs_{previous['year']}/data/wav/raw/{previous['relative_path']}"
        )
        key, _ = identity(int(previous["year"]), previous["site"], relative)
        current = logical_index.get(key, {})
        reconciliation.append(
            {
                "logical_id": key,
                "previous_relative_path": relative,
                "previous_score": previous["target_max_confidence"],
                "previous_in_high_list": previous["in_existing_high_list"],
                "current_exists": key in logical_index,
                **{
                    field: current.get(field, "")
                    for field in (
                        "scope_status",
                        "target_max_confidence",
                        "csv_status",
                        "expected_size_bytes",
                        "old_anomaly_kind",
                    )
                },
            }
        )
    write_rows(
        output / "candidate_list_reconciliation.csv",
        reconciliation,
        list(reconciliation[0]) if reconciliation else ["logical_id"],
    )
    current_counts = Counter(record["scope_status"] for record in logical)
    old_counts = Counter(record["scope_status"] for record in old_logical.values())
    scope_summary = []
    for year in (2023, 2024):
        for site in SITES:
            group = [
                record
                for record in logical
                if record["year"] == year and record["site"] == site
            ]
            selected = [
                record for record in group if record["scope_status"] == SCORE_SELECTED
            ]
            scope_summary.append(
                {
                    "year": year,
                    "site": site,
                    "logical_count": len(group),
                    "selected_count": len(selected),
                    "selected_bytes": sum(
                        record["expected_size_bytes"] for record in selected
                    ),
                    "unknown_count": sum(
                        record["scope_status"] == SCORE_UNKNOWN for record in group
                    ),
                }
            )
    existing_results = [parsed[path] for path in csv_files]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "threshold": args.threshold,
        "comparison": ">",
        "target_scientific_name": TARGET,
        "scope_basis": "inventory representative detection CSV; each file maximum target score; not probability",
        "physical_mp3": len(physical),
        "logical_mp3": len(logical),
        "physical_bytes": sum(
            item["expected_size_bytes"] for item in physical.values()
        ),
        "old_physical_mp3": len(old_physical),
        "old_logical_mp3": len(old_logical),
        "current_scope_counts": dict(current_counts),
        "old_scope_counts": dict(old_counts),
        "selected_representative_bytes": sum(
            record["expected_size_bytes"]
            for record in logical
            if record["scope_status"] == SCORE_SELECTED
        ),
        "selected_zero_byte_logical": sum(
            record["scope_status"] == SCORE_SELECTED
            and record["expected_size_bytes"] == 0
            for record in logical
        ),
        "scope_difference_count": sum(
            bool(record["comparison_change"]) for record in logical
        ),
        "missing_old_logical_ids": missing_logical,
        "physical_inventory_changes": len(changes),
        "stat_errors": stat_errors,
        "existing_detection_csv": len(csv_files),
        "orphan_detection_csv": len(orphan_paths),
        "physical_companion_status_counts": dict(
            Counter(parsed[detection_path(path)]["csv_status"] for path in physical)
        ),
        "logical_companion_status_counts": dict(
            Counter(record["csv_status"] for record in logical)
        ),
        "existing_csv_status_counts": dict(
            Counter(item["csv_status"] for item in existing_results)
        ),
        "detection_totals_existing_csv": {
            field: sum(item[field] for item in existing_results)
            for field in DETECTION_FIELDS
            if field.endswith("_rows") or field == "target_window_count"
        },
        "detection_rows_total_note": "includes duplicate physical CSV and orphan CSV; not logical/independent acoustic observations",
        "exact_threshold_logical_maxima": sum(
            decimal_value(record["target_max_confidence"]) == threshold
            for record in logical
            if record["csv_status"] in {"ok", "empty"}
        ),
        "duplicate_groups": sum(len(group) > 1 for group in groups.values()),
        "duplicate_extra_files": len(physical) - len(logical),
        "duplicate_score_conflict_groups": len(conflict_rows),
        "duplicate_scope_conflict_groups": sum(
            record["scope_conflict"] for record in conflict_rows
        ),
        "old_anomaly_logical": len(anomaly),
        "old_anomaly_current_scope_counts": dict(
            Counter(row["scope_status"] for row in anomaly_rows)
        ),
        "candidate_list_rows": len(reconciliation),
        "candidate_list_current_scope_counts": dict(
            Counter(row["scope_status"] for row in reconciliation)
        ),
        "jobs_full": len(jobs),
        "jobs_full_roles": dict(Counter(row["role"] for row in jobs)),
        "jobs_pilot": len(pilot),
        "jobs_pilot_roles": dict(Counter(row["role"] for row in pilot)),
        "pilot_strata": strata,
        "year_site": scope_summary,
        "engineering_pilot_selection": "lowest SHA256(relative path) physical IDs per year/site; 10 representatives each; plus every known old anomaly and selected sample aliases; not biological validation sample",
        "elapsed_seconds": time.monotonic() - started,
        "limitations": [
            "本步骤只 stat 音频；音频 SHA-256、解码和重复内容检查由后续 engine 独立完成。",
            "未入选低分/未知原始录音未做全量音频哈希与解码；CSV独立解析范围为全物理树。",
            "窗口超出名义30秒只记录诊断；须与解码时长和BirdNET末端滑窗实现分开解释。",
            "有效CSV无目标行记为范围外，但不证明没有树麻雀；缺表、错表不记为阴性。",
            "重复副本范围冲突保持显式标识；发布前需裁决，不从副本挑高分替代历史代表。",
            "逻辑键含年份、站点、文件名时间戳；名称时间错误未被修复，也不证明独立个体。",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sources = [
        old_metadata_path,
        old_physical_path,
        high_path,
        duplicate_path,
        Path(__file__),
    ]
    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "data_root": str(root),
        "parameters": {
            "threshold": args.threshold,
            "workers": args.workers,
            "pilot_per_stratum": args.pilot_per_stratum,
        },
        "source_files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sources
        ],
        "outputs": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "provenance.json"
        ],
        "readonly_sources": True,
        "audio_decoded": False,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "physical_mp3",
                    "logical_mp3",
                    "current_scope_counts",
                    "selected_representative_bytes",
                    "scope_difference_count",
                    "jobs_full",
                    "jobs_pilot",
                    "elapsed_seconds",
                )
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--threshold", default="0.5")
    parser.add_argument("--pilot-per-stratum", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or args.pilot_per_stratum < 1:
        parser.error("workers 和 pilot-per-stratum 必须为正数")
    run(args)


if __name__ == "__main__":
    main()
