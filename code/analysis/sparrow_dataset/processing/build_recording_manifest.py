#!/usr/bin/env python3
"""建立原始录音清单并审计六个样地的采样平衡性。

清单同时保留物理文件和按“年份-样地-时间戳”去重后的逻辑录音。
脚本只读取原始目录，不修改或移动音频文件。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass
class Recording:
    year: int
    site: str
    timestamp: str
    date: str
    time: str
    hour: int | None
    relative_path: str
    file_size_bytes: int
    companion_csv_exists: bool
    filename_valid: bool
    duplicate_group_size: int = 1
    is_canonical: bool = True

    @property
    def logical_key(self) -> tuple[int, str, str]:
        return self.year, self.site, self.timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-2023", type=Path, required=True)
    parser.add_argument("--root-2024", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--ffprobe-per-site-year",
        type=int,
        default=5,
        help="每个年份×样地抽取多少条录音核查编码参数，0表示跳过",
    )
    return parser.parse_args()


def timestamp_from_name(name: str) -> tuple[str, str, str, int | None, bool]:
    match = FILENAME_RE.match(name)
    if match is None:
        return "", "", "", None, False
    date = match["date"]
    time = f"{match['hour']}:{match['minute']}:{match['second']}"
    try:
        stamp = datetime.fromisoformat(f"{date}T{time}")
    except ValueError:
        return "", "", "", None, False
    return stamp.isoformat(), date, time, stamp.hour, True


def scan_root(year: int, root: Path) -> list[Recording]:
    if not root.is_dir():
        raise FileNotFoundError(f"{year}年原始录音目录不存在：{root}")

    recordings: list[Recording] = []
    for path in sorted(root.rglob("*.mp3")):
        relative = path.relative_to(root)
        site = relative.parts[0].lower() if len(relative.parts) >= 2 else ""
        timestamp, date, time, hour, valid = timestamp_from_name(path.name)
        companion = path.with_name(f"{path.stem}.wav.csv")
        recordings.append(
            Recording(
                year=year,
                site=site,
                timestamp=timestamp,
                date=date,
                time=time,
                hour=hour,
                relative_path=relative.as_posix(),
                file_size_bytes=path.stat().st_size,
                companion_csv_exists=companion.is_file(),
                filename_valid=valid,
            )
        )

    groups: dict[tuple[int, str, str], list[Recording]] = defaultdict(list)
    for recording in recordings:
        # 无法解析的文件以相对路径作为唯一键，避免错误合并。
        key = (
            recording.logical_key
            if recording.filename_valid
            else (year, recording.site, recording.relative_path)
        )
        groups[key].append(recording)

    for group in groups.values():
        group.sort(
            key=lambda row: (len(Path(row.relative_path).parts), row.relative_path)
        )
        for index, recording in enumerate(group):
            recording.duplicate_group_size = len(group)
            recording.is_canonical = index == 0
    return recordings


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def coefficient_of_variation(values: Iterable[int]) -> float:
    values = list(values)
    if not values or statistics.mean(values) == 0:
        return math.nan
    return statistics.pstdev(values) / statistics.mean(values)


def period(hour: int) -> str:
    if 4 <= hour < 8:
        return "dawn_04_08"
    if 8 <= hour < 12:
        return "morning_08_12"
    if 12 <= hour < 18:
        return "afternoon_12_18"
    if 18 <= hour < 22:
        return "evening_18_22"
    return "night_22_04"


def iso_week(date: str) -> str:
    parsed = datetime.fromisoformat(date)
    iso_year, week, _ = parsed.isocalendar()
    return f"{iso_year}-W{week:02d}"


def probe_audio(path: Path) -> dict[str, str]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=codec_name,sample_rate,channels,bit_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout)
        stream = payload.get("streams", [{}])[0]
        audio_format = payload.get("format", {})
        return {
            "probe_ok": "1",
            "codec": str(stream.get("codec_name", "")),
            "sample_rate_hz": str(stream.get("sample_rate", "")),
            "channels": str(stream.get("channels", "")),
            "bit_rate_bps": str(stream.get("bit_rate", "")),
            "duration_s": str(audio_format.get("duration", "")),
            "probe_error": "",
        }
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        return {
            "probe_ok": "0",
            "codec": "",
            "sample_rate_hz": "",
            "channels": "",
            "bit_rate_bps": "",
            "duration_s": "",
            "probe_error": str(error),
        }


def build_outputs(
    recordings: list[Recording],
    roots: dict[int, Path],
    output_dir: Path,
    probe_per_group: int,
) -> dict:
    physical_fields = [
        "year",
        "site",
        "timestamp",
        "date",
        "time",
        "hour",
        "relative_path",
        "file_size_bytes",
        "companion_csv_exists",
        "filename_valid",
        "duplicate_group_size",
        "is_canonical",
    ]
    write_csv(
        output_dir / "recording_manifest_physical.csv",
        physical_fields,
        (vars(row) for row in recordings),
    )

    logical = [row for row in recordings if row.is_canonical]
    write_csv(
        output_dir / "recording_manifest_logical.csv",
        physical_fields,
        (vars(row) for row in logical),
    )

    valid = [
        row for row in logical if row.filename_valid and row.site in EXPECTED_SITES
    ]
    by_year_site: dict[tuple[int, str], list[Recording]] = defaultdict(list)
    by_year_site_date: Counter[tuple[int, str, str]] = Counter()
    by_year_site_hour: Counter[tuple[int, str, int]] = Counter()
    by_year_site_period: Counter[tuple[int, str, str]] = Counter()
    for row in valid:
        by_year_site[(row.year, row.site)].append(row)
        by_year_site_date[(row.year, row.site, row.date)] += 1
        assert row.hour is not None
        by_year_site_hour[(row.year, row.site, row.hour)] += 1
        by_year_site_period[(row.year, row.site, period(row.hour))] += 1

    site_summary_rows = []
    for (year, site), rows in sorted(by_year_site.items()):
        date_counts = Counter(row.date for row in rows)
        sizes = [row.file_size_bytes for row in rows]
        site_summary_rows.append(
            {
                "year": year,
                "site": site,
                "logical_recordings": len(rows),
                "sampling_dates": len(date_counts),
                "first_date": min(date_counts),
                "last_date": max(date_counts),
                "median_recordings_per_date": f"{statistics.median(date_counts.values()):.1f}",
                "min_recordings_per_date": min(date_counts.values()),
                "max_recordings_per_date": max(date_counts.values()),
                "median_file_size_bytes": int(statistics.median(sizes)),
            }
        )
    write_csv(
        output_dir / "balance_by_year_site.csv",
        list(site_summary_rows[0]),
        site_summary_rows,
    )

    date_rows = [
        {"year": year, "site": site, "date": date, "recordings": count}
        for (year, site, date), count in sorted(by_year_site_date.items())
    ]
    write_csv(
        output_dir / "balance_by_year_site_date.csv",
        ["year", "site", "date", "recordings"],
        date_rows,
    )

    hour_rows = [
        {"year": year, "site": site, "hour": hour, "recordings": count}
        for (year, site, hour), count in sorted(by_year_site_hour.items())
    ]
    write_csv(
        output_dir / "balance_by_year_site_hour.csv",
        ["year", "site", "hour", "recordings"],
        hour_rows,
    )

    period_rows = [
        {"year": year, "site": site, "period": time_period, "recordings": count}
        for (year, site, time_period), count in sorted(by_year_site_period.items())
    ]
    write_csv(
        output_dir / "balance_by_year_site_period.csv",
        ["year", "site", "period", "recordings"],
        period_rows,
    )

    date_sites: dict[tuple[int, str], set[str]] = defaultdict(set)
    date_totals: Counter[tuple[int, str]] = Counter()
    for row in valid:
        date_sites[(row.year, row.date)].add(row.site)
        date_totals[(row.year, row.date)] += 1
    date_coverage_rows = [
        {
            "year": year,
            "date": date,
            "n_sites": len(sites),
            "sites": ";".join(sorted(sites)),
            "recordings": date_totals[(year, date)],
        }
        for (year, date), sites in sorted(date_sites.items())
    ]
    write_csv(
        output_dir / "date_site_coverage.csv",
        ["year", "date", "n_sites", "sites", "recordings"],
        date_coverage_rows,
    )

    pairwise_rows = []
    for year in sorted(roots):
        dates = {
            site: {row.date for row in valid if row.year == year and row.site == site}
            for site in EXPECTED_SITES
        }
        for index, site_a in enumerate(EXPECTED_SITES):
            for site_b in EXPECTED_SITES[index + 1 :]:
                overlap = dates[site_a] & dates[site_b]
                union = dates[site_a] | dates[site_b]
                pairwise_rows.append(
                    {
                        "year": year,
                        "site_a": site_a,
                        "site_b": site_b,
                        "overlap_dates": len(overlap),
                        "union_dates": len(union),
                        "jaccard": f"{len(overlap) / len(union):.4f}" if union else "",
                    }
                )
    write_csv(
        output_dir / "pairwise_date_overlap.csv",
        ["year", "site_a", "site_b", "overlap_dates", "union_dates", "jaccard"],
        pairwise_rows,
    )

    cell_sites: dict[tuple[int, str, int], set[str]] = defaultdict(set)
    cell_counts: Counter[tuple[int, str, int, str]] = Counter()
    week_cell_sites: dict[tuple[int, str, int], set[str]] = defaultdict(set)
    week_cell_counts: Counter[tuple[int, str, int, str]] = Counter()
    for row in valid:
        assert row.hour is not None
        cell_sites[(row.year, row.date, row.hour)].add(row.site)
        cell_counts[(row.year, row.date, row.hour, row.site)] += 1
        week = iso_week(row.date)
        week_cell_sites[(row.year, week, row.hour)].add(row.site)
        week_cell_counts[(row.year, week, row.hour, row.site)] += 1
    common_cells = {
        cell for cell, sites in cell_sites.items() if sites == set(EXPECTED_SITES)
    }
    matched_rows = []
    for year, date, hour in sorted(common_cells):
        counts = {
            site: cell_counts[(year, date, hour, site)] for site in EXPECTED_SITES
        }
        matched_rows.append(
            {
                "year": year,
                "date": date,
                "hour": hour,
                **counts,
                "minimum_available_per_site": min(counts.values()),
            }
        )
    write_csv(
        output_dir / "common_date_hour_cells.csv",
        ["year", "date", "hour", *EXPECTED_SITES, "minimum_available_per_site"],
        matched_rows,
    )

    common_week_cells = {
        cell for cell, sites in week_cell_sites.items() if sites == set(EXPECTED_SITES)
    }
    matched_week_rows = []
    for year, week, hour in sorted(common_week_cells):
        counts = {
            site: week_cell_counts[(year, week, hour, site)] for site in EXPECTED_SITES
        }
        matched_week_rows.append(
            {
                "year": year,
                "iso_week": week,
                "hour": hour,
                **counts,
                "minimum_available_per_site": min(counts.values()),
            }
        )
    write_csv(
        output_dir / "common_week_hour_cells.csv",
        ["year", "iso_week", "hour", *EXPECTED_SITES, "minimum_available_per_site"],
        matched_week_rows,
    )

    size_modes = {
        year: Counter(
            row.file_size_bytes for row in valid if row.year == year
        ).most_common(1)[0][0]
        for year in roots
    }
    anomaly_rows = [
        {
            "year": row.year,
            "site": row.site,
            "relative_path": row.relative_path,
            "file_size_bytes": row.file_size_bytes,
            "expected_mode_size_bytes": size_modes[row.year],
            "zero_byte": row.file_size_bytes == 0,
            "reason": (
                "zero_byte" if row.file_size_bytes == 0 else "non_modal_file_size"
            ),
        }
        for row in valid
        if row.file_size_bytes != size_modes[row.year]
    ]
    if anomaly_rows:
        write_csv(
            output_dir / "file_anomalies.csv",
            list(anomaly_rows[0]),
            anomaly_rows,
        )

    probe_rows: list[dict[str, str | int]] = []
    if probe_per_group > 0 and shutil.which("ffprobe"):
        for (year, site), rows in sorted(by_year_site.items()):
            rows = sorted(rows, key=lambda row: (row.timestamp, row.relative_path))
            if len(rows) <= probe_per_group:
                selected = rows
            elif probe_per_group == 1:
                selected = [rows[len(rows) // 2]]
            else:
                selected = [
                    rows[round(index * (len(rows) - 1) / (probe_per_group - 1))]
                    for index in range(probe_per_group)
                ]
            for row in selected:
                probe_rows.append(
                    {
                        "year": year,
                        "site": site,
                        "relative_path": row.relative_path,
                        **probe_audio(roots[year] / row.relative_path),
                    }
                )
    if probe_rows:
        write_csv(
            output_dir / "audio_metadata_sample.csv", list(probe_rows[0]), probe_rows
        )

    year_stats = {}
    for year in sorted(roots):
        physical_year = [row for row in recordings if row.year == year]
        logical_year = [row for row in logical if row.year == year]
        valid_year = [row for row in valid if row.year == year]
        site_counts = Counter(row.site for row in valid_year)
        dates_by_site = {
            site: {row.date for row in valid_year if row.site == site}
            for site in EXPECTED_SITES
        }
        common_dates = set.intersection(*dates_by_site.values())
        union_dates = set.union(*dates_by_site.values())
        common_year_cells = [cell for cell in common_cells if cell[0] == year]
        common_year_week_cells = [cell for cell in common_week_cells if cell[0] == year]
        year_mode_size = size_modes[year]
        year_stats[year] = {
            "physical": len(physical_year),
            "logical": len(logical_year),
            "duplicates": len(physical_year) - len(logical_year),
            "invalid_names": sum(not row.filename_valid for row in physical_year),
            "unexpected_sites": sorted(
                {row.site for row in physical_year if row.site not in EXPECTED_SITES}
            ),
            "missing_companion_csv": sum(
                not row.companion_csv_exists for row in physical_year
            ),
            "site_counts": dict(site_counts),
            "count_cv": coefficient_of_variation(site_counts.values()),
            "count_ratio_max_min": max(site_counts.values())
            / min(site_counts.values()),
            "union_dates": len(union_dates),
            "common_dates": len(common_dates),
            "common_date_hour_cells": len(common_year_cells),
            "matched_capacity": sum(
                min(
                    cell_counts[(cell[0], cell[1], cell[2], site)]
                    for site in EXPECTED_SITES
                )
                for cell in common_year_cells
            ),
            "common_week_hour_cells": len(common_year_week_cells),
            "week_matched_capacity": sum(
                min(
                    week_cell_counts[(cell[0], cell[1], cell[2], site)]
                    for site in EXPECTED_SITES
                )
                for cell in common_year_week_cells
            ),
            "mode_file_size_bytes": year_mode_size,
            "mode_file_size_fraction": sum(
                row.file_size_bytes == year_mode_size for row in valid_year
            )
            / len(valid_year),
            "zero_byte_files": sum(row.file_size_bytes == 0 for row in valid_year),
            "non_modal_size_files": sum(
                row.file_size_bytes != year_mode_size for row in valid_year
            ),
        }

    return {
        "physical_count": len(recordings),
        "logical_count": len(logical),
        "year_stats": year_stats,
        "site_summary_rows": site_summary_rows,
        "probe_rows": probe_rows,
    }


def report_markdown(summary: dict, output_dir: Path) -> str:
    lines = [
        "# 声景分析原始录音清单与平衡性审计",
        "",
        "本审计以样地目录和文件名时间戳建立清单；同一“年份—样地—时间戳”"
        "出现多个物理文件时，仅保留目录层级最浅的文件作为逻辑录音。",
        "",
        "## 1. 全量清单",
        "",
        f"- 物理MP3文件：{summary['physical_count']:,}条。",
        f"- 去重后逻辑录音：{summary['logical_count']:,}条。",
        f"- 输出目录：`{output_dir}`。",
        "",
        "## 2. 年份与样地汇总",
        "",
        "| 年份 | 样地 | 逻辑录音数 | 采样日期数 | 日期范围 | 每日录音中位数 |",
        "|---:|---|---:|---:|---|---:|",
    ]
    for row in summary["site_summary_rows"]:
        lines.append(
            f"| {row['year']} | {row['site']} | "
            f"{row['logical_recordings']:,} | {row['sampling_dates']} | "
            f"{row['first_date']}—{row['last_date']} | "
            f"{row['median_recordings_per_date']} |"
        )

    lines.extend(["", "## 3. 平衡性诊断", ""])
    for year, stats in summary["year_stats"].items():
        ratio = stats["count_ratio_max_min"]
        cv = stats["count_cv"]
        lines.extend(
            [
                f"### {year}年",
                "",
                f"- 物理文件{stats['physical']:,}条，逻辑录音"
                f"{stats['logical']:,}条，重复副本{stats['duplicates']:,}条。",
                f"- 六地总录音量最大/最小比为{ratio:.2f}，变异系数为{cv:.3f}。",
                f"- 全部日期并集{stats['union_dates']}天，其中六地共同有录音"
                f"{stats['common_dates']}天。",
                f"- 六地共同覆盖的“日期×小时”单元有"
                f"{stats['common_date_hour_cells']:,}个；若每个单元按最少样地"
                f"等量抽样，每个样地可提供{stats['matched_capacity']:,}条录音。",
                f"- 六地共同覆盖的“ISO周×小时”单元有"
                f"{stats['common_week_hour_cells']:,}个；按最少样地等量抽样时，"
                f"每个样地可提供{stats['week_matched_capacity']:,}条录音。",
                f"- 缺少同名`.wav.csv`伴随文件的MP3有"
                f"{stats['missing_companion_csv']:,}条；这不影响以MP3建立声景清单，"
                "但需要在使用BirdNET检测结果时单独处理。",
                f"- 众数文件大小为{stats['mode_file_size_bytes']:,}字节，占该年"
                f"逻辑录音的{stats['mode_file_size_fraction']:.2%}。",
                f"- 零字节文件{stats['zero_byte_files']:,}条，非众数文件大小"
                f"{stats['non_modal_size_files']:,}条，详见`file_anomalies.csv`。",
                "",
            ]
        )

    probe_rows = summary["probe_rows"]
    if probe_rows:
        successful = [row for row in probe_rows if row["probe_ok"] == "1"]
        configurations = Counter(
            (
                row["codec"],
                row["sample_rate_hz"],
                row["channels"],
                row["bit_rate_bps"],
                round(float(row["duration_s"]), 3),
            )
            for row in successful
        )
        lines.extend(
            [
                "## 4. 分层音频元数据抽检",
                "",
                f"按年份×样地等距抽检{len(probe_rows)}条，成功读取"
                f"{len(successful)}条。配置组合如下：",
                "",
                "| 编码 | 采样率（Hz） | 声道 | 比特率（bit/s） | 时长（s） | 数量 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for config, count in sorted(configurations.items()):
            lines.append(
                f"| {config[0]} | {config[1]} | {config[2]} | "
                f"{config[3]} | {config[4]:.3f} | {count} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 5. 结论与后续使用",
            "",
            "1. 六地均有大量原始30 s录音，但总量、采样日期数和每日录音数"
            "并不平衡，不能直接把全部录音混合后进行简单组间检验。",
            "2. 2023年存在嵌套日期目录造成的重复副本，声景分析应使用"
            "`recording_manifest_logical.csv`，避免重复计数。",
            "3. 2023年声景指数分析可在`common_date_hour_cells.csv`定义的六地"
            "共同日期×小时单元内分层等量抽样。2024年因采样人员有限且刘家峡"
            "距离较远，没有六地同日覆盖，应采用至少两个样地同日同小时的"
            "不完全区组设计，并在模型中显式控制日期×小时区组。",
            "4. 文件大小和抽样编码参数用于检查格式一致性，但不能替代设备增益、"
            "安装高度、天气及背景噪声元数据。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    roots = {2023: args.root_2023.resolve(), 2024: args.root_2024.resolve()}
    recordings: list[Recording] = []
    for year, root in roots.items():
        recordings.extend(scan_root(year, root))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_outputs(
        recordings,
        roots,
        args.output_dir,
        max(0, args.ffprobe_per_site_year),
    )
    report = report_markdown(summary, args.output_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    (args.output_dir / "balance_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
