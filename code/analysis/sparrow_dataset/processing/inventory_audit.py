"""Audit archive headers, companion detections and duplicate file identities. Run inventory first into the declared inventory directory."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

SITE_ORDER = ("liujiaxia", "yongxing", "liangzhuang", "shuanghe", "minqin", "guanyinya")
SOURCE_RE = re.compile(
    r"^(?P<site>[a-z]+)_(?P<stem>\d{4}-\d{2}-\d{2}-birdnet-\d{2}_\d{2}_\d{2})(?P<suffix>.*)$"
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--inventory-dir",
        type=Path,
        required=True,
        help="Contains soundscape_inventory/recording_manifest_physical.csv; derived audit tables are written here",
    )
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    args = parser.parse_args()
    HERE = args.inventory_dir.resolve()
    data_root = args.data_root.resolve()
    if (HERE / "recording_metadata.csv.gz").exists():
        raise FileExistsError(
            "Inventory audit already exists; use a fresh inventory directory"
        )
    roots = {y: data_root / f"birdsongs_{y}/data/wav/raw" for y in (2023, 2024)}
    inventory = HERE / "soundscape_inventory/recording_manifest_physical.csv"
    physical = read_rows(inventory)
    logical = [r for r in physical if r["is_canonical"] == "True"]
    assert len(logical) > 0
    indexed = {
        (int(r["year"]), r["site"], Path(r["relative_path"]).stem): r for r in logical
    }
    filetypes, orphans = [], []
    for year, root in roots.items():
        counts, sizes = Counter(), Counter()
        for parent, _, names in os.walk(root):
            for name in names:
                path = Path(parent) / name
                ext = path.suffix.lower()
                counts[ext] += 1
                sizes[ext] += path.stat().st_size
                if (
                    name.endswith(".wav.csv")
                    and not path.with_name(name[:-8] + ".mp3").exists()
                ):
                    orphans.append(
                        {
                            "year": year,
                            "relative_path": path.relative_to(root).as_posix(),
                        }
                    )
        filetypes.extend(
            {"year": year, "extension": e, "files": counts[e], "bytes": sizes[e]}
            for e in sorted(counts)
        )
    write_rows(HERE / "filetypes.csv", filetypes)
    write_rows(HERE / "orphan_detection_csv.csv", orphans, ["year", "relative_path"])

    duplicate_groups = defaultdict(list)
    for row in physical:
        if int(row["duplicate_group_size"]) > 1:
            duplicate_groups[(row["year"], row["site"], row["timestamp"])].append(row)
    duplicates = []
    for key, group in duplicate_groups.items():
        hashes = []
        for row in group:
            hashes.append(sha256(roots[int(row["year"])] / row["relative_path"]))
        for row, digest in zip(group, hashes):
            duplicates.append(
                {
                    "year": key[0],
                    "site": key[1],
                    "timestamp": key[2],
                    "relative_path": row["relative_path"],
                    "sha256": digest,
                    "same_bytes_within_group": len(set(hashes)) == 1,
                    "is_inventory_representative": row["is_canonical"],
                }
            )
    write_rows(HERE / "duplicate_content_check.csv", duplicates)
    print(f"Duplicate groups checked: {len(duplicate_groups)}", flush=True)

    high_lists, high_sources = {}, []
    for year in roots:
        path = data_root / f"birdsongs_{year}/data/scp/raw/wav_confidence_ge_0.7.scp"
        names = [
            Path(line.strip()).stem
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        high_lists[year] = set(names)
        high_sources.append(
            {
                "year": year,
                "file": str(path),
                "sha256": sha256(path),
                "lines": len(names),
                "unique_names": len(set(names)),
            }
        )

    groups = {}
    for year in roots:
        for site in SITE_ORDER:
            groups[(year, site)] = {
                "year": year,
                "site": site,
                "physical_mp3": 0,
                "physical_bytes": 0,
                "logical_mp3": 0,
                "logical_bytes": 0,
                "zero_byte": 0,
                "header_ok": 0,
                "header_error_nonzero": 0,
                "duration_s": 0.0,
                "duration_not_30s": 0,
                "missing_detection_csv": 0,
                "bad_detection_csv": 0,
                "empty_detection_csv": 0,
                "detection_rows": 0,
                "target_detection_rows": 0,
                "target_files_ge_0_7": 0,
                "in_existing_high_list": 0,
            }
    for row in physical:
        group = groups[(int(row["year"]), row["site"])]
        group["physical_mp3"] += 1
        group["physical_bytes"] += int(row["file_size_bytes"])
    formats, species = Counter(), Counter()
    durations = Counter()
    date_sets = defaultdict(set)
    monthly = Counter()
    issues, mismatches = [], []
    metadata_index = {}
    fields = [
        "year",
        "site",
        "relative_path",
        "timestamp",
        "bytes",
        "header_ok",
        "sample_rate",
        "channels",
        "frames",
        "duration_s",
        "format",
        "subtype",
        "header_error",
        "detection_csv_status",
        "detection_rows",
        "target_detection_rows",
        "target_max_confidence",
        "in_existing_high_list",
    ]
    with gzip.open(
        HERE / "recording_metadata.csv.gz", "wt", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(logical, 1):
            year, site = int(row["year"]), row["site"]
            group = groups[(year, site)]
            path = roots[year] / row["relative_path"]
            before = path.stat()
            item = {k: "" for k in fields}
            item.update(
                year=year,
                site=site,
                relative_path=row["relative_path"],
                timestamp=row["timestamp"],
                bytes=before.st_size,
                header_ok=False,
            )
            group["logical_mp3"] += 1
            group["logical_bytes"] += before.st_size
            date_sets[(year, site)].add(row["date"])
            monthly[(year, site, row["date"][:7])] += 1
            if before.st_size == 0:
                group["zero_byte"] += 1
                item["header_error"] = "zero_byte"
            else:
                try:
                    info = sf.info(path)
                    assert info.frames > 0
                    item.update(
                        header_ok=True,
                        sample_rate=info.samplerate,
                        channels=info.channels,
                        frames=info.frames,
                        duration_s=info.duration,
                        format=info.format,
                        subtype=info.subtype,
                    )
                    group["header_ok"] += 1
                    group["duration_s"] += info.duration
                    group["duration_not_30s"] += (
                        abs(info.duration - 30) > 1 / info.samplerate
                    )
                    formats[
                        (info.format, info.subtype, info.samplerate, info.channels)
                    ] += 1
                    durations[round(info.duration, 9)] += 1
                except (RuntimeError, AssertionError) as error:
                    group["header_error_nonzero"] += 1
                    item["header_error"] = str(error) or "nonpositive_frames"
            csv_path = path.with_name(path.stem + ".wav.csv")
            nr = nt = 0
            confidence = None
            if not csv_path.is_file():
                item["detection_csv_status"] = "missing"
                group["missing_detection_csv"] += 1
            else:
                try:
                    with csv_path.open(encoding="utf-8-sig", newline="") as cf:
                        reader = csv.DictReader(cf, delimiter=";")
                        assert reader.fieldnames and {
                            "Scientific name",
                            "Confidence",
                        }.issubset(reader.fieldnames)
                        for detection in reader:
                            value = float(detection["Confidence"])
                            assert 0 <= value <= 1
                            taxon = detection["Scientific name"].strip()
                            nr += 1
                            species[taxon] += 1
                            if taxon == "Passer montanus":
                                nt += 1
                                confidence = (
                                    value
                                    if confidence is None
                                    else max(confidence, value)
                                )
                    item["detection_csv_status"] = "ok" if nr else "empty"
                    group["empty_detection_csv"] += nr == 0
                except (UnicodeError, ValueError, KeyError, AssertionError, csv.Error):
                    item["detection_csv_status"] = "parse_error"
                    group["bad_detection_csv"] += 1
            high = f"{site}_{path.stem}" in high_lists[year]
            item.update(
                detection_rows=nr,
                target_detection_rows=nt,
                target_max_confidence=confidence if confidence is not None else "",
                in_existing_high_list=high,
            )
            group["detection_rows"] += nr
            group["target_detection_rows"] += nt
            group["target_files_ge_0_7"] += confidence is not None and confidence >= 0.7
            group["in_existing_high_list"] += high
            if high != (confidence is not None and confidence >= 0.7):
                mismatches.append(
                    {
                        k: item[k]
                        for k in (
                            "year",
                            "site",
                            "relative_path",
                            "target_max_confidence",
                            "in_existing_high_list",
                            "detection_csv_status",
                        )
                    }
                )
            if (
                not item["header_ok"]
                or (
                    item["duration_s"] != ""
                    and abs(item["duration_s"] - 30) > 1 / item["sample_rate"]
                )
                or item["detection_csv_status"] != "ok"
            ):
                issues.append(item)
            after = path.stat()
            assert (
                before.st_size == after.st_size
                and before.st_mtime_ns == after.st_mtime_ns
            ), path
            metadata_index[(year, site, path.stem)] = {
                k: item[k]
                for k in (
                    "header_ok",
                    "duration_s",
                    "target_max_confidence",
                    "in_existing_high_list",
                )
            }
            writer.writerow(item)
            if i % 50000 == 0:
                print(
                    f"Read {i}/{len(logical)} metadata headers and detection tables; {time.monotonic()-started:.1f}s",
                    flush=True,
                )

    write_rows(HERE / "file_issues.csv", issues, fields)
    write_rows(
        HERE / "high_confidence_reconciliation.csv",
        mismatches,
        [
            "year",
            "site",
            "relative_path",
            "target_max_confidence",
            "in_existing_high_list",
            "detection_csv_status",
        ],
    )
    for key, group in groups.items():
        dates = sorted(date_sets[key])
        group.update(
            dates_with_files=len(dates),
            first_date=dates[0],
            last_date=dates[-1],
            header_duration_hours=group["duration_s"] / 3600,
        )
    write_rows(HERE / "year_site_summary.csv", list(groups.values()))
    write_rows(
        HERE / "monthly_counts.csv",
        [
            {"year": y, "site": s, "month": m, "logical_mp3": n}
            for (y, s, m), n in sorted(monthly.items())
        ],
    )
    write_rows(
        HERE / "duration_distribution.csv",
        [{"duration_s": d, "files": n} for d, n in sorted(durations.items())],
    )

    baseline = args.canonical_manifest
    annotations = read_rows(baseline)
    stems = Counter(r["source_stem"] for r in annotations)
    crosswalk = []
    for stem, n in sorted(stems.items()):
        match = SOURCE_RE.match(stem)
        key = (int(match["stem"][:4]), match["site"], match["stem"]) if match else None
        found = indexed.get(key)
        crosswalk.append(
            {
                "source_stem": stem,
                "syllables": n,
                "year": key[0] if key else "",
                "site": key[1] if key else "",
                "raw_stem": key[2] if key else "",
                "derived_suffix": match["suffix"] if match else "",
                "raw_relative_path": found["relative_path"] if found else "",
                "matching_basis": (
                    "site_and_timestamp_in_filename_only" if found else "unmatched"
                ),
                **metadata_index.get(key, {}),
            }
        )
    write_rows(HERE / "canonical_source_crosswalk.csv", crosswalk)
    source_keys = {
        (r["year"], r["site"], r["raw_stem"])
        for r in crosswalk
        if r["raw_relative_path"]
    }
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_roots": {str(y): str(p) for y, p in roots.items()},
        "inventory_sha256": sha256(inventory),
        "soundfile_version": sf.__version__,
        "libsndfile_version": sf.__libsndfile_version__,
        "physical_mp3": len(physical),
        "logical_mp3": len(logical),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_extra_files": len(physical) - len(logical),
        "duplicate_groups_with_content_conflicts": len(
            {
                (r["year"], r["site"], r["timestamp"])
                for r in duplicates
                if not r["same_bytes_within_group"]
            }
        ),
        "formats": [
            {"format": a, "subtype": b, "sample_rate": c, "channels": d, "files": n}
            for (a, b, c, d), n in formats.items()
        ],
        "species_detection_rows": dict(species),
        "orphan_detection_csv": len(orphans),
        "high_confidence_lists": high_sources,
        "high_confidence_differences": len(mismatches),
        "canonical": {
            "manifest": str(baseline),
            "sha256": sha256(baseline),
            "syllables": len(annotations),
            "source_stems": len(stems),
            "families": len({r["final_group_code"] for r in annotations}),
            "leaf_buckets": len({r["final_bucket_code"] for r in annotations}),
            "structure_counts": dict(
                Counter(r["syllable_structure"] for r in annotations)
            ),
            "matched_source_stems": sum(
                bool(r["raw_relative_path"]) for r in crosswalk
            ),
            "raw_keys_by_filename": len(source_keys),
            "boundary_mapping_verified": False,
        },
        "year_site": list(groups.values()),
        "limitations": [
            "目录站点由用户确认；逐日设备映射尚缺",
            "按站点与时间戳建立逻辑键；仅对同键副本做全文件SHA-256，不是全库内容去重",
            "时长来自libsndfile帧数元数据，不代表完成全量解码/人工审听或设备运行时长",
            "检测CSV行是模型窗口输出，不是独立录音、个体或鸣唱事件",
            "Canonical原文件关联只核对名称与存在性，切段起点及逐样本内容血统另行验证",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    (HERE / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    output_hashes = [
        {
            "file": p.relative_to(HERE).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
        }
        for p in sorted(HERE.rglob("*"))
        if p.is_file()
        and p.suffix in {".csv", ".gz", ".json", ".py"}
        and p.name != "audit_output_manifest.json"
    ]
    (HERE / "audit_output_manifest.json").write_text(
        json.dumps(output_hashes, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "logical": len(logical),
                "seconds": summary["elapsed_seconds"],
                "canonical": summary["canonical"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
