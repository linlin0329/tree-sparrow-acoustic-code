"""Build the public acoustic data package from accepted QC and review inputs."""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from ..io import digest, read_rows
from ..schema import public_rows
from .song_release_index import load_sources, assemble_index

VERSION = "1.0.1"
DATA_VERSION = "1.0.0"
RAW_FIELDS = [
    "raw_recording_id",
    "source_recording_key",
    "archive_year",
    "site_id",
    "original_filename",
    "audio_path",
    "detection_csv_path",
    "sha256",
    "bytes",
    "duration_s",
    "sample_rate_hz",
    "channels",
    "target_max_confidence",
    "score_comparison",
    "score_threshold",
    "time_status",
    "filename_timestamp",
    "nominal_timezone",
    "calendar_date_status",
    "corrected_recorded_at",
    "strict_decode_pass",
    "detection_csv_sha256",
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def write_new(path, data):
    """断点恢复只复用逐字节一致文件；不覆盖已有不同内容。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(
            path.is_file() and not path.is_symlink() and path.read_bytes() == data,
            f"拒绝覆盖不同内容：{path}",
        )
        return
    with path.open("xb") as stream:
        stream.write(data)


def write_csv(path, records, fields=None):
    import io

    records = [
        {k: str(v).lower() if isinstance(v, bool) else v for k, v in row.items()}
        for row in records
    ]
    fields = fields or list(records[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    write_new(path, stream.getvalue().encode())


def safe_target(root, relative):
    path = Path(relative)
    require(
        not path.is_absolute() and ".." not in path.parts and "\\" not in relative,
        "不安全的包内路径",
    )
    target = root / path
    require(
        target.resolve().is_relative_to(root.resolve()),
        "包内路径逃逸或指向外部符号链接",
    )
    return target


def copy_verified(source, target, expected_sha256, expected_bytes, *, resume=False):
    source, target = Path(source), Path(target)
    expected_bytes = int(expected_bytes)
    before = source.stat()
    require(before.st_size == expected_bytes, f"源大小与证据不符：{source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        require(
            resume and not target.is_symlink() and target.is_file(),
            f"拒绝覆盖已有目标：{target}",
        )
        current = target.stat()
        require(
            current.st_nlink == 1
            and (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino),
            "目标不是独立字节副本",
        )
        require(
            current.st_size == expected_bytes and digest(target) == expected_sha256,
            f"断点目标哈希不符：{target}",
        )
        require(digest(source) == expected_sha256, f"恢复时源哈希不符：{source}")
        return dict(
            package_path=str(target),
            bytes=expected_bytes,
            sha256=expected_sha256,
            reused=True,
        )
    partial = target.with_name(target.name + ".copy-partial")
    if partial.exists():
        require(
            resume and partial.is_file() and not partial.is_symlink(),
            f"未授权的残留临时文件：{partial}",
        )
        partial.unlink()  # 仅本构建协议的未完成目标，绝不删除源文件。
    h = hashlib.sha256()
    try:
        with source.open("rb") as inp, partial.open("xb") as out:
            for block in iter(lambda: inp.read(1024 * 1024), b""):
                h.update(block)
                out.write(block)
        after = source.stat()
        require(
            (before.st_size, before.st_mtime_ns, before.st_ino)
            == (after.st_size, after.st_mtime_ns, after.st_ino),
            f"复制期间源文件变化：{source}",
        )
        require(
            h.hexdigest() == expected_sha256
            and partial.stat().st_size == expected_bytes,
            f"源载荷与已验收哈希不符：{source}",
        )
        require(digest(partial) == expected_sha256, f"目标独立复读哈希不符：{target}")
        require(not target.exists(), f"目标在复制期间被创建：{target}")
        partial.rename(target)
    except BaseException:
        if partial.exists() and partial.is_file() and not partial.is_symlink():
            partial.unlink()
        raise
    return dict(
        package_path=str(target),
        bytes=expected_bytes,
        sha256=expected_sha256,
        reused=False,
    )


def input_identity(path, root, data_root):
    path = Path(path)
    resolved = path.resolve()
    for base, anchor in [
        ("evidence_root", root.resolve()),
        ("data_root", data_root.resolve()),
        ("code_root", Path(__file__).resolve().parent),
    ]:
        if resolved.is_relative_to(anchor):
            return dict(
                base=base,
                path=resolved.relative_to(anchor).as_posix(),
                sha256=digest(path),
            )
    raise ValueError(f"未声明输入根：{path}")


def prepare(root, data_root, output, *, prepared_root):
    from .release_bundle_annotations import plan_core

    ds = root
    qc_dir = ds / "qc/final"
    qc = json.loads((qc_dir / "summary.json").read_text())
    require(
        json.loads((qc_dir / "validation.json").read_text())["passed"],
        "原始发布QC未通过",
    )
    for item in json.loads((qc_dir / "provenance.json").read_text())["outputs"]:
        require(digest(qc_dir / item["path"]) == item["sha256"], "发布QC产物哈希变化")
    require(
        qc["score_selection"]["comparison"] == "ge"
        and Decimal(qc["score_selection"]["threshold"]) == Decimal(".7"),
        "原始门槛不是明确>=0.7",
    )
    require(
        qc["publication_policy"]["exclude_known_clock_errors"] is False,
        "错误时钟应保留标记",
    )
    sources = load_sources(root, data_root)
    scope_path = ds / "review/release_scope.json"
    scope = json.loads(scope_path.read_text())
    jobs, raw, audit = [], [], {}
    ids, keys = set(), set()
    for r in read_rows(qc_dir / "final_logical_inventory.csv.gz"):
        key = f"{r['site']}_{Path(r['relative_path']).stem}"
        selected = str(r["technical_eligible"]).lower() == "true"
        if (
            key in sources["requested_keys"]
            or r["logical_id"] in sources["audit_raw_ids"]
        ):
            require(key not in audit, "鸣唱血统键在逻辑档案中重复")
            audit[key] = dict(
                raw_recording_id=r["logical_id"],
                target_max_confidence=r["target_max_confidence"],
                raw_release_included=selected,
                time_status=r["time_status"],
                scope_status=r["scope_status"],
            )
        if not selected:
            continue
        require(r["logical_id"] not in ids and key not in keys, "公开raw身份重复")
        ids.add(r["logical_id"])
        keys.add(key)
        require(
            Decimal(r["target_max_confidence"]) >= Decimal(".7")
            and r["hold_reasons"] == ""
            and r["corrected_recorded_at"] == "",
            "公开raw分数/QC/时间证据不符",
        )
        filename = Path(r["relative_path"]).name
        audio_path = f"audio/raw/{r['year']}/{r['site']}/{filename}"
        csv_path = f"detections/{r['year']}/{r['site']}/{Path(filename).stem}.wav.csv"
        item = dict(
            raw_recording_id=r["logical_id"],
            source_recording_key=key,
            archive_year=r["year"],
            site_id=r["site"],
            original_filename=filename,
            audio_path=audio_path,
            detection_csv_path=csv_path,
            sha256=r["sha256"],
            bytes=r["actual_size_bytes"],
            duration_s=r["duration_seconds"],
            sample_rate_hz=r["native_sample_rate"],
            channels=r["native_channels"],
            target_max_confidence=r["target_max_confidence"],
            score_comparison="ge",
            score_threshold="0.7",
            time_status=r["time_status"],
            filename_timestamp=r["filename_timestamp"],
            nominal_timezone=r["nominal_timezone"],
            calendar_date_status=r["calendar_date_status"],
            corrected_recorded_at="",
            strict_decode_pass="true",
            detection_csv_sha256=r["csv_sha256"],
        )
        raw.append(item)
        jobs.append(
            dict(
                source=str(data_root / r["relative_path"]),
                package_path=audio_path,
                expected_sha256=r["sha256"],
                expected_bytes=int(r["actual_size_bytes"]),
                kind="raw_audio",
            )
        )
        csv_source = data_root / r["csv_relative_path"]
        jobs.append(
            dict(
                source=str(csv_source),
                package_path=csv_path,
                expected_sha256=r["csv_sha256"],
                expected_bytes=csv_source.stat().st_size,
                kind="original_detection_csv",
            )
        )
    raw.sort(key=lambda r: (r["archive_year"], r["site_id"], r["original_filename"]))
    published = {r["source_recording_key"]: r for r in raw}
    song, song_evidence = assemble_index(
        sources,
        published,
        audit,
        review_policy=scope.get("later_fragment_conflict_policy"),
    )
    core_plan = plan_core(root, data_root=data_root, prepared_root=prepared_root)
    total_raw_bytes = sum(int(r["bytes"]) for r in raw)
    require(
        len(raw) == 64811
        and total_raw_bytes == 77900553615
        and len(song) == scope.get("expected_current_song_recordings"),
        "获批发布规模或最新复核口径发生变化",
    )
    for job in jobs:
        require(
            Path(job["source"]).is_file()
            and Path(job["source"]).stat().st_size == job["expected_bytes"],
            "拟复制源文件不存在或大小变化",
        )
        safe_target(output / "package", job["package_path"])
    core_copies = core_plan["copies"]
    core_bytes = sum(int(j["expected_bytes"]) for j in core_copies)
    require(
        len(core_copies) == 138 and core_bytes == 286940780,
        "本包仅公开138 prepared父源WAV",
    )
    inputs = set(sources["paths"].values()) | {
        qc_dir / "summary.json",
        qc_dir / "validation.json",
        qc_dir / "provenance.json",
        qc_dir / "final_logical_inventory.csv.gz",
    }
    inputs.add(ds / "review/release_scope.json")
    inputs.update(Path(p) for p in core_plan.get("input_paths", []))
    inputs.update(root / p for p in core_plan.get("input_sha256", {}))
    inputs.update(
        Path(__file__).with_name(name)
        for name in (
            "build_dataset_release_bundle.py",
            "song_release_index.py",
            "release_bundle_annotations.py",
        )
    )
    identities = [input_identity(p, root, data_root) for p in sorted(inputs)]
    contract = dict(
        version=VERSION,
        core_version=DATA_VERSION,
        inputs=identities,
        raw_files=len(raw),
        raw_bytes=total_raw_bytes,
        prepared_files=len(core_copies),
        prepared_bytes=core_bytes,
        copy_jobs_digest=hashlib.sha256(
            json.dumps(jobs, sort_keys=True).encode()
        ).hexdigest(),
    )
    parent = output
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    required = sum(j["expected_bytes"] for j in jobs) + core_bytes
    require(free > required + 2 * 1024**3, "剩余磁盘不足以保存真实副本及余量")
    plan = dict(
        version=VERSION,
        state=(
            "unfinished_existing_package_plan"
            if (output / "build_contract.json").exists()
            else "planned_not_copied"
        ),
        raw_audio_files=len(raw),
        raw_audio_bytes=total_raw_bytes,
        original_detection_csv_files=len(raw),
        original_detection_csv_bytes=sum(
            j["expected_bytes"] for j in jobs if j["kind"] == "original_detection_csv"
        ),
        song_recordings=len(song),
        prepared_audio_files=len(core_copies),
        prepared_audio_bytes=core_bytes,
        total_audio_files=len(raw) + len(core_copies),
        total_audio_bytes=total_raw_bytes + core_bytes,
        known_clock_raw_files=sum(r["time_status"] == "known_clock_error" for r in raw),
        bytes_to_copy=required,
        free_bytes=free,
        workers=4,
        source_audio_decoding=False,
        source_audio_modified=False,
        output_package=str(output / "package"),
        source_qc_frame_candidates=qc["source_qc_candidate_logical_files"],
        current_score_candidates=qc["selected_candidates"],
        contract_sha256=hashlib.sha256(json_bytes(contract)).hexdigest(),
    )
    plan["song_routes"] = song_evidence
    plan["existing_raw_and_detection_copy_paths"] = sum(
        (output / "package" / j["package_path"]).is_file() for j in jobs
    )
    plan["existing_copy_sha256_verification_required_at_completion"] = True
    return dict(
        plan=plan,
        contract=contract,
        jobs=jobs,
        raw=raw,
        song=song,
        song_evidence=song_evidence,
        core_plan=core_plan,
    )


def verify_package(package, *, expected_manifest=None):
    manifest_path = package / "manifest.csv"
    if expected_manifest:
        require(digest(manifest_path) == expected_manifest, "包清单自身SHA不符")
    entries = list(read_rows(manifest_path))
    listed = set()
    for r in entries:
        p = safe_target(package, r["path"])
        require(
            not p.is_symlink() and p.is_file() and p.stat().st_nlink == 1,
            "包内文件非独立真实文件",
        )
        require(
            p.stat().st_size == int(r["bytes"]) and digest(p) == r["sha256"],
            f'包内校验失败：{r["path"]}',
        )
        require(r["path"] not in listed, "manifest重复路径")
        listed.add(r["path"])
    actual = {
        p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()
    }
    require(
        actual == listed | {"manifest.csv", "manifest.sha256"}, "包内有未列出或缺失文件"
    )
    expected = {r["path"]: r["sha256"] for r in entries}
    expected["manifest.csv"] = digest(manifest_path)
    checksum = {
        line[66:]: line[:64]
        for line in (package / "manifest.sha256").read_text().splitlines()
    }
    require(checksum == expected, "SHA256清单与文件清单不一致")
    return dict(
        passed=True,
        files_checked=len(entries),
        manifest_sha256=digest(manifest_path),
        checksum_file_sha256=digest(package / "manifest.sha256"),
        unlisted_files=0,
        bytes_checked=sum(int(r["bytes"]) for r in entries),
        actual_file_count=len(actual),
        symlinks_or_source_hardlinks=False,
    )


def create_manifest(package, expected_hashes=None):
    entries = []
    for path in sorted(package.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(package).as_posix()
        if name in ("manifest.csv", "manifest.sha256"):
            continue
        require(
            not path.is_symlink() and path.stat().st_nlink == 1,
            "包内符号链接或硬链接不允许",
        )
        require(not name.endswith(".copy-partial"), "包内残留未完成复制")
        actual = digest(path)
        if expected_hashes is not None and name in expected_hashes:
            require(
                actual == expected_hashes[name], f"已复制文件不再匹配源证据：{name}"
            )
        entries.append(dict(path=name, bytes=path.stat().st_size, sha256=actual))
    if expected_hashes is not None:
        require(
            set(expected_hashes) <= {e["path"] for e in entries}, "既有复制文件有缺失"
        )
    write_csv(package / "manifest.csv", entries)
    checksum = "".join(f"{r['sha256']}  {r['path']}\n" for r in entries)
    checksum += f"{digest(package/'manifest.csv')}  manifest.csv\n"
    write_new(package / "manifest.sha256", checksum.encode())
    return entries


def validate_public_tables(package, specs):
    tables = {
        s["file_name"]: list(read_rows(safe_target(package, s["file_name"])))
        for s in specs
    }
    keys = {}
    for spec in specs:
        rows = tables[spec["file_name"]]
        names = [f["name"] for f in spec["fields"]]
        require(len(rows) == spec["row_count"], "公开表行数与schema不符")
        seen = set()
        for row in rows:
            require(list(row) == names and None not in row, "公开表字段与schema不符")
            key = tuple(row[k] for k in spec["primary_key"])
            require(key not in seen, "公开表主键重复")
            seen.add(key)
            for field in spec["fields"]:
                value = row[field["name"]]
                if value == "":
                    require(field["nullable"], "不可空公开字段缺失")
                    continue
                kind = field["type"]
                if kind in ("integer", "number"):
                    number = Decimal(value)
                    require(number.is_finite(), "非有限元数据数值")
                    if kind == "integer":
                        require(number == number.to_integral_value(), "整数字段非整数")
                if kind == "boolean":
                    require(value in ("true", "false"), "布尔字段非true/false")
                if field["name"].endswith("_path"):
                    require(
                        safe_target(package, value).is_file(),
                        f"公开表引用文件缺失：{value}",
                    )
        keys[spec["file_name"]] = seen
    for spec in specs:
        for fk in spec.get("foreign_keys", []):
            parent = {
                tuple(r[k] for k in fk["reference_fields"])
                for r in tables[fk["reference_table"]]
            }
            require(
                all(
                    tuple(r[k] for k in fk["fields"]) in parent
                    for r in tables[spec["file_name"]]
                ),
                "公开表外键不闭合",
            )
    return dict(
        passed=True,
        table_count=len(specs),
        table_rows={s["file_name"]: s["row_count"] for s in specs},
        fields=sum(len(s["fields"]) for s in specs),
        foreign_keys_checked=True,
        referenced_files_exist=True,
    )


def build(args):
    root = args.evidence_root.resolve()
    data_root = args.data_root.resolve()
    output = args.output_root.resolve()
    package = output / "package"
    if args.verify_only:
        print(json.dumps(verify_package(package), ensure_ascii=False))
        return
    if (output / "validation.json").exists() and json.loads(
        (output / "validation.json").read_text()
    ).get("passed"):
        raise FileExistsError("已完成包禁止覆盖；使用--verify-only或新的输出目录")
    work = prepare(root, data_root, output, prepared_root=args.prepared_root.resolve())
    output.mkdir(parents=True, exist_ok=True)
    if args.plan_only:
        write_new(output / "plan.json", json_bytes(work["plan"]))
        print(json.dumps(work["plan"], ensure_ascii=False))
        return
    state = output / "build_contract.json"
    if state.exists():
        require(args.resume, "已有契约须使用--resume")
        require(
            state.read_bytes() == json_bytes(work["contract"]),
            "Resume requires the same inputs, code and output contract",
        )
    else:
        require(
            not package.exists() or not any(package.iterdir()), "非空包缺少构建契约"
        )
        write_new(state, json_bytes(work["contract"]))
    package.mkdir(parents=True, exist_ok=True)
    jobs = work["jobs"]
    if args.finalize_existing_copies:
        require(args.resume, "仅完成元数据须明确--resume")
        for j in jobs:
            p = safe_target(package, j["package_path"])
            require(
                p.is_file()
                and not p.is_symlink()
                and p.stat().st_nlink == 1
                and p.stat().st_size == j["expected_bytes"],
                "已有复制文件缺失/大小变化/并非真实副本",
            )
        print(
            json.dumps(
                dict(
                    existing_copy_files=len(jobs),
                    independent_sha256_verification="required_during_manifest_creation",
                )
            ),
            flush=True,
        )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    copy_verified,
                    j["source"],
                    safe_target(package, j["package_path"]),
                    j["expected_sha256"],
                    j["expected_bytes"],
                    resume=args.resume,
                )
                for j in jobs
            ]
            for i, future in enumerate(as_completed(futures), 1):
                future.result()
                if i % 2000 == 0 or i == len(jobs):
                    print(
                        json.dumps(dict(copied_or_verified=i, total=len(jobs))),
                        flush=True,
                    )
    from .release_bundle_annotations import deliver_core

    core = deliver_core(
        root, package, data_root=data_root, prepared_root=args.prepared_root.resolve()
    )
    require(core.get("validation", {}).get("passed", False), "核心交付验收失败")
    write_csv(package / "metadata/raw_recordings.csv", work["raw"], RAW_FIELDS)
    write_csv(
        package / "metadata/song_recordings.csv",
        public_rows("song_recordings.csv", work["song"]),
    )
    clock = [
        {
            k: r[k]
            for k in (
                "raw_recording_id",
                "source_recording_key",
                "audio_path",
                "time_status",
                "filename_timestamp",
                "calendar_date_status",
                "corrected_recorded_at",
            )
        }
        for r in work["raw"]
        if r["time_status"] == "known_clock_error"
    ]
    write_csv(package / "metadata/known_clock_errors.csv", clock)
    write_new(package / "provenance/input_identity.json", json_bytes(work["contract"]))
    metadata_validation = write_public_docs(package, work, core)
    expected_hashes = {
        j["package_path"]: j["expected_sha256"]
        for j in jobs + work["core_plan"]["copies"]
    }
    entries = create_manifest(package, expected_hashes)
    verification = verify_package(package)
    summary = dict(
        version=VERSION,
        core_version=DATA_VERSION,
        state="complete_local_not_uploaded",
        raw_audio_files=len(work["raw"]),
        raw_audio_bytes=sum(int(r["bytes"]) for r in work["raw"]),
        raw_duration_seconds=sum(Decimal(r["duration_s"]) for r in work["raw"]),
        prepared_audio_files=138,
        prepared_audio_bytes=286940780,
        total_audio_files=64949,
        total_audio_bytes=78187494395,
        song_recordings=len(work["song"]),
        high_quality_clips=138,
        canonical_syllables=2556,
        final_raven_tables=138,
        known_clock_raw_files=len(clock),
        original_detection_csv_files=len(work["raw"]),
        public_metadata_tables=[
            p.name for p in sorted((package / "metadata").glob("*.csv"))
        ],
        package_files=verification["actual_file_count"],
        package_bytes=sum(p.stat().st_size for p in package.rglob("*") if p.is_file()),
        source_audio_reprocessed=False,
        public_upload_performed=False,
        core=core.get("summary", {}),
        song_routes=work["song_evidence"],
        manifest_sha256=verification["manifest_sha256"],
    )
    summary["raw_duration_seconds"] = str(summary["raw_duration_seconds"])
    validation = dict(
        passed=True,
        version=VERSION,
        copy_source_and_destination_sha256_checked=True,
        portable_real_files_not_links=True,
        manifest=verification,
        core=core["validation"],
        metadata=metadata_validation,
        raw_threshold="Decimal >= 0.7",
        parent_versions_modified=False,
        current_registry_check_not_part_of_parent_snapshot=True,
    )
    write_new(output / "summary.json", json_bytes(summary))
    write_new(output / "validation.json", json_bytes(validation))
    write_new(output / "core_delivery.json", json_bytes(core))
    print(json.dumps(summary, ensure_ascii=False))


def write_public_docs(package, work, core):
    # 小型读取/重切示例由核心模块提供；此处补全raw和鸣唱索引的公开口径。
    text = f"""# Eurasian tree sparrow acoustic dataset

Version: {VERSION}. Data licence: CC BY 4.0. Attribution: Researcher(s).

This release contains 64,811 MP3 recordings, {len(work['song'])} confirmed-song
recordings, 138 prepared WAV clips and 138 labelled Raven tables containing
2,556 syllables (27 families and 41 terminal categories).

The raw-recording gate is maximum Passer montanus score >=0.7 plus technical
integrity checks. Scores are model scores, not calibrated probabilities. Known
clock errors remain flagged. Original MP3 offsets are unverified; use the
prepared WAVs and integer-frame metadata for exact syllable extraction.

Install the companion ../code/analysis component and run sparrow-data inspect
with --package pointing to this directory. The seven metadata tables and
original detection CSVs retain the source identities and measurement values.
Validate distributed file bytes with sha256sum -c manifest.sha256.
"""
    write_new(package / "README.md", text.encode())
    descriptions = {
        "raw_recording_id": "Stable archive logical recording key; not a bird identity.",
        "source_recording_key": "Site plus original filename stem used to join historical lists.",
        "archive_year": "Archive directory year, not an independently corrected recording year.",
        "site_id": "Study site identifier.",
        "original_filename": "Unchanged original MP3 filename.",
        "audio_path": "Package-relative audio path.",
        "detection_csv_path": "Complete original BirdNET CSV path.",
        "sha256": "SHA-256 of original MP3 bytes and copied file.",
        "bytes": "File size in bytes.",
        "duration_s": "Previously fully decoded duration in seconds.",
        "sample_rate_hz": "Native decoded sample rate in Hz.",
        "channels": "Number of native channels.",
        "target_max_confidence": "Maximum valid Passer montanus model score; not calibrated probability.",
        "score_comparison": "ge means >=; threshold equality included.",
        "score_threshold": "Exact decimal threshold 0.7.",
        "time_status": "known_clock_error or not_independently_verified.",
        "filename_timestamp": "Nominal local filename timestamp, not corrected time.",
        "nominal_timezone": "Asia/Shanghai as reported; does not validate device clock.",
        "calendar_date_status": "Filename calendar-date evidence status.",
        "corrected_recorded_at": "Blank; no recovered absolute timestamp.",
        "strict_decode_pass": "Existing full decoding/integrity test passed, not human species validation.",
        "detection_csv_sha256": "Original companion CSV SHA-256.",
        "song_recording_id": "Stable song-index identifier linked to raw_recording_id.",
        "provenance_route": "early_confirmed_seed / songiness_deployment_review / legacy_rule_candidate_core_supplement.",
        "route_detail": "Recorded source-route qualifier; pipeline origin does not imply no human review.",
        "route_rank": "Rank in the relevant historical candidate list; blank for seed.",
        "confirmation_evidence": "Most recent applicable evidence; author_reported_repeat_listening is reported by the dataset author, not independent blinded validation.",
        "historical_confirmation_evidence": "Original discovery-stage confirmation retained even when a later scoped author review adds evidence.",
        "high_quality_clip_count": "Number of current core clips linked to this raw recording.",
        "canonical_syllable_count": "Number of released syllables linked to this raw recording.",
        "latest_fragment_review_group": "Most recent reviewed local-fragment category, if available; does not classify all raw intervals.",
        "latest_review_disposition": "Current handling of local-fragment evidence, including separate positive core evidence or historical full-recording evidence.",
        "author_correction_id": "Links to provenance/researcher_review.csv; blank outside the 11 author-reheard items.",
        "author_reviewed_unit": "same_seed_manual_wav or original_30s_raw_mp3; a raw-only review does not relabel the cropped fragment.",
    }
    specs = []
    lines = [
        "# Package data dictionary",
        "",
        "All seven public tables use UTF-8 CSV. Blank means missing; booleans use true/false. Frame coordinates are zero-based half-open intervals.",
        "",
    ]
    integers = {
        "archive_year",
        "bytes",
        "sample_rate_hz",
        "channels",
        "route_rank",
        "high_quality_clip_count",
        "canonical_syllable_count",
    }
    numbers = {"duration_s", "target_max_confidence", "score_threshold"}
    for filename in [
        "raw_recordings.csv",
        "song_recordings.csv",
        "known_clock_errors.csv",
    ]:
        records = list(read_rows(package / "metadata" / filename))
        fields = list(records[0])
        spec = dict(
            file_name="metadata/" + filename,
            row_count=len(records),
            primary_key=[
                (
                    "song_recording_id"
                    if filename == "song_recordings.csv"
                    else "raw_recording_id"
                )
            ],
            fields=[
                dict(
                    name=f,
                    type=(
                        "integer"
                        if f in integers
                        else (
                            "number"
                            if f in numbers
                            else "boolean" if f == "strict_decode_pass" else "string"
                        )
                    ),
                    unit=(
                        "Hz"
                        if f == "sample_rate_hz"
                        else (
                            "s"
                            if f == "duration_s"
                            else "byte" if f == "bytes" else None
                        )
                    ),
                    nullable=any(r[f] == "" for r in records),
                    description=descriptions[f],
                )
                for f in fields
            ],
            foreign_keys=[],
        )
        if filename != "raw_recordings.csv":
            spec["foreign_keys"] = [
                dict(
                    fields=["raw_recording_id"],
                    reference_table="metadata/raw_recordings.csv",
                    reference_fields=["raw_recording_id"],
                )
            ]
        specs.append(spec)
        lines += [f"## {filename}", "", "| Field | Meaning |", "| --- | --- |"]
        lines += [f"| {f} | {descriptions[f]} |" for f in fields]
        lines.append("")
    core_specs = core["table_specs"]
    for spec in core_specs:
        if any(f["name"] == "raw_recording_id" for f in spec["fields"]):
            spec["foreign_keys"].append(
                dict(
                    fields=["raw_recording_id"],
                    reference_table="metadata/raw_recordings.csv",
                    reference_fields=["raw_recording_id"],
                )
            )
        lines += [
            f"## {spec['file_name']}",
            "",
            "| Field | Type / unit | Nullable | Meaning |",
            "| --- | --- | --- | --- |",
        ]
        for field in spec["fields"]:
            lines.append(
                f"| {field['name']} | {field['type']} / {field.get('unit') or '—'} | {field['nullable']} | {field['description_zh']} |"
            )
        lines.append("")
    specs += core_specs
    write_new(package / "DATA_DICTIONARY.md", "\n".join(lines).encode())
    write_new(
        package / "metadata/schema.json",
        json_bytes(dict(version=VERSION, tables=specs)),
    )
    write_new(
        package / "LOCAL_RELEASE_STATUS.json",
        json_bytes(
            dict(
                version=VERSION,
                status="complete",
                repository=None,
                doi=None,
                data_licence="CC-BY-4.0",
                example_code_licence="MIT",
                licence_status="declared",
                upload_performed=False,
            )
        ),
    )
    return validate_public_tables(package, specs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--evidence-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--prepared-root", type=Path, required=True)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument(
        "--finalize-existing-copies",
        action="store_true",
        help="不再次复制已完成的raw/CSV，最终manifest仍逐文件独立复读并对源哈希核验",
    )
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    require(1 <= a.workers <= 16, "workers须在1—16")
    build(a)


if __name__ == "__main__":
    main()
