"""Assemble prepared clips, labelled Raven tables and sample-exact syllable metadata."""

from __future__ import annotations

from ..schema import public_rows
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys

DATA_VERSION = "1.0.0"
META = "annotations/metadata"
REVISION = "annotations/reference"
HISTORY = "processing/parents.csv"
RAVEN_FIELDS = [
    "Selection",
    "View",
    "Channel",
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "syllable_id",
    "family_id",
    "leaf_id",
    "structure",
    "original_selection_id",
    "source_raven_row",
    "original_view",
]


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_rows(path, delimiter=","):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def safe_path(base, relative):
    """拒绝越界路径；已登记外置数据根内的路径由项目配置解析。"""
    value = Path(relative)
    require(
        not value.is_absolute()
        and value.parts
        and all(p not in (".", "..") for p in value.parts),
        f"非法相对路径：{relative}",
    )
    unresolved = Path(base) / value
    for index in range(1, len(value.parts) + 1):
        require(
            not (Path(base).joinpath(*value.parts[:index])).is_symlink(),
            f"拒绝符号链接目标：{relative}",
        )
    target = unresolved.resolve()
    require(target.is_relative_to(Path(base).resolve()), f"路径越界：{relative}")
    return target


def csv_bytes(rows, fields=None, delimiter=","):
    require(bool(rows) or fields is not None, "空表须提供字段")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fields or list(rows[0]),
        delimiter=delimiter,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write_same_or_new(path, payload):
    """断点复用只接受完全一致的文件，不覆盖异值目标。"""
    path = Path(path)
    if path.exists():
        require(
            path.is_file() and not path.is_symlink() and path.read_bytes() == payload,
            f"拒绝覆盖不一致的交付文件：{path}",
        )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def raven_rows(syllables, original_rows):
    """按原数据行号连接；Selection不是行号，当前秒界与频率框原样保留。"""
    result = []
    for number, syllable in enumerate(
        sorted(syllables, key=lambda s: int(s["source_raven_row"])), 1
    ):
        row_number = int(syllable["source_raven_row"])
        require(1 <= row_number <= len(original_rows), "Raven原行号越界")
        original = original_rows[row_number - 1]
        comparisons = {
            "selection_id": "Selection",
            "annotation_channel": "Channel",
            "begin_s": "Begin Time (s)",
            "end_s": "End Time (s)",
            "low_freq_hz": "Low Freq (Hz)",
            "high_freq_hz": "High Freq (Hz)",
        }
        for field, column in comparisons.items():
            require(
                Decimal(syllable[field]) == Decimal(original[column]),
                f"输入与当前Raven字段不同：{syllable['syllable_id']} {field}",
            )
        require(
            original["View"] in ("Spectrogram 1", "Waveform 1"), "未知Raven原始视图"
        )
        require(
            Decimal(syllable["end_s"]) > Decimal(syllable["begin_s"]) >= 0,
            "终审Raven时间范围无效",
        )
        result.append(
            dict(
                zip(
                    RAVEN_FIELDS,
                    [
                        str(number),
                        "Spectrogram 1",
                        syllable["annotation_channel"],
                        syllable["begin_s"],
                        syllable["end_s"],
                        syllable["low_freq_hz"],
                        syllable["high_freq_hz"],
                        syllable["syllable_id"],
                        syllable["family_id"],
                        syllable["leaf_id"],
                        syllable["syllable_structure"],
                        original["Selection"],
                        str(row_number),
                        original["View"],
                    ],
                )
            )
        )
    require(len({r["syllable_id"] for r in result}) == len(result), "重复终审音节ID")
    return result


def clip_row(manual, prepared, link):
    """正式raw区间不从候选相关峰推定；负偏移的未覆盖前导单独说明。"""
    stem = manual["source_stem"]
    require(re.fullmatch(r"[A-Za-z0-9_-]+", stem), "非法clip_id")
    require(
        link["release_offset_s"] == ""
        and link["release_offset_status"] == "not_verified_remains_missing",
        "正式偏移证据发生变化，需先复核后更新发布规则",
    )
    candidate = link["proposed_offset_s"]
    negative = bool(candidate) and Decimal(candidate) < 0
    return dict(
        clip_id=stem,
        source_stem=stem,
        raw_recording_id=manual["raw_recording_id"],
        site_id=manual["site_id"],
        analysis_wav_path=f"audio/analysis_clips/{stem}.wav",
        raven_table_path=f"annotations/raven/{stem}.Table.1.selections.txt",
        audio_layer="prepared_analysis_parent",
        data_version=DATA_VERSION,
        local_start_s="0",
        local_end_s=prepared["duration_s"],
        local_start_frame="0",
        local_stop_frame=prepared["decoded_frames"],
        sample_rate_hz=prepared["sample_rate_hz"],
        channels=prepared["channels"],
        wav_subtype=prepared["wav_subtype"],
        size_bytes=prepared["size_bytes"],
        sha256=prepared["sha256"],
        canonical_syllable_count=manual["canonical_syllable_count"],
        processing_profile_id=prepared["processing_profile_id"],
        legacy_bandpass_match_registered=prepared["legacy_bandpass_match_registered"],
        raw_release_included=manual["raw_release_included"],
        raw_audio_path=link["raw_planned_package_path"],
        raw_sha256=link["raw_sha256"],
        raw_time_status=manual["time_status"],
        raw_start_s="",
        raw_end_s="",
        raw_interval_status="unverified_not_supplied",
        release_offset_s="",
        release_offset_status=link["release_offset_status"],
        source_link_evidence=link["link_evidence"],
        content_review_status=link["content_review_status"],
        candidate_raw_offset_s=candidate,
        candidate_offset_score=link["proposed_offset_score"],
        candidate_offset_method=link["proposed_offset_method"],
        candidate_coverage_status=(
            "partial_negative_offset_unverified" if negative else "unverified"
        ),
        candidate_unmatched_manual_head_frames=link["unmatched_manual_head_frames"],
        candidate_unmatched_manual_head_s=link["unmatched_manual_head_s"],
        raw_to_prepared_sample_equivalence="not_established",
    )


def table_specs(tables):
    descriptions = {
        "clip_id": "现有source_stem；138份prepared分析父源的稳定标识，不是新的原始录音ID。",
        "source_stem": "现有来源文件stem，等于clip_id，保留历史连接键。",
        "raw_recording_id": "关联原始MP3逻辑标识；138分析父源对应116个raw ID。",
        "site_id": "实际采集地归档标识，不由设备检测坐标推定。",
        "analysis_wav_path": "包内prepared分析父源WAV路径；不指原manual输入WAV。",
        "analysis_clip_path": "包内prepared分析父源WAV；按整数帧可重切当前音节。",
        "raven_table_path": "包内终审Raven表路径；绑定该prepared父源的局部时间坐标。",
        "audio_layer": "固定为prepared_analysis_parent，已处理的分析父源。",
        "data_version": "公开坐标数据版本。",
        "local_start_s": "分析父源局部起点为0；不是原始MP3中的起点。",
        "local_end_s": "分析父源解码长度；不是已验收的raw裁剪结束时间。",
        "local_start_frame": "分析父源局部0起点帧号。",
        "local_stop_frame": "分析父源总帧数，半开区间右端。",
        "sample_rate_hz": "文件原生采样率；本次交付不重采样。",
        "channels": "解码声道数；本批为单声道。",
        "wav_subtype": "现存prepared父源WAV编码子类型；本次直接复制。",
        "size_bytes": "实际复制文件的字节数。",
        "sha256": "整个prepared WAV文件的SHA-256，不是仅波形数据的哈希。",
        "canonical_syllable_count": "该分析父源的当前终审音节数，不是鸟个体数。",
        "processing_profile_id": "既有1.5–15 kHz Butterworth SOS双向带通配置；order设计参数5。",
        "legacy_bandpass_match_registered": "输入manual已有历史带通匹配记录；true表示prepared又经过一次带通。",
        "raw_release_included": "关联raw是否在本拟发布范围；不代表来源精确偏移已验证。",
        "raw_audio_path": "关联原始MP3的包内路径，不是该prepared父源的未处理复制件。",
        "raw_sha256": "关联原始MP3文件SHA-256。",
        "raw_time_status": "known_clock_error或not_independently_verified；未标错不等于准确。",
        "raw_start_s": "经过正式验收的原始MP3内片段起点；本版全部空，不能用候选值代填。",
        "raw_end_s": "经过正式验收的原始MP3内片段终点；本版全部空。",
        "raw_interval_status": "正式原始区间未核实，固定unverified_not_supplied。",
        "release_offset_s": "已有正式发布偏移字段，本版全部空；不从candidate_raw_offset_s升级。",
        "release_offset_status": "已有正式偏移验收状态，本版not_verified_remains_missing。",
        "source_link_evidence": "既有来源关联证据等级，不是新增人工对齐验收。",
        "content_review_status": "内容对应确认状态，与精确偏移验收分开。",
        "candidate_raw_offset_s": "未验收候选：raw_time=manual/prepared_local_time+offset；不能作正式裁剪指令。",
        "candidate_offset_score": "既有相关对齐分数，不是物种概率或人工验收。",
        "candidate_offset_method": "既有候选生成方法，未在打包时重新估计。",
        "candidate_coverage_status": "partial_negative_offset_unverified表示负偏移下人工前导无raw覆盖；其他仍unverified。",
        "candidate_unmatched_manual_head_frames": "候选负偏移未覆盖的人工前导帧数；其他为空。",
        "candidate_unmatched_manual_head_s": "候选负偏移未覆盖的人工前导秒数；不自动删除或补零。",
        "raw_to_prepared_sample_equivalence": "raw与prepared存在处理差异，逐样本等价未建立。",
        "syllable_id": "当前公开终审音节稳定ID，沿用现有身份，不依赖表内新Selection序号。",
        "raven_selection": "新发布Raven表内从1开始的唯一Selection编号。",
        "original_selection_id": "原始／修订Raven表中的Selection值，不等于原数据行号。",
        "source_raven_row": "原始输入Raven表不含表头的1起始行号，用于追溯连接。",
        "original_view": "该原始行的视图；新表统一显示Spectrogram 1而不增加重复音节。",
        "annotation_channel": "Raven的1起始声道编号。",
        "begin_s": "保留输入人工/准备父源局部秒起点；Raven显示用，不重新四舍五入生成重切帧。",
        "end_s": "保留输入人工/准备父源局部秒终点；精确重切使用source_frame_stop。",
        "low_freq_hz": "原Raven人工频率框下界，不是物种频带或滤波截止。",
        "high_freq_hz": "原Raven人工频率框上界，不是物种频带或滤波截止。",
        "source_frame_start": "当前公开登记的prepared父源0起始重切帧；区间左闭。",
        "source_frame_stop": "当前公开登记的prepared父源重切停止帧；区间右开。",
        "decoded_frames": "音节重切后的解码帧数，等于stop-start。",
        "duration_s": "音节解码帧数除以采样率；与显示秒界之差可能小于采样帧。",
        "family_id": "27个操作性类型族的稳定标签，不是自然类别已获独立验证。",
        "leaf_id": "41个末级类型标签，包含主体与已存变体。",
        "variant_code": "变体的完整叶标签；主体为空，不用0替代。",
        "structure": "single/double/triple主体元素数类别，轻短变体尾部不额外计数。",
        "syllable_structure": "single/double/triple主体元素数类别，沿用当前操作判据。",
        "label_version": "公开标签版本。",
        "reference_syllable_wav_sha256": "内部现存参考切片WAV哈希；本包不含该独立音节文件，重新编码容器可能不同。",
        "decoded_pcm_f32le_sha256": "按帧顺序、声道交错的little-endian float32解码样本SHA-256；可用包内脚本验证。",
        "body_element_count": "主体声学元素计数1/2/3，不包括轻短变体尾音。",
        "leaf_count": "该family在当前标签字典中的末级类别数。",
        "syllable_count": "当前终审音节在该标签中的数量。",
        "manual_source_count": "含该标签的不同人工／prepared来源数，不是独立个体数。",
        "site_count": "含该标签的不同采集站点数。",
        "is_variant": "是否属于既有变体叶标签。",
    }
    integers = {
        "local_start_frame",
        "local_stop_frame",
        "sample_rate_hz",
        "channels",
        "size_bytes",
        "canonical_syllable_count",
        "candidate_unmatched_manual_head_frames",
        "raven_selection",
        "original_selection_id",
        "source_raven_row",
        "annotation_channel",
        "source_frame_start",
        "source_frame_stop",
        "decoded_frames",
        "body_element_count",
        "leaf_count",
        "syllable_count",
        "manual_source_count",
        "site_count",
    }
    numbers = {
        "local_start_s",
        "local_end_s",
        "raw_start_s",
        "raw_end_s",
        "release_offset_s",
        "candidate_raw_offset_s",
        "candidate_offset_score",
        "candidate_unmatched_manual_head_s",
        "begin_s",
        "end_s",
        "low_freq_hz",
        "high_freq_hz",
        "duration_s",
    }
    boolean = {"legacy_bandpass_match_registered", "raw_release_included", "is_variant"}
    primary = {
        "high_quality_clips.csv": "clip_id",
        "syllables.csv": "syllable_id",
        "label_families.csv": "family_id",
        "label_leaves.csv": "leaf_id",
    }
    result = []
    for name, rows in tables.items():
        fields = []
        for key in rows[0]:
            unit = (
                "Hz"
                if key.endswith("_hz")
                else (
                    "s"
                    if key.endswith("_s")
                    else "byte" if key in {"size_bytes"} else None
                )
            )
            if key in {
                "source_frame_start",
                "source_frame_stop",
                "decoded_frames",
                "local_start_frame",
                "local_stop_frame",
                "candidate_unmatched_manual_head_frames",
            }:
                unit = "sample frame"
            fields.append(
                dict(
                    name=key,
                    type=(
                        "integer"
                        if key in integers
                        else (
                            "number"
                            if key in numbers
                            else "boolean" if key in boolean else "string"
                        )
                    ),
                    unit=unit,
                    nullable=(
                        False
                        if key == "decoded_pcm_f32le_sha256"
                        else any(row[key] == "" for row in rows)
                    ),
                    description_zh=descriptions[key],
                )
            )
        foreign = []
        if name == "syllables.csv":
            foreign = [
                dict(
                    fields=[key],
                    reference_table="metadata/" + table,
                    reference_fields=[key],
                )
                for key, table in (
                    ("clip_id", "high_quality_clips.csv"),
                    ("family_id", "label_families.csv"),
                    ("leaf_id", "label_leaves.csv"),
                )
            ]
        elif name == "label_leaves.csv":
            foreign = [
                dict(
                    fields=["family_id"],
                    reference_table="metadata/label_families.csv",
                    reference_fields=["family_id"],
                )
            ]
        result.append(
            dict(
                file_name="metadata/" + name,
                row_count=len(rows),
                primary_key=[primary[name]],
                fields=fields,
                foreign_keys=foreign,
            )
        )
    return result


def plan_core(root: Path, *, data_root: Path, prepared_root: Path) -> dict:
    """只读计划：不解码、不复制音频；解析配置、校验小型输入、核对已有文件大小。"""
    root = Path(root).resolve()
    metadata, revision = root / META, root / REVISION
    manifest_files = {
        r["file_name"]: r
        for r in json.loads((metadata / "file_manifest.json").read_text())["files"]
    }
    names = [
        "manual_recordings.csv",
        "prepared_recordings.csv",
        "source_links.csv",
        "raven_tables.csv",
        "syllables.csv",
        "label_families.csv",
        "label_leaves.csv",
    ]
    tables = {}
    input_hashes = {}
    for name in names:
        path = metadata / "tables" / name
        input_hashes[f"{META}/tables/{name}"] = digest(path)
        require(
            input_hashes[f"{META}/tables/{name}"]
            == manifest_files[f"tables/{name}"]["sha256"],
            f"核心元数据哈希不符：{name}",
        )
        tables[name] = read_rows(path)
    syllables = tables["syllables.csv"]
    require(
        len(syllables) == len({s["syllable_id"] for s in syllables}) == 2556,
        "须为当前2556唯一音节",
    )
    groups = defaultdict(list)
    for s in syllables:
        require(bool(s["data_version"]), "核心边界版本未声明")
        groups[s["manual_recording_id"]].append(s)
    require(len(groups) == 138, "须为138分析父源")
    manual = {r["manual_recording_id"]: r for r in tables["manual_recordings.csv"]}
    prepared = {r["manual_recording_id"]: r for r in tables["prepared_recordings.csv"]}
    links = {r["manual_recording_id"]: r for r in tables["source_links.csv"]}
    ravens = {r["manual_recording_id"]: r for r in tables["raven_tables.csv"]}
    history = {r["source_stem"]: r for r in read_rows(root / HISTORY)}
    references = {r["stable_id"]: r for r in read_rows(revision / "manifest.csv")}
    require(
        set(references) == {s["syllable_id"] for s in syllables}, "参考音频清单ID不匹配"
    )
    input_hashes[HISTORY] = digest(root / HISTORY)
    input_hashes[f"{REVISION}/manifest.csv"] = digest(revision / "manifest.csv")
    copies, clips, output_syllables, reference_audio = [], [], [], {}
    output_ravens = {}
    for key in sorted(groups):
        m, p, link, table = manual[key], prepared[key], links[key], ravens[key]
        stem = m["source_stem"]
        require(
            int(m["canonical_syllable_count"]) == len(groups[key]),
            "每父源终审行数与元数据不符",
        )
        require(
            m["raw_release_included"] == "true", "当前核心父源关联raw须在拟发布范围"
        )
        clip = clip_row(m, p, link)
        source = safe_path(
            prepared_root, history[stem]["prepared_relative_to_tmp_root"]
        )
        require(
            source.is_file() and source.stat().st_size == int(p["size_bytes"]),
            f"分析父源缺失／大小不符：{stem}",
        )
        require(
            history[stem]["actual_prepared_sha256"] == p["sha256"],
            "准备父源证据哈希不一致",
        )
        copies.append(
            dict(
                source=str(source),
                package_path=clip["analysis_wav_path"],
                expected_sha256=p["sha256"],
                expected_bytes=int(p["size_bytes"]),
            )
        )
        clips.append(clip)
        original = (
            safe_path(data_root, "review_audio/manual/" + table["original_filename"])
            if table["table_version"] == "original_historical"
            else safe_path(revision, "annotations/raven/" + table["original_filename"])
        )
        require(digest(original) == table["sha256"], f"现行Raven表哈希不符：{stem}")
        original_rows = read_rows(original, delimiter="\t")
        require(
            len(original_rows) == int(table["raw_row_count"]), "原Raven历史行数改变"
        )
        annotated = raven_rows(groups[key], original_rows)
        output_ravens[clip["raven_table_path"]] = annotated
        new_selection = {r["syllable_id"]: r["Selection"] for r in annotated}
        for s in sorted(groups[key], key=lambda r: int(r["source_raven_row"])):
            ref = references[s["syllable_id"]]
            require(ref["output_audio_sha256"] == s["sha256"], "参考切片哈希不匹配")
            require(
                ref["source_stem"] == stem
                and ref["source_frame_start"] == s["source_frame_start"]
                and ref["source_frame_stop"] == s["source_frame_stop"],
                "输入父源／帧边界不匹配",
            )
            reference_audio[s["syllable_id"]] = dict(
                source=str(safe_path(revision, ref["output_relative_path"])),
                expected_sha256=s["sha256"],
            )
            output_syllables.append(
                dict(
                    syllable_id=s["syllable_id"],
                    clip_id=stem,
                    raw_recording_id=s["raw_recording_id"],
                    site_id=s["site_id"],
                    analysis_clip_path=clip["analysis_wav_path"],
                    raven_table_path=clip["raven_table_path"],
                    raven_selection=new_selection[s["syllable_id"]],
                    original_selection_id=s["selection_id"],
                    source_raven_row=s["source_raven_row"],
                    original_view=annotated[int(new_selection[s["syllable_id"]]) - 1][
                        "original_view"
                    ],
                    annotation_channel=s["annotation_channel"],
                    begin_s=s["begin_s"],
                    end_s=s["end_s"],
                    low_freq_hz=s["low_freq_hz"],
                    high_freq_hz=s["high_freq_hz"],
                    source_frame_start=s["source_frame_start"],
                    source_frame_stop=s["source_frame_stop"],
                    decoded_frames=s["decoded_frames"],
                    sample_rate_hz=s["sample_rate_hz"],
                    channels=s["channels"],
                    duration_s=s["duration_s"],
                    family_id=s["family_id"],
                    leaf_id=s["leaf_id"],
                    variant_code=s["variant_code"],
                    structure=s["syllable_structure"],
                    label_version=s["label_version"],
                    data_version=s["data_version"],
                    boundary_status=s["boundary_status"],
                    supersedes_syllable_id=s["supersedes_syllable_id"],
                    reference_syllable_wav_sha256=s["sha256"],
                    decoded_pcm_f32le_sha256="",
                )
            )
    require(
        sum(c["expected_bytes"] for c in copies) == 286940780, "138分析父源字节总量变化"
    )
    require(len({c["raw_recording_id"] for c in clips}) == 116, "核心raw关联数变化")
    require(
        len(tables["label_families.csv"]) == 27
        and len(tables["label_leaves.csv"]) == 41,
        "标签字典数量变化",
    )
    for key, table in (
        ("family_id", "label_families.csv"),
        ("leaf_id", "label_leaves.csv"),
    ):
        counts = Counter(s[key] for s in output_syllables)
        declared = {r[key]: int(r["syllable_count"]) for r in tables[table]}
        require(
            counts == declared and len(declared) == len(tables[table]),
            "标签字典计数与发布标注不同",
        )
    require(
        sum(
            c["candidate_coverage_status"] == "partial_negative_offset_unverified"
            for c in clips
        )
        == 3,
        "负偏移候选数量变化，须复核来源证据",
    )
    output_tables = {
        "high_quality_clips.csv": clips,
        "syllables.csv": output_syllables,
        "label_families.csv": tables["label_families.csv"],
        "label_leaves.csv": tables["label_leaves.csv"],
    }
    output_tables = {
        name: public_rows(name, rows) for name, rows in output_tables.items()
    }
    return dict(
        copies=copies,
        tables=output_tables,
        table_specs=table_specs(output_tables),
        raven_tables=output_ravens,
        reference_audio=reference_audio,
        input_sha256=input_hashes,
        summary=dict(
            analysis_clips=138,
            analysis_clip_bytes=286940780,
            raven_tables=138,
            syllables=2556,
            families=27,
            leaves=41,
            core_raw_files=116,
            raw_intervals_verified=0,
            negative_offset_candidates=3,
        ),
    )


def pcm_hash(samples):
    import numpy as np

    return hashlib.sha256(
        np.asarray(samples, dtype="<f4").tobytes(order="C")
    ).hexdigest()


def verify_reconstruction(parent, parent_rate, syllable, reference):
    """比较解码波形而非WAV容器字节，登记整数帧为精确重切依据。"""
    import numpy as np
    import soundfile as sf

    start, stop = int(syllable["source_frame_start"]), int(
        syllable["source_frame_stop"]
    )
    require(0 <= start < stop <= len(parent), "重切帧范围越界")
    require(parent_rate == int(syllable["sample_rate_hz"]), "父源采样率不匹配")
    require(parent.shape[1] == int(syllable["channels"]), "父源声道数不匹配")
    require(stop - start == int(syllable["decoded_frames"]), "重切帧数不匹配")
    require(
        Decimal(syllable["begin_s"]) >= 0
        and Decimal(syllable["end_s"]) <= Decimal(len(parent)) / Decimal(parent_rate),
        "Raven秒界越界",
    )
    require(
        0
        <= Decimal(syllable["low_freq_hz"])
        < Decimal(syllable["high_freq_hz"])
        <= Decimal(parent_rate) / 2,
        "Raven频率框越界",
    )
    ref = Path(reference["source"])
    require(digest(ref) == reference["expected_sha256"], "参考音节文件哈希改变")
    expected, rate = sf.read(ref, dtype="float32", always_2d=True)
    actual = parent[start:stop]
    require(
        rate == parent_rate and expected.shape == actual.shape,
        "重切与参考采样率／形状不一致",
    )
    require(
        np.isfinite(actual).all() and np.array_equal(actual, expected),
        "重切与参考逐样本不一致",
    )
    require(pcm_hash(actual) == pcm_hash(expected), "重切与参考解码PCM字节不一致")
    return pcm_hash(actual)


def raven_runtime_status():
    executables = [
        p
        for name in ("Raven", "raven", "RavenPro", "raven-pro")
        if (p := shutil.which(name))
    ]
    return dict(
        executables_on_path=executables,
        display_available=bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ),
        actual_gui_load_performed=False,
        license_verified=False,
        actual_load_status="pending_manual_raven_load_acceptance",
    )


def deliver_core(
    root: Path,
    package_dir: Path,
    *,
    data_root: Path,
    prepared_root: Path,
    copy_file=None,
) -> dict:
    """主生成器调用一次；支持相同目标复用，不覆盖不同内容。"""
    import soundfile as sf

    plan = plan_core(root, data_root=data_root, prepared_root=prepared_root)
    if copy_file is None:
        from .build_dataset_release_bundle import copy_verified

        copy_file = copy_verified
    package_dir = Path(package_dir).resolve()
    package_dir.mkdir(parents=True, exist_ok=True)
    for item in plan["copies"]:
        target = safe_path(package_dir, item["package_path"])
        copy_file(
            Path(item["source"]),
            target,
            item["expected_sha256"],
            item["expected_bytes"],
            resume=True,
        )
        require(
            target.is_file()
            and target.stat().st_nlink == 1
            and target.stat().st_size == item["expected_bytes"]
            and digest(target) == item["expected_sha256"],
            "分析父源须为字节一致的独立文件，不能是硬链接",
        )
    groups = defaultdict(list)
    for s in plan["tables"]["syllables.csv"]:
        groups[s["clip_id"]].append(s)
    checked = 0
    for clip in plan["tables"]["high_quality_clips.csv"]:
        parent, rate = sf.read(
            safe_path(package_dir, clip["analysis_wav_path"]),
            dtype="float32",
            always_2d=True,
        )
        require(
            len(parent) == int(clip["local_stop_frame"])
            and rate == int(clip["sample_rate_hz"])
            and parent.shape[1] == int(clip["channels"]),
            "复制父源解码格式或长度不符",
        )
        for syllable in groups[clip["clip_id"]]:
            syllable["decoded_pcm_f32le_sha256"] = verify_reconstruction(
                parent, rate, syllable, plan["reference_audio"][syllable["syllable_id"]]
            )
            checked += 1
    generated = [c["package_path"] for c in plan["copies"]]
    for relative, rows in plan["raven_tables"].items():
        write_same_or_new(
            safe_path(package_dir, relative), csv_bytes(rows, RAVEN_FIELDS, "\t")
        )
        generated.append(relative)
    for name, rows in plan["tables"].items():
        relative = "metadata/" + name
        write_same_or_new(safe_path(package_dir, relative), csv_bytes(rows))
        generated.append(relative)
    example = "examples/reconstruct_syllables.py"
    write_same_or_new(safe_path(package_dir, example), EXAMPLE_SCRIPT.encode("utf-8"))
    generated.append(example)
    validation = dict(
        passed=True,
        analysis_clip_copies_byte_identical=138,
        reconstructed_syllables_sample_exact=checked,
        raven_rows=checked,
        raven_annotations_complete=True,
        original_raven_rows_not_overwritten=True,
        raven_time_frequency_bounds_checked=True,
        source_raven_row_join_not_selection=True,
        original_waveform_view_rows_retained_once=sum(
            s["original_view"] == "Waveform 1" for s in plan["tables"]["syllables.csv"]
        ),
        formal_raw_intervals_populated=0,
        raw_offset_candidates_not_promoted=True,
        runtime=raven_runtime_status(),
        standalone_example_provided=True,
        no_independent_syllable_audio_published=True,
        no_manual_input_wav_published=True,
    )
    return dict(
        summary=plan["summary"],
        validation=validation,
        copies=plan["copies"],
        table_specs=table_specs(plan["tables"]),
        generated_files=generated,
        input_sha256=plan["input_sha256"],
    )


EXAMPLE_SCRIPT = '''#!/usr/bin/env python3
"""Reconstruct published syllables with the separately installed analysis tools."""
import sys
from sparrow_dataset.cli import main
if __name__ == "__main__":
    main(["reconstruct", *sys.argv[1:]])
'''
