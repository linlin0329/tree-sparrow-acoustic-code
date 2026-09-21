#!/usr/bin/env python3
"""审计最终鸣唱语料、统一带通并规范化 Raven 音节表。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

TABLE_SUFFIX = ".Table.1.selections.txt"
REQUIRED_COLUMNS = [
    "Selection",
    "View",
    "Channel",
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
]
BOX_COLUMNS = [
    "Selection",
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_syllable_id(stem: str, row: pd.Series) -> str:
    payload = (
        f"{stem}|{int(row['Selection'])}|"
        f"{float(row['Begin Time (s)']):.9f}|{float(row['End Time (s)']):.9f}|"
        f"{float(row['Low Freq (Hz)']):.3f}|{float(row['High Freq (Hz)']):.3f}"
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def normalize_raven_table(
    table_path: Path,
    audio_path: Path,
    output_path: Path,
    exclusions_path: Path,
    min_duration: float,
    max_duration: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """去除重复视图和无效时间框，不合并相邻音节。"""
    frame = pd.read_csv(table_path, sep="\t", encoding="utf-8-sig")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{table_path.name} 缺少列：{missing}")

    frame = frame[REQUIRED_COLUMNS].copy()
    frame.insert(0, "source_row", np.arange(1, len(frame) + 1))
    for column in [
        "Selection",
        "Channel",
        "Begin Time (s)",
        "End Time (s)",
        "Low Freq (Hz)",
        "High Freq (Hz)",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    reasons = pd.Series("", index=frame.index, dtype=object)
    invalid_numeric = frame[BOX_COLUMNS + ["Channel"]].isna().any(axis=1)
    reasons.loc[invalid_numeric] = "invalid_numeric"

    duplicate = frame.duplicated(BOX_COLUMNS, keep="first") & ~invalid_numeric
    reasons.loc[duplicate] = "duplicate_view"

    info = sf.info(audio_path)
    begin = frame["Begin Time (s)"]
    end = frame["End Time (s)"]
    duration = end - begin
    available = reasons.eq("")

    nonpositive = available & ((begin < 0) | (end <= begin))
    reasons.loc[nonpositive] = "nonpositive_duration"
    available = reasons.eq("")
    out_of_bounds = available & (end > info.duration + 1e-4)
    reasons.loc[out_of_bounds] = "out_of_bounds"
    available = reasons.eq("")
    duration_outside = available & (
        (duration < min_duration) | (duration > max_duration)
    )
    reasons.loc[duration_outside] = "duration_outside_limits"

    excluded = frame.loc[~reasons.eq("")].copy()
    excluded["exclusion_reason"] = reasons.loc[excluded.index]

    valid = frame.loc[reasons.eq("")].copy()
    valid["stable_id"] = [
        stable_syllable_id(audio_path.stem, row) for _, row in valid.iterrows()
    ]
    valid["source_stem"] = audio_path.stem
    valid["source_audio"] = str(audio_path.resolve())
    valid["source_table"] = str(table_path.resolve())
    valid["audio_duration_s"] = info.duration
    valid["syllable_duration_s"] = valid["End Time (s)"] - valid["Begin Time (s)"]
    valid["frequency_box_valid"] = (
        (valid["Low Freq (Hz)"] >= 0)
        & (valid["High Freq (Hz)"] > valid["Low Freq (Hz)"])
        & (valid["High Freq (Hz)"] <= info.samplerate / 2 + 1)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exclusions_path.parent.mkdir(parents=True, exist_ok=True)
    valid.to_csv(output_path, sep="\t", index=False)
    excluded.to_csv(exclusions_path, index=False)
    return valid, excluded


def bandpass_one(
    input_path: Path,
    output_path: Path,
    lowcut: float,
    highcut: float,
    order: int,
) -> dict:
    try:
        audio, sample_rate = sf.read(input_path, dtype="float64", always_2d=False)
        if audio.ndim != 1:
            raise ValueError(f"仅支持单声道，实际 shape={audio.shape}")
        if highcut >= sample_rate / 2:
            raise ValueError(f"highcut={highcut} 必须小于 Nyquist={sample_rate / 2}")
        sos = butter(
            order,
            [lowcut, highcut],
            btype="bandpass",
            fs=sample_rate,
            output="sos",
        )
        filtered = sosfiltfilt(sos, audio)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, filtered.astype(np.float32), sample_rate, subtype="FLOAT")
        return {
            "success": True,
            "output_sha256": sha256_file(output_path),
            "error": "",
        }
    except Exception as error:  # pragma: no cover - integration failure path
        return {"success": False, "output_sha256": "", "error": str(error)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lowcut", type=float, default=1500.0)
    parser.add_argument("--highcut", type=float, default=15000.0)
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--min-duration", type=float, default=0.03)
    parser.add_argument("--max-duration", type=float, default=1.0)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="仅审计并规范化表，不生成带通音频",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    if input_dir == output_dir or input_dir in output_dir.parents:
        # 输出允许位于数据根，但不能嵌在源语料目录中。
        if output_dir.parent == input_dir or input_dir == output_dir:
            raise ValueError("输出目录不得位于输入语料目录内")

    audio_files = sorted(input_dir.glob("*.wav"))
    table_files = sorted(input_dir.glob(f"*{TABLE_SUFFIX}"))
    table_by_stem = {path.name[: -len(TABLE_SUFFIX)]: path for path in table_files}
    old_bandpass_dir = input_dir.parent / "wav_manul_seg_final_bandpass"

    output_dir.mkdir(parents=True, exist_ok=True)
    bandpass_dir = output_dir / "bandpass_audio"
    normalized_dir = output_dir / "normalized_raven"
    exclusion_dir = output_dir / "raven_exclusions"

    manifest_rows: list[dict] = []
    for audio_path in audio_files:
        info = sf.info(audio_path)
        table_path = table_by_stem.get(audio_path.stem)
        legacy_path = old_bandpass_dir / audio_path.name
        manifest_rows.append(
            {
                "stem": audio_path.stem,
                "input_audio": str(audio_path.resolve()),
                "input_table": str(table_path.resolve()) if table_path else "",
                "pair_status": "paired" if table_path else "bandpass_only",
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "input_subtype": info.subtype,
                "duration_s": info.duration,
                "input_sha256": sha256_file(audio_path),
                "matches_legacy_bandpass": (
                    legacy_path.is_file()
                    and sha256_file(legacy_path) == sha256_file(audio_path)
                ),
                "bandpass_output": str((bandpass_dir / audio_path.name).resolve()),
            }
        )
    manifest = pd.DataFrame(manifest_rows)

    if not args.audit_only:
        results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(
                    bandpass_one,
                    Path(row.input_audio),
                    Path(row.bandpass_output),
                    args.lowcut,
                    args.highcut,
                    args.order,
                ): row.stem
                for row in manifest.itertuples()
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        manifest["bandpass_success"] = manifest["stem"].map(
            lambda stem: results[stem]["success"]
        )
        manifest["output_sha256"] = manifest["stem"].map(
            lambda stem: results[stem]["output_sha256"]
        )
        manifest["bandpass_error"] = manifest["stem"].map(
            lambda stem: results[stem]["error"]
        )
    else:
        manifest["bandpass_success"] = False
        manifest["output_sha256"] = ""
        manifest["bandpass_error"] = "audit_only"

    all_valid: list[pd.DataFrame] = []
    all_excluded: list[pd.DataFrame] = []
    for row in manifest.loc[manifest["pair_status"].eq("paired")].itertuples():
        source_audio = Path(row.input_audio)
        clustering_audio = (
            Path(row.bandpass_output) if not args.audit_only else source_audio
        )
        valid, excluded = normalize_raven_table(
            Path(row.input_table),
            source_audio,
            normalized_dir / f"{row.stem}{TABLE_SUFFIX}",
            exclusion_dir / f"{row.stem}.excluded.csv",
            args.min_duration,
            args.max_duration,
        )
        valid["clustering_audio"] = str(clustering_audio.resolve())
        all_valid.append(valid)
        if not excluded.empty:
            excluded["source_stem"] = row.stem
            all_excluded.append(excluded)

    syllables = pd.concat(all_valid, ignore_index=True)
    exclusions = (
        pd.concat(all_excluded, ignore_index=True) if all_excluded else pd.DataFrame()
    )
    manifest.to_csv(output_dir / "corpus_manifest.csv", index=False)
    syllables.to_csv(output_dir / "syllable_manifest.csv", index=False)
    exclusions.to_csv(output_dir / "raven_exclusions.csv", index=False)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "filter": {
            "method": "Butterworth SOS zero-phase",
            "lowcut_hz": args.lowcut,
            "highcut_hz": args.highcut,
            "order": args.order,
            "output_subtype": "FLOAT",
        },
        "counts": {
            "audio": len(audio_files),
            "tables": len(table_files),
            "paired": int(manifest["pair_status"].eq("paired").sum()),
            "bandpass_only": int(manifest["pair_status"].eq("bandpass_only").sum()),
            "bandpass_success": int(manifest["bandpass_success"].sum()),
            "valid_syllables": len(syllables),
            "excluded_rows": len(exclusions),
            "invalid_frequency_boxes_kept": int(
                (~syllables["frequency_box_valid"]).sum()
            ),
        },
        "notes": [
            "只使用输入目录中的同名Raven表。",
            "Raven 表不执行小于 5 ms 合并。",
            "输入与登记带通参考的匹配状态由 matches_legacy_bandpass 标记。",
        ],
    }
    with (output_dir / "preprocess_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.audit_only and not manifest["bandpass_success"].all():
        print("存在带通失败文件，详见 corpus_manifest.csv", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
