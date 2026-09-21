#!/usr/bin/env python3
"""Audio QC：只读发布候选音频完整解码、SHA-256 与分批检查点。

ffmpeg 严格解码到本机采样率/声道的 float32 WAV 管道；不写 PCM、不重采样、
不归一化、不修改原音频。时长、全零及幅值越界仅为诊断，不设生物学排除阈值。
所有角色均执行同一解码；timeout 是未决结果，不等同于损坏。
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import selectors
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import numpy as np


SCHEMA_VERSION = 2
STREAM_ACCUMULATION_BYTES = 256 * 1024
REQUIRED_COLUMNS = (
    "physical_id",
    "logical_id",
    "relative_path",
    "year",
    "site",
    "role",
    "expected_size_bytes",
)
ROLES = {"selected_representative", "selected_duplicate", "diagnostic_anomaly"}
STAT_FIELDS = ("source_state", "size_bytes", "mtime_ns", "ctime_ns", "inode", "device")
QC_FIELDS = (
    "qc_status",
    "strict_decode_pass",
    "sha256",
    "actual_size_bytes",
    "expected_size_matches",
    "source_stat_matches",
    "source_mtime_ns",
    "source_ctime_ns",
    "source_inode",
    "source_device",
    "native_sample_rate",
    "native_channels",
    "decoded_frames",
    "decoded_samples",
    "duration_seconds",
    "duration_difference_from_30s",
    "finite_samples",
    "nonfinite_samples",
    "peak_abs",
    "rms",
    "all_zero",
    "samples_abs_ge_1",
    "fraction_abs_ge_1",
    "samples_abs_eq_1",
    "samples_abs_gt_1",
    "mean_dc",
    "mean_dc_per_channel_json",
    "longest_zero_run_frames",
    "longest_constant_run_frames",
    "window_rms_window_ms",
    "window_rms_hop_ms",
    "window_rms_count",
    "window_rms_partial_count",
    "window_rms_nonfinite_count",
    "window_rms_min",
    "window_rms_median",
    "window_rms_max",
    "decode_complete",
    "decoder_return_code",
    "stderr_bytes",
    "stderr_excerpt",
    "has_decoder_error",
    "decoder_error_lines",
    "decoder_error_excerpt",
    "diagnostic_flags",
    "attempt_count",
    "attempt_history_json",
    "elapsed_seconds",
    "completed_at",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hash_file(path: Path, stop: threading.Event | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            if stop is not None and stop.is_set():
                raise InterruptedError("run interrupted")
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Any, *, replace: bool = False) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists() and not replace:
        temporary.unlink()
        raise FileExistsError(path)
    os.replace(temporary, path)


def open_csv(path: Path):
    return (
        gzip.open(path, "rt", encoding="utf-8-sig", newline="")
        if path.suffix == ".gz"
        else path.open("r", encoding="utf-8-sig", newline="")
    )


def write_csv_atomic(path: Path, fields: list[str], rows) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
                writer.writeheader()
                writer.writerows(rows)
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, path)


def resolve_audio(data_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"音频路径必须为 data_root 内相对路径：{relative_path}")
    path = (data_root / relative).resolve()
    if not path.is_relative_to(data_root):
        raise ValueError(f"音频链接逃逸 data_root：{relative_path}")
    return path


def source_stat(path: Path) -> dict[str, Any]:
    result = dict.fromkeys(STAT_FIELDS, "")
    try:
        value = path.stat()
    except FileNotFoundError:
        result["source_state"] = "missing"
        return result
    except OSError as error:
        result["source_state"] = f"stat_error:{type(error).__name__}"
        return result
    result.update(
        source_state="regular" if stat.S_ISREG(value.st_mode) else "not_regular",
        size_bytes=value.st_size,
        mtime_ns=value.st_mtime_ns,
        ctime_ns=value.st_ctime_ns,
        inode=value.st_ino,
        device=value.st_dev,
    )
    return result


def stat_equal(left: dict, right: dict) -> bool:
    return all(str(left.get(key, "")) == str(right.get(key, "")) for key in STAT_FIELDS)


class FloatWaveAccumulator:
    """解析 ffmpeg WAV 管道，累计全部声道样本；不保留解码音频。"""

    def __init__(self) -> None:
        self.header = bytearray()
        self.pending = b""
        self.in_data = False
        self.rate = 0
        self.channels = 0
        self.samples = 0
        self.finite = 0
        self.nonfinite = 0
        self.peak = 0.0
        self.sumsq = 0.0
        self.ge_one = 0
        self.zero = True
        self.eq_one = 0
        self.sum_values = 0.0
        self.channel_sums = None
        self.channel_finite_counts = None
        self.last_values = None
        self.zero_tails = None
        self.constant_tails = None
        self.longest_zero = 0
        self.longest_constant = 0
        self.window_powers = np.empty(0, dtype=np.float64)
        self.window_invalid = np.empty(0, dtype=np.int64)
        self.window_rms = []
        self.window_count = 0
        self.window_partial_count = 0
        self.window_nonfinite_count = 0
        self.windows_finished = False

    def feed(self, block: bytes) -> None:
        if not self.in_data:
            self.header.extend(block)
            if len(self.header) < 12:
                return
            if (
                self.header[:4] not in (b"RIFF", b"RF64")
                or self.header[8:12] != b"WAVE"
            ):
                raise ValueError("ffmpeg output is not WAV")
            offset = 12
            while len(self.header) >= offset + 8:
                name = bytes(self.header[offset : offset + 4])
                size = struct.unpack_from("<I", self.header, offset + 4)[0]
                if name == b"data":
                    if not self.rate or not self.channels:
                        raise ValueError("WAV data before valid fmt")
                    self.in_data = True
                    block = bytes(self.header[offset + 8 :])
                    self.header.clear()
                    break
                if size > 1024 * 1024:
                    raise ValueError("unexpectedly large WAV metadata chunk")
                if len(self.header) < offset + 8 + size:
                    return
                if name == b"fmt ":
                    if size < 16:
                        raise ValueError("incomplete WAV fmt")
                    tag, channels, rate, _, alignment, bits = struct.unpack_from(
                        "<HHIIHH", self.header, offset + 8
                    )
                    if tag == 0xFFFE and size >= 40:
                        tag = struct.unpack_from("<H", self.header, offset + 8 + 24)[0]
                    if tag != 3 or bits != 32 or alignment != channels * 4:
                        raise ValueError("expected native float32 WAV output")
                    self.rate, self.channels = rate, channels
                offset += 8 + size + (size % 2)
            if not self.in_data:
                return
        block = self.pending + block
        usable = len(block) - len(block) % (4 * self.channels)
        self.pending = block[usable:]
        if not usable:
            return
        values = np.frombuffer(block[:usable], dtype="<f4")
        mask = np.isfinite(values)
        self.samples += int(values.size)
        self.finite += int(mask.sum())
        self.nonfinite += int((~mask).sum())
        finite = values[mask].astype(np.float64)
        if finite.size:
            absolute = np.abs(finite)
            self.peak = max(self.peak, float(absolute.max()))
            self.sumsq += float(np.sum(finite * finite))
            self.sum_values += float(finite.sum())
            self.eq_one += int((absolute == 1.0).sum())
            self.ge_one += int((absolute >= 1.0).sum())
            self.zero = self.zero and not bool(np.any(finite != 0))
        if not np.all(mask):
            self.zero = False
        matrix = values.reshape(-1, self.channels)
        finite_mask = mask.reshape(-1, self.channels)
        safe = np.where(finite_mask, matrix, 0).astype(np.float64)
        if self.channel_sums is None:
            self.channel_sums = np.zeros(self.channels, dtype=np.float64)
            self.channel_finite_counts = np.zeros(self.channels, dtype=np.int64)
            self.last_values = np.full(self.channels, np.nan)
            self.zero_tails = np.zeros(self.channels, dtype=np.int64)
            self.constant_tails = np.zeros(self.channels, dtype=np.int64)
        self.channel_sums += safe.sum(axis=0)
        self.channel_finite_counts += finite_mask.sum(axis=0)
        for channel in range(self.channels):
            channel_values = matrix[:, channel]
            # Runs are counted within each channel, never across interleaved channels.
            is_zero = channel_values == 0
            changes = np.flatnonzero(np.r_[True, is_zero[1:] != is_zero[:-1], True])
            lengths = np.diff(changes)
            zero_lengths = lengths[is_zero[changes[:-1]]]
            if zero_lengths.size:
                if is_zero[0]:
                    zero_lengths[0] += self.zero_tails[channel]
                self.longest_zero = max(self.longest_zero, int(zero_lengths.max()))
            self.zero_tails[channel] = int(zero_lengths[-1]) if is_zero[-1] else 0
            changes = np.flatnonzero(
                np.r_[True, channel_values[1:] != channel_values[:-1], True]
            )
            lengths = np.diff(changes)
            if channel_values[0] == self.last_values[channel] and np.isfinite(
                channel_values[0]
            ):
                lengths[0] += self.constant_tails[channel]
            eligible = np.isfinite(channel_values[changes[:-1]])
            if np.any(eligible):
                self.longest_constant = max(
                    self.longest_constant, int(lengths[eligible].max())
                )
            self.constant_tails[channel] = (
                int(lengths[-1]) if np.isfinite(channel_values[-1]) else 0
            )
            self.last_values[channel] = channel_values[-1]
        powers = np.mean(safe * safe, axis=1)
        invalid = (~finite_mask.all(axis=1)).astype(np.int64)
        self.window_powers = np.concatenate((self.window_powers, powers))
        self.window_invalid = np.concatenate((self.window_invalid, invalid))
        self._consume_windows(partial=False)

    def _consume_windows(self, partial: bool) -> None:
        if not self.rate:
            return
        size = max(1, round(self.rate * 0.1))
        hop = max(1, round(self.rate * 0.05))
        available = len(self.window_powers)
        stop = available if partial else max(0, available - size + 1)
        starts = np.arange(0, stop, hop, dtype=np.int64)
        if not len(starts):
            return
        ends = np.minimum(starts + size, available)
        sums = np.r_[0.0, np.cumsum(self.window_powers)]
        invalid = np.r_[0, np.cumsum(self.window_invalid)]
        valid_windows = invalid[ends] == invalid[starts]
        rms = np.sqrt(np.maximum(0, (sums[ends] - sums[starts]) / (ends - starts)))
        self.window_rms.extend(rms[valid_windows].tolist())
        self.window_count += len(starts)
        self.window_partial_count += int(((ends - starts) < size).sum())
        self.window_nonfinite_count += int((~valid_windows).sum())
        consumed = int(starts[-1] + hop)
        self.window_powers = self.window_powers[consumed:]
        self.window_invalid = self.window_invalid[consumed:]

    def result(self) -> dict:
        if not self.windows_finished:
            self._consume_windows(partial=True)
            self.windows_finished = True
        frames = self.samples // self.channels if self.channels else 0
        duration = frames / self.rate if self.rate else None
        return {
            "native_sample_rate": self.rate or "",
            "native_channels": self.channels or "",
            "decoded_frames": frames,
            "decoded_samples": self.samples,
            "duration_seconds": duration if duration is not None else "",
            "duration_difference_from_30s": (
                duration - 30.0 if duration is not None else ""
            ),
            "finite_samples": self.finite,
            "nonfinite_samples": self.nonfinite,
            "peak_abs": self.peak if self.finite else "",
            "rms": math.sqrt(self.sumsq / self.finite) if self.finite else "",
            "all_zero": bool(self.samples and self.zero),
            "samples_abs_ge_1": self.ge_one,
            "fraction_abs_ge_1": self.ge_one / self.samples if self.samples else "",
            "samples_abs_eq_1": self.eq_one,
            "samples_abs_gt_1": self.ge_one - self.eq_one,
            "mean_dc": self.sum_values / self.finite if self.finite else "",
            "mean_dc_per_channel_json": (
                json.dumps(
                    [
                        float(total / count) if count else None
                        for total, count in zip(
                            self.channel_sums, self.channel_finite_counts
                        )
                    ]
                )
                if self.channel_sums is not None
                else "[]"
            ),
            "longest_zero_run_frames": self.longest_zero,
            "longest_constant_run_frames": self.longest_constant,
            "window_rms_window_ms": 100,
            "window_rms_hop_ms": 50,
            "window_rms_count": self.window_count,
            "window_rms_partial_count": self.window_partial_count,
            "window_rms_nonfinite_count": self.window_nonfinite_count,
            "window_rms_min": min(self.window_rms) if self.window_rms else "",
            "window_rms_median": (
                float(np.median(self.window_rms)) if self.window_rms else ""
            ),
            "window_rms_max": max(self.window_rms) if self.window_rms else "",
        }


def strict_decode(
    path: Path, ffmpeg: str, timeout: float, stop: threading.Event
) -> dict:
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-v",
        "repeat+level+warning",
        "-threads",
        "1",
        "-xerror",
        "-err_detect",
        "explode",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-c:a",
        "pcm_f32le",
        "-threads",
        "1",
        "-f",
        "wav",
        "pipe:1",
    ]
    accumulator = FloatWaveAccumulator()
    accumulated_output = bytearray()
    started = time.monotonic()
    status, error_text = "", ""
    process = None
    with tempfile.TemporaryFile() as errors:
        try:
            decoder_env = {**os.environ, "AV_LOG_FORCE_NOCOLOR": "1"}
            decoder_env.pop("AV_LOG_FORCE_COLOR", None)
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=errors,
                bufsize=0,
                env=decoder_env,
            )
            assert process.stdout is not None
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    if stop.is_set():
                        status = "interrupted"
                        break
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        status = "timeout_indeterminate"
                        break
                    if selector.select(min(0.2, remaining)):
                        block = os.read(process.stdout.fileno(), 256 * 1024)
                        if not block:
                            break
                        accumulated_output.extend(block)
                        if len(accumulated_output) >= STREAM_ACCUMULATION_BYTES:
                            accumulator.feed(bytes(accumulated_output))
                            accumulated_output.clear()
                if accumulated_output:
                    accumulator.feed(bytes(accumulated_output))
                    accumulated_output.clear()
                if not status:
                    try:
                        process.wait(
                            timeout=max(0.01, timeout - (time.monotonic() - started))
                        )
                    except subprocess.TimeoutExpired:
                        status = "timeout_indeterminate"
        except (OSError, ValueError) as error:
            status = "io_error" if isinstance(error, OSError) else "decode_error"
            error_text = f"{type(error).__name__}: {error}"
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait()
                if process.stdout is not None:
                    process.stdout.close()
        errors.seek(0, os.SEEK_END)
        stderr_bytes = errors.tell()
        # Scan the COMPLETE stderr stream, not only the retained tail: FFmpeg
        # can return 0 after a decoder thread reports an error and emits some PCM.
        # Severity prefixes are produced by level+warning, avoiding keyword
        # false positives in filenames or ordinary warning text.
        errors.seek(0)
        decoder_error_lines = 0
        error_excerpt_lines = []
        excerpt_bytes = 0
        for line in errors:
            prefix = re.match(rb"^\[([^]\r\n]+)\](?:\s+\[([^]\r\n]+)\])?", line)
            severity = None
            if prefix:
                # A line may have one FFmpeg component prefix before its level.
                # Stop at the first real severity so '[error]' in warning text
                # or a quoted file path cannot become a false decoder failure.
                levels = {
                    b"panic",
                    b"fatal",
                    b"error",
                    b"warning",
                    b"info",
                    b"verbose",
                    b"debug",
                    b"trace",
                }
                severity = prefix[1] if prefix[1] in levels else prefix[2]
            if severity in {b"error", b"fatal", b"panic"}:
                decoder_error_lines += 1
                if excerpt_bytes < 16384:
                    saved = line[: min(len(line), 16384 - excerpt_bytes)]
                    error_excerpt_lines.append(
                        saved.decode("utf-8", errors="replace").rstrip()
                    )
                    excerpt_bytes += len(saved)
        errors.seek(max(0, stderr_bytes - 16384))
        stderr = errors.read().decode("utf-8", errors="replace")
    value = accumulator.result()
    return_code = process.returncode if process is not None else ""
    complete = (
        not status
        and return_code == 0
        and decoder_error_lines == 0
        and accumulator.in_data
        and not accumulator.pending
        and not (accumulator.channels and accumulator.samples % accumulator.channels)
    )
    if not status:
        if not complete:
            status = "decode_error"
        elif not accumulator.samples:
            status = "no_decoded_samples"
        elif accumulator.nonfinite:
            status = "nonfinite_samples"
        else:
            status = "decoded_with_warnings" if stderr else "decoded"
    value.update(
        qc_status=status,
        strict_decode_pass=status in {"decoded", "decoded_with_warnings"},
        decode_complete=complete,
        decoder_return_code=return_code,
        stderr_bytes=stderr_bytes,
        stderr_excerpt=(stderr + ("\n" + error_text if error_text else "")).strip(),
        has_decoder_error=bool(decoder_error_lines),
        decoder_error_lines=decoder_error_lines,
        decoder_error_excerpt="\n".join(error_excerpt_lines),
    )
    return value


def audit_one(
    row: dict,
    snapshot: dict,
    data_root: Path,
    ffmpeg: str,
    timeout: float,
    retries: int,
    retry_delay: float,
    stop: threading.Event,
) -> dict:
    started = time.monotonic()
    result = dict(row)
    result.update(dict.fromkeys(QC_FIELDS, ""))
    result.update(
        strict_decode_pass=False,
        decode_complete=False,
        attempt_count=0,
        attempt_history_json="[]",
        source_stat_matches=False,
    )
    path = resolve_audio(data_root, row["relative_path"])
    before = source_stat(path)
    result.update(
        actual_size_bytes=before["size_bytes"],
        source_mtime_ns=before["mtime_ns"],
        source_ctime_ns=before["ctime_ns"],
        source_inode=before["inode"],
        source_device=before["device"],
        expected_size_matches=str(before["size_bytes"]) == row["expected_size_bytes"],
        source_stat_matches=stat_equal(before, snapshot),
    )
    flags = []
    try:
        if not stat_equal(before, snapshot):
            result["qc_status"] = "source_changed_before_read"
        elif before["source_state"] != "regular":
            result["qc_status"] = before["source_state"]
        elif before["size_bytes"] == 0:
            result.update(qc_status="zero_byte", sha256=hash_file(path, stop))
        else:
            result["sha256"] = hash_file(path, stop)
            attempts = []
            for attempt in range(retries + 1):
                decoded = strict_decode(path, ffmpeg, timeout, stop)
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "qc_status": decoded["qc_status"],
                        "return_code": decoded["decoder_return_code"],
                        "stderr_excerpt": decoded["stderr_excerpt"],
                        "has_decoder_error": decoded["has_decoder_error"],
                        "decoder_error_lines": decoded["decoder_error_lines"],
                        "decoder_error_excerpt": decoded["decoder_error_excerpt"],
                    }
                )
                result.update(decoded)
                if (
                    decoded["qc_status"] not in {"timeout_indeterminate", "io_error"}
                    or stop.is_set()
                ):
                    break
                if attempt < retries and stop.wait(retry_delay):
                    break
            result.update(
                attempt_count=len(attempts),
                attempt_history_json=json.dumps(attempts, ensure_ascii=False),
            )
            if result.get("all_zero"):
                flags.append("all_decoded_samples_zero")
            if result.get("samples_abs_ge_1", 0):
                flags.append("finite_amplitude_abs_ge_1")
            duration = result.get("duration_seconds")
            if duration != "" and duration is not None:
                if duration < 30:
                    flags.append("duration_less_than_nominal_30s")
                elif duration > 30:
                    flags.append("duration_greater_than_nominal_30s")
        if not result["expected_size_matches"]:
            flags.append("size_differs_from_inventory")
        after = source_stat(path)
        if not stat_equal(before, after):
            result.update(
                qc_status="source_changed_during_read",
                strict_decode_pass=False,
                source_stat_matches=False,
            )
        if stop.is_set():
            result.update(qc_status="interrupted", strict_decode_pass=False)
    except InterruptedError:
        result.update(qc_status="interrupted", strict_decode_pass=False)
    except OSError as error:
        result.update(
            qc_status="io_error",
            strict_decode_pass=False,
            stderr_excerpt=f"{type(error).__name__}: {error}",
        )
    result.update(
        diagnostic_flags=";".join(flags),
        elapsed_seconds=round(time.monotonic() - started, 6),
        completed_at=utc_now(),
    )
    return result


def read_manifest(
    path: Path, data_root: Path, limit: int | None
) -> tuple[list[str], list[dict]]:
    with open_csv(path) as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if set(REQUIRED_COLUMNS) - set(fields):
            raise ValueError(
                f"清单缺少字段：{sorted(set(REQUIRED_COLUMNS) - set(fields))}"
            )
        if set(fields) & set(QC_FIELDS) or len(set(fields)) != len(fields):
            raise ValueError("清单字段与 QC 输出冲突或重复")
        rows = []
        ids, paths = set(), set()
        for row in reader:
            if limit is not None and len(rows) >= limit:
                break
            if None in row or any(
                row[column] is None or row[column] == "" for column in REQUIRED_COLUMNS
            ):
                raise ValueError("清单有缺失字段或列数异常")
            if row["role"] not in ROLES or int(row["expected_size_bytes"]) < 0:
                raise ValueError(f"清单角色或大小无效：{row['physical_id']}")
            resolve_audio(data_root, row["relative_path"])
            if row["physical_id"] in ids or row["relative_path"] in paths:
                raise ValueError("清单必须逐物理文件唯一")
            ids.add(row["physical_id"])
            paths.add(row["relative_path"])
            rows.append(row)
    if not rows:
        raise ValueError("清单为空")
    return fields, rows


def tool_identity(name: str) -> dict:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"未找到 {name}")
    resolved = Path(path).resolve()
    version = subprocess.run(
        [str(resolved), "-version"], capture_output=True, text=True, check=True
    ).stdout
    return {
        "path": str(resolved),
        "binary_sha256": hash_file(resolved),
        "version": version,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--timeout-seconds", type=float, default=60)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--retry-delay-seconds", type=float, default=0)
    args = parser.parse_args(argv)
    if (
        args.workers < 1
        or args.workers > 16
        or args.batch_size < 1
        or args.timeout_seconds <= 0
        or args.retries < 0
        or args.retries > 3
        or args.retry_delay_seconds < 0
    ):
        parser.error(
            "workers 1–16；batch-size/timeout >0；retries 0–3；retry delay >=0"
        )
    if args.limit is not None and (args.mode != "pilot" or args.limit <= 0):
        parser.error("--limit 仅允许 pilot 且必须为正数")
    return args


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def aggregate(rows: list[dict]) -> dict:
    statuses = Counter(row["qc_status"] for row in rows)
    roles = Counter(row["role"] for row in rows)
    role_status = Counter(f"{row['role']}:{row['qc_status']}" for row in rows)
    passed = sum(str(row["strict_decode_pass"]).lower() == "true" for row in rows)
    return {
        "row_count": len(rows),
        "status_counts": dict(statuses),
        "role_counts": dict(roles),
        "role_status_counts": dict(role_status),
        "strict_decode_pass_count": passed,
    }


def run(args: argparse.Namespace) -> dict:
    manifest = args.manifest.resolve()
    data_root = args.data_root.resolve()
    output = args.output_dir.resolve()
    fields, rows = read_manifest(manifest, data_root, args.limit)
    ffmpeg = tool_identity("ffmpeg")
    contract = {
        "schema_version": SCHEMA_VERSION,
        "manifest": str(manifest),
        "manifest_sha256": hash_file(manifest),
        "data_root": str(data_root),
        "mode": args.mode,
        "limit": args.limit,
        "row_count": len(rows),
        "manifest_fields": fields,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "timeout_seconds": args.timeout_seconds,
        "retries": args.retries,
        "retry_delay_seconds": args.retry_delay_seconds,
        "script_sha256": hash_file(Path(__file__).resolve()),
        "ffmpeg": ffmpeg,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "decode_policy": "all roles; first audio stream; native rate/channels; float32 WAV stdout; -xerror -err_detect explode; no audio output files",
        "stream_accumulation_bytes": STREAM_ACCUMULATION_BYTES,
        "decoder_log_policy": "FFmpeg repeat+level+warning; scan complete stderr for error/fatal/panic severities; any such line fails strict decoding even with exit 0 and decoded PCM; retain first16KiB error evidence and last16KiB general log",
        "nominal_duration_seconds": 30,
        "signal_diagnostics": "per-channel exact-zero/constant longest runs; DC; abs==1/abs>1; 100ms RMS every50ms with trailing partial windows; nonfinite windows omitted from RMS summaries but counted",
        "duration_and_amplitude_flags": "diagnostic_only_no_biological_exclusion",
        "timeout_policy": "indeterminate_not_corruption; only timeout/io_error retried; all attempts retained",
    }
    contract_sha = canonical_hash(contract)
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise FileNotFoundError("--resume 需要已存在的任务目录")
    lock = (output / ".run.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("同一输出目录已有正在运行的执行器")
    stop = threading.Event()
    run_started = time.monotonic()
    old_handler = signal.getsignal(signal.SIGTERM)
    if threading.current_thread() is threading.main_thread():
        signal.signal(
            signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt())
        )
    try:
        if args.resume:
            config = load_json(output / "run_config.json")
            if (
                config.get("contract") != contract
                or config.get("contract_sha256") != contract_sha
            ):
                raise ValueError(
                    "resume 拒绝：manifest/config/code/tool/environment 与初次运行不一致"
                )
            snapshot_path = output / "source_snapshot.csv.gz"
            if hash_file(snapshot_path) != config["source_snapshot_sha256"]:
                raise ValueError("resume 拒绝：源 stat 快照校验失败")
            with open_csv(snapshot_path) as handle:
                snapshots = list(csv.DictReader(handle))
            if len(snapshots) != len(rows):
                raise ValueError("resume 拒绝：源 stat 快照行数不符")
        else:
            snapshots = [
                {
                    "physical_id": row["physical_id"],
                    **source_stat(resolve_audio(data_root, row["relative_path"])),
                }
                for row in rows
            ]
            snapshot_path = output / "source_snapshot.csv.gz"
            write_csv_atomic(snapshot_path, ["physical_id", *STAT_FIELDS], snapshots)
            config = {
                "created_at": utc_now(),
                "command": sys.argv,
                "contract": contract,
                "contract_sha256": contract_sha,
                "source_snapshot_sha256": hash_file(snapshot_path),
            }
            atomic_json(output / "run_config.json", config)
            (output / "batches").mkdir()
        for row, snapshot in zip(rows, snapshots):
            if row["physical_id"] != snapshot["physical_id"]:
                raise ValueError("源 stat 快照行序与输入不符")
        batch_count = (len(rows) + args.batch_size - 1) // args.batch_size
        completed_batches: dict[int, dict] = {}
        counts = Counter()
        role_counts = Counter()
        role_status_counts = Counter()
        completed_count = passed_count = 0
        for batch_index in range(batch_count):
            batch_name = f"batch_{batch_index:05d}"
            checkpoint_path = output / "batches" / f"{batch_name}.json"
            results_path = output / "batches" / f"{batch_name}.csv.gz"
            if not checkpoint_path.exists():
                if results_path.exists():
                    orphan = output / "interrupted_orphans"
                    orphan.mkdir(exist_ok=True)
                    suffix = f"{time.time_ns()}_{results_path.name}"
                    results_path.rename(orphan / suffix)
                continue
            checkpoint = load_json(checkpoint_path)
            start, end = batch_index * args.batch_size, min(
                (batch_index + 1) * args.batch_size, len(rows)
            )
            if (
                checkpoint.get("contract_sha256") != contract_sha
                or checkpoint.get("start") != start
                or checkpoint.get("end") != end
                or checkpoint.get("results_sha256") != hash_file(results_path)
            ):
                raise ValueError(f"resume 拒绝：批次校验失败 {batch_name}")
            with open_csv(results_path) as handle:
                completed = list(csv.DictReader(handle))
            if len(completed) != end - start or [
                x["physical_id"] for x in completed
            ] != [x["physical_id"] for x in rows[start:end]]:
                raise ValueError(f"resume 拒绝：批次行序失败 {batch_name}")
            changed = [
                row["physical_id"]
                for row, snapshot in zip(rows[start:end], snapshots[start:end])
                if not stat_equal(
                    source_stat(resolve_audio(data_root, row["relative_path"])),
                    snapshot,
                )
            ]
            if changed:
                rejection = {
                    "checked_at": utc_now(),
                    "reason": "completed_source_stat_changed",
                    "batch": batch_name,
                    "changed_physical_ids": changed,
                    "reused_rows": 0,
                }
                atomic_json(
                    output / f"resume_rejection_{time.time_ns()}.json", rejection
                )
                raise ValueError(
                    f"resume 拒绝：已完成批次中 {len(changed)} 个源文件 stat 改变"
                )
            stats = aggregate(completed)
            if stats != checkpoint.get("summary"):
                raise ValueError(f"resume 拒绝：批次统计与内容不符 {batch_name}")
            completed_batches[batch_index] = checkpoint
            counts.update(stats["status_counts"])
            role_counts.update(stats["role_counts"])
            role_status_counts.update(stats["role_status_counts"])
            completed_count += len(completed)
            passed_count += stats["strict_decode_pass_count"]

        def progress(
            state: str, in_batch: int = 0, batch_index: int | None = None
        ) -> None:
            atomic_json(
                output / "progress.json",
                {
                    "state": state,
                    "updated_at": utc_now(),
                    "mode": args.mode,
                    "manifest_sha256": contract["manifest_sha256"],
                    "contract_sha256": contract_sha,
                    "total_rows": len(rows),
                    "committed_rows": completed_count,
                    "active_batch_index": batch_index,
                    "active_batch_completed_rows": in_batch,
                    "completed_batches": len(completed_batches),
                    "total_batches": batch_count,
                    "status_counts_committed": dict(counts),
                    "elapsed_this_invocation_seconds": round(
                        time.monotonic() - run_started, 3
                    ),
                    "workers": args.workers,
                },
                replace=True,
            )

        progress("running")
        executor = ThreadPoolExecutor(max_workers=args.workers)
        try:
            for batch_index in range(batch_count):
                if batch_index in completed_batches:
                    continue
                start, end = batch_index * args.batch_size, min(
                    (batch_index + 1) * args.batch_size, len(rows)
                )
                batch_started = time.monotonic()
                results = [None] * (end - start)
                futures = {
                    executor.submit(
                        audit_one,
                        row,
                        snapshots[start + index],
                        data_root,
                        ffmpeg["path"],
                        args.timeout_seconds,
                        args.retries,
                        args.retry_delay_seconds,
                        stop,
                    ): index
                    for index, row in enumerate(rows[start:end])
                }
                done = 0
                last_progress = time.monotonic()
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
                    done += 1
                    if time.monotonic() - last_progress >= 5:
                        progress("running", done, batch_index)
                        last_progress = time.monotonic()
                if stop.is_set() or any(
                    result["qc_status"] == "interrupted" for result in results
                ):
                    raise KeyboardInterrupt
                batch_name = f"batch_{batch_index:05d}"
                results_path = output / "batches" / f"{batch_name}.csv.gz"
                write_csv_atomic(results_path, fields + list(QC_FIELDS), results)
                checkpoint = {
                    "batch_index": batch_index,
                    "start": start,
                    "end": end,
                    "committed_at": utc_now(),
                    "contract_sha256": contract_sha,
                    "results_sha256": hash_file(results_path),
                    "summary": aggregate(results),
                    "elapsed_seconds": round(time.monotonic() - batch_started, 6),
                }
                atomic_json(output / "batches" / f"{batch_name}.json", checkpoint)
                completed_batches[batch_index] = checkpoint
                stats = checkpoint["summary"]
                completed_count += len(results)
                passed_count += stats["strict_decode_pass_count"]
                counts.update(stats["status_counts"])
                role_counts.update(stats["role_counts"])
                role_status_counts.update(stats["role_status_counts"])
                progress("running")
                print(
                    json.dumps(
                        {
                            "batch": batch_index,
                            "committed_rows": completed_count,
                            "total_rows": len(rows),
                            "status_counts": stats["status_counts"],
                            "elapsed_seconds": checkpoint["elapsed_seconds"],
                        }
                    ),
                    flush=True,
                )
        except BaseException:
            stop.set()
            executor.shutdown(wait=True, cancel_futures=True)
            progress("interrupted")
            raise
        else:
            executor.shutdown(wait=True)

        def all_results():
            for index in range(batch_count):
                with open_csv(
                    output / "batches" / f"batch_{index:05d}.csv.gz"
                ) as handle:
                    yield from csv.DictReader(handle)

        final_results = output / "results.csv.gz"
        summary_path = output / "summary.json"
        if final_results.exists():
            if not summary_path.exists():
                orphan = output / "interrupted_orphans"
                orphan.mkdir(exist_ok=True)
                final_results.rename(orphan / f"{time.time_ns()}_results.csv.gz")
            elif hash_file(final_results) != load_json(summary_path)["results_sha256"]:
                raise ValueError("resume 拒绝：合并结果校验失败")
        if not final_results.exists():
            write_csv_atomic(final_results, fields + list(QC_FIELDS), all_results())
        summary = {
            "schema_version": SCHEMA_VERSION,
            "state": "complete",
            "mode": args.mode,
            "completed_at": utc_now(),
            "contract_sha256": contract_sha,
            "row_count": completed_count,
            "strict_decode_pass_count": passed_count,
            "status_counts": dict(counts),
            "role_counts": dict(role_counts),
            "role_status_counts": dict(role_status_counts),
            "results_sha256": hash_file(final_results),
            "batch_count": batch_count,
            "total_committed_batch_elapsed_seconds": sum(
                x["elapsed_seconds"] for x in completed_batches.values()
            ),
            "limitations": [
                "First audio stream only; no species verification",
                "Finite/amplitude statistics cover decoded samples; failed decodes can be partial",
                "MP3 strict decoding does not establish an undamaged pre-encoding waveform or exact field recording duration",
                "Timeout is indeterminate, not corruption; ordinary warnings are retained; error/fatal/panic severities reject strict completion even with exit0",
                "No nominal-duration or amplitude threshold automatically excludes a recording",
            ],
        }
        if not summary_path.exists():
            atomic_json(summary_path, summary)
        else:
            saved_summary = load_json(summary_path)
            # completed_at describes the original completion and is intentionally
            # not regenerated on resume; every deterministic summary field must agree.
            expected_fields = {
                key: value for key, value in summary.items() if key != "completed_at"
            }
            actual_fields = {
                key: value
                for key, value in saved_summary.items()
                if key != "completed_at"
            }
            if actual_fields != expected_fields:
                raise ValueError("resume 拒绝：最终 summary 与已校验批次重算统计不一致")
            summary = saved_summary
        progress("complete")
        return summary
    finally:
        stop.set()
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, old_handler)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def main(argv: list[str] | None = None) -> int:
    try:
        summary = run(parse_args(argv))
    except KeyboardInterrupt:
        print(
            "运行中断：仅完整提交的批次可在相同配置下 --resume 复用。", file=sys.stderr
        )
        return 130
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
