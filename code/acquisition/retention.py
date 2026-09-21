#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Complete-recording retention, Python 3.9+, standard library only.

The producer MUST publish a completion marker only after analysis has closed the
WAV and CSV. CSV existence, size, age, and stable mtime are never completion.
This worker does not run BirdNET, calibrate confidence, or manage a microphone.

CSV schema is the historical five-column semicolon schema. Windows may extend
past the 30 s recording through model zero padding: start must be <30, width
<=3 s, and end <=33 s. Padding never extends the archived waveform.

All source inputs remain by default. Negative quarantine makes a verified copy;
it deliberately does not delete inputs. Only a verified committed positive can
authorize optional deletion of its matching source WAV. Directories must be
distinct, non-nested, private to the operator, and contain no symlink components.
"""

import argparse
import contextlib
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
import wave


VERSION = "1.0.0"
SCHEMA = 1
HEADER = ["Start (s)", "End (s)", "Scientific name", "Common name", "Confidence"]
FRAMES = 30 * 48000
DEFAULTS = {
    "schema_version": 1,
    "scientific_name": "Passer montanus",
    "common_name": "麻雀",
    "threshold": 0.2,
    "comparison": "gte",
    "source_wav_policy": "retain",
    "negative_policy": "retain",
    "min_free_bytes": 64 * 1024 * 1024,
    "capacity_buffer_bytes": 16 * 1024 * 1024,
    "ffmpeg": "ffmpeg",
    "ffprobe": "ffprobe",
    "command_timeout_seconds": 300,
    "lock_timeout_seconds": 30,
}


class RetentionError(Exception):
    """Unknown/failed input: never a negative observation."""


class Waiting(RetentionError):
    """Retry after completion, available space, or another worker."""


def sha256(path):
    """Return a file content digest; callers enforce their own path boundary."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_path(value, *, exists=False, directory=False):
    path = Path(os.path.abspath(os.fspath(value)))
    for component in [*reversed(path.parents), path]:
        if component.is_symlink():
            raise RetentionError("Symlinks are not permitted: " + str(component))
        # Reject Windows directory junctions/reparse points as well as symlinks.
        if (
            component.exists()
            and getattr(component.lstat(), "st_file_attributes", 0) & 0x400
        ):
            raise RetentionError("Reparse points are not permitted: " + str(component))
    if exists and not path.exists():
        raise Waiting("Input does not exist: " + str(path))
    if path.exists():
        if directory and not path.is_dir():
            raise RetentionError("Expected directory: " + str(path))
        if not directory and not stat.S_ISREG(path.lstat().st_mode):
            raise RetentionError("Expected regular file: " + str(path))
    return path


def _basename(value):
    if (
        not isinstance(value, str)
        or not value
        or value in (".", "..")
        or any(c in value for c in "/\\\x00:")
        or Path(value).name != value
    ):
        raise RetentionError("Marker paths must be plain basenames")
    return value


def _finite(value, label):
    if isinstance(value, bool):
        raise RetentionError(label + " must be a finite number")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise RetentionError(label + " must be a finite number") from exc
    if not math.isfinite(number):
        raise RetentionError(label + " must be a finite number")
    return number


def prepare_config(config):
    if not isinstance(config, dict):
        raise RetentionError("Configuration must be a JSON object")
    conf = dict(DEFAULTS)
    conf.update(config)
    if type(conf.get("schema_version")) is not int or conf["schema_version"] != SCHEMA:
        raise RetentionError("Unsupported config schema_version")
    for key in ("pending_dir", "archive_dir"):
        if not conf.get(key):
            raise RetentionError("Missing config: " + key)
    for key in ("pending_dir", "archive_dir", "quarantine_dir"):
        if conf.get(key):
            conf[key] = _safe_path(conf[key], directory=True)
    dirs = [
        conf[key]
        for key in ("pending_dir", "archive_dir", "quarantine_dir")
        if conf.get(key)
    ]
    for i, left in enumerate(dirs):
        for right in dirs[i + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise RetentionError(
                    "Input/archive/quarantine directories must be distinct and non-nested"
                )
    for key in ("scientific_name", "common_name", "ffmpeg", "ffprobe"):
        if not isinstance(conf.get(key), str) or not conf[key].strip():
            raise RetentionError("Missing nonempty config string: " + key)
    conf["threshold"] = _finite(conf["threshold"], "threshold")
    if not 0 <= conf["threshold"] <= 1:
        raise RetentionError("threshold must be between 0 and 1")
    if conf["comparison"] not in ("gte", "gt"):
        raise RetentionError("comparison must be gte or gt")
    if conf["source_wav_policy"] not in ("retain", "delete_after_commit"):
        raise RetentionError("Invalid source_wav_policy")
    if conf["negative_policy"] not in ("retain", "quarantine"):
        raise RetentionError("Invalid negative_policy")
    if conf["negative_policy"] == "quarantine" and not conf.get("quarantine_dir"):
        raise RetentionError("quarantine_dir is required for quarantine")
    for key in ("min_free_bytes", "capacity_buffer_bytes"):
        if type(conf[key]) is not int or conf[key] < 0:
            raise RetentionError(key + " must be a nonnegative integer")
    for key in ("command_timeout_seconds", "lock_timeout_seconds"):
        conf[key] = _finite(conf[key], key)
        if conf[key] <= 0:
            raise RetentionError(key + " must be positive")
    return conf


def _fsync_dir(path):
    # Windows has no portable directory fsync; POSIX power-loss durability is
    # stronger. File fsync and same-filesystem rename are still used on Windows.
    if os.name != "nt":
        fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _write_json(path, value):
    with open(path, "x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _copy_verified(source, destination, expected):
    _safe_path(source, exists=True)
    with open(source, "rb") as incoming, open(destination, "xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if sha256(destination) != expected:
        raise Waiting("Input changed after analysis completion: " + str(source))


@contextlib.contextmanager
def _lock(directory, timeout, name=".retention.lock"):
    directory.mkdir(parents=True, exist_ok=True)
    _safe_path(directory, exists=True, directory=True)
    path = _safe_path(directory / _basename(name))
    with open(path, "a+b") as handle:
        # One byte permits msvcrt byte-range locking; flock uses the whole file.
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    raise Waiting("Another retention process holds the lock")
                time.sleep(0.05)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _load_marker(path):
    _safe_path(path, exists=True)
    if path.stat().st_size > 64 * 1024:
        raise RetentionError("Oversized completion marker")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise RetentionError("Invalid completion marker JSON") from exc
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != SCHEMA
    ):
        raise RetentionError("Unsupported marker schema_version")
    if value.get("completed") is not True:
        raise Waiting("Analysis is not complete")
    if (
        not isinstance(value.get("recording_id"), str)
        or not value["recording_id"].strip()
        or len(value["recording_id"]) > 1024
    ):
        raise RetentionError("Missing or oversized recording_id")
    for key, suffix in (("wav", ".wav"), ("csv", ".csv")):
        name = _basename(value.get(key))
        if not name.lower().endswith(suffix):
            raise RetentionError("Invalid " + key + " extension")
    for key in ("wav_sha256", "csv_sha256"):
        digest = value.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise RetentionError("Missing or invalid " + key)
    if "context" in value:
        if not isinstance(value["context"], dict):
            raise RetentionError("Completion context must be a JSON object")
        try:
            json.dumps(value["context"], allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise RetentionError(
                "Completion context must contain only finite JSON values"
            ) from exc
    return value


def publish_completed(wav_path, csv_path, recording_id, *, context=None):
    """Called by a successful analysis producer, never by a filesystem watcher.

    Both files must be closed and in the same non-symlink directory. The marker
    is published atomically; an incompatible existing marker is never replaced.
    The producer remains responsible for checking the analyzer's success.
    """
    wav_path = _safe_path(wav_path, exists=True)
    csv_path = _safe_path(csv_path, exists=True)
    if wav_path.parent != csv_path.parent:
        raise RetentionError("Completion inputs must share one pending directory")
    marker = wav_path.with_name(wav_path.name + ".ready.json")
    value = {
        "schema_version": SCHEMA,
        "completed": True,
        "recording_id": recording_id,
        "wav": wav_path.name,
        "csv": csv_path.name,
        "wav_sha256": sha256(wav_path),
        "csv_sha256": sha256(csv_path),
    }
    if context is not None:
        if not isinstance(context, dict):
            raise RetentionError("Completion context must be a JSON object")
        try:
            # Deep-copy JSON metadata to prevent caller mutation during publish.
            value["context"] = json.loads(json.dumps(context, allow_nan=False))
        except (ValueError, TypeError) as exc:
            raise RetentionError(
                "Completion context must contain only finite JSON values"
            ) from exc
    with _lock(wav_path.parent, 30):
        if marker.exists():
            if _load_marker(marker) != value:
                raise RetentionError("Conflicting existing completion marker")
            return marker
        temp = marker.with_name(marker.name + ".writing")
        _safe_path(temp)
        if temp.exists():
            raise RetentionError(
                "Incomplete producer marker requires operator inspection: " + str(temp)
            )
        _write_json(temp, value)
        _load_marker(temp)
        os.replace(temp, marker)
        _fsync_dir(marker.parent)
    return marker


def _identity(marker):
    core = {
        key: marker[key]
        for key in ("recording_id", "wav", "csv", "wav_sha256", "csv_sha256")
    }
    digest = hashlib.sha256(
        json.dumps(core, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return digest, core


def parse_detections(csv_path, conf):
    """Validate every row before deciding; one good hit cannot mask corruption."""
    positive = 0
    rows = 0
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter=";", strict=True)
            if next(reader, None) != HEADER:
                raise RetentionError("Unsupported CSV header or delimiter")
            for row in reader:
                if len(row) != len(HEADER):
                    raise RetentionError("Malformed/truncated CSV row")
                start, end = _finite(row[0], "window start"), _finite(
                    row[1], "window end"
                )
                score = _finite(row[4], "confidence")
                if (
                    not 0 <= start < 30
                    or not start < end <= 33
                    or end - start > 3.000001
                ):
                    raise RetentionError(
                        "Invalid detection window (30 s recording; at most 3 s padded window)"
                    )
                if not 0 <= score <= 1 or not row[2].strip() or not row[3].strip():
                    raise RetentionError("Invalid confidence or species labels")
                rows += 1
                above = (
                    score >= conf["threshold"]
                    if conf["comparison"] == "gte"
                    else score > conf["threshold"]
                )
                if (
                    row[2] == conf["scientific_name"]
                    and row[3] == conf["common_name"]
                    and above
                ):
                    positive += 1
    except (UnicodeError, csv.Error) as exc:
        raise RetentionError("Unreadable or malformed CSV") from exc
    return {
        "rows": rows,
        "qualified_target_rows": positive,
        "classification": "positive" if positive else "negative",
    }


def validate_wav(path):
    """Validate a complete 30 s PCM16 WAV, raising RetentionError or Waiting."""
    path = _safe_path(path, exists=True)
    try:
        with wave.open(str(path), "rb") as recording:
            info = {
                "sample_rate": recording.getframerate(),
                "channels": recording.getnchannels(),
                "sample_width_bytes": recording.getsampwidth(),
                "frames": recording.getnframes(),
                "compression": recording.getcomptype(),
            }
            if info != {
                "sample_rate": 48000,
                "channels": 1,
                "sample_width_bytes": 2,
                "frames": FRAMES,
                "compression": "NONE",
            }:
                raise RetentionError("Expected full 30 s, 48 kHz, mono, 16-bit PCM WAV")
            decoded_bytes = len(recording.readframes(FRAMES + 1))
            if decoded_bytes != FRAMES * 2:
                raise RetentionError("Truncated input WAV data")
            return info
    except (wave.Error, EOFError) as exc:
        raise RetentionError("Invalid PCM WAV") from exc


def _command(arguments, conf):
    try:
        result = subprocess.run(
            arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=conf["command_timeout_seconds"],
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RetentionError(
            "External command could not complete: " + str(exc)
        ) from exc
    if result.returncode != 0:
        raise RetentionError(
            "External command failed (exit %s): %s"
            % (result.returncode, result.stderr.decode("utf-8", "replace")[-2048:])
        )
    return result.stdout


def _verify_mp3(path, conf):
    _safe_path(path, exists=True)
    if path.stat().st_size == 0:
        raise RetentionError("Empty MP3 output")
    try:
        probe = json.loads(
            _command(
                [
                    conf["ffprobe"],
                    "-v",
                    "error",
                    "-show_streams",
                    "-show_format",
                    "-of",
                    "json",
                    str(path),
                ],
                conf,
            )
        )
        streams = probe["streams"]
        stream = streams[0]
        if (
            len(streams) != 1
            or stream["codec_name"] != "mp3"
            or int(stream["sample_rate"]) != 48000
            or int(stream["channels"]) != 1
            or int(stream["bit_rate"]) != 320000
        ):
            raise RetentionError(
                "Output codec/rate/channels/bitrate differ from required MP3 profile"
            )
        duration = _finite(probe["format"]["duration"], "MP3 container duration")
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise RetentionError("Unrecognized ffprobe output") from exc
    pcm = _command(
        [
            conf["ffmpeg"],
            "-nostdin",
            "-xerror",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "s16le",
            "-c:a",
            "pcm_s16le",
            "-",
        ],
        conf,
    )
    if len(pcm) != FRAMES * 2:
        raise RetentionError(
            "Decoded MP3 does not contain exactly 1440000 mono PCM frames"
        )
    return {
        "codec": "mp3",
        "sample_rate": 48000,
        "channels": 1,
        "bit_rate": 320000,
        "decoded_frames": len(pcm) // 2,
        "container_duration_seconds": duration,
    }


def _verify_commit(final, identity, conf, outcome="positive", expected_completion=None):
    _safe_path(final, exists=True, directory=True)
    receipt_path = _safe_path(final / "receipt.json", exists=True)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise RetentionError(
            "Corrupt committed receipt; operator inspection required"
        ) from exc
    if (
        not isinstance(receipt, dict)
        or type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != SCHEMA
        or receipt.get("identity") != identity
        or receipt.get("outcome") != outcome
    ):
        raise RetentionError("Archive identity collision; refusing overwrite")
    expected_rule = {
        key: conf[key]
        for key in ("scientific_name", "common_name", "threshold", "comparison")
    }
    if receipt.get("rule") != expected_rule:
        raise RetentionError(
            "Archive uses a different target/threshold rule; refusing reinterpretation"
        )
    completion_path = _safe_path(final / "completion.json", exists=True)
    completion = _load_marker(completion_path)
    if _identity(completion)[1] != identity or sha256(completion_path) != receipt.get(
        "completion_sha256"
    ):
        raise RetentionError("Committed completion metadata hash/identity mismatch")
    if expected_completion is not None and completion != expected_completion:
        raise RetentionError(
            "Completion metadata differs for an existing recording identity"
        )
    csv_path = _safe_path(final / "detections.csv", exists=True)
    if sha256(csv_path) != identity["csv_sha256"]:
        raise RetentionError("Committed CSV hash mismatch")
    detection = parse_detections(csv_path, conf)
    if detection != receipt.get("detections") or detection["classification"] != outcome:
        raise RetentionError("Committed CSV and receipt decision disagree")
    if outcome == "positive":
        audio = _safe_path(final / "audio.mp3", exists=True)
        if sha256(audio) != receipt.get("mp3_sha256"):
            raise RetentionError("Committed MP3 hash mismatch")
        _verify_mp3(audio, conf)
    else:
        audio = _safe_path(final / "source.wav", exists=True)
        if sha256(audio) != identity["wav_sha256"]:
            raise RetentionError("Quarantined WAV hash mismatch")
        validate_wav(audio)
    return receipt


def _discard_stage(stage):
    """Remove only known intermediate files under our locked staging location."""
    _safe_path(stage, exists=True, directory=True)
    allowed = {
        "source.wav",
        "detections.csv",
        "audio.mp3",
        "receipt.json",
        "completion.json",
    }
    children = list(stage.iterdir())
    for child in children:
        if child.name not in allowed:
            raise RetentionError(
                "Unrecognized staging content; operator inspection required"
            )
        _safe_path(child, exists=True)
    for child in children:
        child.unlink()
    stage.rmdir()


def _delete_source_if_requested(wav_path, final, identity, conf):
    if conf["source_wav_policy"] != "delete_after_commit" or not wav_path.exists():
        return False
    _verify_commit(final, identity, conf)
    _safe_path(wav_path, exists=True)
    if sha256(wav_path) != identity["wav_sha256"]:
        raise RetentionError("Source changed after commit; source deletion refused")
    csv_path = _safe_path(wav_path.parent / identity["csv"], exists=True)
    if sha256(csv_path) != identity["csv_sha256"]:
        raise RetentionError("Source CSV changed after commit; source deletion refused")
    wav_path.unlink()
    _fsync_dir(wav_path.parent)
    return True


def _process_locked(marker_path, conf):
    marker = _load_marker(marker_path)
    analysis = marker.get("context", {}).get("analysis")
    if isinstance(analysis, dict) and "inference_contract" in analysis:
        contract = analysis["inference_contract"]
        saved = contract.get("configuration") if isinstance(contract, dict) else None
        if not isinstance(saved, dict) or any(
            saved.get(key) != conf[key]
            for key in ("scientific_name", "common_name", "threshold", "comparison")
        ):
            raise RetentionError(
                "CSV producer inference contract differs from the retention rule"
            )
    if marker_path.name != marker["wav"] + ".ready.json":
        raise RetentionError(
            "Completion marker basename does not match its declared WAV"
        )
    key, identity = _identity(marker)
    wav_path = _safe_path(conf["pending_dir"] / marker["wav"])
    csv_path = _safe_path(conf["pending_dir"] / marker["csv"])
    final = _safe_path(conf["archive_dir"] / key, directory=True)
    if final.exists():
        _verify_commit(final, identity, conf, expected_completion=marker)
        removed = _delete_source_if_requested(wav_path, final, identity, conf)
        return {
            "status": "already_archived",
            "recording_id": marker["recording_id"],
            "archive": str(final),
            "source_wav_deleted": removed,
        }
    _safe_path(wav_path, exists=True)
    _safe_path(csv_path, exists=True)
    if (
        sha256(wav_path) != marker["wav_sha256"]
        or sha256(csv_path) != marker["csv_sha256"]
    ):
        raise Waiting(
            "Input hash differs from completed analysis marker; keep originals"
        )
    # Validate both inputs even for a header-only result; an incomplete WAV is
    # an unknown state, never a valid negative recording.
    input_info = validate_wav(wav_path)
    detection = parse_detections(csv_path, conf)
    if (
        detection["classification"] == "negative"
        and conf["negative_policy"] == "retain"
    ):
        return {
            "status": "negative_retained",
            "recording_id": marker["recording_id"],
            **detection,
        }
    outcome = detection["classification"]
    destination = (
        conf["archive_dir"] if outcome == "positive" else conf["quarantine_dir"]
    )
    destination.mkdir(parents=True, exist_ok=True)
    _safe_path(destination, exists=True, directory=True)
    final = _safe_path(destination / key, directory=True)
    if final.exists():
        _verify_commit(final, identity, conf, outcome, expected_completion=marker)
        return {
            "status": "already_quarantined",
            "recording_id": marker["recording_id"],
            "archive": str(final),
        }
    needed = (
        conf["min_free_bytes"]
        + conf["capacity_buffer_bytes"]
        + wav_path.stat().st_size
        + csv_path.stat().st_size
        + 2 * 1200000
        + 65536
    )
    if shutil.disk_usage(destination).free < needed:
        raise Waiting(
            "Insufficient free space; originals retained (required free bytes: %s)"
            % needed
        )
    staging = _safe_path(destination / ".staging", directory=True)
    staging.mkdir(exist_ok=True)
    stage = _safe_path(staging / key, directory=True)
    recovered = False
    if stage.exists():
        # A process may have died before rename; only a complete, verified stage
        # can be committed. Partial staging files are recreated from originals.
        for child in stage.iterdir():
            if child.name not in {
                "source.wav",
                "detections.csv",
                "audio.mp3",
                "receipt.json",
                "completion.json",
            }:
                raise RetentionError(
                    "Unrecognized staging content; operator inspection required"
                )
            _safe_path(child, exists=True)
        if (stage / "receipt.json").exists():
            try:
                _verify_commit(
                    stage, identity, conf, outcome, expected_completion=marker
                )
                recovered = True
            except RetentionError:
                _discard_stage(stage)
        else:
            _discard_stage(stage)
    if not recovered:
        stage.mkdir()
        _copy_verified(wav_path, stage / "source.wav", marker["wav_sha256"])
        _copy_verified(csv_path, stage / "detections.csv", marker["csv_sha256"])
        # Re-parse the staged immutable copy; catches concurrent writers whose
        # contents were different from the CSV we classified above.
        if parse_detections(stage / "detections.csv", conf) != detection:
            raise Waiting("CSV changed during staging")
        _write_json(stage / "completion.json", marker)
        receipt = {
            "schema_version": SCHEMA,
            "software_version": VERSION,
            "identity": identity,
            "outcome": outcome,
            "completed_at_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "input": input_info,
            "detections": detection,
            "rule": {
                key: conf[key]
                for key in ("scientific_name", "common_name", "threshold", "comparison")
            },
            "source_wav_policy_requested": conf["source_wav_policy"],
            "negative_policy": conf["negative_policy"],
            "completion_sha256": sha256(stage / "completion.json"),
        }
        if outcome == "positive":
            command = [
                conf["ffmpeg"],
                "-nostdin",
                "-xerror",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(stage / "source.wav"),
                "-map",
                "0:a:0",
                "-map_metadata",
                "-1",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "320k",
                "-ac",
                "1",
                "-y",
                str(stage / "audio.mp3"),
            ]
            _command(command, conf)
            receipt["output"] = _verify_mp3(stage / "audio.mp3", conf)
            receipt["mp3_sha256"] = sha256(stage / "audio.mp3")
            receipt["transcode_arguments"] = [
                "ffmpeg",
                "-nostdin",
                "-xerror",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "source.wav",
                "-map",
                "0:a:0",
                "-map_metadata",
                "-1",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "320k",
                "-ac",
                "1",
                "-y",
                "audio.mp3",
            ]
            with open(stage / "audio.mp3", "r+b") as handle:
                os.fsync(handle.fileno())
            (stage / "source.wav").unlink()
        _write_json(stage / "receipt.json", receipt)
        _fsync_dir(stage)
    # No overwrite: all writers use one OS lock, and unexpected final directories
    # are rejected. Staging and final are on the same filesystem.
    if final.exists():
        raise RetentionError("Archive appeared during processing; refusing overwrite")
    os.rename(stage, final)
    _fsync_dir(destination)
    _fsync_dir(staging)
    _verify_commit(final, identity, conf, outcome, expected_completion=marker)
    removed = (
        _delete_source_if_requested(wav_path, final, identity, conf)
        if outcome == "positive"
        else False
    )
    return {
        "status": "archived" if outcome == "positive" else "negative_quarantined",
        "recording_id": marker["recording_id"],
        "archive": str(final),
        "recovered_stage": recovered,
        "source_wav_deleted": removed,
        **detection,
    }


def process_marker(marker_path, config):
    """Return structured status. Errors preserve originals; KeyboardInterrupt propagates."""
    try:
        conf = prepare_config(config)
        marker_path = _safe_path(marker_path, exists=True)
        if marker_path.parent != conf["pending_dir"] or not marker_path.name.endswith(
            ".ready.json"
        ):
            raise RetentionError(
                "Marker must be a *.ready.json directly inside pending_dir"
            )
        _safe_path(conf["pending_dir"], exists=True, directory=True)
        # Sort every shared storage root to prevent deadlocks across configurations.
        # A common quarantine root must serialize writers even when archives differ.
        roots = {conf["pending_dir"], conf["archive_dir"]}
        if conf["negative_policy"] == "quarantine":
            roots.add(conf["quarantine_dir"])
        with contextlib.ExitStack() as locks:
            for root in sorted(roots, key=str):
                locks.enter_context(_lock(root, conf["lock_timeout_seconds"]))
            return _process_locked(marker_path, conf)
    except Waiting as exc:
        return {"status": "waiting", "marker": str(marker_path), "message": str(exc)}
    except (RetentionError, OSError, ValueError) as exc:
        return {"status": "error", "marker": str(marker_path), "message": str(exc)}


def process_pending(config):
    conf = prepare_config(config)
    pending = _safe_path(conf["pending_dir"], exists=True, directory=True)
    return [process_marker(path, conf) for path in sorted(pending.glob("*.ready.json"))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON retention configuration")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process a single snapshot (the only supported mode)",
    )
    parser.add_argument(
        "--publish",
        nargs=3,
        metavar=("WAV", "CSV", "RECORDING_ID"),
        help="Publish a marker after producer-confirmed successful analysis",
    )
    args = parser.parse_args(argv)
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if args.publish:
            conf = prepare_config(config)
            if any(
                _safe_path(p, exists=True).parent != conf["pending_dir"]
                for p in args.publish[:2]
            ):
                raise RetentionError(
                    "Published inputs must be in configured pending_dir"
                )
            marker = publish_completed(*args.publish)
            print(
                json.dumps(
                    {"status": "published", "marker": str(marker)}, ensure_ascii=False
                )
            )
            return 0
        results = process_pending(config)
        for result in results:
            print(json.dumps(result, ensure_ascii=False))
        if any(item["status"] == "error" for item in results):
            return 1
        return 2 if any(item["status"] == "waiting" for item in results) else 0
    except (RetentionError, OSError, ValueError) as exc:
        print(
            json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
