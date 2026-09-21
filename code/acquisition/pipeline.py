#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Maintained capture -> pinned model -> explicit completion -> retention.

No legacy service is started or modified. `run` has an independent capture
thread and one serial model consumer. Each arecord invocation requests 30 s;
reopening the device introduces a gap, so this is not gapless recording.
Partial capture/analysis files are retained and never become negative results.
"""

import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import threading
import uuid
from zoneinfo import ZoneInfo

import retention
from model_bridge import LABELS_SHA256, MODEL_SHA256, SOURCE_SHA256, ModelBridge

DEFAULTS = {
    "model": "BirdNET_6K_GLOBAL_MODEL",
    "model_path": None,
    "labels_path": None,
    "device_id": "recorder-01",
    "recording_device": None,
    "timezone": "Asia/Shanghai",
    "latitude": None,
    "longitude": None,
    "sensitivity": 1.25,
    "overlap": 1.5,
    "privacy_threshold": 0,
    "arecord": "arecord",
    "capture_timeout_seconds": 45,
}
FRAMES = retention.FRAMES


def validate_config(config, require_runtime=False):
    """Lightweight schema validation; no model, microphone or network access.

    False accepts null deployment placeholders. True requires the fields needed to
    run the entire pipeline and local model files. Unknown fields are rejected.
    """
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    allowed = (
        set(retention.DEFAULTS)
        | set(DEFAULTS)
        | {
            "pending_dir",
            "archive_dir",
            "quarantine_dir",
            "runtime_user",
            "runtime_python",
        }
    )
    unknown = set(config) - allowed
    if unknown:
        raise ValueError("Unknown configuration fields: " + ", ".join(sorted(unknown)))
    conf = dict(retention.DEFAULTS)
    conf.update(DEFAULTS)
    conf.update(config)
    try:
        normalized = retention.prepare_config(conf)
    except retention.RetentionError as exc:
        raise ValueError(str(exc)) from exc
    for key in ("pending_dir", "archive_dir", "quarantine_dir"):
        if conf.get(key) and not (
            Path(conf[key]).is_absolute() or PurePosixPath(str(conf[key])).is_absolute()
        ):
            raise ValueError(key + " must be an absolute deployment path")
    for key, lower, upper in (
        ("latitude", -90, 90),
        ("longitude", -180, 180),
        ("sensitivity", 0.5, 1.5),
        ("overlap", 0, 2.9),
        ("threshold", 0.01, 0.99),
        ("capture_timeout_seconds", 31, 600),
    ):
        value = conf[key]
        if value is None and key in ("latitude", "longitude") and not require_runtime:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not lower <= value <= upper
        ):
            raise ValueError(
                key + " must be a finite number in [%s, %s]" % (lower, upper)
            )
    if conf["privacy_threshold"] != 0 or isinstance(conf["privacy_threshold"], bool):
        raise ValueError("Only historical privacy_threshold=0 is supported")
    if conf["model"] != "BirdNET_6K_GLOBAL_MODEL":
        raise ValueError("Only BirdNET_6K_GLOBAL_MODEL is supported")
    if not isinstance(conf["device_id"], str) or not re.fullmatch(
        r"[A-Za-z0-9_-]{1,64}", conf["device_id"]
    ):
        raise ValueError(
            "device_id must be 1-64 letters, digits, underscores or hyphens"
        )
    if not isinstance(conf["timezone"], str) or not conf["timezone"]:
        raise ValueError("timezone must be an IANA time zone name")
    if not isinstance(conf["arecord"], str) or not conf["arecord"].strip():
        raise ValueError("arecord must be a command or executable path")
    for key in ("recording_device", "model_path", "labels_path"):
        if conf.get(key) is None and not require_runtime:
            continue
        if not isinstance(conf.get(key), str) or not conf[key].strip():
            raise ValueError("Set deployment field " + key)
    ZoneInfo(conf["timezone"])
    if require_runtime:
        for key in ("model_path", "labels_path"):
            retention._safe_path(conf[key], exists=True)
    for key in ("pending_dir", "archive_dir", "quarantine_dir"):
        if normalized.get(key):
            conf[key] = str(normalized[key])
    return conf


def _json_atomic(path, value):
    path = retention._safe_path(path)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with open(temp, "x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    retention._fsync_dir(path.parent)


def _capture_audio_info(path):
    """Map shared PCM validation to the existing capture receipt schema."""
    try:
        info = retention.validate_wav(path)
    except retention.RetentionError as exc:
        raise ValueError(str(exc)) from exc
    return {
        "channels": info["channels"],
        "sample_width_bytes": info["sample_width_bytes"],
        "sample_rate_hz": info["sample_rate"],
        "frames": info["frames"],
    }


def _capacity(conf):
    required = (
        conf["min_free_bytes"] + conf["capacity_buffer_bytes"] + FRAMES * 2 + 4096
    )
    for key in ("pending_dir", "archive_dir"):
        root = retention._safe_path(conf[key], directory=True)
        root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(root).free < required:
            raise retention.Waiting("Insufficient free space before capture on " + key)


def capture_once(config, runner=subprocess.run, now=None):
    """Serialize microphone ownership even when called without the CLI."""
    conf = validate_config(config)
    with retention._lock(
        Path(conf["pending_dir"]), conf["lock_timeout_seconds"], ".capture.lock"
    ):
        return _capture_once(conf, runner, now)


def _capture_once(config, runner=subprocess.run, now=None):
    """Publish capture receipt only after arecord and complete PCM validation."""
    conf = validate_config(config)
    if not conf.get("recording_device"):
        raise ValueError("Set recording_device before capture")
    _capacity(conf)
    root = Path(conf["pending_dir"])
    stamp = now or datetime.datetime.now(ZoneInfo(conf["timezone"]))
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Recording timestamp needs an explicit UTC offset")
    recording_id = (
        conf["device_id"]
        + "_"
        + stamp.strftime("%Y%m%dT%H%M%S%f%z")
        + "_"
        + uuid.uuid4().hex[:12]
    )
    wav = root / (recording_id + ".wav")
    partial = root / (recording_id + ".wav.incomplete")
    for candidate in (partial, wav, wav.with_name(wav.name + ".capture.json")):
        retention._safe_path(candidate)
        if candidate.exists():
            raise ValueError("Capture identity collision; existing files retained")
    result = runner(
        [
            conf["arecord"],
            "-D",
            conf["recording_device"],
            "-f",
            "S16_LE",
            "-r",
            "48000",
            "-c",
            "1",
            "-t",
            "wav",
            "-d",
            "30",
            str(partial),
        ],
        check=False,
        capture_output=True,
        timeout=conf["capture_timeout_seconds"],
    )
    if result.returncode != 0:
        raise RuntimeError(
            "arecord failed; incomplete audio retained (exit %s)" % result.returncode
        )
    audio_info = _capture_audio_info(partial)
    with open(partial, "r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(partial, wav)
    retention._fsync_dir(root)
    receipt = {
        "schema_version": 1,
        "capture_completed": True,
        "recording_id": recording_id,
        "device_id": conf["device_id"],
        "recorded_at": stamp.isoformat(),
        "timezone": conf["timezone"],
        "recording_device": conf["recording_device"],
        "wav": wav.name,
        "wav_sha256": retention.sha256(wav),
        "audio": audio_info,
        "requested_duration_seconds": 30,
        "clock_source": "device wall clock; not retrospectively corrected",
    }
    marker = root / (recording_id + ".wav.capture.json")
    _json_atomic(marker, receipt)
    return marker


def _capture_metadata(path, conf):
    path = retention._safe_path(path, exists=True)
    if path.parent != Path(conf["pending_dir"]).absolute():
        raise ValueError("Capture receipt is outside pending_dir")
    if path.stat().st_size > 64 * 1024:
        raise ValueError("Oversized capture receipt")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(receipt, dict)
        or type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or receipt.get("capture_completed") is not True
    ):
        raise ValueError("Unknown or unfinished capture receipt")
    wav_name = retention._basename(receipt.get("wav"))
    if path.name != wav_name + ".capture.json":
        raise ValueError("Capture receipt name mismatch")
    stamp = datetime.datetime.fromisoformat(receipt["recorded_at"])
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Capture timestamp is not timezone aware")
    if not isinstance(receipt.get("recording_id"), str) or not receipt["recording_id"]:
        raise ValueError("Missing recording identity")
    wav = path.parent / wav_name
    digest = receipt.get("wav_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Missing or invalid capture WAV SHA-256")
    return receipt, wav


def _capture_checked(path, conf):
    receipt, wav = _capture_metadata(path, conf)
    _capture_audio_info(wav)
    if retention.sha256(wav) != receipt.get("wav_sha256"):
        raise ValueError("Captured WAV has changed")
    return receipt, wav


def _analysis_contract(conf):
    """Bind every parameter that affects historical inference or CSV filtering."""
    return {
        "configuration": {
            key: conf[key]
            for key in (
                "latitude",
                "longitude",
                "sensitivity",
                "overlap",
                "privacy_threshold",
                "threshold",
                "comparison",
                "scientific_name",
                "common_name",
                "model",
            )
        },
        "model_sha256": MODEL_SHA256,
        "labels_sha256": LABELS_SHA256,
        "source_sha256": SOURCE_SHA256,
    }


def _ready_bound(path, conf):
    """Bind ready to immutable capture provenance even after source deletion.

    The engine verifies a saved archive commitment before accepting a missing
    WAV on retry. No WAV read is needed merely to bind these two receipts.
    """
    receipt, wav = _capture_metadata(path, conf)
    ready = wav.with_name(wav.name + ".ready.json")
    marker = retention._load_marker(retention._safe_path(ready, exists=True))
    if any(
        marker.get(key) != receipt[key] for key in ("recording_id", "wav", "wav_sha256")
    ):
        raise ValueError("Completion marker does not belong to this capture")
    if marker.get("csv") != wav.name + ".csv":
        raise ValueError("Completion CSV does not belong to this capture")
    context = marker.get("context")
    if not isinstance(context, dict) or context.get("capture") != receipt:
        raise ValueError("Completion capture provenance differs from capture receipt")
    analysis = context.get("analysis")
    if (
        not isinstance(analysis, dict)
        or analysis.get("capture_sha256") != retention.sha256(path)
        or analysis.get("recording_id") != receipt["recording_id"]
        or analysis.get("recorded_at") != receipt["recorded_at"]
    ):
        raise ValueError("Completion analysis does not bind this capture receipt")
    if analysis.get("schema_version") != 2 or analysis.get(
        "inference_contract"
    ) != _analysis_contract(conf):
        raise ValueError(
            "Completed analysis uses a different or missing inference contract; "
            "keep prior outputs and use a separate pending directory for reanalysis"
        )
    return ready


def _cache_fingerprint(capture, conf):
    ready = capture.with_name(capture.name[: -len(".capture.json")] + ".ready.json")
    if not ready.exists():
        return None
    for path in (capture, ready):
        retention._safe_path(path, exists=True)
        if path.stat().st_size > 64 * 1024:
            raise ValueError("Oversized capture/completion receipt")
    configuration = json.dumps(
        conf, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return (
        retention.sha256(capture),
        retention.sha256(ready),
        hashlib.sha256(configuration).hexdigest(),
    )


def analyze_capture(path, config, bridge):
    """Serialize analysis and marker publication across processes and API calls."""
    conf = validate_config(config)
    with retention._lock(
        Path(conf["pending_dir"]), conf["lock_timeout_seconds"], ".analysis.lock"
    ):
        return _analyze_capture(path, conf, bridge)


def _analyze_capture(path, config, bridge):
    """Never infer readiness from an unmarked WAV or an existing CSV."""
    conf = validate_config(config)
    receipt, wav = _capture_metadata(path, conf)
    csv = wav.with_name(wav.name + ".csv")
    ready = wav.with_name(wav.name + ".ready.json")
    if ready.exists():
        return retention.process_marker(_ready_bound(path, conf), conf)
    receipt, wav = _capture_checked(path, conf)
    # Any uncommitted previous CSV is preserved as a diagnostic. A failed run
    # can be retried, but never silently overwrite unknown source material.
    temp = wav.with_name(wav.name + ".csv.incomplete-" + uuid.uuid4().hex)
    model_info = bridge.analyze(wav, temp, receipt["recorded_at"])
    with open(temp, "r+b") as handle:
        os.fsync(handle.fileno())
    retention.parse_detections(temp, retention.prepare_config(conf))
    if retention.sha256(wav) != receipt["wav_sha256"]:
        raise ValueError("WAV changed during inference; no completion marker published")
    if csv.exists():
        retention._safe_path(csv, exists=True)
        if csv.read_bytes() != temp.read_bytes():
            raise ValueError(
                "Conflicting uncommitted CSV; retained both versions for review"
            )
        temp.unlink()
    else:
        os.replace(temp, csv)
        retention._fsync_dir(csv.parent)
    analysis_info = {
        "schema_version": 2,
        "inference_contract": _analysis_contract(conf),
        "recording_id": receipt["recording_id"],
        "recorded_at": receipt["recorded_at"],
        "capture_sha256": retention.sha256(path),
        "model": model_info,
    }
    _json_atomic(wav.with_name(wav.name + ".analysis.json"), analysis_info)
    marker = retention.publish_completed(
        wav,
        csv,
        receipt["recording_id"],
        context={"capture": receipt, "analysis": analysis_info},
    )
    return retention.process_marker(marker, conf)


def analyze_pending(config, bridge, completed=None):
    conf = validate_config(config)
    root = retention._safe_path(conf["pending_dir"], directory=True)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for capture in sorted(root.glob("*.capture.json")):
        # Cache only for this run. A restart/standalone command verifies saved
        # commitments again; a long-running session need not repeatedly decode
        # every already archived MP3 on each one-second scan.
        try:
            fingerprint = (
                _cache_fingerprint(capture, conf) if completed is not None else None
            )
            if (
                completed is not None
                and fingerprint is not None
                and completed.get(capture.name) == fingerprint
            ):
                continue
            result = analyze_capture(capture, conf, bridge)
            results.append({"capture": capture.name, **result})
            if completed is not None and result["status"] in (
                "archived",
                "already_archived",
                "negative_retained",
                "negative_quarantined",
                "already_quarantined",
            ):
                completed[capture.name] = _cache_fingerprint(capture, conf)
        except Exception as exc:
            results.append(
                {"capture": capture.name, "status": "error", "error": str(exc)}
            )
    return results


@contextlib.contextmanager
def pipeline_lock(directory):
    """One owning pipeline CLI process; crash releases the OS file lock."""
    root = retention._safe_path(directory, directory=True)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".pipeline.lock"
    retention._safe_path(lock_path)
    with open(lock_path, "a+b") as handle:
        handle.seek(0)
        handle.write(b"0")
        handle.flush()
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run(config, bridge):
    stop = threading.Event()
    errors = []
    completed = {}

    def capture_loop():
        try:
            while not stop.is_set():
                marker = capture_once(config)
                print(json.dumps({"capture_completed": marker.name}), flush=True)
        except Exception as exc:
            errors.append(str(exc))
            stop.set()

    thread = threading.Thread(target=capture_loop, name="serial-capture", daemon=False)
    thread.start()
    try:
        while not stop.is_set():
            results = analyze_pending(config, bridge, completed)
            for result in results:
                if result["status"] not in (
                    "already_archived",
                    "already_quarantined",
                    "negative_retained",
                ):
                    print(json.dumps(result, ensure_ascii=False), flush=True)
            if any(item["status"] in ("error", "waiting") for item in results):
                errors.append("Analysis/retention needs attention; capture is stopping")
                stop.set()
            stop.wait(1)
    finally:
        stop.set()
        # An in-flight 30 s capture finishes and publishes its receipt. Never
        # interrupt arecord then mistake its partial output for complete audio.
        thread.join(config["capture_timeout_seconds"] + 5)
    if errors:
        raise RuntimeError("; ".join(errors))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("mode", choices=("capture-once", "analyze-pending", "run"))
    args = parser.parse_args(argv)
    try:
        conf = validate_config(
            json.loads(Path(args.config).read_text(encoding="utf-8")),
            require_runtime=args.mode != "capture-once",
        )
        with pipeline_lock(conf["pending_dir"]):
            if args.mode == "capture-once":
                print(capture_once(conf))
                return 0
            bridge = ModelBridge(conf)
            if args.mode == "run":
                run(conf, bridge)
                return 0
            results = analyze_pending(conf, bridge)
            print(json.dumps(results, ensure_ascii=False, indent=2))
            return (
                1
                if any(item["status"] in ("error", "waiting") for item in results)
                else 0
            )
    except (Exception, KeyboardInterrupt) as exc:
        print("Pipeline stopped; inputs retained: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
