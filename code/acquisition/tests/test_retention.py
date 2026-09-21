# SPDX-License-Identifier: MIT
"""Temporary-directory safety/retention tests; never accesses field recordings.

Run: python -m unittest discover -s tests -p test_retention.py -v
Mocked codec tests isolate failure paths. RealCodecTests separately requires
local FFmpeg/FFprobe and asserts full decode-frame preservation and endpoints.
"""

import array
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import retention as r


HEADER = ";".join(r.HEADER) + "\n"
TARGET = "0;3;Passer montanus;麻雀;0.2\n"
OTHER = "0;3;Parus major;大山雀;0.9\n"


def create_wav(path, *, seconds=30, rate=48000, channels=1, width=2, endpoints=False):
    frames = seconds * rate
    if endpoints:
        samples = array.array("h", [0]) * frames
        for start, end in ((0, rate // 2), (frames - rate // 2, frames)):
            for index in range(start, end):
                samples[index] = int(
                    12000 * math.sin(2 * math.pi * 4000 * index / rate)
                )
        if sys.byteorder != "little":
            samples.byteswap()
        payload = samples.tobytes()
    else:
        payload = b"\0" * frames * channels * width
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(payload)


class TemporaryCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="birdnet retention tests ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pending = self.root / "pending recordings"
        self.pending.mkdir()
        self.archive = self.root / "archive recordings"
        self.quarantine = self.root / "quarantine recordings"
        self.config = {
            "pending_dir": str(self.pending),
            "archive_dir": str(self.archive),
            "quarantine_dir": str(self.quarantine),
            "min_free_bytes": 0,
            "capacity_buffer_bytes": 0,
        }
        self.transcodes = []

    def inputs(
        self, rows=TARGET, name="song with spaces # & $ 鸟", raw_csv=None, **wav_options
    ):
        wav = self.pending / (name + ".wav")
        csv = self.pending / (name + ".wav.csv")
        create_wav(wav, **wav_options)
        csv.write_bytes(
            raw_csv if raw_csv is not None else (HEADER + rows).encode("utf-8")
        )
        marker = r.publish_completed(wav, csv, "site-A/session-001/" + name)
        return wav, csv, marker

    def fake_command(self, args, conf):
        if args[0] == conf["ffprobe"]:
            return json.dumps(
                {
                    "streams": [
                        {
                            "codec_name": "mp3",
                            "sample_rate": "48000",
                            "channels": 1,
                            "bit_rate": "320000",
                        }
                    ],
                    "format": {"duration": "30.024"},
                }
            ).encode()
        if args[-1] == "-":
            return b"\0" * r.FRAMES * 2
        self.transcodes.append(args)
        Path(args[-1]).write_bytes(b"test MP3 stub; real decoding tested separately")
        return b""

    def finals(self, directory=None):
        directory = directory or self.archive
        return (
            [
                p
                for p in directory.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            ]
            if directory.exists()
            else []
        )

    def assert_preserved(self, wav, csv):
        self.assertTrue(wav.is_file())
        self.assertTrue(csv.is_file())

    def run_fake(self, marker, **overrides):
        with mock.patch.object(r, "_command", side_effect=self.fake_command):
            return r.process_marker(marker, {**self.config, **overrides})


class DetectionTests(TemporaryCase):
    def test_threshold_boundaries_and_both_operators(self):
        for comparison in ("gte", "gt"):
            for value in (0.199999, 0.2, 0.200001):
                with self.subTest(comparison=comparison, value=value):
                    wav, csv, marker = self.inputs(
                        rows=TARGET.replace("0.2\n", str(value) + "\n"),
                        name=comparison + str(value),
                    )
                    result = self.run_fake(marker, comparison=comparison)
                    expected = value >= 0.2 if comparison == "gte" else value > 0.2
                    self.assertEqual(
                        result["status"],
                        "archived" if expected else "negative_retained",
                    )
                    self.assert_preserved(wav, csv)

    def test_header_only_valid_negative(self):
        wav, csv, marker = self.inputs(rows="")
        self.assertEqual(self.run_fake(marker)["status"], "negative_retained")
        self.assertEqual(self.finals(), [])
        self.assert_preserved(wav, csv)

    def test_exact_60_bytes_is_parsed_not_guessed(self):
        valid = b"\xef\xbb\xbf" + HEADER.encode()
        self.assertEqual(len(valid), 60)
        wav, csv, marker = self.inputs(raw_csv=valid, name="valid sixty")
        self.assertEqual(self.run_fake(marker)["status"], "negative_retained")
        wav2, csv2, marker2 = self.inputs(raw_csv=b"x" * 60, name="invalid sixty")
        self.assertEqual(self.run_fake(marker2)["status"], "error")
        self.assert_preserved(wav, csv)
        self.assert_preserved(wav2, csv2)

    def test_other_species_and_exact_label_mapping(self):
        for index, row in enumerate(
            (
                OTHER,
                TARGET.replace("麻雀", "Eurasian Tree Sparrow"),
                TARGET.replace("Passer montanus", " Passer montanus"),
            )
        ):
            with self.subTest(row=row):
                _, _, marker = self.inputs(rows=row, name=str(index))
                self.assertEqual(self.run_fake(marker)["status"], "negative_retained")

    def test_many_hits_single_archive_preserves_csv(self):
        original = (
            HEADER + TARGET + "1.5;4.5;Passer montanus;麻雀;0.95\n" + OTHER
        ).encode()
        wav, csv, marker = self.inputs(raw_csv=original)
        result = self.run_fake(marker)
        self.assertEqual(result["status"], "archived")
        self.assertEqual(result["qualified_target_rows"], 2)
        self.assertEqual(len(self.finals()), 1)
        self.assertEqual((self.finals()[0] / "detections.csv").read_bytes(), original)
        self.assertEqual(len(self.transcodes), 1)
        self.assert_preserved(wav, csv)

    def test_malformed_truncated_unknown_and_nonfinite_are_errors(self):
        cases = [
            b"",
            HEADER.replace(";", ",").encode(),
            (HEADER + "0;3;Passer montanus\n").encode(),
            (HEADER + TARGET + "bad\n").encode(),
            (HEADER + '"unterminated').encode(),
            (HEADER + "\n").encode(),
            HEADER.encode() + b"\xff",
        ]
        for value in ("NaN", "inf", "-inf", "1.1", "-0.01", "bad"):
            cases.append((HEADER + TARGET.replace("0.2\n", value + "\n")).encode())
        for index, content in enumerate(cases):
            with self.subTest(index=index):
                wav, csv, marker = self.inputs(raw_csv=content, name=str(index))
                self.assertEqual(self.run_fake(marker)["status"], "error")
                self.assert_preserved(wav, csv)
        self.assertEqual(self.finals(), [])

    def test_padded_final_window_keeps_nominal_audio_duration(self):
        _, _, marker = self.inputs(rows="28.5;31.5;Passer montanus;麻雀;0.8\n")
        result = self.run_fake(marker)
        receipt = json.loads((Path(result["archive"]) / "receipt.json").read_text())
        self.assertEqual(receipt["input"]["frames"], 1440000)
        self.assertEqual(receipt["output"]["decoded_frames"], 1440000)
        args = self.transcodes[0]
        for forbidden in ("-ss", "-t", "-to", "-af", "-ar", "-filter_complex"):
            self.assertNotIn(forbidden, args)
        self.assertEqual(args[args.index("-ac") + 1], "1")

    def test_invalid_windows_remain_unknown(self):
        for index, window in enumerate(
            ("-1;2", "30;33", "0;4", "29;33.1", "3;2", "NaN;3")
        ):
            wav, csv, marker = self.inputs(
                rows=window + ";Passer montanus;麻雀;0.8\n", name=str(index)
            )
            self.assertEqual(self.run_fake(marker)["status"], "error")
            self.assert_preserved(wav, csv)


class InputAndPolicyTests(TemporaryCase):
    def test_without_completion_marker_nothing_happens(self):
        wav, csv, marker = self.inputs()
        marker.unlink()
        self.assertEqual(r.process_pending(self.config), [])
        self.assert_preserved(wav, csv)

    def test_incomplete_marker_and_writing_csv_wait(self):
        wav, csv, marker = self.inputs()
        value = json.loads(marker.read_text())
        value["completed"] = False
        marker.write_text(json.dumps(value))
        self.assertEqual(self.run_fake(marker)["status"], "waiting")
        value["completed"] = True
        marker.write_text(json.dumps(value))
        csv.write_text(HEADER + TARGET + "0;3;", encoding="utf-8")
        self.assertEqual(self.run_fake(marker)["status"], "waiting")
        self.assert_preserved(wav, csv)
        self.assertEqual(self.finals(), [])

    def test_missing_wav_is_waiting(self):
        wav, csv, marker = self.inputs()
        wav.unlink()
        self.assertEqual(self.run_fake(marker)["status"], "waiting")
        self.assertTrue(csv.exists())

    def test_missing_csv_and_marker_are_waiting(self):
        wav, csv, marker = self.inputs()
        csv.unlink()
        self.assertEqual(self.run_fake(marker)["status"], "waiting")
        marker.unlink()
        self.assertEqual(self.run_fake(marker)["status"], "waiting")
        self.assertTrue(wav.exists())

    def test_marker_schema_hash_and_nonboolean_completion_are_rejected(self):
        wav, csv, marker = self.inputs()
        original = json.loads(marker.read_text())
        for updates in (
            {"schema_version": True},
            {"schema_version": 9},
            {"wav_sha256": ""},
            {"csv_sha256": "G" * 64},
            {"recording_id": ""},
            {"wav": "a:b.wav"},
        ):
            marker.write_text(json.dumps({**original, **updates}))
            self.assertEqual(self.run_fake(marker)["status"], "error")
        marker.write_text(json.dumps({**original, "completed": "true"}))
        self.assertEqual(self.run_fake(marker)["status"], "waiting")
        self.assert_preserved(wav, csv)

    def test_invalid_configuration_cannot_touch_inputs(self):
        wav, csv, marker = self.inputs()
        for updates in (
            {"threshold": "NaN"},
            {"threshold": True},
            {"threshold": -1},
            {"comparison": ">="},
            {"negative_policy": "delete"},
            {"source_wav_policy": "delete"},
            {"capacity_buffer_bytes": -1},
            {"command_timeout_seconds": 0},
            {"scientific_name": ""},
        ):
            with self.subTest(updates=updates):
                self.assertEqual(self.run_fake(marker, **updates)["status"], "error")
        self.assert_preserved(wav, csv)
        self.assertEqual(self.finals(), [])

    def test_invalid_audio_profile_is_error_even_for_negative(self):
        for index, options in enumerate(
            (
                {"seconds": 3},
                {"seconds": 15},
                {"rate": 44100},
                {"channels": 2},
                {"width": 1},
            )
        ):
            wav, csv, marker = self.inputs(rows="", name=str(index), **options)
            self.assertEqual(self.run_fake(marker)["status"], "error")
            self.assert_preserved(wav, csv)

    def test_truncated_wav_rejected(self):
        wav, csv, marker = self.inputs()
        marker.unlink()
        wav.write_bytes(wav.read_bytes()[:-20])
        marker = r.publish_completed(wav, csv, "truncated")
        self.assertEqual(self.run_fake(marker)["status"], "error")
        self.assert_preserved(wav, csv)

    def test_negative_quarantine_is_copy_only_and_idempotent(self):
        wav, csv, marker = self.inputs(rows=OTHER)
        first = self.run_fake(
            marker,
            negative_policy="quarantine",
            source_wav_policy="delete_after_commit",
        )
        second = self.run_fake(marker, negative_policy="quarantine")
        self.assertEqual(first["status"], "negative_quarantined")
        self.assertEqual(second["status"], "already_quarantined")
        self.assertEqual(len(self.finals(self.quarantine)), 1)
        self.assertEqual(
            r.sha256(self.finals(self.quarantine)[0] / "source.wav"), r.sha256(wav)
        )
        self.assert_preserved(wav, csv)

    def test_source_delete_only_after_committed_verified_positive(self):
        wav, csv, marker = self.inputs()
        result = self.run_fake(marker, source_wav_policy="delete_after_commit")
        self.assertEqual(result["status"], "archived")
        self.assertTrue(result["source_wav_deleted"])
        self.assertFalse(wav.exists())
        self.assertTrue(csv.exists())
        again = self.run_fake(marker, source_wav_policy="delete_after_commit")
        self.assertEqual(again["status"], "already_archived")
        self.assertEqual(len(self.transcodes), 1)

    def test_changed_source_or_rule_prevents_later_deletion(self):
        wav, csv, marker = self.inputs()
        self.assertEqual(self.run_fake(marker)["status"], "archived")
        self.assertEqual(
            self.run_fake(
                marker, threshold=0.9, source_wav_policy="delete_after_commit"
            )["status"],
            "error",
        )
        self.assertTrue(wav.exists())
        wav.write_bytes(wav.read_bytes() + b"changed")
        self.assertEqual(
            self.run_fake(marker, source_wav_policy="delete_after_commit")["status"],
            "error",
        )
        self.assert_preserved(wav, csv)

    def test_changed_source_csv_prevents_later_deletion(self):
        wav, csv, marker = self.inputs()
        self.run_fake(marker)
        csv.write_text(HEADER + OTHER, encoding="utf-8")
        self.assertEqual(
            self.run_fake(marker, source_wav_policy="delete_after_commit")["status"],
            "error",
        )
        self.assert_preserved(wav, csv)

    def test_changed_completed_inputs_are_not_reclassified_negative(self):
        wav, csv, marker = self.inputs()
        csv.write_text(HEADER, encoding="utf-8")
        self.assertEqual(self.run_fake(marker)["status"], "waiting")
        self.assert_preserved(wav, csv)


class FailureAndRecoveryTests(TemporaryCase):
    def test_ffmpeg_missing_and_nonzero_exit(self):
        wav, csv, marker = self.inputs()
        result = r.process_marker(
            marker, {**self.config, "ffmpeg": str(self.root / "missing-ffmpeg")}
        )
        self.assertEqual(result["status"], "error")
        self.assert_preserved(wav, csv)
        failed = subprocess.CompletedProcess([], 2, b"", b"intentional codec failure")
        with mock.patch.object(r.subprocess, "run", return_value=failed):
            result = r.process_marker(marker, self.config)
        self.assertEqual(result["status"], "error")
        self.assertEqual(self.finals(), [])

    def test_command_timeout_preserves_sources(self):
        wav, csv, marker = self.inputs()
        with mock.patch.object(
            r.subprocess, "run", side_effect=subprocess.TimeoutExpired("ffmpeg", 1)
        ):
            result = r.process_marker(marker, self.config)
        self.assertEqual(result["status"], "error")
        self.assert_preserved(wav, csv)
        self.assertEqual(self.finals(), [])

    def test_empty_unreadable_and_wrong_decoded_length_fail_closed(self):
        for mode in (
            "empty",
            "unreadable",
            "short_decode",
            "wrong_rate",
            "wrong_codec",
            "wrong_bitrate",
            "stereo",
        ):
            with self.subTest(mode=mode):
                wav, csv, marker = self.inputs(name=mode)

                def command(args, conf):
                    if args[0] == conf["ffprobe"]:
                        if mode == "unreadable":
                            raise r.RetentionError("invalid MP3")
                        value = json.loads(self.fake_command(args, conf))
                        if mode == "wrong_rate":
                            value["streams"][0]["sample_rate"] = "44100"
                        if mode == "wrong_codec":
                            value["streams"][0]["codec_name"] = "aac"
                        if mode == "wrong_bitrate":
                            value["streams"][0]["bit_rate"] = "64000"
                        if mode == "stereo":
                            value["streams"][0]["channels"] = 2
                        return json.dumps(value).encode()
                    if mode == "short_decode" and args[-1] == "-":
                        return b"\0" * 3 * 48000 * 2
                    result = self.fake_command(args, conf)
                    if mode == "empty" and args[-1] != "-":
                        Path(args[-1]).write_bytes(b"")
                    return result

                with mock.patch.object(r, "_command", side_effect=command):
                    result = r.process_marker(
                        marker,
                        {**self.config, "source_wav_policy": "delete_after_commit"},
                    )
                self.assertEqual(result["status"], "error")
                self.assert_preserved(wav, csv)
        self.assertEqual(self.finals(), [])

    def test_unwritable_destination(self):
        wav, csv, marker = self.inputs()
        self.archive.mkdir()
        with mock.patch.object(
            r,
            "_copy_verified",
            side_effect=PermissionError("simulated read-only target"),
        ):
            result = self.run_fake(marker)
        self.assertEqual(result["status"], "error")
        self.assert_preserved(wav, csv)
        self.assertEqual(self.finals(), [])

    def test_capacity_boundary_waits_without_deleting(self):
        wav, csv, marker = self.inputs()
        needed = wav.stat().st_size + csv.stat().st_size + 2 * 1200000 + 65536
        disk = shutil.disk_usage(self.pending)
        with mock.patch.object(
            r.shutil,
            "disk_usage",
            return_value=type(disk)(needed * 2, needed + 1, needed - 1),
        ):
            self.assertEqual(self.run_fake(marker)["status"], "waiting")
        self.assert_preserved(wav, csv)
        self.assertEqual(self.finals(), [])
        with mock.patch.object(
            r.shutil, "disk_usage", return_value=type(disk)(needed * 2, needed, needed)
        ):
            self.assertEqual(self.run_fake(marker)["status"], "archived")

    def test_interruption_partial_stage_retries_without_false_final(self):
        wav, csv, marker = self.inputs()

        def interrupted(args, conf):
            Path(args[-1]).write_bytes(b"partial")
            raise KeyboardInterrupt()

        with mock.patch.object(r, "_command", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                r.process_marker(marker, self.config)
        self.assertEqual(self.finals(), [])
        self.assert_preserved(wav, csv)
        result = self.run_fake(marker)
        self.assertEqual(result["status"], "archived")
        self.assertFalse(result["recovered_stage"])

    def test_verified_stage_recovered_after_precommit_termination(self):
        wav, csv, marker = self.inputs()
        with mock.patch.object(r.os, "rename", side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                self.run_fake(marker)
        self.assertEqual(self.finals(), [])
        self.assert_preserved(wav, csv)
        result = self.run_fake(marker)
        self.assertTrue(result["recovered_stage"])
        self.assertEqual(len(self.transcodes), 1)

    def test_committed_output_survives_postcommit_interruption(self):
        wav, csv, marker = self.inputs()
        with mock.patch.object(
            r, "_delete_source_if_requested", side_effect=KeyboardInterrupt()
        ):
            with self.assertRaises(KeyboardInterrupt):
                self.run_fake(marker, source_wav_policy="delete_after_commit")
        self.assertEqual(len(self.finals()), 1)
        self.assert_preserved(wav, csv)
        again = self.run_fake(marker, source_wav_policy="delete_after_commit")
        self.assertEqual(again["status"], "already_archived")
        self.assertFalse(wav.exists())
        self.assertEqual(len(self.transcodes), 1)

    def test_committed_corruption_prevents_source_deletion(self):
        wav, csv, marker = self.inputs()
        self.run_fake(marker)
        (self.finals()[0] / "audio.mp3").write_bytes(b"tampered")
        result = self.run_fake(marker, source_wav_policy="delete_after_commit")
        self.assertEqual(result["status"], "error")
        self.assert_preserved(wav, csv)

    def test_context_preserved_and_tampering_blocks_source_deletion(self):
        wav, csv, marker = self.inputs()
        marker.unlink()
        context = {
            "recorded_at_utc": "2026-09-21T01:23:45Z",
            "device_id": "recorder-01",
            "model_sha256": "f" * 64,
            "analysis": {"overlap_seconds": 1.5, "sensitivity": 1.25},
        }
        marker = r.publish_completed(wav, csv, "context-example", context=context)
        result = self.run_fake(marker)
        self.assertEqual(result["status"], "archived")
        completion_path = Path(result["archive"]) / "completion.json"
        saved = json.loads(completion_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["context"], context)
        receipt = json.loads((Path(result["archive"]) / "receipt.json").read_text())
        self.assertEqual(receipt["completion_sha256"], r.sha256(completion_path))
        saved["context"]["device_id"] = "wrong-recorder"
        completion_path.write_text(json.dumps(saved), encoding="utf-8")
        self.assertEqual(
            self.run_fake(marker, source_wav_policy="delete_after_commit")["status"],
            "error",
        )
        self.assert_preserved(wav, csv)

    def test_changed_pending_context_cannot_reinterpret_committed_identity(self):
        wav, csv, marker = self.inputs()
        result = self.run_fake(marker)
        self.assertEqual(result["status"], "archived")
        changed = json.loads(marker.read_text())
        changed["context"] = {"device_id": "changed-after-commit"}
        marker.write_text(json.dumps(changed), encoding="utf-8")
        self.assertEqual(self.run_fake(marker)["status"], "error")
        self.assert_preserved(wav, csv)


class IdentityAndPathTests(TemporaryCase):
    def test_copied_ready_marker_cannot_impersonate_another_recording(self):
        wav_a, csv_a, marker_a = self.inputs(name="recording A")
        wav_b, csv_b, marker_b = self.inputs(name="recording B")
        before = {path: path.read_bytes() for path in (wav_a, csv_a, wav_b, csv_b)}
        marker_b.write_bytes(marker_a.read_bytes())
        result = self.run_fake(marker_b, source_wav_policy="delete_after_commit")
        self.assertEqual(result["status"], "error")
        self.assertIn("basename", result["message"])
        self.assertEqual(self.finals(), [])
        self.assertEqual(len(self.transcodes), 0)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        self.assertEqual(self.run_fake(marker_a)["status"], "archived")

    def test_repeated_and_concurrent_workers_transcode_once(self):
        wav, csv, marker = self.inputs()
        original = self.fake_command

        def slow(args, conf):
            if args[0] == conf["ffmpeg"] and args[-1] != "-":
                time.sleep(0.1)
            return original(args, conf)

        with mock.patch.object(r, "_command", side_effect=slow):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(
                    pool.map(lambda _: r.process_marker(marker, self.config), range(2))
                )
        self.assertEqual(
            {x["status"] for x in results}, {"archived", "already_archived"}
        )
        self.assertEqual(len(self.transcodes), 1)
        self.assertEqual(self.run_fake(marker)["status"], "already_archived")
        self.assert_preserved(wav, csv)

    def test_same_basename_different_identity_never_overwrites(self):
        wav, csv, marker = self.inputs()
        first = self.run_fake(marker)
        first_bytes = (Path(first["archive"]) / "detections.csv").read_bytes()
        marker.unlink()
        csv.write_text(HEADER + TARGET.replace("0.2", "0.8"), encoding="utf-8")
        marker = r.publish_completed(wav, csv, "new recording identity")
        second = self.run_fake(marker)
        self.assertNotEqual(first["archive"], second["archive"])
        self.assertEqual(len(self.finals()), 2)
        self.assertEqual(
            (Path(first["archive"]) / "detections.csv").read_bytes(), first_bytes
        )

    def test_existing_conflicting_final_refused(self):
        wav, csv, marker = self.inputs()
        key, _ = r._identity(json.loads(marker.read_text()))
        final = self.archive / key
        final.mkdir(parents=True)
        (final / "receipt.json").write_text('{"identity": "other"}')
        result = self.run_fake(marker)
        self.assertEqual(result["status"], "error")
        self.assertEqual((final / "receipt.json").read_text(), '{"identity": "other"}')
        self.assert_preserved(wav, csv)

    def test_directory_boundaries_and_traversal(self):
        wav, csv, marker = self.inputs()
        for archive in (self.pending, self.pending / "nested", self.root):
            self.assertEqual(
                self.run_fake(marker, archive_dir=str(archive))["status"], "error"
            )
        value = json.loads(marker.read_text())
        value["wav"] = "../outside.wav"
        marker.write_text(json.dumps(value))
        self.assertEqual(self.run_fake(marker)["status"], "error")
        self.assert_preserved(wav, csv)

    def test_symlink_input_refused(self):
        wav, csv, marker = self.inputs()
        original = self.root / "unrelated.wav"
        wav.rename(original)
        try:
            wav.symlink_to(original)
        except OSError as exc:
            self.skipTest("Host cannot create symlinks: " + str(exc))
        result = self.run_fake(marker, source_wav_policy="delete_after_commit")
        self.assertEqual(result["status"], "error")
        self.assertTrue(original.exists())
        self.assertEqual(self.finals(), [])

    def test_symlink_rejection_branch_without_host_privileges(self):
        wav, csv, marker = self.inputs()
        actual_is_symlink = Path.is_symlink

        def simulated_symlink(path):
            return path == wav or actual_is_symlink(path)

        with mock.patch.object(Path, "is_symlink", simulated_symlink):
            result = self.run_fake(marker, source_wav_policy="delete_after_commit")
        self.assertEqual(result["status"], "error")
        self.assert_preserved(wav, csv)

    def test_symlinked_archive_parent_refused(self):
        wav, csv, marker = self.inputs()
        actual_is_symlink = Path.is_symlink

        def simulated_symlink(path):
            return path == self.archive or actual_is_symlink(path)

        with mock.patch.object(Path, "is_symlink", simulated_symlink):
            self.assertEqual(self.run_fake(marker)["status"], "error")
        self.assert_preserved(wav, csv)

    def test_producer_republication_of_identical_completion_is_idempotent(self):
        wav, csv, marker = self.inputs()
        before = marker.read_bytes()
        recording_id = json.loads(before)["recording_id"]
        self.assertEqual(r.publish_completed(wav, csv, recording_id), marker)
        self.assertEqual(marker.read_bytes(), before)

    def test_unrelated_small_files_remain_untouched(self):
        wav, csv, marker = self.inputs(rows="")
        unrelated = self.pending / "unrelated small note"
        unrelated.write_bytes(b"hi")
        self.run_fake(marker)
        self.assertEqual(unrelated.read_bytes(), b"hi")
        self.assert_preserved(wav, csv)

    def test_producer_conflicting_marker_never_overwritten(self):
        wav, csv, marker = self.inputs()
        before = marker.read_bytes()
        csv.write_text(HEADER + OTHER, encoding="utf-8")
        with self.assertRaises(r.RetentionError):
            r.publish_completed(wav, csv, "different")
        self.assertEqual(marker.read_bytes(), before)

    def test_lock_timeout_is_waiting(self):
        wav, csv, marker = self.inputs()
        with r._lock(self.archive, 1):
            result = self.run_fake(marker, lock_timeout_seconds=0.1)
        self.assertEqual(result["status"], "waiting")
        self.assert_preserved(wav, csv)

    def test_unknown_staging_content_never_removed(self):
        wav, csv, marker = self.inputs()
        key, _ = r._identity(json.loads(marker.read_text()))
        stage = self.archive / ".staging" / key
        stage.mkdir(parents=True)
        unknown = stage / "personal.txt"
        unknown.write_text("preserve")
        self.assertEqual(self.run_fake(marker)["status"], "error")
        self.assertEqual(unknown.read_text(), "preserve")
        self.assert_preserved(wav, csv)


class AdditionalSafetyTests(TemporaryCase):
    def test_shared_quarantine_serializes_distinct_archive_configurations(self):
        wav, csv, marker = self.inputs(rows="")
        configs = [
            {
                **self.config,
                "archive_dir": str(self.root / ("archive-" + str(i))),
                "negative_policy": "quarantine",
            }
            for i in range(3)
        ]
        with ThreadPoolExecutor(max_workers=3) as workers:
            results = list(
                workers.map(lambda conf: r.process_marker(marker, conf), configs)
            )
        self.assertEqual(
            sum(item["status"] == "negative_quarantined" for item in results), 1
        )
        self.assertEqual(
            sum(item["status"] == "already_quarantined" for item in results), 2
        )
        self.assert_preserved(wav, csv)

    def test_malformed_commit_receipt_is_structured_error_and_keeps_wav(self):
        wav, csv, marker = self.inputs()
        result = self.run_fake(marker)
        receipt = Path(result["archive"]) / "receipt.json"
        receipt.write_text("[]")
        result = self.run_fake(marker, source_wav_policy="delete_after_commit")
        self.assertEqual(result["status"], "error")
        self.assert_preserved(wav, csv)


class RealCodecTests(TemporaryCase):
    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/FFprobe unavailable"
    )
    def test_real_30_second_full_context_encode_decode(self):
        wav, csv, marker = self.inputs(
            endpoints=True, rows="28.5;31.5;Passer montanus;麻雀;0.8\n"
        )
        result = r.process_marker(marker, self.config)
        self.assertEqual(result["status"], "archived", result)
        final = Path(result["archive"])
        receipt = json.loads((final / "receipt.json").read_text())
        self.assertEqual(receipt["input"]["frames"], 1440000)
        self.assertEqual(receipt["output"]["decoded_frames"], 1440000)
        self.assertEqual(receipt["output"]["sample_rate"], 48000)
        self.assertEqual(receipt["output"]["bit_rate"], 320000)
        self.assertEqual((final / "detections.csv").read_bytes(), csv.read_bytes())
        decoded = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(final / "audio.mp3"),
                "-f",
                "s16le",
                "-c:a",
                "pcm_s16le",
                "-",
            ],
            check=True,
            capture_output=True,
        ).stdout
        samples = array.array("h")
        samples.frombytes(decoded)
        if sys.byteorder != "little":
            samples.byteswap()
        self.assertEqual(len(samples), 1440000)
        self.assertGreater(max(abs(x) for x in samples[:24000]), 5000)
        self.assertGreater(max(abs(x) for x in samples[-24000:]), 5000)
        self.assert_preserved(wav, csv)
        self.assertEqual(
            r.process_marker(marker, self.config)["status"], "already_archived"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
