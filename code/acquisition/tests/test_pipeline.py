# SPDX-License-Identifier: MIT
"""Producer/bridge tests with synthetic PCM and mocked microphone/model only."""

import ast
import datetime
from concurrent.futures import ThreadPoolExecutor
import shutil
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_bridge
import pipeline
import retention

HEADER = ";".join(retention.HEADER) + "\n"
STAMP = datetime.datetime(
    2024, 5, 1, 8, 30, tzinfo=datetime.timezone(datetime.timedelta(hours=8))
)


def write_wav(path, frames=pipeline.FRAMES):
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
        handle.writeframes(b"\x00\x00" * frames)


class FakeModel:
    def __init__(self, rows=HEADER):
        self.rows = rows
        self.calls = 0

    def analyze(self, wav, csv, recorded_at):
        self.calls += 1
        Path(csv).write_text(self.rows, encoding="utf-8")
        return {"inference": "MOCK ONLY", "recorded_at": recorded_at}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="retention pipeline ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.conf = {
            "pending_dir": str(self.root / "pending space"),
            "archive_dir": str(self.root / "archive"),
            "quarantine_dir": str(self.root / "negative"),
            "device_id": "test-01",
            "recording_device": "mock",
            "latitude": 36.55,
            "longitude": 104.18,
            "min_free_bytes": 0,
            "capacity_buffer_bytes": 0,
        }
        self.commands = []

    def recorder(self, command, **kwargs):
        self.commands.append(command)
        write_wav(command[-1])
        return subprocess.CompletedProcess(command, 0)

    def capture(self):
        return pipeline.capture_once(self.conf, runner=self.recorder, now=STAMP)

    def test_capture_receipt_after_pcm_validation(self):
        path = self.capture()
        receipt = json.loads(path.read_text(encoding="utf-8"))
        wav = path.parent / receipt["wav"]
        self.assertTrue(receipt["capture_completed"])
        self.assertEqual(receipt["wav_sha256"], retention.sha256(wav))
        self.assertEqual(receipt["audio"]["frames"], 1440000)
        self.assertEqual(receipt["recorded_at"], STAMP.isoformat())
        self.assertEqual(
            self.commands[0][1:-1],
            [
                "-D",
                "mock",
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
            ],
        )
        self.assertFalse(list(path.parent.glob("*.ready.json")))

    def test_capture_nonzero_preserves_partial(self):
        def fail(command, **kwargs):
            write_wav(command[-1], 100)
            return subprocess.CompletedProcess(command, 1)

        with self.assertRaises(RuntimeError):
            pipeline.capture_once(self.conf, runner=fail, now=STAMP)
        pending = Path(self.conf["pending_dir"])
        self.assertEqual(len(list(pending.glob("*.incomplete"))), 1)
        self.assertFalse(list(pending.glob("*.capture.json")))

    def test_short_successful_capture_is_not_complete(self):
        def short(command, **kwargs):
            write_wav(command[-1], 48000)
            return subprocess.CompletedProcess(command, 0)

        with self.assertRaises(ValueError):
            pipeline.capture_once(self.conf, runner=short, now=STAMP)
        self.assertFalse(list(Path(self.conf["pending_dir"]).glob("*.capture.json")))

    def test_capacity_failure_never_invokes_microphone(self):
        with mock.patch.object(
            pipeline.shutil, "disk_usage", return_value=SimpleNamespace(free=0)
        ):
            with self.assertRaises(retention.Waiting):
                self.capture()
        self.assertEqual(self.commands, [])

    def test_complete_negative_and_repeat_do_not_repeat_model(self):
        capture = self.capture()
        model = FakeModel()
        result = pipeline.analyze_capture(capture, self.conf, model)
        self.assertEqual(result["status"], "negative_retained")
        results = pipeline.analyze_pending(self.conf, model)
        self.assertEqual(results[0]["status"], "negative_retained")
        self.assertEqual(model.calls, 1)
        self.assertEqual(len(list(capture.parent.glob("*.ready.json"))), 1)
        self.assertEqual(len(list(capture.parent.glob("*.analysis.json"))), 1)

    def test_positive_keeps_all_rows_and_receipt_identity(self):
        capture = self.capture()
        rows = (
            HEADER
            + "0;3;Passer montanus;麻雀;0.2\n28.5;31.5;Passer montanus;麻雀;0.9\n"
        )
        with mock.patch.object(
            retention, "process_marker", return_value={"status": "archived"}
        ):
            result = pipeline.analyze_capture(capture, self.conf, FakeModel(rows))
        self.assertEqual(result["status"], "archived")
        ready = json.loads(
            next(capture.parent.glob("*.ready.json")).read_text(encoding="utf-8")
        )
        receipt = json.loads(capture.read_text(encoding="utf-8"))
        self.assertEqual(ready["recording_id"], receipt["recording_id"])
        self.assertEqual(
            (capture.parent / ready["csv"]).read_text(encoding="utf-8"), rows
        )
        self.assertEqual(ready["context"]["capture"]["recorded_at"], STAMP.isoformat())
        self.assertEqual(ready["context"]["capture"]["device_id"], "test-01")
        self.assertEqual(
            ready["context"]["analysis"]["inference_contract"]["configuration"][
                "threshold"
            ],
            0.2,
        )
        self.assertEqual(
            ready["context"]["analysis"]["capture_sha256"], retention.sha256(capture)
        )

    def test_copied_ready_cannot_complete_another_capture(self):
        first, second = self.capture(), self.capture()
        model = FakeModel()
        pipeline.analyze_capture(first, self.conf, model)
        first_ready = first.with_name(
            first.name.replace(".capture.json", ".ready.json")
        )
        second_ready = second.with_name(
            second.name.replace(".capture.json", ".ready.json")
        )
        second_ready.write_bytes(first_ready.read_bytes())
        completed = {}
        results = pipeline.analyze_pending(self.conf, model, completed)
        result = next(item for item in results if item["capture"] == second.name)
        self.assertEqual(result["status"], "error")
        self.assertNotIn(second.name, completed)
        self.assertEqual(model.calls, 1)

    def test_same_filename_but_different_recording_identity_rejected(self):
        capture = self.capture()
        model = FakeModel()
        pipeline.analyze_capture(capture, self.conf, model)
        ready = next(capture.parent.glob("*.ready.json"))
        content = json.loads(ready.read_text(encoding="utf-8"))
        content["recording_id"] = "different-recording"
        ready.write_text(json.dumps(content), encoding="utf-8")
        with self.assertRaises(ValueError):
            pipeline.analyze_capture(capture, self.conf, model)
        self.assertEqual(model.calls, 1)

    def test_mutated_capture_provenance_invalidates_completed_cache(self):
        capture = self.capture()
        model = FakeModel()
        completed = {}
        self.assertEqual(
            pipeline.analyze_pending(self.conf, model, completed)[0]["status"],
            "negative_retained",
        )
        self.assertIn(capture.name, completed)
        content = json.loads(capture.read_text(encoding="utf-8"))
        content["recorded_at"] = "2024-05-02T08:30:00+08:00"
        capture.write_text(json.dumps(content), encoding="utf-8")
        results = pipeline.analyze_pending(self.conf, model, completed)
        self.assertEqual(results[0]["status"], "error")
        self.assertEqual(model.calls, 1)

    def test_changed_ready_context_rejected_even_with_matching_identity(self):
        capture = self.capture()
        model = FakeModel()
        pipeline.analyze_capture(capture, self.conf, model)
        ready = next(capture.parent.glob("*.ready.json"))
        content = json.loads(ready.read_text(encoding="utf-8"))
        content["context"]["capture"]["device_id"] = "another-device"
        ready.write_text(json.dumps(content), encoding="utf-8")
        with self.assertRaises(ValueError):
            pipeline.analyze_capture(capture, self.conf, model)

    def test_verified_committed_positive_retry_after_source_wav_deleted(self):
        capture = self.capture()
        conf = dict(self.conf, source_wav_policy="delete_after_commit")
        model = FakeModel(HEADER + "0;3;Passer montanus;麻雀;0.9\n")

        def fake_codec(command, config):
            if command[0] == config["ffprobe"]:
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
            if command[-1] == "-":
                return b"\x00\x00" * pipeline.FRAMES
            Path(command[-1]).write_bytes(b"mock-codec-output")
            return b""

        with mock.patch.object(retention, "_command", side_effect=fake_codec):
            result = pipeline.analyze_capture(capture, conf, model)
            self.assertEqual(result["status"], "archived")
            self.assertFalse(list(capture.parent.glob("*.wav")))
            result = pipeline.analyze_capture(capture, conf, model)
            self.assertEqual(result["status"], "already_archived")
            self.assertEqual(
                pipeline.analyze_pending(conf, model)[0]["status"], "already_archived"
            )
        self.assertEqual(model.calls, 1)

    def test_active_unmarked_wav_is_ignored(self):
        root = Path(self.conf["pending_dir"])
        root.mkdir()
        write_wav(root / "active.wav")
        model = FakeModel()
        self.assertEqual(pipeline.analyze_pending(self.conf, model), [])
        self.assertEqual(model.calls, 0)

    def test_inference_failure_never_publishes_ready_and_can_retry(self):
        capture = self.capture()

        class FailingModel:
            def analyze(self, wav, csv, recorded_at):
                Path(csv).write_text(HEADER, encoding="utf-8")
                raise RuntimeError("simulated model crash")

        with self.assertRaises(RuntimeError):
            pipeline.analyze_capture(capture, self.conf, FailingModel())
        self.assertFalse(list(capture.parent.glob("*.ready.json")))
        self.assertTrue(list(capture.parent.glob("*.csv.incomplete-*")))
        self.assertEqual(
            pipeline.analyze_capture(capture, self.conf, FakeModel())["status"],
            "negative_retained",
        )

    def test_malformed_model_csv_not_negative(self):
        capture = self.capture()
        with self.assertRaises(retention.RetentionError):
            pipeline.analyze_capture(
                capture, self.conf, FakeModel(HEADER + "0;3;broken\n")
            )
        self.assertFalse(list(capture.parent.glob("*.ready.json")))
        self.assertTrue(list(capture.parent.glob("*.wav")))

    def test_changed_capture_rejected(self):
        capture = self.capture()
        wav = next(capture.parent.glob("*.wav"))
        with open(wav, "r+b") as handle:
            handle.seek(100)
            handle.write(b"\x01")
        with self.assertRaises(ValueError):
            pipeline.analyze_capture(capture, self.conf, FakeModel())
        self.assertFalse(list(capture.parent.glob("*.ready.json")))

    def test_existing_conflicting_csv_not_overwritten(self):
        capture = self.capture()
        wav = next(capture.parent.glob("*.wav"))
        csv = wav.with_name(wav.name + ".csv")
        original = HEADER + "0;3;Passer montanus;麻雀;0.8\n"
        csv.write_text(original, encoding="utf-8")
        with self.assertRaises(ValueError):
            pipeline.analyze_capture(capture, self.conf, FakeModel())
        self.assertEqual(csv.read_text(encoding="utf-8"), original)
        self.assertFalse(list(capture.parent.glob("*.ready.json")))

    def test_runtime_placeholders_do_not_prevent_staging(self):
        conf = dict(self.conf, latitude=None, longitude=None, recording_device=None)
        self.assertIsNone(pipeline.validate_config(conf)["latitude"])
        with self.assertRaises(ValueError):
            pipeline.validate_config(conf, require_runtime=True)

    def test_unknown_model_privacy_and_nan_rejected(self):
        for extra in (
            {"model": "BirdNET_GLOBAL_3K_V2.2_Model_FP16"},
            {"privacy_threshold": 1},
            {"sensitivity": float("nan")},
            {"threshold": 0},
            {"overlap": 3},
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                pipeline.validate_config(dict(self.conf, **extra))

    def test_single_cli_owner_lock(self):
        with pipeline.pipeline_lock(self.conf["pending_dir"]):
            with self.assertRaises(OSError):
                with pipeline.pipeline_lock(self.conf["pending_dir"]):
                    self.fail("Concurrent owner was permitted")

    def test_unknown_configuration_and_invalid_timezone_are_refused(self):
        for update in ({"threshhold": 0.9}, {"timezone": "not/a/timezone"}):
            with self.subTest(update=update), self.assertRaises((ValueError, KeyError)):
                pipeline.validate_config({**self.conf, **update})

    def test_capture_identity_collision_does_not_overwrite_wav(self):
        token = SimpleNamespace(hex="a" * 32)
        with mock.patch.object(pipeline.uuid, "uuid4", return_value=token):
            capture = self.capture()
            wav = capture.parent / json.loads(capture.read_text())["wav"]
            original = wav.read_bytes()
            with self.assertRaisesRegex(ValueError, "identity collision"):
                self.capture()
        self.assertEqual(wav.read_bytes(), original)
        self.assertEqual(len(self.commands), 1)

    def test_completed_csv_cannot_be_reused_after_lowering_filter_threshold(self):
        capture = self.capture()
        first = pipeline.analyze_capture(capture, self.conf, FakeModel())
        self.assertEqual(first["status"], "negative_retained")
        changed = FakeModel(HEADER + "0;3;Passer montanus;麻雀;0.15\n")
        with self.assertRaisesRegex(
            ValueError, "different or missing inference contract"
        ):
            pipeline.analyze_capture(capture, {**self.conf, "threshold": 0.1}, changed)
        self.assertEqual(changed.calls, 0)
        self.assertTrue(capture.exists())

    def test_standalone_retention_refuses_changed_model_csv_filter_rule(self):
        capture = self.capture()
        pipeline.analyze_capture(capture, self.conf, FakeModel())
        ready = capture.with_name(capture.name.replace(".capture.json", ".ready.json"))
        result = retention.process_marker(ready, {**self.conf, "threshold": 0.1})
        self.assertEqual(result["status"], "error")
        self.assertIn("producer inference contract", result["message"])

    def test_completed_cache_checks_configuration_identity(self):
        capture = self.capture()
        cache = {}
        pipeline.analyze_pending(self.conf, FakeModel(), cache)
        results = pipeline.analyze_pending(
            {**self.conf, "threshold": 0.1}, FakeModel(), cache
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "error")
        self.assertIn("inference contract", results[0]["error"])
        self.assertTrue(capture.exists())

    def test_saved_contract_binds_target_and_fixed_model_without_defaults(self):
        capture = self.capture()
        pipeline.analyze_capture(capture, self.conf, FakeModel())
        for update in ({"common_name": "树麻雀"}, {"latitude": 35.0}, {"overlap": 0.0}):
            with (
                self.subTest(update=update),
                self.assertRaisesRegex(ValueError, "inference contract"),
            ):
                pipeline.analyze_capture(capture, {**self.conf, **update}, FakeModel())
        ready = capture.with_name(capture.name.replace(".capture.json", ".ready.json"))
        original = json.loads(ready.read_text())
        for key in ("model_sha256", "labels_sha256", "source_sha256"):
            changed = json.loads(json.dumps(original))
            changed["context"]["analysis"]["inference_contract"].pop(key)
            ready.write_text(json.dumps(changed))
            with (
                self.subTest(missing=key),
                self.assertRaisesRegex(ValueError, "inference contract"),
            ):
                pipeline.analyze_capture(capture, self.conf, FakeModel())
        ready.write_text(json.dumps(original))

    def test_public_analysis_api_serializes_duplicate_work(self):
        capture = self.capture()
        model = FakeModel()
        with ThreadPoolExecutor(max_workers=3) as workers:
            results = list(
                workers.map(
                    lambda _: pipeline.analyze_capture(capture, self.conf, model),
                    range(3),
                )
            )
        self.assertEqual(model.calls, 1)
        self.assertTrue(
            all(result["status"] == "negative_retained" for result in results)
        )

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required"
    )
    def test_capture_model_retention_chain_with_real_codec(self):
        capture = self.capture()
        model = FakeModel(HEADER + "0;3;Passer montanus;麻雀;0.3\n")
        result = pipeline.analyze_capture(capture, self.conf, model)
        self.assertEqual(result["status"], "archived")
        archive = Path(result["archive"])
        saved = json.loads((archive / "receipt.json").read_text())
        self.assertEqual(saved["output"]["decoded_frames"], 1440000)
        self.assertEqual(saved["software_version"], retention.VERSION)
        self.assertTrue(
            (capture.parent / json.loads(capture.read_text())["wav"]).exists()
        )


class BridgeTests(unittest.TestCase):
    def test_only_allowlisted_definitions_compile(self):
        tree = model_bridge.core_ast()
        self.assertEqual({node.name for node in tree.body}, set(model_bridge.FUNCTIONS))
        self.assertTrue(all(isinstance(node, ast.FunctionDef) for node in tree.body))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertNotIn("userDir", names)
        self.assertNotIn("socket", names)
        self.assertNotIn("sqlite3", names)
        self.assertNotIn("sendAppriseNotifications", names)

    def test_source_guard_rejects_even_small_change(self):
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "server.py"
            changed.write_bytes(model_bridge.SOURCE.read_bytes() + b"\n")
            with self.assertRaises(ValueError):
                model_bridge.core_ast(changed)

    def test_padded_last_window_has_20_windows(self):
        config = {
            "model_path": "unused",
            "labels_path": "unused",
            "pending_dir": "unused",
            "scientific_name": "Passer montanus",
            "common_name": "麻雀",
        }
        namespace = model_bridge.compile_core(
            config, {"np": SimpleNamespace(zeros=lambda shape: [0] * shape)}
        )
        chunks = namespace["splitSignal"]([1] * pipeline.FRAMES, 48000, 1.5)
        self.assertEqual(len(chunks), 20)
        self.assertEqual(len(chunks[-1]), 144000)
        self.assertEqual(chunks[-1][:72000], [1] * 72000)
        self.assertEqual(chunks[-1][72000:], [0] * 72000)

    def test_timestamp_week_and_sensitivity_mapping(self):
        bridge = model_bridge.ModelBridge.__new__(model_bridge.ModelBridge)
        bridge.config = {
            "sensitivity": 1.25,
            "overlap": 1.5,
            "latitude": 36.55,
            "longitude": 104.18,
            "threshold": 0.2,
        }
        inference = mock.Mock(return_value={})
        writer = mock.Mock()
        bridge.core = {
            "readAudioData": mock.Mock(return_value=[1]),
            "analyzeAudioData": inference,
            "writeResultsToFile": writer,
        }
        info = bridge.analyze("unused.wav", "unused.csv", "2020-12-31T10:00:00+08:00")
        self.assertEqual(info["inference_week"], 48)
        self.assertEqual(info["sigmoid_sensitivity"], 0.75)
        self.assertEqual(inference.call_args.args[3:], (48, 0.75, 1.5))
        writer.assert_called_once_with({}, 0.2, "unused.csv")
        with self.assertRaises(ValueError):
            bridge.analyze("unused.wav", "unused.csv", "2020-12-31T10:00:00")


if __name__ == "__main__":
    unittest.main()
