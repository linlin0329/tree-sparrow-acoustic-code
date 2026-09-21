"""DP-11 只读流式音频审计：真实编解码与检查点恢复行为。"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest import mock
import wave

import numpy as np
import soundfile as sf

from sparrow_dataset.processing import audit_release_audio as module


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg unavailable")
class ReleaseAudioQCTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "audio"
        self.data.mkdir()
        self.ffmpeg = shutil.which("ffmpeg")
        self.stop = threading.Event()

    def audio(self, name="tone.wav", rate=32000, channels=2, duration=0.2):
        time = np.arange(round(rate * duration)) / rate
        signal = np.sin(2 * np.pi * 500 * time) * 0.25
        values = np.column_stack([signal * (i + 1) / channels for i in range(channels)])
        path = self.data / name
        sf.write(path, values, rate, subtype="PCM_16")
        return path

    def row(self, path, physical_id="id1", role="selected_representative"):
        return dict(
            physical_id=physical_id,
            logical_id="logical-" + physical_id,
            relative_path=path.name,
            year="2023",
            site="liujiaxia",
            role=role,
            expected_size_bytes=str(path.stat().st_size if path.exists() else 0),
        )

    def audit(self, path, row=None):
        return module.audit_one(
            row or self.row(path),
            module.source_stat(path),
            self.data,
            self.ffmpeg,
            5.0,
            0,
            0,
            self.stop,
        )

    def manifest(self, paths):
        path = self.root / "manifest.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=module.REQUIRED_COLUMNS)
            writer.writeheader()
            for index, item in enumerate(paths):
                writer.writerow(self.row(item, "id" + str(index)))
        return path

    def args(self, manifest, output=None, extra=()):
        return module.parse_args(
            [
                "--manifest",
                str(manifest),
                "--data-root",
                str(self.data),
                "--output-dir",
                str(output or self.root / "result"),
                "--mode",
                "pilot",
                "--workers",
                "1",
                "--batch-size",
                "1",
                *extra,
            ]
        )

    def test_native_stereo_frames_duration_hash_and_signal_statistics(self):
        path = self.audio()
        original = path.read_bytes()
        result = self.audit(path)
        decoded, rate = sf.read(path, always_2d=True)
        self.assertTrue(result["strict_decode_pass"])
        self.assertTrue(result["decode_complete"])
        self.assertEqual(result["native_sample_rate"], 32000)
        self.assertEqual(result["native_channels"], 2)
        self.assertEqual(result["decoded_frames"], 6400)
        self.assertEqual(result["decoded_samples"], 12800)
        self.assertAlmostEqual(result["duration_seconds"], 0.2)
        self.assertEqual(result["sha256"], hashlib.sha256(original).hexdigest())
        self.assertAlmostEqual(result["rms"], np.sqrt(np.mean(decoded**2)), places=12)
        self.assertAlmostEqual(result["mean_dc"], np.mean(decoded), places=12)
        self.assertEqual(result["window_rms_count"], 4)
        self.assertEqual(result["window_rms_partial_count"], 1)
        self.assertTrue(result["source_stat_matches"])
        self.assertEqual(path.read_bytes(), original)

    def test_real_mp3_decodes_without_forcing_48k_mono(self):
        wav = self.audio(channels=2, rate=22050, duration=0.7)
        mp3 = self.data / "compressed.mp3"
        subprocess.run(
            [
                self.ffmpeg,
                "-v",
                "error",
                "-i",
                str(wav),
                "-codec:a",
                "libmp3lame",
                str(mp3),
            ],
            check=True,
        )
        result = self.audit(mp3)
        self.assertTrue(result["strict_decode_pass"], result["stderr_excerpt"])
        self.assertEqual(result["native_sample_rate"], 22050)
        self.assertEqual(result["native_channels"], 2)
        self.assertAlmostEqual(result["duration_seconds"], 0.7, places=3)
        self.assertEqual(result["sha256"], hashlib.sha256(mp3.read_bytes()).hexdigest())

    def test_missing_zero_byte_corrupt_and_truncated_wav_are_distinct(self):
        missing = self.data / "missing.mp3"
        empty = self.data / "empty.mp3"
        empty.touch()
        corrupt = self.data / "corrupt.mp3"
        corrupt.write_bytes(b"This is not an audio file" * 30)
        truncated = self.audio("truncated.wav", channels=1)
        truncated.write_bytes(truncated.read_bytes()[:1000])
        for path, status in [
            (missing, "missing"),
            (empty, "zero_byte"),
            (corrupt, "decode_error"),
            (truncated, "decode_error"),
        ]:
            result = self.audit(path)
            self.assertEqual(result["qc_status"], status, result)
            self.assertFalse(result["strict_decode_pass"])
        self.assertEqual(self.audit(empty)["sha256"], hashlib.sha256(b"").hexdigest())

    def test_constant_zero_and_float_out_of_range_are_diagnostics(self):
        path = self.data / "float.wav"
        values = np.r_[
            np.zeros(250), np.ones(100), np.full(50, 1.5), np.full(100, -1.0)
        ].astype(np.float32)
        sf.write(path, values, 1000, subtype="FLOAT")
        result = self.audit(path)
        self.assertTrue(result["strict_decode_pass"], result)
        self.assertEqual(result["longest_zero_run_frames"], 250)
        self.assertEqual(result["longest_constant_run_frames"], 250)
        self.assertEqual(result["samples_abs_eq_1"], 200)
        self.assertEqual(result["samples_abs_gt_1"], 50)
        self.assertEqual(result["samples_abs_ge_1"], 250)
        self.assertEqual(result["window_rms_count"], 10)
        zero = self.data / "zero.wav"
        sf.write(zero, np.zeros(500), 1000, subtype="PCM_16")
        result = self.audit(zero)
        self.assertTrue(result["strict_decode_pass"])
        self.assertTrue(result["all_zero"])
        self.assertEqual(result["longest_zero_run_frames"], 500)
        self.assertEqual(result["longest_constant_run_frames"], 500)

    def test_nonfinite_samples_not_silently_replaced_or_passed(self):
        path = self.data / "nonfinite.wav"
        values = np.r_[np.ones(100) * 0.25, np.nan, np.inf, np.ones(98) * 0.25].astype(
            np.float32
        )
        sf.write(path, values, 1000, subtype="FLOAT")
        result = self.audit(path)
        self.assertEqual(result["qc_status"], "nonfinite_samples")
        self.assertEqual(result["nonfinite_samples"], 2)
        self.assertEqual(result["finite_samples"], 198)
        self.assertFalse(result["strict_decode_pass"])
        self.assertGreater(result["window_rms_nonfinite_count"], 0)

    def test_stream_chunk_boundaries_preserve_runs_and_partial_windows(self):
        path = self.data / "chunks.wav"
        values = np.r_[np.zeros(113), np.full(122, 0.25), np.ones(12)].astype(
            np.float32
        )
        sf.write(path, np.column_stack([values, -values]), 1000, subtype="FLOAT")
        raw = path.read_bytes()
        acc = module.FloatWaveAccumulator()
        for offset in range(0, len(raw), 37):
            acc.feed(raw[offset : offset + 37])
        result = acc.result()
        self.assertEqual(result["decoded_frames"], 247)
        self.assertEqual(result["longest_zero_run_frames"], 113)
        self.assertEqual(result["longest_constant_run_frames"], 122)
        self.assertEqual(result["window_rms_count"], 5)
        self.assertEqual(result["window_rms_partial_count"], 2)
        expected = [
            np.sqrt(np.mean(values[start : start + 100].astype(float) ** 2))
            for start in range(0, len(values), 50)
        ]
        self.assertAlmostEqual(result["window_rms_min"], min(expected))
        self.assertAlmostEqual(result["window_rms_median"], np.median(expected))
        self.assertAlmostEqual(result["window_rms_max"], max(expected))
        self.assertEqual(result, acc.result())

    def test_signal_statistics_invariant_to_accumulation_chunk_size(self):
        path = self.data / "random_float.wav"
        values = (
            np.random.default_rng(42)
            .normal(0.15, 0.5, size=(16047, 2))
            .astype(np.float32)
        )
        values[140:730, 0] = 0
        values[1200:2010, 1] = 0.75
        values[3000, 0] = np.nan
        sf.write(path, values, 16000, subtype="FLOAT")
        raw = path.read_bytes()
        results = []
        for size in (37, 8192, 262144):
            acc = module.FloatWaveAccumulator()
            for offset in range(0, len(raw), size):
                acc.feed(raw[offset : offset + size])
            results.append(acc.result())
        for result in results[1:]:
            for key, expected in results[0].items():
                if key == "mean_dc_per_channel_json":
                    np.testing.assert_allclose(
                        json.loads(result[key]),
                        json.loads(expected),
                        rtol=1e-12,
                        atol=1e-14,
                    )
                elif isinstance(expected, float):
                    self.assertAlmostEqual(result[key], expected, places=12, msg=key)
                else:
                    self.assertEqual(result[key], expected, key)

    def test_decoder_exit_zero_with_error_level_fails_even_if_error_precedes_log_tail(
        self,
    ):
        path = self.audio()
        payload = self.data / "float_payload.wav"
        sf.write(payload, np.ones(100, dtype=np.float32) * 0.2, 1000, subtype="FLOAT")
        fake = self.root / "decoder_exit_zero"
        fake.write_text(
            "#!/usr/bin/env python3\nimport sys\nfrom pathlib import Path\n"
            "sys.stderr.write('[mp3float @ 0x123] [error] Header missing\\n')\n"
            "sys.stderr.write('[decoder @ 0x456] [error] Error submitting packet: Invalid data\\n')\n"
            "sys.stderr.write('[warning] subsequent ordinary message ' * 1000 + '\\n')\n"
            f"sys.stdout.buffer.write(Path({str(payload)!r}).read_bytes())\n"
        )
        fake.chmod(0o755)
        result = module.strict_decode(path, str(fake), 5, self.stop)
        self.assertEqual(result["decoder_return_code"], 0)
        self.assertEqual(result["decoded_frames"], 100)
        self.assertEqual(result["qc_status"], "decode_error")
        self.assertFalse(result["strict_decode_pass"])
        self.assertFalse(result["decode_complete"])
        self.assertTrue(result["has_decoder_error"])
        self.assertEqual(result["decoder_error_lines"], 2)
        self.assertIn("Header missing", result["decoder_error_excerpt"])
        self.assertNotIn("Header missing", result["stderr_excerpt"])

    def test_warning_severity_with_error_words_is_not_decoder_error(self):
        path = self.audio()
        payload = self.data / "warning_payload.wav"
        sf.write(payload, np.zeros(100, dtype=np.float32), 1000, subtype="FLOAT")
        fake = self.root / "decoder_warning"
        fake.write_text(
            "#!/usr/bin/env python3\nimport sys\nfrom pathlib import Path\n"
            "sys.stderr.write('[mp3 @ 0x123] [warning] Estimating duration; filename=/tmp/[error].mp3\\n')\n"
            "sys.stderr.write('[warning] [error] quoted filename is not a severity change\\n')\n"
            f"sys.stdout.buffer.write(Path({str(payload)!r}).read_bytes())\n"
        )
        fake.chmod(0o755)
        result = module.strict_decode(path, str(fake), 5, self.stop)
        self.assertEqual(result["qc_status"], "decoded_with_warnings")
        self.assertTrue(result["strict_decode_pass"])
        self.assertFalse(result["has_decoder_error"])
        self.assertEqual(result["decoder_error_lines"], 0)

    def test_timeout_is_indeterminate_and_retry_history_retained(self):
        path = self.audio()
        fake = self.root / "sleep_decoder"
        fake.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(10)\n")
        fake.chmod(0o755)
        result = module.audit_one(
            self.row(path),
            module.source_stat(path),
            self.data,
            str(fake),
            0.05,
            1,
            0,
            self.stop,
        )
        self.assertEqual(result["qc_status"], "timeout_indeterminate")
        self.assertEqual(result["attempt_count"], 2)
        self.assertEqual(len(json.loads(result["attempt_history_json"])), 2)
        self.assertFalse(result["decode_complete"])
        self.assertFalse(result["strict_decode_pass"])

    def test_changed_stat_not_reused_and_inventory_size_is_diagnostic(self):
        path = self.audio()
        snapshot = module.source_stat(path)
        path.write_bytes(path.read_bytes() + b"changed")
        result = module.audit_one(
            self.row(path), snapshot, self.data, self.ffmpeg, 5, 0, 0, self.stop
        )
        self.assertEqual(result["qc_status"], "source_changed_before_read")
        self.assertFalse(result["strict_decode_pass"])
        clean = self.audio("clean.wav")
        row = self.row(clean)
        row["expected_size_bytes"] = "1"
        result = self.audit(clean, row)
        self.assertTrue(result["strict_decode_pass"])
        self.assertFalse(result["expected_size_matches"])
        self.assertIn("size_differs_from_inventory", result["diagnostic_flags"])

    def test_no_overwrite_and_exact_resume_preserve_results(self):
        path = self.audio()
        manifest = self.manifest([path])
        args = self.args(manifest)
        summary = module.run(args)
        result_bytes = (args.output_dir / "results.csv.gz").read_bytes()
        with self.assertRaises(FileExistsError):
            module.run(args)
        args.resume = True
        with mock.patch.object(
            module,
            "audit_one",
            side_effect=AssertionError("completed file decoded twice"),
        ):
            self.assertEqual(module.run(args), summary)
        self.assertEqual(
            (args.output_dir / "results.csv.gz").read_bytes(), result_bytes
        )
        args.timeout_seconds += 1
        with self.assertRaisesRegex(ValueError, "resume 拒绝"):
            module.run(args)

    def test_resume_rejects_source_change_and_result_tampering(self):
        path = self.audio()
        manifest = self.manifest([path])
        args = self.args(manifest)
        module.run(args)
        args.resume = True
        current = path.stat()
        os.utime(path, ns=(current.st_atime_ns, current.st_mtime_ns + 1))
        with self.assertRaisesRegex(ValueError, "stat 改变"):
            module.run(args)
        batch = args.output_dir / "batches/batch_00000.csv.gz"
        batch.write_bytes(batch.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "批次校验失败"):
            module.run(args)

    def test_resume_rejects_tampered_final_summary_without_overwriting_it(self):
        path = self.audio()
        manifest = self.manifest([path])
        args = self.args(manifest)
        module.run(args)
        summary_path = args.output_dir / "summary.json"
        summary = module.load_json(summary_path)
        summary["row_count"] = 999
        summary_path.write_text(json.dumps(summary))
        before = summary_path.read_bytes()
        args.resume = True
        with self.assertRaisesRegex(ValueError, "最终 summary"):
            module.run(args)
        self.assertEqual(summary_path.read_bytes(), before)

    def test_interrupted_batch_is_not_committed_and_resume_finishes(self):
        paths = [self.audio("a.wav"), self.audio("b.wav")]
        manifest = self.manifest(paths)
        args = self.args(manifest)
        original = module.audit_one

        def interrupt_second(row, *rest):
            if row["physical_id"] == "id1":
                raise KeyboardInterrupt
            return original(row, *rest)

        with mock.patch.object(module, "audit_one", side_effect=interrupt_second):
            with self.assertRaises(KeyboardInterrupt):
                module.run(args)
        self.assertTrue((args.output_dir / "batches/batch_00000.json").exists())
        self.assertFalse((args.output_dir / "batches/batch_00001.json").exists())
        self.assertEqual(
            module.load_json(args.output_dir / "progress.json")["state"], "interrupted"
        )
        args.resume = True
        self.assertEqual(module.run(args)["row_count"], 2)

    def test_manifest_unique_relative_paths_and_full_limit_rejected(self):
        path = self.audio()
        manifest = self.manifest([path, path])
        with self.assertRaisesRegex(ValueError, "唯一"):
            module.read_manifest(manifest, self.data, None)
        with self.assertRaises(ValueError):
            module.resolve_audio(self.data, "../escape.wav")
        with self.assertRaises(SystemExit):
            module.parse_args(
                [
                    "--manifest",
                    str(manifest),
                    "--data-root",
                    str(self.data),
                    "--output-dir",
                    "unused",
                    "--mode",
                    "full",
                    "--limit",
                    "1",
                ]
            )


if __name__ == "__main__":
    unittest.main()
