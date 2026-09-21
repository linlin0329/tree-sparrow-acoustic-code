"""发布Raven的行号连接、候选偏移边界与整数帧重切验收。"""

import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from sparrow_dataset.processing import release_bundle_annotations as m


class AnnotationBundleTests(unittest.TestCase):
    def syllable(self):
        return dict(
            syllable_id="abc123",
            source_raven_row="2",
            selection_id="19",
            annotation_channel="1",
            begin_s="0.20001",
            end_s="0.70001",
            low_freq_hz="0",
            high_freq_hz="4",
            family_id="single_group_01",
            leaf_id="single_group_01_v1",
            syllable_structure="single",
            source_frame_start="2",
            source_frame_stop="7",
            decoded_frames="5",
            sample_rate_hz="10",
            channels="1",
        )

    def raven(self):
        return [
            dict(
                Selection="1",
                View="Spectrogram 1",
                Channel="1",
                **{
                    "Begin Time (s)": "0",
                    "End Time (s)": "0.1",
                    "Low Freq (Hz)": "0",
                    "High Freq (Hz)": "3",
                },
            ),
            dict(
                Selection="19",
                View="Waveform 1",
                Channel="1",
                **{
                    "Begin Time (s)": "0.20001",
                    "End Time (s)": "0.70001",
                    "Low Freq (Hz)": "0",
                    "High Freq (Hz)": "4",
                },
            ),
        ]

    def test_original_row_number_and_waveform_are_not_discarded_or_misjoined(self):
        rows = m.raven_rows([self.syllable()], self.raven())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Selection"], "1")
        self.assertEqual(rows[0]["original_selection_id"], "19")
        self.assertEqual(rows[0]["source_raven_row"], "2")
        self.assertEqual(rows[0]["original_view"], "Waveform 1")
        self.assertEqual(rows[0]["View"], "Spectrogram 1")
        self.assertEqual(rows[0]["leaf_id"], "single_group_01_v1")
        self.assertEqual(rows[0]["Begin Time (s)"], "0.20001")
        reopened = list(
            csv.DictReader(
                io.StringIO(m.csv_bytes(rows, m.RAVEN_FIELDS, "\t").decode()),
                delimiter="\t",
            )
        )
        self.assertEqual(reopened, rows)

    def test_stale_raven_boundary_or_frequency_cannot_be_silently_used(self):
        for field in ("begin_s", "end_s", "low_freq_hz", "selection_id"):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "输入与当前Raven"),
            ):
                m.raven_rows([dict(self.syllable(), **{field: "999"})], self.raven())

    def fixtures(self):
        manual = dict(
            source_stem="source",
            raw_recording_id="raw",
            site_id="site",
            canonical_syllable_count="1",
            raw_release_included="true",
            time_status="known_clock_error",
        )
        prepared = dict(
            duration_s="1.0",
            decoded_frames="10",
            sample_rate_hz="10",
            channels="1",
            wav_subtype="FLOAT",
            size_bytes="120",
            sha256="a" * 64,
            processing_profile_id="prepared_filter",
            legacy_bandpass_match_registered="true",
        )
        link = dict(
            release_offset_s="",
            release_offset_status="not_verified_remains_missing",
            proposed_offset_s="-0.04702083333333333",
            proposed_offset_score="0.999999",
            proposed_offset_method="candidate",
            raw_planned_package_path="audio/raw/raw.mp3",
            raw_sha256="b" * 64,
            link_evidence="filename_and_existing_alignment_candidate",
            content_review_status="human_content_confirmed",
            unmatched_manual_head_frames="2257",
            unmatched_manual_head_s="0.04702083333333333",
        )
        return manual, prepared, link

    def test_high_correlation_and_content_confirmation_do_not_verify_raw_interval(self):
        manual, prepared, link = self.fixtures()
        row = m.clip_row(manual, prepared, link)
        self.assertEqual(
            row["candidate_coverage_status"], "partial_negative_offset_unverified"
        )
        self.assertEqual(
            (row["raw_start_s"], row["raw_end_s"], row["release_offset_s"]),
            ("", "", ""),
        )
        self.assertEqual(row["analysis_wav_path"], "audio/analysis_clips/source.wav")
        self.assertEqual(row["local_start_s"], "0")
        with self.assertRaisesRegex(ValueError, "正式偏移证据发生变化"):
            m.clip_row(
                manual, prepared, dict(link, release_offset_s="-0.04702083333333333")
            )

    def test_positive_candidate_does_not_become_formal_crop_start(self):
        manual, prepared, link = self.fixtures()
        row = m.clip_row(
            manual,
            prepared,
            dict(
                link,
                proposed_offset_s="12.3",
                unmatched_manual_head_frames="",
                unmatched_manual_head_s="",
            ),
        )
        self.assertEqual(row["candidate_coverage_status"], "unverified")
        self.assertEqual(row["raw_start_s"], "")
        self.assertEqual(row["candidate_raw_offset_s"], "12.3")

    def test_integer_frame_reconstruction_compares_waveform_not_wav_header(self):
        samples = np.arange(10, dtype=np.float32).reshape(-1, 1) / 20
        with tempfile.TemporaryDirectory() as directory:
            ref, other = (
                Path(directory) / "reference.wav",
                Path(directory) / "other.wav",
            )
            sf.write(ref, samples[2:7], 10, subtype="FLOAT", format="WAVEX")
            sf.write(other, samples[2:7], 10, subtype="FLOAT", format="WAV")
            self.assertNotEqual(m.digest(ref), m.digest(other))
            reference = dict(source=str(ref), expected_sha256=m.digest(ref))
            actual = m.verify_reconstruction(samples, 10, self.syllable(), reference)
            self.assertEqual(actual, m.pcm_hash(samples[2:7]))
            with self.assertRaisesRegex(ValueError, "逐样本不一致"):
                m.verify_reconstruction(
                    samples + np.float32(0.1), 10, self.syllable(), reference
                )
            with self.assertRaisesRegex(ValueError, "帧范围越界"):
                m.verify_reconstruction(
                    samples,
                    10,
                    dict(self.syllable(), source_frame_stop="11"),
                    reference,
                )
            with self.assertRaisesRegex(ValueError, "采样率不匹配"):
                m.verify_reconstruction(samples, 11, self.syllable(), reference)

    def test_resume_text_reuses_equal_content_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.txt"
            m.write_same_or_new(path, b"original")
            m.write_same_or_new(path, b"original")
            with self.assertRaisesRegex(ValueError, "拒绝覆盖"):
                m.write_same_or_new(path, b"changed")
            self.assertEqual(path.read_bytes(), b"original")

    def test_output_path_cannot_leave_package_or_resolve_a_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "非法相对路径"):
                m.safe_path(root, "../escaped")
            (root / "actual").write_text("existing")
            (root / "alias").symlink_to(root / "actual")
            with self.assertRaisesRegex(ValueError, "符号链接"):
                m.safe_path(root, "alias")

    def test_standalone_example_reconstructs_and_detects_sample_corruption(self):
        samples = np.arange(10, dtype=np.float32).reshape(-1, 1) / 20
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "examples").mkdir()
            (root / "metadata").mkdir()
            (root / "audio").mkdir()
            script = root / "examples/reconstruct_syllables.py"
            script.write_text(m.EXAMPLE_SCRIPT)
            audio = root / "audio/parent.wav"
            sf.write(audio, samples, 10, subtype="FLOAT")
            row = dict(
                self.syllable(),
                analysis_clip_path="audio/parent.wav",
                decoded_pcm_f32le_sha256=m.pcm_hash(samples[2:7]),
            )
            (root / "metadata/syllables.csv").write_bytes(m.csv_bytes([row]))
            result = subprocess.run(
                [sys.executable, str(script), "--package", str(root)],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(json.loads(result.stdout)["checked_syllables"], 1)
            self.assertFalse((root / "syllables").exists())
            sf.write(audio, samples + np.float32(0.1), 10, subtype="FLOAT")
            failed = subprocess.run(
                [sys.executable, str(script), "--package", str(root)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("PCM hash mismatch", failed.stderr)


if __name__ == "__main__":
    unittest.main()
