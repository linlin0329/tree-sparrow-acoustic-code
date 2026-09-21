"""Immutable MP3 fixtures, read-only plans and exact-field reference checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from sparrow_dataset.assessments import mp3_assessment as module


class Mp3AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixtures = self.root / "fixtures"
        self.fixtures.mkdir()
        sf.write(self.fixtures / "S01.wav", np.zeros(6000), 48000, subtype="PCM_16")
        # Fixture identity checking does not decode or create MP3s.
        (self.fixtures / "S01.mp3").write_bytes(b"synthetic identity test payload")
        item = {"id": "S01"}
        for extension in ("wav", "mp3"):
            path = self.fixtures / ("S01." + extension)
            item[extension + "_bytes"] = path.stat().st_size
            item[extension + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.fixtures / "manifest.json").write_text(json.dumps([item]))
        self.contract = {
            "fixture_ids": ["S01"],
            "sample_rate_hz": 48000,
            "channels": 1,
            "files": [],
        }
        for path in self.fixtures.iterdir():
            self.contract["files"].append(
                {
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        self.mock_contract = patch.object(
            module, "fixture_contract", return_value=self.contract
        )
        self.mock_contract.start()
        self.addCleanup(self.mock_contract.stop)

    def test_dry_run_checks_original_hashes_and_creates_no_output(self):
        output = self.root / "new-output"
        with patch.object(module, "run") as compute:
            result = module.replay(self.fixtures, output, dry_run=True)
        self.assertEqual(
            result["identity"]["original_encoder_manifest_hashes_matched"], 2
        )
        self.assertFalse(output.exists())
        compute.assert_not_called()

    def test_corrupt_fixture_is_rejected_before_output_creation(self):
        (self.fixtures / "S01.mp3").write_bytes(b"changed")
        output = self.root / "new-output"
        with self.assertRaisesRegex(ValueError, "identity changed"):
            module.replay(self.fixtures, output, dry_run=True)
        self.assertFalse(output.exists())

    def test_reauthoring_local_manifest_does_not_replace_fixed_identity(self):
        path = self.fixtures / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest[0]["mp3_sha256"] = "0" * 64
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "identity changed"):
            module.validate_fixtures(self.fixtures)

    def test_existing_output_is_never_reused_or_overwritten(self):
        output = self.root / "output"
        output.mkdir()
        sentinel = output / "summary.json"
        sentinel.write_bytes(b"previous result")
        with self.assertRaises(FileExistsError):
            module.replay(self.fixtures, output)
        self.assertEqual(sentinel.read_bytes(), b"previous result")

    def test_reference_comparison_accepts_roundoff_only(self):
        result = module.compare_reference(
            {"peak_hz": 4000.0 + 1e-9}, {"peak_hz": 4000.0}
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["numeric_values"], 1)
        self.assertLess(result["max_absolute_difference"], 1e-8)

    def test_reference_comparison_rejects_changed_metrics_or_omitted_fields(self):
        expected = {"peak_hz": 4000.0, "centroid_hz": 4500.0, "bandwidth90_hz": 1200.0}
        for field in expected:
            with self.subTest(field=field):
                changed = dict(expected)
                changed[field] += 1.0
                with self.assertRaisesRegex(ValueError, "numeric reference differs"):
                    module.compare_reference(changed, expected)
        with self.assertRaisesRegex(ValueError, "fields differ"):
            module.compare_reference({"peak_hz": 4000.0}, expected)


if __name__ == "__main__":
    unittest.main()
