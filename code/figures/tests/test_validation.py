"""Numerical and filesystem guards for the standalone figure entry point."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sparrow_figures import validation
from sparrow_figures.stft import calculate_spectrogram


class FigureValidationTests(unittest.TestCase):
    def test_stft_does_not_pad_partial_tail(self):
        result = calculate_spectrogram(np.zeros(632, dtype=np.float32), 48000)
        self.assertEqual(result['stft_frame_count'], 2)
        self.assertEqual(result['stft_last_window_stop_frame'], 608)
        self.assertEqual(result['stft_uncovered_tail_frames'], 24)
        self.assertTrue(np.all(result['db'] == -70))

    def test_relative_spectrum_is_invariant_to_scalar_gain(self):
        signal = np.sin(2 * np.pi * 4000 * np.arange(4096) / 48000)
        a = calculate_spectrogram(signal, 48000)
        b = calculate_spectrogram(signal * .5, 48000)
        np.testing.assert_allclose(a['db'], b['db'], rtol=0, atol=1e-10)

    def test_stft_rejects_invalid_samples(self):
        for samples in [np.zeros(511), np.zeros((600,2)), np.full(600,np.nan)]:
            with self.subTest(shape=samples.shape), self.assertRaises(ValueError):
                calculate_spectrogram(samples, 48000)
        with self.assertRaises(ValueError):
            calculate_spectrogram(np.zeros(600), 22050)

    def test_relative_path_cannot_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'inputs'
            root.mkdir()
            for value in ['../outside', str(Path(tmp)/'outside')]:
                with self.assertRaises(ValueError):
                    validation.safe_join(root, value)
            (root/'link').symlink_to(Path(tmp)/'outside')
            with self.assertRaises(ValueError):
                validation.safe_join(root,'link')

    def test_manifest_checks_payload_and_unlisted_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root/'assets'
            assets.mkdir()
            (assets/'sample').write_bytes(b'fixed input')
            (root/'assets_manifest.json').write_text(json.dumps({'files':[{
                'path':'sample','bytes':11,'sha256':validation.sha256(assets/'sample')}]}))
            with patch.object(validation,'ASSETS',assets),patch.object(validation,'COMPONENT',root):
                self.assertEqual(validation.check_assets(),1)
                (assets/'extra').write_text('unexpected')
                with self.assertRaises(ValueError):validation.check_assets()
                (assets/'extra').unlink()
                (assets/'sample').write_bytes(b'other input')
                with self.assertRaises(ValueError):validation.check_assets()

    def test_checks_survive_python_optimization(self):
        import subprocess
        import sys
        result = subprocess.run([sys.executable,'-O','-c',
            'from sparrow_figures.validation import require; require(False,"expected")'],
            capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('expected',result.stderr)


if __name__ == '__main__':
    unittest.main()
