"""最终鸣唱语料审计与 Raven 规范化的轻量测试。"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from sparrow_dataset.processing import prepare_corpus as module


def load_script():
    return module


class FinalSongCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script()

    def test_stable_id_is_deterministic(self) -> None:
        row = pd.Series(
            {
                "Selection": 3,
                "Begin Time (s)": 0.1,
                "End Time (s)": 0.3,
                "Low Freq (Hz)": 1500.0,
                "High Freq (Hz)": 9000.0,
            }
        )
        first = self.module.stable_syllable_id("song", row)
        second = self.module.stable_syllable_id("song", row.copy())
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)

    def test_normalize_deduplicates_views_and_logs_invalid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_path = root / "song.wav"
            table_path = root / "song.Table.1.selections.txt"
            output_path = root / "normalized/song.Table.1.selections.txt"
            exclusion_path = root / "excluded/song.csv"
            sf.write(audio_path, np.zeros(48000, dtype=np.float32), 48000)
            rows = [
                [1, "Waveform 1", 1, 0.1, 0.3, 1500, 9000],
                [1, "Spectrogram 1", 1, 0.1, 0.3, 1500, 9000],
                [2, "Spectrogram 1", 1, 0.4, 0.4, 2000, 8000],
                [3, "Spectrogram 1", 1, 0.5, 0.7, 2000, 8000],
            ]
            pd.DataFrame(rows, columns=self.module.REQUIRED_COLUMNS).to_csv(
                table_path, sep="\t", index=False
            )

            valid, excluded = self.module.normalize_raven_table(
                table_path,
                audio_path,
                output_path,
                exclusion_path,
                0.03,
                1.0,
            )

            self.assertEqual(len(valid), 2)
            self.assertEqual(valid["stable_id"].nunique(), 2)
            self.assertCountEqual(
                excluded["exclusion_reason"],
                ["duplicate_view", "nonpositive_duration"],
            )
            self.assertTrue(output_path.is_file())
            self.assertTrue(exclusion_path.is_file())


if __name__ == "__main__":
    unittest.main()
