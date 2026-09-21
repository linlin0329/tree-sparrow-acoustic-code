"""2556宏音节体系分层内部验证的单元与live-contract测试。"""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


from sparrow_dataset.assessments import taxonomy as module


def load_script():
    return module


class MacroSyllableTaxonomyValidation2556Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script()

    def test_recording_blocked_splits_have_no_source_leakage(self) -> None:
        labels = np.repeat(np.arange(3), 12)
        groups = np.asarray(
            [
                f"class_{label}_recording_{index // 4}"
                for label in range(3)
                for index in range(12)
            ]
        )
        folds = self.module.recording_blocked_splits(
            labels, groups, n_splits=3, seed=42
        )
        self.assertEqual(len(folds), 3)
        for train, test in folds:
            self.assertFalse(set(groups[train]) & set(groups[test]))

    def test_source_balanced_units_equalize_recordings(self) -> None:
        frame = pd.DataFrame(
            {
                "label": ["a", "a", "a", "b", "b"],
                "source_stem": ["r1", "r1", "r2", "r3", "r4"],
            }
        )
        vectors = np.asarray([[0.0], [2.0], [10.0], [20.0], [22.0]])
        units, prototypes = self.module.source_balanced_units(vectors, frame, "label")
        self.assertEqual(len(units), 4)
        self.assertEqual(units.groupby("label").size().to_dict(), {"a": 2, "b": 2})
        self.assertEqual(prototypes[:, 0].tolist(), [1.0, 10.0, 20.0, 22.0])

    def test_fixed_seed_geometry_is_reproducible(self) -> None:
        units = pd.DataFrame(
            {
                "label": ["a"] * 4 + ["b"] * 4,
                "source_stem": [f"r{i}" for i in range(8)],
            }
        )
        vectors = np.asarray([[0.0], [0.1], [0.2], [0.3], [2.0], [2.1], [2.2], [2.3]])
        first = self.module.bootstrap_geometry(
            units,
            vectors,
            n_bootstrap=10,
            n_permutations=10,
            seed=42,
        )
        second = self.module.bootstrap_geometry(
            units,
            vectors,
            n_bootstrap=10,
            n_permutations=10,
            seed=42,
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
