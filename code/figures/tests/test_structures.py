"""Regression checks for the accepted 27-family catalogue and shared scales."""

import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sparrow_figures import structures
from sparrow_figures.resources import ASSETS
from sparrow_figures.validation import rows


class StructuresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selected = [
            row for row in rows(ASSETS / "selected_syllables.csv")
            if row["figure"] == "figure5"
        ]
        cls.spectra = {}
        for row in cls.selected:
            with np.load(ASSETS / "spectra" / f'{row["stable_id"]}.npz', allow_pickle=False) as archive:
                cls.spectra[row["stable_id"]] = {key: archive[key] for key in archive.files}
        with plt.rc_context():
            cls.figure = structures.draw(cls.selected, cls.spectra)
        cls.axes = cls.figure.axes[:27]
        cls.renderer = cls.figure.canvas.get_renderer()

    @classmethod
    def tearDownClass(cls):
        plt.close(cls.figure)

    def test_complete_order_and_one_example_per_family(self):
        labels = ([f"S{i:02d}" for i in range(1, 15)]
                  + [f"D{i:02d}" for i in range(1, 7)]
                  + [f"T{i:02d}" for i in range(1, 8)])
        self.assertEqual([axis.get_label() for axis in self.axes], labels)
        self.assertEqual([row["panel"] for row in self.selected], labels)
        self.assertEqual(len(self.figure.axes), 28)  # 27 spectra and one power scale
        self.assertEqual(len({row["stable_id"] for row in self.selected}), 27)

    def test_physical_panels_rows_and_gutters(self):
        figure = self.figure
        np.testing.assert_allclose(figure.get_size_inches() * 25.4,
                                   [180, 93.53934426229509], rtol=0, atol=1e-8)
        boxes = np.array([
            axis.get_window_extent(self.renderer)
                .transformed(figure.dpi_scale_trans.inverted()).bounds
            for axis in self.axes
        ]) * 25.4
        np.testing.assert_allclose(boxes[:, 2], 16.444444444444443, rtol=0, atol=1e-8)
        np.testing.assert_allclose(boxes[:, 3], 19.713114754098364, rtol=0, atol=1e-8)
        for start in (0, 9, 18):
            row = boxes[start:start + 9]
            np.testing.assert_allclose(row[:, 1], row[0, 1], rtol=0, atol=1e-8)
            np.testing.assert_allclose(np.diff(row[:, 0]) - row[:-1, 2], 2,
                                       rtol=0, atol=1e-8)
        for column in range(9):
            np.testing.assert_allclose(boxes[column::9, 0], boxes[column, 0],
                                       rtol=0, atol=1e-8)
        self.assertGreaterEqual(float(boxes[:, 1].min()), 18.5 - 1e-8)

    def test_display_keeps_the_selected_arrays_and_full_common_range(self):
        for axis, row in zip(self.axes, self.selected):
            source = self.spectra[row["stable_id"]]
            image = axis.images[0]
            np.testing.assert_array_equal(image.get_array(), source["db"])
            np.testing.assert_array_equal(image.get_extent(), source["extent"])
            self.assertEqual(axis.get_xlim(), (0, 0.7))
            self.assertEqual(axis.get_ylim(), (0, 16))
            self.assertEqual(image.get_clim(), (-45, 0))
            self.assertEqual(image.get_interpolation(), "nearest")
            self.assertEqual(len(axis.get_xticks()), 0)

    def test_shared_time_bar_matches_data_coordinates(self):
        bar = next(artist for artist in self.figure.artists
                   if artist.get_gid() == "shared_time_scale")
        ends = bar.get_transform().transform(bar.get_xydata())
        data_ends = self.axes[0].transData.transform([[0, 0], [0.4, 0]])
        self.assertAlmostEqual(ends[1, 0] - ends[0, 0],
                               data_ends[1, 0] - data_ends[0, 0], places=8)
        self.assertLess(ends[0, 1], min(axis.bbox.y0 for axis in self.axes))

    def test_labels_are_below_their_row_baselines(self):
        texts = {text.get_text(): text for text in self.figure.texts}
        for axis in self.axes:
            text = texts[axis.get_label()]
            self.assertEqual(text.get_fontsize(), 6.3)
            self.assertLess(text.get_window_extent(self.renderer).y1, axis.bbox.y0)
        for row in range(3):
            baseline = next(artist for artist in self.figure.artists
                            if artist.get_gid() == f"row_baseline_{row + 1}")
            ends = baseline.get_transform().transform(baseline.get_xydata())
            self.assertAlmostEqual(ends[0, 1], self.axes[row * 9].bbox.y0, places=8)
            self.assertAlmostEqual(ends[0, 0], self.axes[row * 9].bbox.x0, places=8)
            self.assertAlmostEqual(ends[1, 0], self.axes[row * 9 + 8].bbox.x1, places=8)

    def test_rejects_partial_or_reordered_catalogue(self):
        for selected in (self.selected[:-1], self.selected[::-1]):
            with self.subTest(count=len(selected)), self.assertRaises(ValueError):
                structures.draw(selected, self.spectra)

    def test_rejects_duplicate_or_missing_syllables(self):
        duplicate = [dict(row) for row in self.selected]
        duplicate[-1]["stable_id"] = duplicate[0]["stable_id"]
        with self.assertRaises(ValueError):
            structures.draw(duplicate, self.spectra)
        with self.assertRaises(ValueError):
            structures.draw(self.selected, {})


if __name__ == "__main__":
    unittest.main()
