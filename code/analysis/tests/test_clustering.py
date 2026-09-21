"""Small fixed synthetic checks for the retained clustering aid, never corpus fitting."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from sparrow_dataset import cluster, clustering
from sparrow_dataset.assessment import DEFAULT_ASSETS


class ClusteringTests(unittest.TestCase):
    def points(self):
        generator = np.random.default_rng(42)
        return np.vstack(
            [generator.normal(center, 0.08, size=(20, 4)) for center in (0, 5, 10)]
        )

    def test_hdbscan_uses_leaf_selection_without_mutating_input(self):
        points = self.points()
        before = points.copy()
        model, labels, metrics = clustering.fit_hdbscan(
            points, min_cluster_size=5, min_samples=3
        )
        self.assertEqual(model.cluster_selection_method, "leaf")
        self.assertTrue(model.gen_min_span_tree)
        self.assertTrue(model.prediction_data)
        self.assertEqual(labels.shape, (60,))
        self.assertGreaterEqual(metrics["n_clusters"], 2)
        self.assertTrue(np.isfinite(model.probabilities_).all())
        np.testing.assert_array_equal(points, before)

    def test_fixed_seed_umap_is_reproducible_on_small_synthetic_input(self):
        points = self.points()
        reducer, first = clustering.build_embedding(points, 5, 0.0, 2, 42)
        _, second = clustering.build_embedding(points, 5, 0.0, 2, 42)
        self.assertEqual(reducer.metric, "euclidean")
        self.assertEqual(reducer.random_state, reducer.transform_seed)
        self.assertEqual(first.shape, (60, 2))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_allclose(first, second, rtol=0, atol=1e-7)

    def test_seed_stability_ignores_arbitrary_cluster_numbering(self):
        rows = []
        labels = {}
        for seed, assignment in [(42, [0, 0, 1, 1]), (52, [8, 8, 3, 3])]:
            rows.append(
                dict(
                    n_neighbors=5,
                    min_dist=0.0,
                    min_cluster_size=2,
                    min_samples=1,
                    seed=seed,
                    n_clusters=2,
                    noise_ratio=0.0,
                    silhouette=0.9,
                    relative_validity=0.8,
                    mean_cluster_persistence=0.7,
                )
            )
            labels[(5, 0.0, 2, 1, seed)] = np.asarray(assignment)
        result = clustering.robust_aggregate(pd.DataFrame(rows), labels)
        self.assertEqual(result.iloc[0].stability_ari, 1.0)
        self.assertEqual(
            clustering.rank_robust_candidates(result).iloc[0].median_clusters, 2
        )

    def test_candidate_assignments_preserve_original_row_identity(self):
        rows = pd.DataFrame(
            {
                "stable_id": ["a", "b"],
                "source_row": [7, 9],
                "source_raven_row": [19, 20],
            }
        )
        result = clustering.assignment_frame(
            rows, np.array([3, -1]), np.array([0.9, 0.0]), np.zeros((2, 2)), np.ones(2)
        )
        pd.testing.assert_frame_equal(result[rows.columns], rows)
        self.assertEqual(result.micro_type.tolist(), [3, -1])
        self.assertNotIn("family_id", result)

    def write_inputs(self, directory, duplicate=False):
        n = 30
        np.save(directory / "features_105d.npy", np.zeros((n, 105), dtype=np.float32))
        np.save(directory / "rms_energy.npy", np.ones(n))
        names = json.loads(
            (DEFAULT_ASSETS / "label_assessment/run/feature_names.json").read_text()
        )
        (directory / "feature_names.json").write_text(json.dumps(names))
        pd.DataFrame(
            {
                "stable_id": (
                    ["duplicate"] * n if duplicate else [f"id{i}" for i in range(n)]
                ),
                "source_stem": ["synthetic"] * n,
                "source_file": ["synthetic.wav"] * n,
                "selection_id": range(n),
                "begin_time": np.arange(n),
                "end_time": np.arange(n) + 0.1,
                "low_freq": [1500] * n,
                "high_freq": [10000] * n,
            }
        ).to_csv(directory / "feature_rows.csv", index=False)

    def test_input_loader_rejects_duplicate_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_inputs(directory, duplicate=True)
            with self.assertRaisesRegex(ValueError, "stable_id"):
                cluster.load_inputs(directory)

    def test_existing_output_cannot_start_clustering(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = cluster.parse_args(["--output-dir", temporary])
            with patch.object(clustering, "select_robust") as fit:
                with self.assertRaisesRegex(ValueError, "Output must be new"):
                    cluster.execute(args)
                fit.assert_not_called()

    def test_dry_run_checks_inputs_without_fitting_or_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_inputs(directory)
            output = directory / "new-output"
            args = cluster.parse_args(
                [
                    "--input-dir",
                    str(directory),
                    "--output-dir",
                    str(output),
                    "--mode",
                    "robust_hd",
                    "--robust-neighbors",
                    "5",
                    "--dry-run",
                ]
            )
            with patch.object(clustering, "select_robust") as fit:
                result = cluster.execute(args)
                fit.assert_not_called()
            self.assertEqual(result["input_rows"], 30)
            self.assertEqual(result["status"], "planned")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
