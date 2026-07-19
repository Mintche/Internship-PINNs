import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import QuadMesh

from tools.compare_sound_speed import (
    build_grid,
    build_ground_truth,
    compute_misfit,
    create_comparison_figure,
    create_misfit_figure,
    create_reconstruction_figure,
)
from tools.uv_checkpoint import UVCheckpoint
from tools.ground_truth import build_registered_sound_speed


class SoundSpeedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.checkpoint = UVCheckpoint(
            b_base=np.empty((0, 2)),
            length=1.0,
            height=0.6,
            c0=340.0,
            cases={},
            metadata={"defect_name": "barhalf", "contrast_ratio": 0.8},
        )

    def tearDown(self):
        plt.close("all")

    def test_default_ground_truth_uses_checkpoint_contrast(self):
        x_grid, y_grid = build_grid(self.checkpoint, nx=11, ny=7)
        ground_truth = build_ground_truth(self.checkpoint, x_grid, y_grid)
        center = ground_truth[3, 5]
        exterior = ground_truth[-1, -1]
        self.assertAlmostEqual(center, 272.0)
        self.assertAlmostEqual(exterior, 340.0)

    def test_metrics_are_zero_for_identical_fields(self):
        values = np.full((4, 5), 340.0)
        metrics = compute_misfit(values, values, background_speed=340.0)
        self.assertEqual(metrics.rmse, 0.0)
        self.assertEqual(metrics.relative_l1, 0.0)
        self.assertIsNone(metrics.anomaly_relative_l1)
        self.assertIsNone(metrics.anomaly_mean_absolute)
        self.assertEqual(metrics.relative_l2, 0.0)
        self.assertEqual(metrics.mean_absolute, 0.0)
        self.assertEqual(metrics.background_mean_absolute, 0.0)
        self.assertEqual(metrics.homogeneous_mean_absolute, 0.0)
        self.assertEqual(metrics.max_absolute, 0.0)

    def test_homogeneous_prediction_scores_one_when_it_misses_the_anomaly(self):
        ground_truth = np.asarray([[340.0, 272.0], [340.0, 272.0]])
        reconstructed = np.full_like(ground_truth, 340.0)
        metrics = compute_misfit(
            reconstructed,
            ground_truth,
            background_speed=340.0,
        )

        self.assertAlmostEqual(metrics.anomaly_relative_l1, 1.0)
        self.assertAlmostEqual(metrics.improvement_over_homogeneous, 0.0)
        self.assertAlmostEqual(metrics.anomaly_mean_absolute, 68.0)
        self.assertAlmostEqual(metrics.background_mean_absolute, 0.0)
        self.assertAlmostEqual(metrics.homogeneous_mean_absolute, 34.0)

    def test_all_maps_are_rasterized(self):
        x_grid, y_grid = build_grid(self.checkpoint, nx=11, ny=7)
        ground_truth = build_ground_truth(self.checkpoint, x_grid, y_grid)
        reconstructed = ground_truth + 1.0
        metrics = compute_misfit(
            reconstructed,
            ground_truth,
            background_speed=self.checkpoint.c0,
        )
        figures = (
            create_reconstruction_figure(x_grid, y_grid, reconstructed),
            create_comparison_figure(
                x_grid, y_grid, reconstructed, ground_truth
            ),
            create_misfit_figure(
                x_grid, y_grid, reconstructed, ground_truth, metrics
            ),
        )
        for figure in figures:
            meshes = [
                collection
                for axis in figure.axes
                for collection in axis.collections
                if isinstance(collection, QuadMesh)
            ]
            self.assertTrue(meshes)
            self.assertTrue(all(mesh.get_rasterized() for mesh in meshes))

    def test_ground_truth_function_supports_multiple_independent_speeds(self):
        x_grid, y_grid = build_grid(self.checkpoint, nx=11, ny=7)

        def multiple_defects(x, y, checkpoint):
            values = np.full(x.shape, checkpoint.c0)
            values[(x < -0.4) & (y < 0.2)] = 250.0
            values[(x > 0.4) & (y > 0.4)] = 300.0
            return values

        with patch(
            "tools.compare_sound_speed.ground_truth_sound_speed",
            side_effect=multiple_defects,
        ):
            ground_truth = build_ground_truth(self.checkpoint, x_grid, y_grid)

        self.assertIn(250.0, ground_truth)
        self.assertIn(300.0, ground_truth)
        self.assertIn(340.0, ground_truth)

    def test_registry_uses_checkpoint_defect_name(self):
        x = np.asarray([-0.2, 0.2, 0.0])
        y = np.asarray([0.2, 0.2, 0.5])

        left = build_registered_sound_speed(
            x,
            y,
            defect_name="circlebottomleft",
            c0=340.0,
            contrast_ratio=0.8,
        )
        right = build_registered_sound_speed(
            x,
            y,
            defect_name="circlebottomright",
            c0=340.0,
            contrast_ratio=0.8,
        )

        np.testing.assert_allclose(left, [272.0, 340.0, 340.0])
        np.testing.assert_allclose(right, [340.0, 272.0, 340.0])

    def test_registry_rejects_unknown_defect(self):
        with self.assertRaisesRegex(ValueError, "Unknown ground-truth defect"):
            build_registered_sound_speed(
                np.asarray([0.0]),
                np.asarray([0.0]),
                defect_name="typo",
                c0=340.0,
                contrast_ratio=0.8,
            )


if __name__ == "__main__":
    unittest.main()
