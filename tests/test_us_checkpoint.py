import csv
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from tools.compare_scattered_pinn_fem import prepare_comparisons
from tools.compare_scattered_sound_speed import build_grid, compute_misfit
from tools.data_loader import SymmetricCOOMatrix
from tools.us_checkpoint import (
    MSSoundSpeedModel,
    USCase,
    USCheckpoint,
    export_predictions_csv,
    load_us_checkpoint,
    save_us_checkpoint,
)


MS_MIN = -1.0e-7
MS_MAX = 1.0e-5


def synthetic_parameters():
    params_us = {
        (600.0, 0): {
            "sigma": np.asarray([1.5, 0.5], dtype=np.float32),
            "layers": [
                {
                    "W": np.arange(8, dtype=np.float32).reshape(4, 2) / 20.0,
                    "b": np.asarray([0.1, -0.2], dtype=np.float32),
                }
            ],
        }
    }
    layers_ms = [
        {
            "W": np.asarray(
                [[0.25, -0.1, 0.2], [0.05, 0.3, -0.15]], dtype=np.float32
            ),
            "b": np.asarray([0.0, 0.1, -0.1], dtype=np.float32),
        },
        {
            "W": np.asarray([[0.2], [-0.3], [0.4]], dtype=np.float32),
            "b": np.asarray([0.05], dtype=np.float32),
        },
    ]
    return params_us, layers_ms


def save_synthetic_checkpoint(directory: str | Path) -> USCheckpoint:
    params_us, layers_ms = synthetic_parameters()
    b_base = np.asarray([[0.2, 0.4], [-0.1, 0.3]], dtype=np.float32)
    path = Path(directory) / "scattered.npz"
    save_us_checkpoint(
        path,
        params_us,
        b_base,
        {(600.0, 0): 2.5},
        length=1.0,
        height=0.6,
        c0=340.0,
        layers_ms=layers_ms,
        ms_min=MS_MIN,
        ms_max=MS_MAX,
        network_config={
            "us_layers": [4, 2],
            "ms_layers": [2, 3, 1],
            "fourier_features": 2,
            "hidden_activation": "tanh",
            "us_feature_mapping": "random_fourier_cos_sin",
            "ms_output_parameterization": "bounded_scattered_slowness_tanh",
        },
        best_validation_losses={
            "warmup": [{"package": 2, "weighted_total": 0.3}],
            "inverse": [{"package": 1, "weighted_total": 0.1}],
        },
        random_seed=7,
        metadata={
            "defect_name": "circlebottomleft",
            "contrast_ratio": 0.8,
            "field_formulation": "u_total = u0 + us",
        },
        provenance={
            "created_at_utc": "2026-07-05T00:00:00+00:00",
            "git_commit": "abc123",
            "git_dirty": False,
            "python": "3.test",
            "packages": {},
        },
    )
    return load_us_checkpoint(path)


class USCheckpointTests(unittest.TestCase):
    def test_existing_checkpoint_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            save_synthetic_checkpoint(directory)
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                save_synthetic_checkpoint(directory)

    def test_round_trip_preserves_fields_and_sound_speed(self):
        params_us, layers_ms = synthetic_parameters()
        b_base = np.asarray([[0.2, 0.4], [-0.1, 0.3]], dtype=np.float32)
        reference = USCheckpoint(
            b_base=b_base,
            length=1.0,
            height=0.6,
            c0=340.0,
            cases={
                (600.0, 0): USCase(
                    frequency=600.0,
                    mode=0,
                    field_norm=2.5,
                    sigma=params_us[(600.0, 0)]["sigma"],
                    layers=params_us[(600.0, 0)]["layers"],
                )
            },
            ms_model=MSSoundSpeedModel(
                layers=layers_ms,
                c0=340.0,
                ms_min=MS_MIN,
                ms_max=MS_MAX,
            ),
            metadata={"defect_name": "circlebottomleft", "contrast_ratio": 0.8},
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_synthetic_checkpoint(directory)

            self.assertEqual(checkpoint.format_version, 1)
            self.assertEqual(checkpoint.random_seed, 7)
            self.assertEqual(
                checkpoint.best_validation_losses["inverse"][0]["weighted_total"],
                0.1,
            )
            self.assertEqual(
                checkpoint.best_validation_losses["warmup"][0]["weighted_total"],
                0.3,
            )

            x = np.asarray([-0.5, 0.0, 0.5])
            y = np.asarray([0.1, 0.3, 0.5])
            np.testing.assert_allclose(
                checkpoint.predict_us_physical(600.0, 0, x, y),
                reference.predict_us_physical(600.0, 0, x, y),
            )
            np.testing.assert_allclose(
                checkpoint.predict_incident_physical(600.0, 0, x, y),
                reference.predict_incident_physical(600.0, 0, x, y),
            )
            np.testing.assert_allclose(
                checkpoint.predict_total_physical(600.0, 0, x, y),
                reference.predict_total_physical(600.0, 0, x, y),
            )
            np.testing.assert_allclose(
                checkpoint.predict_sound_speed_physical(x, y),
                reference.predict_sound_speed_physical(x, y),
            )
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_total_is_scattered_plus_incident_and_scaling_is_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_synthetic_checkpoint(directory)
            x_norm = np.asarray([-0.5, 0.0, 0.5])
            y_norm = np.asarray([-0.2, 0.0, 0.2])
            us = checkpoint.predict_us(600.0, 0, x_norm, y_norm)
            incident = checkpoint.predict_incident(600.0, 0, x_norm, y_norm)
            total = checkpoint.predict_total(600.0, 0, x_norm, y_norm)
            np.testing.assert_allclose(total, us + incident)

            us_normalized = checkpoint.predict_us(
                600.0, 0, x_norm, y_norm, physical_units=False
            )
            np.testing.assert_allclose(us, 2.5 * us_normalized)
            incident_normalized = checkpoint.predict_incident(
                600.0, 0, x_norm, y_norm, physical_units=False
            )
            np.testing.assert_allclose(incident, 2.5 * incident_normalized)

    def test_loader_rejects_uv_checkpoint_format(self):
        manifest = {
            "format": "uv_checkpoint_npz",
            "format_version": 2,
            "geometry": {"length": 1.0, "height": 0.6},
            "c0": 340.0,
            "cases": [],
            "metadata": {"defect_name": "barhalf", "contrast_ratio": 0.8},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "uv.npz"
            with path.open("wb") as stream:
                np.savez_compressed(stream, manifest_json=np.asarray(json.dumps(manifest)))
            with self.assertRaisesRegex(ValueError, "us_checkpoint_npz"):
                load_us_checkpoint(path)

    def test_invalid_inputs_are_rejected(self):
        params_us, layers_ms = synthetic_parameters()
        b_base = np.asarray([[0.2, 0.4], [-0.1, 0.3]], dtype=np.float32)
        common_kwargs = dict(
            length=1.0,
            height=0.6,
            c0=340.0,
            layers_ms=layers_ms,
            ms_min=MS_MIN,
            ms_max=MS_MAX,
            network_config={"us_layers": [4, 2], "ms_layers": [2, 3, 1]},
            best_validation_losses={},
            random_seed=0,
            metadata={"defect_name": "circlebottomleft", "contrast_ratio": 0.8},
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "ms bounds"):
                save_us_checkpoint(
                    Path(directory) / "bad_bounds.npz",
                    params_us,
                    b_base,
                    {(600.0, 0): 2.5},
                    **{**common_kwargs, "ms_min": 1.0e-5, "ms_max": 1.0e-7},
                )
            with self.assertRaisesRegex(ValueError, "ms bounds"):
                save_us_checkpoint(
                    Path(directory) / "positive_bounds.npz",
                    params_us,
                    b_base,
                    {(600.0, 0): 2.5},
                    **{**common_kwargs, "ms_min": 1.0e-7, "ms_max": 1.0e-5},
                )
            with self.assertRaisesRegex(KeyError, "field_norm"):
                save_us_checkpoint(
                    Path(directory) / "missing_norm.npz",
                    params_us,
                    b_base,
                    {},
                    **common_kwargs,
                )
            bad_params = {
                (600.0, 0): {
                    "sigma": np.asarray([1.0, 1.0]),
                    "layers": [{"W": np.zeros((3, 2)), "b": np.zeros(2)}],
                }
            }
            with self.assertRaisesRegex(ValueError, "2\\*fourier_features"):
                save_us_checkpoint(
                    Path(directory) / "bad_layers.npz",
                    bad_params,
                    b_base,
                    {(600.0, 0): 2.5},
                    **common_kwargs,
                )

            checkpoint = save_synthetic_checkpoint(directory)
            with self.assertRaisesRegex(ValueError, "outside"):
                checkpoint.predict_total_physical(
                    600.0,
                    0,
                    np.asarray([2.0]),
                    np.asarray([0.3]),
                )

    def test_export_writes_total_field_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_synthetic_checkpoint(directory)
            output = Path(directory) / "predictions.csv"
            grid = {
                "grid_i": np.asarray([0, 1]),
                "x": np.asarray([-0.5, 0.5]),
                "y": np.asarray([0.2, 0.4]),
                "x_norm": np.asarray([-0.5, 0.5]),
                "y_norm": np.asarray([2.0 * 0.2 / 0.6 - 1.0, 2.0 * 0.4 / 0.6 - 1.0]),
            }
            export_predictions_csv(output, checkpoint, grid)
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

            self.assertEqual(set(rows[0].keys()), {
                "f",
                "k0",
                "mode",
                "grid_i",
                "x",
                "y",
                "x_norm",
                "y_norm",
                "Re_U",
                "Im_U",
                "abs_U",
            })
            expected = checkpoint.predict_total(
                600.0, 0, grid["x_norm"], grid["y_norm"]
            )
            self.assertAlmostEqual(float(rows[0]["Re_U"]), expected[0].real)
            self.assertAlmostEqual(float(rows[0]["Im_U"]), expected[0].imag)

    def test_sound_speed_comparison_accepts_us_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_synthetic_checkpoint(directory)
            x_grid, y_grid = build_grid(checkpoint, nx=5, ny=4)
            reconstructed = checkpoint.predict_sound_speed_physical(x_grid, y_grid)
            metrics = compute_misfit(reconstructed, reconstructed, checkpoint.c0)
            self.assertEqual(metrics.mean_absolute, 0.0)
            self.assertEqual(metrics.relative_l2, 0.0)

    def test_scattered_fem_comparison_has_zero_misfit_for_matching_total_field(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_synthetic_checkpoint(directory)
            x = np.asarray([-0.5, 0.0, 0.5])
            y = np.asarray([0.1, 0.3, 0.5])
            values = checkpoint.predict_total_physical(600.0, 0, x, y)
            fem_data = _FakeFEMData(x=x, y=y, values=values, c0=checkpoint.c0)
            mass_matrix = SymmetricCOOMatrix(
                size=x.size,
                rows=np.arange(x.size),
                columns=np.arange(x.size),
                values=np.ones(x.size),
            )
            result = prepare_comparisons(
                checkpoint,
                fem_data,
                mass_matrix,
                stiffness_matrix=None,
                cases=[(600.0, 0)],
            )[0]
            self.assertEqual(result.metrics.l2_absolute, 0.0)
            self.assertEqual(result.metrics.l2_relative, 0.0)


@dataclass(frozen=True)
class _FakeFEMCase:
    k0: float
    values: np.ndarray


class _FakeFEMData:
    def __init__(self, x: np.ndarray, y: np.ndarray, values: np.ndarray, c0: float):
        self.x = x
        self.y = y
        self.size = x.size
        self.available_cases = ((600.0, 0),)
        self._case = _FakeFEMCase(k0=2.0 * np.pi * 600.0 / c0, values=values)

    def has_case(self, frequency: float, mode: int) -> bool:
        return np.isclose(frequency, 600.0) and mode == 0

    def case(self, frequency: float, mode: int) -> _FakeFEMCase:
        if not self.has_case(frequency, mode):
            raise KeyError((frequency, mode))
        return self._case


if __name__ == "__main__":
    unittest.main()
