import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.uv_checkpoint import (
    SoundSpeedModel,
    UVCase,
    UVCheckpoint,
    load_uv_checkpoint,
    save_uv_checkpoint,
)


def synthetic_parameters():
    params_uv = {
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
    layers_m = [
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
    return params_uv, layers_m


class UVCheckpointTests(unittest.TestCase):
    def test_v2_round_trip_preserves_uv_and_sound_speed_predictions(self):
        params_uv, layers_m = synthetic_parameters()
        b_base = np.asarray([[0.2, 0.4], [-0.1, 0.3]], dtype=np.float32)
        losses = {
            "forward": [{"frequency": 600.0, "mode": 0, "weighted_total": 0.2}],
            "inverse": [{"frequency": 600.0, "weighted_total": 0.1}],
        }
        provenance = {
            "created_at_utc": "2026-07-05T00:00:00+00:00",
            "git_commit": "abc123",
            "git_dirty": False,
            "python": "3.test",
            "packages": {},
        }
        reference = UVCheckpoint(
            b_base=b_base,
            length=1.0,
            height=0.6,
            c0=340.0,
            cases={
                (600.0, 0): UVCase(
                    frequency=600.0,
                    mode=0,
                    u_norm=2.5,
                    sigma=params_uv[(600.0, 0)]["sigma"],
                    layers=params_uv[(600.0, 0)]["layers"],
                )
            },
            metadata={"defect_name": "barhalf", "contrast_ratio": 0.8},
            sound_speed=SoundSpeedModel(
                layers=layers_m,
                c_min=204.0,
                c_max=476.0,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            save_uv_checkpoint(
                path,
                params_uv,
                b_base,
                {(600.0, 0): 2.5},
                length=1.0,
                height=0.6,
                c0=340.0,
                layers_m=layers_m,
                c_min=204.0,
                c_max=476.0,
                network_config={
                    "uv_layers": [4, 2],
                    "m_layers": [2, 3, 1],
                    "fourier_features": 2,
                    "hidden_activation": "tanh",
                    "uv_feature_mapping": "random_fourier_cos_sin",
                    "m_output_parameterization": "bounded_slowness_sigmoid",
                },
                best_validation_losses=losses,
                random_seed=7,
                metadata={"defect_name": "barhalf", "contrast_ratio": 0.8},
                provenance=provenance,
            )
            checkpoint = load_uv_checkpoint(path)

            self.assertEqual(checkpoint.format_version, 2)
            self.assertEqual(checkpoint.random_seed, 7)
            self.assertEqual(checkpoint.provenance, provenance)
            self.assertEqual(checkpoint.best_validation_losses, losses)
            self.assertIsNotNone(checkpoint.sound_speed)

            x = np.asarray([-0.5, 0.0, 0.5])
            y = np.asarray([0.1, 0.3, 0.5])
            uv = checkpoint.predict_physical(600.0, 0, x, y)
            speed = checkpoint.predict_sound_speed_physical(x, y)
            np.testing.assert_allclose(
                uv,
                reference.predict_physical(600.0, 0, x, y),
            )
            np.testing.assert_allclose(
                speed,
                reference.predict_sound_speed_physical(x, y),
            )
            self.assertEqual(uv.shape, x.shape)
            self.assertEqual(speed.shape, x.shape)
            self.assertTrue(np.isfinite(uv).all())
            self.assertTrue(np.isfinite(speed).all())
            self.assertTrue((speed >= 204.0).all())
            self.assertTrue((speed <= 476.0).all())
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_v1_checkpoint_stays_loadable(self):
        manifest = {
            "format": "uv_checkpoint_npz",
            "format_version": 1,
            "geometry": {"length": 1.0, "height": 0.6},
            "c0": 340.0,
            "cases": [
                {
                    "prefix": "case_0",
                    "frequency": 600.0,
                    "mode": 0,
                    "u_norm": 1.0,
                    "n_layers": 1,
                }
            ],
            "metadata": {"defect_name": "barhalf", "contrast_ratio": 0.8},
        }
        arrays = {
            "manifest_json": np.asarray(json.dumps(manifest)),
            "b_base": np.asarray([[0.25, 0.5]], dtype=np.float32),
            "case_0_sigma": np.asarray([1.0, 1.0], dtype=np.float32),
            "case_0_layer_0_W": np.zeros((2, 2), dtype=np.float32),
            "case_0_layer_0_b": np.asarray([1.0, -1.0], dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1.npz"
            with path.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            checkpoint = load_uv_checkpoint(path)

        self.assertEqual(checkpoint.format_version, 1)
        self.assertIsNone(checkpoint.sound_speed)
        values = checkpoint.predict(600.0, 0, np.asarray([0.0]), np.asarray([0.0]))
        np.testing.assert_allclose(values, np.asarray([1.0 - 1.0j]))
        with self.assertRaisesRegex(ValueError, "v2"):
            checkpoint.predict_sound_speed_physical(
                np.asarray([0.0]), np.asarray([0.3])
            )

    def test_existing_checkpoint_is_not_overwritten(self):
        params_uv, layers_m = synthetic_parameters()
        arguments = dict(
            length=1.0,
            height=0.6,
            c0=340.0,
            layers_m=layers_m,
            c_min=204.0,
            c_max=343.4,
            network_config={"uv_layers": [4, 2], "m_layers": [2, 3, 1]},
            best_validation_losses={},
            random_seed=0,
            metadata={"defect_name": "barhalf", "contrast_ratio": 0.8},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            save_uv_checkpoint(
                path,
                params_uv,
                np.ones((2, 2)),
                {(600.0, 0): 1.0},
                **arguments,
            )
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                save_uv_checkpoint(
                    path,
                    params_uv,
                    np.ones((2, 2)),
                    {(600.0, 0): 1.0},
                    **arguments,
                )

    def test_invalid_sound_speed_bounds_are_rejected(self):
        params_uv, layers_m = synthetic_parameters()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "bounds"):
                save_uv_checkpoint(
                    Path(directory) / "invalid.npz",
                    params_uv,
                    np.ones((2, 2)),
                    {(600.0, 0): 1.0},
                    length=1.0,
                    height=0.6,
                    c0=340.0,
                    layers_m=layers_m,
                    c_min=400.0,
                    c_max=300.0,
                    network_config={"uv_layers": [4, 2], "m_layers": [2, 3, 1]},
                    best_validation_losses={},
                    random_seed=0,
                    metadata={"defect_name": "barhalf", "contrast_ratio": 0.8},
                )


if __name__ == "__main__":
    unittest.main()
