import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.experiment_manifest import (
    build_run_tag,
    canonical_manifest,
    configuration_id,
    ensure_outputs_available,
    write_manifest_exclusive,
)


class ExperimentManifestTests(unittest.TestCase):
    def test_configuration_id_is_order_independent_and_sensitive(self):
        first = {
            "optimizer": {"steps": 100, "weights": (1.0, 3.0, 10.0)},
            "seed": np.int64(2),
        }
        reordered = {
            "seed": 2,
            "optimizer": {"weights": [1.0, 3.0, 10.0], "steps": 100},
        }
        changed = {
            "seed": 2,
            "optimizer": {"weights": [1.0, 3.0, 10.0], "steps": 101},
        }

        self.assertEqual(configuration_id(first), configuration_id(reordered))
        self.assertNotEqual(configuration_id(first), configuration_id(changed))

    def test_manifest_rejects_non_finite_values(self):
        with self.assertRaises(ValueError):
            canonical_manifest({"learning_rate": float("nan")})

    def test_run_tag_exposes_primary_identity_fields(self):
        configuration = {"seed": 7, "collocation": [4096, 256, 128]}
        tag = build_run_tag(
            formulation="scattered",
            random_seed=7,
            c_min=204.0,
            c_max=343.4,
            configuration=configuration,
        )

        self.assertRegex(
            tag,
            r"^scattered_seed7_c204to343p4_cfg[0-9a-f]{12}$",
        )

    def test_existing_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "run.npz"
            existing.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                ensure_outputs_available(existing, Path(directory) / "run.json")

    def test_sidecar_is_created_exclusively(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.config.json"
            configuration = {"random_seed": 3, "path": Path("FEM/pinn_data")}
            write_manifest_exclusive(
                path,
                configuration,
                experiment_id="total_seed3_cfg123",
            )
            payload = json.loads(path.read_text())

            self.assertEqual(payload["experiment_id"], "total_seed3_cfg123")
            self.assertEqual(payload["configuration"]["random_seed"], 3)
            self.assertEqual(payload["configuration"]["path"], "FEM/pinn_data")
            with self.assertRaises(FileExistsError):
                write_manifest_exclusive(
                    path,
                    configuration,
                    experiment_id="total_seed3_cfg123",
                )


if __name__ == "__main__":
    unittest.main()
