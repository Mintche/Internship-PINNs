from __future__ import annotations

import jax
import numpy as np

from inverse_PINN.checkpoints import (
    load_material_checkpoint,
    load_pressure_checkpoint,
    save_material_checkpoint,
    save_pressure_checkpoint,
)
from inverse_PINN.config import Case, GeometryConfig, ModelConfig
from inverse_PINN.models import initialize_field_model, initialize_material_model
from inverse_PINN.variants import parse_variant


def test_npz_checkpoints_round_trip_without_pickle(tmp_path):
    geometry = GeometryConfig(0.6, 1.0, 340.0, (0.7, 1.2), ())
    model_config = ModelConfig(3, (4,), (5,))
    case = Case(1200.0, 0, -1)
    field = initialize_field_model(
        jax.random.key(0), model_config, geometry, case, parse_variant("fourier_total")
    )
    material = initialize_material_model(jax.random.key(1), model_config, geometry)
    pressure_path = tmp_path / "pressure.npz"
    material_path = tmp_path / "material.npz"
    save_pressure_checkpoint(
        pressure_path, {case: field}, {case: 2.5}, variant="fourier_total",
        package_index=0, monitor_loss=1.2,
    )
    save_material_checkpoint(
        material_path, material, variant="fourier_total", package_index=0,
        monitor_loss=1.2,
    )
    loaded_fields, scales, pressure_metadata = load_pressure_checkpoint(pressure_path)
    loaded_material, material_metadata = load_material_checkpoint(material_path)
    assert scales[case] == 2.5
    assert pressure_metadata["variant"] == material_metadata["variant"] == "fourier_total"
    for expected, actual in zip(
        jax.tree_util.tree_leaves(field.params),
        jax.tree_util.tree_leaves(loaded_fields[case].params),
    ):
        np.testing.assert_array_equal(expected, actual)
    for expected, actual in zip(
        jax.tree_util.tree_leaves(material), jax.tree_util.tree_leaves(loaded_material)
    ):
        np.testing.assert_array_equal(expected, actual)

