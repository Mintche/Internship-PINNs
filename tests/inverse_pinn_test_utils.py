from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def _write_csv(path: Path, fieldnames, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_inverse_config(
    root: Path,
    *,
    frequencies=(1200.0,),
    package_frequencies=None,
    warmup_steps=0,
    inverse_steps=1,
    lbfgs_cycles=0,
    lbfgs_field_steps=0,
    lbfgs_material_steps=0,
    snapshot_fractions=(1.0,),
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    height, half_length, c0 = 0.6, 1.0, 340.0
    nodes = [
        (0, -1.0, 0.0, 0),
        (1, 1.0, 0.0, 0),
        (2, -1.0, height, 0),
        (3, 0.0, 0.0, 0),
        (4, 0.0, height / 2.0, 0),
        (5, -1.0, height / 2.0, 0),
    ]
    boundary_left, boundary_right, fem_rows = [], [], []
    modal_scale = np.sqrt(1.0 / height)
    for frequency in frequencies:
        k0 = 2.0 * np.pi * frequency / c0
        for x, target in ((-half_length, boundary_left), (half_length, boundary_right)):
            incident = modal_scale * np.exp(1j * k0 * x)
            value = incident + 0.2 + 0.1j
            for y in (0.0, height / 2.0, height):
                target.append(
                    {
                        "incidence": -1,
                        "f": frequency,
                        "k0": k0,
                        "mode": 0,
                        "x": x,
                        "y": y,
                        "Re_U": value.real,
                        "Im_U": value.imag,
                    }
                )
        for node_id, x, y, _ in nodes:
            incident = modal_scale * np.exp(1j * k0 * x)
            value = incident + (0.2 + 0.1j) * (1.0 - 0.2 * x * y)
            fem_rows.append(
                {
                    "incidence": -1,
                    "f": frequency,
                    "k0": k0,
                    "mode": 0,
                    "node_id": node_id,
                    "x": x,
                    "y": y,
                    "Re_U": value.real,
                    "Im_U": value.imag,
                }
            )
    boundary_columns = ("incidence", "f", "k0", "mode", "x", "y", "Re_U", "Im_U")
    _write_csv(root / "left.csv", boundary_columns, boundary_left)
    _write_csv(root / "right.csv", boundary_columns, boundary_right)
    _write_csv(
        root / "field.csv",
        (*boundary_columns[:4], "node_id", "x", "y", "Re_U", "Im_U"),
        fem_rows,
    )
    _write_csv(
        root / "nodes.csv", ("node_id", "x", "y", "ref"),
        [dict(zip(("node_id", "x", "y", "ref"), row)) for row in nodes],
    )
    _write_csv(
        root / "triangles.csv",
        ("triangle_id", "n0", "n1", "n2", "n3", "n4", "n5", "ref"),
        [{"triangle_id": 0, "n0": 0, "n1": 1, "n2": 2, "n3": 3, "n4": 4, "n5": 5, "ref": 0}],
    )
    matrix_rows = [{"row": index, "col": index, "value": 1.0} for index in range(6)]
    _write_csv(root / "mass.csv", ("row", "col", "value"), matrix_rows)
    _write_csv(root / "stiffness.csv", ("row", "col", "value"), matrix_rows)

    package_frequencies = package_frequencies or (tuple(frequencies),)
    packages = []
    for package in package_frequencies:
        packages.append(
            {
                "cases": {"-1": {f"{frequency:g}": [0] for frequency in package}},
                "warmup": {"adam_steps": warmup_steps, "lbfgs_steps": 0},
                "inverse": {
                    "adam_steps": inverse_steps,
                    "lbfgs_cycles": lbfgs_cycles,
                    "lbfgs_field_steps": lbfgs_field_steps,
                    "lbfgs_material_steps": lbfgs_material_steps,
                },
            }
        )
    config = {
        "geometry": {
            "height": height,
            "half_length": half_length,
            "c0": c0,
            "celerity_ratio_bounds": [0.7, 1.2],
            "truth_regions": [
                {"shape": "circle", "center": [0.0, 0.2], "radius": 0.1, "speed_ratio": 0.8}
            ],
        },
        "data": {
            "boundary_left": str(root / "left.csv"),
            "boundary_right": str(root / "right.csv"),
            "fem_field": str(root / "field.csv"),
            "mass_matrix": str(root / "mass.csv"),
            "stiffness_matrix": str(root / "stiffness.csv"),
            "mesh_nodes": str(root / "nodes.csv"),
            "mesh_triangles": str(root / "triangles.csv"),
        },
        "models": {
            "fourier_features": 2,
            "field_hidden_layers": [4],
            "material_hidden_layers": [4],
        },
        "loss": {
            "field_weights": [1.0, 1.0, 1.0, 1.0],
            "field_adweights": {
                "epsilon": 1e-4,
                "alpha": 0.1,
                "initial_lambdas": [0.01, 0.01, 0.01, 0.01],
                "custom_weights": [1.0, 1.0, 1.0, 1.0],
                "update_interval_adam": 1,
            },
            "material_adweights": {
                "epsilon": 1e-4,
                "alpha": 0.1,
                "initial_lambda": 0.01,
                "custom_weight": 1.0,
                "update_interval_adam": 1,
            },
            "tv": {"weight": 1e-4, "epsilon_squared": 1e-12},
        },
        "optimization": {
            "field_learning_rate": 1e-3,
            "material_learning_rate": 1e-3,
            "cosine_decay_start": 10_000,
            "consine_decay_stop": 20_000,
            "cosine_decay_alpha": 0.1,
            "sigma_learning_rate": 1e-2,
            "sigma_decay_fraction": 0.5,
            "sigma_cosine_alpha": 1e-3,
            "data_initial_factor": 0.1,
            "data_transition_steps": 10_000,
        },
        "sampling": {
            "adam": [4, 2, 2],
            "monitor": [4, 2, 2],
            "lbfgs": [4, 2, 2],
            "sobol_scramble": True,
            "sobol_seed_offset": 10,
        },
        "logging": {
            "loss_interval_adam": 1,
            "loss_interval_lbfgs": 1,
            "pressure_gradient_interval_adam": 1,
            "sigma_interval_adam": 1,
            "print_interval_adam": 1,
            "material_snapshot_fractions": list(snapshot_fractions),
            "snapshot_grid": [4, 3],
            "prediction_batch_size": 16,
        },
        "training_packages": packages,
        "output_root": str(root / "outputs"),
    }
    path = root / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path
