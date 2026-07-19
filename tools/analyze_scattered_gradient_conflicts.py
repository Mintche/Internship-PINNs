#!/usr/bin/env python3
"""Measure per-acquisition coefficient-gradient conflicts in a US/MS checkpoint.

The scattered PINN data and boundary losses do not depend directly on the
coefficient-network parameters.  Holding the field networks fixed, the
coefficient gradient is therefore exactly the gradient of the PDE loss.  This
tool evaluates those per-case gradients on one common fixed collocation grid.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.flatten_util import ravel_pytree


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pinn_waveguide_2d import pinn_scattered_waveguide as scattered  # noqa: E402
from tools.us_checkpoint import USCheckpoint, load_us_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute PDE-gradient norms and pairwise cosines with respect to the "
            "stored scattered-slowness network for every checkpoint case."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--n-pde", type=int, default=4096)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "gradient_diagnostics",
    )
    return parser.parse_args()


def _jax_layers(layers: list[dict[str, np.ndarray]]) -> list[dict[str, jax.Array]]:
    return [
        {
            "W": jnp.asarray(layer["W"], dtype=jnp.float32),
            "b": jnp.asarray(layer["b"], dtype=jnp.float32),
        }
        for layer in layers
    ]


def _configure_checkpoint_physics(checkpoint: USCheckpoint) -> None:
    """Configure the exact retained residual helpers from a portable checkpoint."""
    scattered.L = float(checkpoint.length)
    scattered.H = float(checkpoint.height)
    scattered.c0 = float(checkpoint.c0)
    scattered.m0 = 1.0 / checkpoint.c0**2
    scattered.ms_min = float(checkpoint.ms_model.ms_min)
    scattered.ms_max = float(checkpoint.ms_model.ms_max)
    scattered.B_base = jnp.asarray(checkpoint.b_base, dtype=jnp.float32)

    maximum_mode = max(mode for _, mode in checkpoint.available_cases())
    mode_count = maximum_mode + 6
    scattered.n_modes = jnp.arange(mode_count, dtype=jnp.float32)
    amplitudes = jnp.sqrt(2.0 / checkpoint.height) * jnp.ones(
        mode_count, dtype=jnp.float32
    )
    scattered.a_n = amplitudes.at[0].set(jnp.sqrt(1.0 / checkpoint.height))


def _normalized_case_parameters(checkpoint: USCheckpoint, frequency: float, mode: int):
    case = checkpoint.case(frequency, mode)
    return {
        "layers": _jax_layers(case.layers),
        "sigma": jnp.asarray(case.sigma, dtype=jnp.float32),
    }, jnp.asarray(case.field_norm, dtype=jnp.float32)


def _make_value_and_gradient(x_pde: jax.Array, y_pde: jax.Array):
    def case_pde_loss(
        layers_ms,
        params_us,
        frequency,
        mode_index,
        field_norm,
    ):
        def residual(x, y):
            us_value, laplacian_us = scattered.us_value_and_physical_laplacian(
                params_us, x, y
            )
            ms_value = scattered.forward_ms(layers_ms, jnp.array([x, y]))
            u0_value = scattered.create_incident_wave_mode(
                x, y, frequency, mode_index, field_norm
            )
            omega = 2.0 * jnp.pi * frequency
            return laplacian_us + omega**2 * (
                scattered.m0 * us_value + ms_value * (us_value + u0_value)
            )

        values = jax.vmap(residual, in_axes=(0, 0))(x_pde, y_pde)
        return jnp.mean(values**2)

    return jax.jit(jax.value_and_grad(case_pde_loss, argnums=0))


def analyze_checkpoint(checkpoint: USCheckpoint, n_pde: int):
    if n_pde <= 0:
        raise ValueError("n_pde must be strictly positive")
    _configure_checkpoint_physics(checkpoint)
    x_pde, y_pde, _, _ = scattered.regular_collocation_points((n_pde, 1, 1))
    value_and_gradient = _make_value_and_gradient(x_pde, y_pde)
    layers_ms = _jax_layers(checkpoint.ms_model.layers)

    rows = []
    gradient_vectors = []
    for frequency, mode in checkpoint.available_cases():
        params_us, field_norm = _normalized_case_parameters(
            checkpoint, frequency, mode
        )
        loss, gradient = value_and_gradient(
            layers_ms,
            params_us,
            jnp.asarray(frequency, dtype=jnp.float32),
            jnp.asarray(mode, dtype=jnp.int32),
            field_norm,
        )
        flat_gradient, _ = ravel_pytree(gradient)
        flat_gradient = np.asarray(flat_gradient.block_until_ready(), dtype=np.float64)
        gradient_norm = float(np.linalg.norm(flat_gradient))
        rows.append(
            {
                "label": f"f{frequency:g}_m{mode}",
                "frequency": frequency,
                "mode": mode,
                "field_norm": float(field_norm),
                "pde_loss": float(loss),
                "gradient_norm": gradient_norm,
            }
        )
        gradient_vectors.append(flat_gradient)

    gradients = np.stack(gradient_vectors)
    norms = np.linalg.norm(gradients, axis=1)
    denominator = norms[:, None] * norms[None, :]
    cosine_matrix = np.divide(
        gradients @ gradients.T,
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 0.0,
    )
    return rows, cosine_matrix


def write_outputs(
    checkpoint_path: Path,
    rows: list[dict[str, float | int | str]],
    cosine_matrix: np.ndarray,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = checkpoint_path.stem
    cases_path = output_dir / f"{stem}_case_gradients.csv"
    matrix_path = output_dir / f"{stem}_gradient_cosines.csv"
    figure_path = output_dir / f"{stem}_gradient_cosines.pdf"

    with cases_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = [str(row["label"]) for row in rows]
    with matrix_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", *labels])
        for label, values in zip(labels, cosine_matrix):
            writer.writerow([label, *values])

    figure_size = max(5.0, 0.75 * len(labels) + 2.0)
    figure, axis = plt.subplots(figsize=(figure_size, figure_size - 0.5))
    image = axis.imshow(cosine_matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels=labels)
    for row_index in range(len(labels)):
        for column_index in range(len(labels)):
            value = cosine_matrix[row_index, column_index]
            text_color = "white" if abs(value) > 0.55 else "black"
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )
    axis.set_title("Coefficient-network PDE-gradient cosine at stored checkpoint")
    figure.colorbar(image, ax=axis, label="cosine")
    figure.tight_layout()
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)
    return cases_path, matrix_path, figure_path


def main() -> None:
    args = parse_args()
    checkpoint = load_us_checkpoint(args.checkpoint)
    rows, cosine_matrix = analyze_checkpoint(checkpoint, args.n_pde)
    paths = write_outputs(
        args.checkpoint, rows, cosine_matrix, args.output_dir
    )

    upper_triangle = cosine_matrix[np.triu_indices_from(cosine_matrix, k=1)]
    finite = upper_triangle[np.isfinite(upper_triangle)]
    negative_fraction = float(np.mean(finite < 0.0)) if finite.size else float("nan")
    gradient_norms = np.asarray(
        [float(row["gradient_norm"]) for row in rows], dtype=np.float64
    )
    summed_gradient_norm = math.sqrt(
        max(float(gradient_norms @ cosine_matrix @ gradient_norms), 0.0)
    )
    cancellation_ratio = summed_gradient_norm / float(np.sum(gradient_norms))
    print(f"Cases: {len(rows)}")
    print(f"Pairwise cosine min/median/max: {finite.min():.6f} / {np.median(finite):.6f} / {finite.max():.6f}")
    print(f"Negative pair fraction: {negative_fraction:.6f}")
    print(f"Gradient norm max/min ratio: {gradient_norms.max() / gradient_norms.min():.6f}")
    print(f"Gradient cancellation ratio: {cancellation_ratio:.6f}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
