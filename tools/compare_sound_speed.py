#!/usr/bin/env python3
"""Display a PINN sound-speed map, its ground truth and their misfit."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.uv_checkpoint import UVCheckpoint, load_uv_checkpoint  # noqa: E402


# ==============================================================================
# USER-EDITABLE GROUND TRUTH
# ==============================================================================
# Keep this label synchronized with the map implemented by
# ground_truth_sound_speed. Set it to None to disable the metadata safeguard.
GROUND_TRUTH_NAME: str | None = "barhalf"


def ground_truth_sound_speed(
    x: np.ndarray,
    y: np.ndarray,
    checkpoint: UVCheckpoint,
) -> np.ndarray:
    """Return the user-defined sound-speed map at physical coordinates.

    The default is the current ``barhalf`` geometry. Assigning different values
    on additional masks is enough to describe several defects with independent
    sound speeds.
    """
    contrast_ratio = float(checkpoint.metadata["contrast_ratio"])
    sound_speed = np.full(x.shape, checkpoint.c0, dtype=np.float64)
    barhalf = (
        (x >= -0.2)
        & (x <= 0.2)
        & (y >= 0.0)
        & (y <= 0.3)
    )
    circlebottomright = (
        (x-0.2)**2 + (y-0.2)**2 <= 0.1**2
    )
    sound_speed[barhalf] = checkpoint.c0 * contrast_ratio

    # Example for an additional defect with an independent speed:
    # second_defect = (x - 0.5) ** 2 + (y - 0.3) ** 2 <= 0.1 ** 2
    # sound_speed[second_defect] = 300.0
    return sound_speed


@dataclass(frozen=True)
class SoundSpeedMisfit:
    mean_absolute: float
    relative_l1: float
    anomaly_relative_l1: float | None
    anomaly_mean_absolute: float | None
    background_mean_absolute: float | None
    homogeneous_mean_absolute: float
    improvement_over_homogeneous: float | None
    rmse: float
    relative_l2: float
    max_absolute: float


def _grid_size(value: str) -> int:
    size = int(value)
    if size < 2:
        raise argparse.ArgumentTypeError("grid sizes must be at least 2")
    return size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Display the reconstructed sound speed, compare it with a "
            "user-defined ground truth, and display the absolute error."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--nx", type=_grid_size, default=201)
    parser.add_argument("--ny", type=_grid_size, default=121)
    return parser.parse_args()


def build_grid(
    checkpoint: UVCheckpoint,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_axis = np.linspace(-checkpoint.length, checkpoint.length, nx)
    y_axis = np.linspace(0.0, checkpoint.height, ny)
    return np.meshgrid(x_axis, y_axis, indexing="xy")


def build_ground_truth(
    checkpoint: UVCheckpoint,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray:
    checkpoint_defect = checkpoint.metadata.get("defect_name")
    if checkpoint_defect is None:
        raise ValueError("The checkpoint does not contain defect_name metadata")
    if GROUND_TRUTH_NAME is not None and str(checkpoint_defect) != GROUND_TRUTH_NAME:
        raise ValueError(
            "Ground truth/checkpoint mismatch: the hardcoded map is "
            f"{GROUND_TRUTH_NAME!r}, while the checkpoint declares "
            f"{checkpoint_defect!r}. Edit GROUND_TRUTH_NAME and "
            "ground_truth_sound_speed before comparing."
        )

    ground_truth = np.asarray(
        ground_truth_sound_speed(x_grid, y_grid, checkpoint),
        dtype=np.float64,
    )
    if ground_truth.shape != x_grid.shape:
        raise ValueError(
            "ground_truth_sound_speed must return a map with the same shape as the grid"
        )
    if not np.isfinite(ground_truth).all() or (ground_truth <= 0.0).any():
        raise ValueError(
            "ground_truth_sound_speed must return finite, strictly positive speeds"
        )
    return ground_truth


def compute_misfit(
    reconstructed: np.ndarray,
    ground_truth: np.ndarray,
    background_speed: float,
) -> SoundSpeedMisfit:
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    if reconstructed.shape != ground_truth.shape:
        raise ValueError(
            "The reconstructed and ground-truth maps have different shapes"
        )
    if not np.isfinite(reconstructed).all() or not np.isfinite(ground_truth).all():
        raise ValueError(
            "The reconstructed or ground-truth map contains NaN or infinite values"
        )
    if not np.isfinite(background_speed) or background_speed <= 0.0:
        raise ValueError(
            "The background sound speed must be finite and strictly positive"
        )

    error = reconstructed - ground_truth
    absolute_error = np.abs(error)
    reference_l1 = np.sum(np.abs(ground_truth))
    reference_l2 = np.linalg.norm(ground_truth.reshape(-1))
    if reference_l1 == 0.0 or reference_l2 == 0.0:
        raise ValueError("The ground-truth map has zero norm")
    anomaly_amplitude = np.abs(ground_truth - background_speed)
    tolerance = 1e-9 * max(1.0, abs(background_speed))
    anomaly_mask = anomaly_amplitude > tolerance
    background_mask = ~anomaly_mask
    anomaly_reference = float(np.sum(anomaly_amplitude))

    anomaly_relative_l1 = None
    anomaly_mean_absolute = None
    improvement_over_homogeneous = None
    if anomaly_reference > 0.0:
        anomaly_relative_l1 = float(np.sum(absolute_error) / anomaly_reference)
        anomaly_mean_absolute = float(np.mean(absolute_error[anomaly_mask]))
        improvement_over_homogeneous = 1.0 - anomaly_relative_l1

    background_mean_absolute = (
        float(np.mean(absolute_error[background_mask]))
        if background_mask.any()
        else None
    )
    return SoundSpeedMisfit(
        mean_absolute=float(np.mean(absolute_error)),
        relative_l1=float(np.sum(absolute_error) / reference_l1),
        anomaly_relative_l1=anomaly_relative_l1,
        anomaly_mean_absolute=anomaly_mean_absolute,
        background_mean_absolute=background_mean_absolute,
        homogeneous_mean_absolute=float(np.mean(anomaly_amplitude)),
        improvement_over_homogeneous=improvement_over_homogeneous,
        rmse=float(np.sqrt(np.mean(error**2))),
        relative_l2=float(np.linalg.norm(error.reshape(-1)) / reference_l2),
        max_absolute=float(np.max(absolute_error)),
    )


def _format_speed_axis(axis: plt.Axes, title: str) -> None:
    axis.set_title(title)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="box")


def create_reconstruction_figure(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    reconstructed: np.ndarray,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8.0, 3.8))
    image = axis.pcolormesh(
        x_grid,
        y_grid,
        reconstructed,
        shading="auto",
        cmap="viridis",
        rasterized=True,
    )
    _format_speed_axis(axis, "Reconstructed sound speed")
    figure.colorbar(image, ax=axis, label="c [m/s]")
    figure.tight_layout()
    return figure


def create_comparison_figure(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    reconstructed: np.ndarray,
    ground_truth: np.ndarray,
) -> plt.Figure:
    speed_min = float(min(np.min(reconstructed), np.min(ground_truth)))
    speed_max = float(max(np.max(reconstructed), np.max(ground_truth)))
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 3.8),
        sharex=True,
        sharey=True,
    )
    for axis, values, title in (
        (axes[0], ground_truth, "Ground-truth sound speed"),
        (axes[1], reconstructed, "Reconstructed sound speed"),
    ):
        image = axis.pcolormesh(
            x_grid,
            y_grid,
            values,
            shading="auto",
            cmap="viridis",
            vmin=speed_min,
            vmax=speed_max,
            rasterized=True,
        )
        _format_speed_axis(axis, title)
        figure.colorbar(image, ax=axis, label="c [m/s]")
    figure.tight_layout()
    return figure


def create_misfit_figure(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    reconstructed: np.ndarray,
    ground_truth: np.ndarray,
    metrics: SoundSpeedMisfit,
) -> plt.Figure:
    absolute_error = np.abs(reconstructed - ground_truth)
    limit = float(np.max(absolute_error))
    if limit == 0.0:
        limit = 1.0
    figure, axis = plt.subplots(figsize=(8.0, 3.8))
    image = axis.pcolormesh(
        x_grid,
        y_grid,
        absolute_error,
        shading="auto",
        cmap="magma",
        vmin=0.0,
        vmax=limit,
        rasterized=True,
    )
    _format_speed_axis(axis, r"Absolute error $|c_{PINN}-c_{GT}|$")
    figure.colorbar(image, ax=axis, label=r"$|\Delta c|$ [m/s]")
    if metrics.anomaly_relative_l1 is not None:
        metric_title = (
            f"anomaly-relative L1={100.0 * metrics.anomaly_relative_l1:.4g} %, "
            f"defect MAE={metrics.anomaly_mean_absolute:.4g} m/s"
        )
    else:
        metric_title = f"global relative L1={100.0 * metrics.relative_l1:.4g} %"
    figure.suptitle(metric_title)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    return figure


def print_metrics(metrics: SoundSpeedMisfit) -> None:
    if metrics.anomaly_relative_l1 is not None:
        print(
            f"Anomaly-relative L1: {metrics.anomaly_relative_l1:.8e} "
            f"({100.0 * metrics.anomaly_relative_l1:.5f} %)"
        )
        print(f"Anomaly MAE: {metrics.anomaly_mean_absolute:.8e} m/s")
        print(
            "Improvement over homogeneous c0 baseline: "
            f"{100.0 * metrics.improvement_over_homogeneous:.5f} %"
        )
    else:
        print("Anomaly-relative metrics: n/a (homogeneous ground truth)")
    if metrics.background_mean_absolute is not None:
        print(f"Background MAE: {metrics.background_mean_absolute:.8e} m/s")
    print(
        "Homogeneous c0 baseline MAE: "
        f"{metrics.homogeneous_mean_absolute:.8e} m/s"
    )
    print(f"Global mean absolute error: {metrics.mean_absolute:.8e} m/s")
    print(
        f"Global relative discrete L1: {metrics.relative_l1:.8e} "
        f"({100.0 * metrics.relative_l1:.5f} %)"
    )
    print(f"RMSE: {metrics.rmse:.8e} m/s")
    print(
        f"Relative discrete L2: {metrics.relative_l2:.8e} "
        f"({100.0 * metrics.relative_l2:.5f} %)"
    )
    print(f"Maximum absolute error: {metrics.max_absolute:.8e} m/s")


def main() -> None:
    args = parse_args()
    checkpoint = load_uv_checkpoint(args.checkpoint)
    if checkpoint.sound_speed is None:
        raise ValueError(
            "The checkpoint does not contain layers_m; a format-v2 checkpoint is required"
        )

    x_grid, y_grid = build_grid(checkpoint, args.nx, args.ny)
    reconstructed = checkpoint.predict_sound_speed_physical(x_grid, y_grid)
    ground_truth = build_ground_truth(checkpoint, x_grid, y_grid)
    metrics = compute_misfit(reconstructed, ground_truth, checkpoint.c0)
    print_metrics(metrics)

    create_reconstruction_figure(x_grid, y_grid, reconstructed)
    create_comparison_figure(x_grid, y_grid, reconstructed, ground_truth)
    create_misfit_figure(x_grid, y_grid, reconstructed, ground_truth, metrics)
    plt.show()


if __name__ == "__main__":
    main()
