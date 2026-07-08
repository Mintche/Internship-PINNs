#!/usr/bin/env python3
"""Compare nodal FEM total fields and scattered PINN checkpoints."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.data_loader import (  # noqa: E402
    FEMFieldData,
    SymmetricCOOMatrix,
    load_symmetric_coo_matrix,
)
from tools.us_checkpoint import USCheckpoint, load_us_checkpoint  # noqa: E402


@dataclass(frozen=True)
class MisfitMetrics:
    l2_absolute: float
    l2_relative: float
    h1_absolute: float | None = None
    h1_relative: float | None = None


@dataclass(frozen=True)
class ComparisonResult:
    frequency: float
    mode: int
    fem_values: np.ndarray
    pinn_values: np.ndarray
    metrics: MisfitMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a scattered PINN checkpoint at FEM P2 degrees of freedom, "
            "compare FEM total U against u0 + us, and display real-part maps."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--mass-matrix", required=True, type=Path)
    parser.add_argument("--stiffness-matrix", type=Path)
    parser.add_argument("--fem-field", required=True, type=Path)
    parser.add_argument("--frequency", type=float)
    parser.add_argument("--mode", type=int)
    return parser.parse_args()


def validate_geometry(checkpoint: USCheckpoint, fem_data: FEMFieldData) -> None:
    expected = np.asarray(
        [-checkpoint.length, checkpoint.length, 0.0, checkpoint.height],
        dtype=np.float64,
    )
    observed = np.asarray(
        [fem_data.x.min(), fem_data.x.max(), fem_data.y.min(), fem_data.y.max()],
        dtype=np.float64,
    )
    tolerance = 1e-7 * max(1.0, checkpoint.length, checkpoint.height)
    if not np.allclose(observed, expected, rtol=0.0, atol=tolerance):
        raise ValueError(
            "FEM/checkpoint geometry mismatch: "
            f"FEM bounds are x=[{observed[0]}, {observed[1]}], "
            f"y=[{observed[2]}, {observed[3]}], while the checkpoint expects "
            f"x=[{expected[0]}, {expected[1]}], y=[{expected[2]}, {expected[3]}]"
        )


def select_cases(
    checkpoint: USCheckpoint,
    fem_data: FEMFieldData,
    frequency: float | None = None,
    mode: int | None = None,
) -> list[tuple[float, int]]:
    cases = checkpoint.available_cases()
    if frequency is not None:
        cases = [case for case in cases if np.isclose(case[0], frequency)]
    if mode is not None:
        cases = [case for case in cases if case[1] == mode]
    if not cases:
        raise ValueError("No checkpoint case matches the requested frequency/mode filters")

    missing = [case for case in cases if not fem_data.has_case(*case)]
    if missing:
        raise ValueError(
            f"Selected checkpoint cases are missing from the FEM field: {missing}; "
            f"available FEM cases: {fem_data.available_cases}"
        )
    return cases


def compute_misfit_metrics(
    fem_values: np.ndarray,
    pinn_values: np.ndarray,
    mass_matrix: SymmetricCOOMatrix,
    stiffness_matrix: SymmetricCOOMatrix | None = None,
) -> MisfitMetrics:
    fem_values = np.asarray(fem_values, dtype=np.complex128)
    pinn_values = np.asarray(pinn_values, dtype=np.complex128)
    if fem_values.shape != pinn_values.shape:
        raise ValueError(
            f"FEM and PINN fields have different shapes: {fem_values.shape}, {pinn_values.shape}"
        )
    if fem_values.ndim != 1 or fem_values.size != mass_matrix.size:
        raise ValueError(
            f"Fields must have shape ({mass_matrix.size},), got {fem_values.shape}"
        )
    if not np.isfinite(fem_values).all() or not np.isfinite(pinn_values).all():
        raise ValueError("FEM/PINN fields contain NaN or infinite values")

    error = pinn_values - fem_values
    l2_error_squared = mass_matrix.quadratic_form(error)
    l2_fem_squared = mass_matrix.quadratic_form(fem_values)
    if l2_fem_squared <= 0.0:
        raise ValueError("The FEM L2 norm is zero; a relative misfit is undefined")
    l2_absolute = np.sqrt(l2_error_squared)
    l2_relative = l2_absolute / np.sqrt(l2_fem_squared)

    if stiffness_matrix is None:
        return MisfitMetrics(
            l2_absolute=float(l2_absolute),
            l2_relative=float(l2_relative),
        )
    if stiffness_matrix.size != mass_matrix.size:
        raise ValueError(
            f"Stiffness matrix size {stiffness_matrix.size} differs from mass matrix "
            f"size {mass_matrix.size}"
        )

    h1_error_squared = l2_error_squared + stiffness_matrix.quadratic_form(error)
    h1_fem_squared = l2_fem_squared + stiffness_matrix.quadratic_form(fem_values)
    if h1_fem_squared <= 0.0:
        raise ValueError("The FEM H1 norm is zero; a relative misfit is undefined")
    h1_absolute = np.sqrt(h1_error_squared)
    h1_relative = h1_absolute / np.sqrt(h1_fem_squared)
    return MisfitMetrics(
        l2_absolute=float(l2_absolute),
        l2_relative=float(l2_relative),
        h1_absolute=float(h1_absolute),
        h1_relative=float(h1_relative),
    )


def prepare_comparisons(
    checkpoint: USCheckpoint,
    fem_data: FEMFieldData,
    mass_matrix: SymmetricCOOMatrix,
    stiffness_matrix: SymmetricCOOMatrix | None,
    cases: list[tuple[float, int]],
) -> list[ComparisonResult]:
    for frequency, mode in cases:
        fem_case = fem_data.case(frequency, mode)
        expected_k0 = 2.0 * np.pi * frequency / checkpoint.c0
        if not np.isclose(fem_case.k0, expected_k0, rtol=1e-7, atol=1e-9):
            raise ValueError(
                f"FEM/checkpoint c0 mismatch for f={frequency}, mode={mode}: "
                f"FEM k0={fem_case.k0}, expected {expected_k0}"
            )

    predictions = {
        case: checkpoint.predict_total_physical(
            case[0], case[1], fem_data.x, fem_data.y, physical_units=True
        )
        for case in cases
    }

    results = []
    for frequency, mode in cases:
        fem_case = fem_data.case(frequency, mode)
        pinn_values = np.asarray(predictions[(frequency, mode)], dtype=np.complex128)
        metrics = compute_misfit_metrics(
            fem_case.values, pinn_values, mass_matrix, stiffness_matrix
        )
        results.append(
            ComparisonResult(
                frequency=frequency,
                mode=mode,
                fem_values=fem_case.values,
                pinn_values=pinn_values,
                metrics=metrics,
            )
        )
    return results


def print_metrics(result: ComparisonResult) -> None:
    metrics = result.metrics
    print(f"f={result.frequency:.8g} Hz, mode={result.mode}")
    print(
        f"  L2(U_total): absolute={metrics.l2_absolute:.8e}, "
        f"relative={metrics.l2_relative:.8e} ({100.0 * metrics.l2_relative:.5f} %)"
    )
    if metrics.h1_absolute is not None and metrics.h1_relative is not None:
        print(
            f"  H1(U_total): absolute={metrics.h1_absolute:.8e}, "
            f"relative={metrics.h1_relative:.8e} ({100.0 * metrics.h1_relative:.5f} %)"
        )


def _symmetric_limit(*arrays: np.ndarray) -> float:
    limit = max(float(np.max(np.abs(array))) for array in arrays)
    return limit if limit > 0.0 else 1.0


def create_comparison_figure(
    triangulation: mtri.Triangulation,
    result: ComparisonResult,
) -> plt.Figure:
    fem_real = result.fem_values.real
    pinn_real = result.pinn_values.real
    real_misfit = pinn_real - fem_real
    field_limit = _symmetric_limit(fem_real, pinn_real)
    misfit_limit = _symmetric_limit(real_misfit)

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharex=True, sharey=True)
    panels = (
        (fem_real, "Re(U FEM)", field_limit),
        (pinn_real, "Re(U0 + US PINN)", field_limit),
        (real_misfit, "Re(U PINN - U FEM)", misfit_limit),
    )
    for axis, (values, title, limit) in zip(axes, panels):
        image = axis.tripcolor(
            triangulation,
            values,
            shading="gouraud",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            rasterized=True,
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_aspect("equal", adjustable="box")
        figure.colorbar(image, ax=axis, shrink=0.85)
    axes[0].set_ylabel("y")

    metrics = result.metrics
    metric_title = f"relative L2={100.0 * metrics.l2_relative:.4g} %"
    if metrics.h1_relative is not None:
        metric_title += f", relative H1={100.0 * metrics.h1_relative:.4g} %"
    figure.suptitle(
        f"FEM-scattered-PINN comparison - f={result.frequency:.8g} Hz, "
        f"mode={result.mode} - {metric_title}"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return figure


def main() -> None:
    args = parse_args()
    checkpoint = load_us_checkpoint(args.checkpoint)
    fem_data = FEMFieldData(args.fem_field)
    validate_geometry(checkpoint, fem_data)

    mass_matrix = load_symmetric_coo_matrix(
        args.mass_matrix, expected_size=fem_data.size
    )
    stiffness_matrix = None
    if args.stiffness_matrix is not None:
        stiffness_matrix = load_symmetric_coo_matrix(
            args.stiffness_matrix, expected_size=fem_data.size
        )

    cases = select_cases(checkpoint, fem_data, frequency=args.frequency, mode=args.mode)
    fem_only_cases = sorted(set(fem_data.available_cases) - set(checkpoint.available_cases()))
    if fem_only_cases:
        print(f"FEM-only cases ignored: {fem_only_cases}")

    results = prepare_comparisons(checkpoint, fem_data, mass_matrix, stiffness_matrix, cases)
    for result in results:
        print_metrics(result)

    triangulation = mtri.Triangulation(fem_data.x, fem_data.y)
    for result in results:
        create_comparison_figure(triangulation, result)
    plt.show()


if __name__ == "__main__":
    main()
