#!/usr/bin/env python3
"""Compare two PINN UV checkpoints on a fixed FEM P2 node set."""

from __future__ import annotations

import argparse
import csv
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
    SymmetricCOOMatrix,
    load_symmetric_coo_matrix,
)
from tools.uv_checkpoint import UVCheckpoint, load_uv_checkpoint  # noqa: E402


@dataclass(frozen=True)
class FEMNodeGrid:
    """Physical P2 nodes in the RCM ordering shared by the FEM matrices."""

    node_ids: np.ndarray
    x: np.ndarray
    y: np.ndarray

    @property
    def size(self) -> int:
        return int(self.node_ids.size)


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
    checkpoint1_values: np.ndarray
    checkpoint2_values: np.ndarray
    metrics: MisfitMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate two PINN checkpoints at the same FEM P2 degrees of freedom, "
            "compute complex L2/H1 misfits, and display real-part field maps."
        )
    )
    parser.add_argument("--checkpoint1", required=True, type=Path)
    parser.add_argument("--checkpoint2", required=True, type=Path)
    parser.add_argument(
        "--fem-field",
        required=True,
        type=Path,
        help=(
            "fem_field_*.csv associated with the matrices; only the coordinates "
            "from the first field block are read"
        ),
    )
    parser.add_argument("--mass-matrix", required=True, type=Path)
    parser.add_argument(
        "--stiffness-matrix",
        type=Path,
        help="Optional matrix required to compute the H1 misfit",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        action="append",
        help="Frequency to compare; this option may be repeated",
    )
    parser.add_argument(
        "--mode",
        type=int,
        action="append",
        help="Mode to compare; this option may be repeated",
    )
    return parser.parse_args()


def _parse_integer(value: str, column: str, filepath: Path) -> int:
    try:
        numeric_value = float(value)
    except ValueError as error:
        raise ValueError(
            f"Non-numeric value in column {column!r} of {filepath}"
        ) from error
    if not np.isfinite(numeric_value) or numeric_value != round(numeric_value):
        raise ValueError(
            f"Invalid integer value in column {column!r} of {filepath}: "
            f"{value!r}"
        )
    return int(round(numeric_value))


def load_fem_node_grid(filepath: str | Path) -> FEMNodeGrid:
    """Read only the first field block, which contains every P2 node once."""

    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Missing FEM field file: {filepath}")

    node_ids: list[int] = []
    x_values: list[float] = []
    y_values: list[float] = []
    first_case: tuple[float, int] | None = None
    required_columns = ("f", "mode", "node_id", "x", "y")

    with filepath.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or ())
        missing = [column for column in required_columns if column not in fieldnames]
        if missing:
            raise ValueError(f"{filepath} is missing columns: {missing}")

        for row in reader:
            try:
                frequency = float(row["f"])
                mode = _parse_integer(row["mode"], "mode", filepath)
                node_id = _parse_integer(row["node_id"], "node_id", filepath)
                x = float(row["x"])
                y = float(row["y"])
            except ValueError as error:
                if "column" in str(error):
                    raise
                raise ValueError(f"Invalid numeric row in {filepath}") from error

            values = np.asarray((frequency, x, y), dtype=np.float64)
            if not np.isfinite(values).all() or frequency <= 0.0 or mode < 0 or node_id < 0:
                raise ValueError(f"Invalid physical value or index in {filepath}")

            case = (frequency, mode)
            if first_case is None:
                first_case = case
            elif case != first_case:
                break

            node_ids.append(node_id)
            x_values.append(x)
            y_values.append(y)

    if not node_ids:
        raise ValueError(f"{filepath} contains no FEM nodes")

    node_id_array = np.asarray(node_ids, dtype=np.int64)
    expected_ids = np.arange(node_id_array.size, dtype=np.int64)
    if not np.array_equal(node_id_array, expected_ids):
        raise ValueError(
            f"The first field block in {filepath} must be ordered by contiguous "
            "node_id values starting at zero"
        )

    x_array = np.asarray(x_values, dtype=np.float64)
    y_array = np.asarray(y_values, dtype=np.float64)
    if np.unique(np.column_stack((x_array, y_array)), axis=0).shape[0] != node_id_array.size:
        raise ValueError(f"The first field block in {filepath} contains duplicate coordinates")

    return FEMNodeGrid(node_ids=node_id_array, x=x_array, y=y_array)


def validate_geometry(
    checkpoint1: UVCheckpoint,
    checkpoint2: UVCheckpoint,
    grid: FEMNodeGrid,
) -> None:
    geometry1 = np.asarray(
        (checkpoint1.length, checkpoint1.height, checkpoint1.c0), dtype=np.float64
    )
    geometry2 = np.asarray(
        (checkpoint2.length, checkpoint2.height, checkpoint2.c0), dtype=np.float64
    )
    if not np.allclose(geometry1, geometry2, rtol=1e-7, atol=1e-9):
        raise ValueError(
            "The checkpoints do not describe the same physical problem: "
            f"(L, H, c0)={tuple(geometry1)} versus {tuple(geometry2)}"
        )

    expected_bounds = np.asarray(
        (-checkpoint1.length, checkpoint1.length, 0.0, checkpoint1.height),
        dtype=np.float64,
    )
    observed_bounds = np.asarray(
        (grid.x.min(), grid.x.max(), grid.y.min(), grid.y.max()), dtype=np.float64
    )
    tolerance = 1e-7 * max(1.0, checkpoint1.length, checkpoint1.height)
    if not np.allclose(observed_bounds, expected_bounds, rtol=0.0, atol=tolerance):
        raise ValueError(
            "FEM/checkpoint geometry mismatch: "
            f"FEM bounds={tuple(observed_bounds)}, expected={tuple(expected_bounds)}"
        )


def get_defect_name(
    checkpoint1: UVCheckpoint,
    checkpoint2: UVCheckpoint,
) -> str:
    defect1 = checkpoint1.metadata.get("defect_name")
    defect2 = checkpoint2.metadata.get("defect_name")
    if defect1 is not None and defect2 is not None and str(defect1) != str(defect2):
        raise ValueError(
            "The checkpoints correspond to different defects: "
            f"{defect1!r} and {defect2!r}"
        )
    defect = defect1 if defect1 is not None else defect2
    return str(defect) if defect is not None else "unspecified"


def _common_cases(
    checkpoint1: UVCheckpoint, checkpoint2: UVCheckpoint
) -> list[tuple[float, int]]:
    common: list[tuple[float, int]] = []
    for frequency1, mode1 in checkpoint1.available_cases():
        matches = [
            (frequency2, mode2)
            for frequency2, mode2 in checkpoint2.available_cases()
            if mode1 == mode2 and np.isclose(frequency1, frequency2)
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous cases in checkpoint 2 for f={frequency1}, mode={mode1}"
            )
        if matches:
            common.append((frequency1, mode1))
    return common


def select_cases(
    checkpoint1: UVCheckpoint,
    checkpoint2: UVCheckpoint,
    frequencies: list[float] | None = None,
    modes: list[int] | None = None,
) -> list[tuple[float, int]]:
    cases = _common_cases(checkpoint1, checkpoint2)
    if frequencies is not None:
        cases = [
            case
            for case in cases
            if any(np.isclose(case[0], frequency) for frequency in frequencies)
        ]
    if modes is not None:
        cases = [case for case in cases if case[1] in modes]
    if not cases:
        raise ValueError(
            "No case shared by the checkpoints matches the requested filters"
        )

    if frequencies is not None:
        missing_frequencies = [
            frequency
            for frequency in frequencies
            if not any(np.isclose(case[0], frequency) for case in cases)
        ]
        if missing_frequencies:
            raise ValueError(
                "Requested frequencies missing from the shared cases: "
                f"{missing_frequencies}"
            )
    if modes is not None:
        missing_modes = [mode for mode in modes if not any(case[1] == mode for case in cases)]
        if missing_modes:
            raise ValueError(f"Requested modes missing from the shared cases: {missing_modes}")
    return cases


def compute_misfit_metrics(
    checkpoint1_values: np.ndarray,
    checkpoint2_values: np.ndarray,
    mass_matrix: SymmetricCOOMatrix,
    stiffness_matrix: SymmetricCOOMatrix | None = None,
) -> MisfitMetrics:
    values1 = np.asarray(checkpoint1_values, dtype=np.complex128)
    values2 = np.asarray(checkpoint2_values, dtype=np.complex128)
    if values1.shape != values2.shape:
        raise ValueError(
            f"The PINN fields have different shapes: {values1.shape} and {values2.shape}"
        )
    if values1.ndim != 1 or values1.size != mass_matrix.size:
        raise ValueError(
            f"Fields must have shape ({mass_matrix.size},), got {values1.shape}"
        )
    if not np.isfinite(values1).all() or not np.isfinite(values2).all():
        raise ValueError("A PINN field contains NaN or infinite values")

    error = values2 - values1
    l2_error_squared = mass_matrix.quadratic_form(error)
    l2_reference_squared = mass_matrix.quadratic_form(values1)
    if l2_reference_squared <= 0.0:
        raise ValueError(
            "The L2 norm of checkpoint 1 is zero; the relative misfit is undefined"
        )
    l2_absolute = np.sqrt(l2_error_squared)
    l2_relative = l2_absolute / np.sqrt(l2_reference_squared)

    if stiffness_matrix is None:
        return MisfitMetrics(
            l2_absolute=float(l2_absolute), l2_relative=float(l2_relative)
        )
    if stiffness_matrix.size != mass_matrix.size:
        raise ValueError(
            f"Stiffness matrix size {stiffness_matrix.size} differs from mass "
            f"matrix size {mass_matrix.size}"
        )

    h1_error_squared = l2_error_squared + stiffness_matrix.quadratic_form(error)
    h1_reference_squared = l2_reference_squared + stiffness_matrix.quadratic_form(
        values1
    )
    if h1_reference_squared <= 0.0:
        raise ValueError(
            "The H1 norm of checkpoint 1 is zero; the relative misfit is undefined"
        )
    h1_absolute = np.sqrt(h1_error_squared)
    h1_relative = h1_absolute / np.sqrt(h1_reference_squared)
    return MisfitMetrics(
        l2_absolute=float(l2_absolute),
        l2_relative=float(l2_relative),
        h1_absolute=float(h1_absolute),
        h1_relative=float(h1_relative),
    )


def prepare_comparisons(
    checkpoint1: UVCheckpoint,
    checkpoint2: UVCheckpoint,
    grid: FEMNodeGrid,
    mass_matrix: SymmetricCOOMatrix,
    stiffness_matrix: SymmetricCOOMatrix | None,
    cases: list[tuple[float, int]],
) -> list[ComparisonResult]:
    results = []
    for frequency, mode in cases:
        values1 = checkpoint1.predict_physical(
            frequency, mode, grid.x, grid.y, physical_units=True
        )
        values2 = checkpoint2.predict_physical(
            frequency, mode, grid.x, grid.y, physical_units=True
        )
        metrics = compute_misfit_metrics(
            values1, values2, mass_matrix, stiffness_matrix
        )
        results.append(
            ComparisonResult(
                frequency=frequency,
                mode=mode,
                checkpoint1_values=np.asarray(values1, dtype=np.complex128),
                checkpoint2_values=np.asarray(values2, dtype=np.complex128),
                metrics=metrics,
            )
        )
    return results


def print_metrics(result: ComparisonResult) -> None:
    metrics = result.metrics
    print(f"f={result.frequency:.8g} Hz, mode={result.mode}")
    print(
        f"  L2(U2-U1): absolute={metrics.l2_absolute:.8e}, "
        f"relative_to_U1={metrics.l2_relative:.8e} "
        f"({100.0 * metrics.l2_relative:.5f} %)"
    )
    if metrics.h1_absolute is not None and metrics.h1_relative is not None:
        print(
            f"  H1(U2-U1): absolute={metrics.h1_absolute:.8e}, "
            f"relative_to_U1={metrics.h1_relative:.8e} "
            f"({100.0 * metrics.h1_relative:.5f} %)"
        )


def _symmetric_limit(*arrays: np.ndarray) -> float:
    limit = max(float(np.max(np.abs(array))) for array in arrays)
    return limit if limit > 0.0 else 1.0


def create_comparison_figure(
    triangulation: mtri.Triangulation,
    result: ComparisonResult,
    defect_name: str,
) -> plt.Figure:
    values1 = result.checkpoint1_values.real
    values2 = result.checkpoint2_values.real
    difference = values2 - values1
    field_limit = _symmetric_limit(values1, values2)
    difference_limit = _symmetric_limit(difference)

    figure, axes = plt.subplots(1, 3, figsize=(16.0, 4.6), sharex=True, sharey=True)
    panels = (
        (values1, r"$\mathrm{Re}(U_1)$", field_limit),
        (values2, r"$\mathrm{Re}(U_2)$", field_limit),
        (difference, r"$\mathrm{Re}(U_2-U_1)$", difference_limit),
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
        f"Defect: {defect_name} — f={result.frequency:.8g} Hz, mode={result.mode} — "
        + metric_title
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return figure


def main() -> None:
    args = parse_args()
    checkpoint1 = load_uv_checkpoint(args.checkpoint1)
    checkpoint2 = load_uv_checkpoint(args.checkpoint2)
    defect_name = get_defect_name(checkpoint1, checkpoint2)
    grid = load_fem_node_grid(args.fem_field)
    validate_geometry(checkpoint1, checkpoint2, grid)

    mass_matrix = load_symmetric_coo_matrix(
        args.mass_matrix, expected_size=grid.size
    )
    stiffness_matrix = None
    if args.stiffness_matrix is not None:
        stiffness_matrix = load_symmetric_coo_matrix(
            args.stiffness_matrix, expected_size=grid.size
        )

    cases = select_cases(
        checkpoint1,
        checkpoint2,
        frequencies=args.frequency,
        modes=args.mode,
    )
    results = prepare_comparisons(
        checkpoint1,
        checkpoint2,
        grid,
        mass_matrix,
        stiffness_matrix,
        cases,
    )
    for result in results:
        print_metrics(result)

    triangulation = mtri.Triangulation(grid.x, grid.y)
    for result in results:
        create_comparison_figure(triangulation, result, defect_name)
    plt.show()


if __name__ == "__main__":
    main()
