#!/usr/bin/env python3
"""Plot FEM mesh convergence against the finest available FEM solution.

The coarse complex fields are linearly interpolated from their exported P2
node clouds onto the nodes of the last (reference) field.  The L2 and H1 norms
are then evaluated with the mass and stiffness matrices of that reference
mesh.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.data_loader import (  # noqa: E402
    FEMFieldCase,
    FEMFieldData,
    SymmetricCOOMatrix,
    load_symmetric_coo_matrix,
)


@dataclass(frozen=True)
class ConvergenceResult:
    degrees_of_freedom: np.ndarray
    relative_l2_errors: np.ndarray
    relative_h1_errors: np.ndarray
    reference_degrees_of_freedom: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot relative L2/H1 FEM convergence. Give the FEM field files from "
            "the coarsest mesh to the finest; the last one is the reference."
        )
    )
    parser.add_argument(
        "fem_fields",
        nargs="+",
        type=Path,
        help="fem_field_*.csv files, ordered from coarsest to finest",
    )
    parser.add_argument("--mass-matrix", required=True, type=Path)
    parser.add_argument("--stiffness-matrix", required=True, type=Path)
    parser.add_argument("--frequency", type=float, default=1200.0)
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--incidence", type=int, choices=[-1, 1], default=-1)
    parser.add_argument("--output", type=Path, help="save the figure instead of showing it")
    return parser.parse_args()


def interpolate_on_reference(
    source: FEMFieldCase,
    reference: FEMFieldCase,
) -> np.ndarray:
    """Linearly interpolate a complex nodal field onto the reference nodes."""
    if np.array_equal(source.x, reference.x) and np.array_equal(source.y, reference.y):
        return np.asarray(source.values, dtype=np.complex128).copy()

    triangulation = mtri.Triangulation(source.x, source.y)
    real = mtri.LinearTriInterpolator(triangulation, source.values.real)(
        reference.x, reference.y
    )
    imaginary = mtri.LinearTriInterpolator(triangulation, source.values.imag)(
        reference.x, reference.y
    )
    if np.ma.is_masked(real) or np.ma.is_masked(imaginary):
        raise ValueError(
            "Some reference nodes lie outside a coarse solution; all meshes must "
            "cover the same domain"
        )
    return np.asarray(real, dtype=np.float64) + 1j * np.asarray(
        imaginary, dtype=np.float64
    )


def compute_convergence(
    solutions: Sequence[FEMFieldData],
    mass_matrix: SymmetricCOOMatrix,
    stiffness_matrix: SymmetricCOOMatrix,
    *,
    frequency: float = 1200.0,
    mode: int = 0,
    incidence: int = -1,
) -> ConvergenceResult:
    """Compute relative errors against the last solution in ``solutions``."""
    if len(solutions) < 2:
        raise ValueError("At least one coarse solution and one reference are required")

    reference_data = solutions[-1]
    reference = reference_data.case(frequency, mode, incidence)
    if mass_matrix.size != reference_data.size or stiffness_matrix.size != reference_data.size:
        raise ValueError("Mass and stiffness matrices must belong to the last solution")

    reference_l2_squared = mass_matrix.quadratic_form(reference.values)
    reference_h1_squared = reference_l2_squared + stiffness_matrix.quadratic_form(
        reference.values
    )

    degrees_of_freedom = []
    relative_l2_errors = []
    relative_h1_errors = []
    for solution in solutions[:-1]:
        transferred = interpolate_on_reference(
            solution.case(frequency, mode, incidence), reference
        )
        error = transferred - reference.values
        l2_error_squared = mass_matrix.quadratic_form(error)
        h1_error_squared = l2_error_squared + stiffness_matrix.quadratic_form(error)

        degrees_of_freedom.append(solution.size)
        relative_l2_errors.append(np.sqrt(l2_error_squared / reference_l2_squared))
        relative_h1_errors.append(np.sqrt(h1_error_squared / reference_h1_squared))

    return ConvergenceResult(
        degrees_of_freedom=np.asarray(degrees_of_freedom, dtype=np.int64),
        relative_l2_errors=np.asarray(relative_l2_errors, dtype=np.float64),
        relative_h1_errors=np.asarray(relative_h1_errors, dtype=np.float64),
        reference_degrees_of_freedom=reference_data.size,
    )


def create_convergence_figure(
    result: ConvergenceResult,
    *,
    frequency: float = 1200.0,
    mode: int = 0,
    incidence: int = -1,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.loglog(
        result.degrees_of_freedom,
        result.relative_l2_errors,
        "o-",
        label=r"$L^2$ relative",
    )
    axis.loglog(
        result.degrees_of_freedom,
        result.relative_h1_errors,
        "s-",
        label=r"$H^1$ relative",
    )
    axis.axvline(
        result.reference_degrees_of_freedom,
        color="0.45",
        linestyle="--",
        linewidth=1.0,
        label=f"reference ({result.reference_degrees_of_freedom} DOFs)",
    )
    axis.set_xlabel("Number of P2 degrees of freedom")
    axis.set_ylabel("Relative error")
    axis.set_title(
        f"FEM mesh convergence f={frequency:g} Hz, mode={mode}, incidence={incidence}"
    )
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure


def print_convergence(result: ConvergenceResult) -> None:
    print("DOFs,relative_L2,relative_H1")
    for ndof, l2_error, h1_error in zip(
        result.degrees_of_freedom,
        result.relative_l2_errors,
        result.relative_h1_errors,
    ):
        print(f"{ndof},{l2_error:.8e},{h1_error:.8e}")
    print(f"Reference: {result.reference_degrees_of_freedom} DOFs")


def main() -> None:
    args = parse_args()
    solutions = [FEMFieldData(path) for path in args.fem_fields]
    reference_size = solutions[-1].size
    mass_matrix = load_symmetric_coo_matrix(
        args.mass_matrix, expected_size=reference_size
    )
    stiffness_matrix = load_symmetric_coo_matrix(
        args.stiffness_matrix, expected_size=reference_size
    )
    result = compute_convergence(
        solutions,
        mass_matrix,
        stiffness_matrix,
        frequency=args.frequency,
        mode=args.mode,
        incidence=args.incidence,
    )
    print_convergence(result)
    figure = create_convergence_figure(
        result, frequency=args.frequency, mode=args.mode, incidence=args.incidence
    )

    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, bbox_inches="tight")
        print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
