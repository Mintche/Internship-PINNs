"""Reusable complex FEM field misfit metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data_loader import SymmetricCOOMatrix


@dataclass(frozen=True)
class MisfitMetrics:
    """Absolute and relative mass/H1 misfits for two complex nodal fields."""

    l2_absolute: float
    l2_relative: float
    h1_absolute: float | None = None
    h1_relative: float | None = None


def compute_misfit_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    mass_matrix: SymmetricCOOMatrix,
    stiffness_matrix: SymmetricCOOMatrix | None = None,
) -> MisfitMetrics:
    """Compute complex L2 and, optionally, H1 metrics using FEM matrices."""
    reference = np.asarray(reference, dtype=np.complex128)
    prediction = np.asarray(prediction, dtype=np.complex128)
    if reference.shape != prediction.shape:
        raise ValueError(
            f"Reference and prediction fields have different shapes: "
            f"{reference.shape}, {prediction.shape}"
        )
    if reference.ndim != 1 or reference.size != mass_matrix.size:
        raise ValueError(
            f"Fields must have shape ({mass_matrix.size},), got {reference.shape}"
        )
    if not np.isfinite(reference).all() or not np.isfinite(prediction).all():
        raise ValueError("Reference/prediction fields contain NaN or infinite values")

    error = prediction - reference
    l2_error_squared = mass_matrix.quadratic_form(error)
    l2_reference_squared = mass_matrix.quadratic_form(reference)
    if l2_reference_squared <= 0.0:
        raise ValueError("The reference L2 norm is zero; a relative misfit is undefined")
    l2_absolute = np.sqrt(l2_error_squared)
    l2_relative = l2_absolute / np.sqrt(l2_reference_squared)

    if stiffness_matrix is None:
        return MisfitMetrics(float(l2_absolute), float(l2_relative))
    if stiffness_matrix.size != mass_matrix.size:
        raise ValueError(
            f"Stiffness matrix size {stiffness_matrix.size} differs from mass matrix "
            f"size {mass_matrix.size}"
        )

    h1_error_squared = l2_error_squared + stiffness_matrix.quadratic_form(error)
    h1_reference_squared = l2_reference_squared + stiffness_matrix.quadratic_form(reference)
    if h1_reference_squared <= 0.0:
        raise ValueError("The reference H1 norm is zero; a relative misfit is undefined")
    h1_absolute = np.sqrt(h1_error_squared)
    h1_relative = h1_absolute / np.sqrt(h1_reference_squared)
    return MisfitMetrics(
        float(l2_absolute),
        float(l2_relative),
        float(h1_absolute),
        float(h1_relative),
    )


def validate_geometry(checkpoint, fem_data) -> None:
    """Validate the rectangular bounds exposed by a legacy checkpoint object."""
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
