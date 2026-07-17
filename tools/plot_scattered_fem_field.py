#!/usr/bin/env python3
"""Display scattered FEM wavefields by subtracting the incident mode."""

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

from tools.data_loader import FEMFieldData  # noqa: E402


@dataclass(frozen=True)
class ScatteredField:
    frequency: float
    mode: int
    values: np.ndarray


def parse_float_list(value: str) -> list[float]:
    items = [item.strip() for item in value.split(",")]
    if not items or any(not item for item in items):
        raise argparse.ArgumentTypeError("expected comma-separated frequencies")

    frequencies = []
    for item in items:
        try:
            frequency = float(item)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"invalid frequency {item!r}"
            ) from error
        if not np.isfinite(frequency) or frequency <= 0.0:
            raise argparse.ArgumentTypeError(
                f"frequency must be finite and positive: {item!r}"
            )
        frequencies.append(frequency)
    return frequencies


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",")]
    if not items or any(not item for item in items):
        raise argparse.ArgumentTypeError("expected comma-separated mode indices")

    modes = []
    for item in items:
        try:
            mode = int(item)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"invalid mode {item!r}") from error
        if mode < 0:
            raise argparse.ArgumentTypeError(
                f"mode index must be non-negative: {item!r}"
            )
        modes.append(mode)
    return modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot FEM scattered wavefields Us = U_total - U_incident for selected "
            "frequency/mode pairs."
        )
    )
    parser.add_argument("--fem-field", required=True, type=Path)
    parser.add_argument("--freqs", required=True, type=parse_float_list)
    parser.add_argument("--modes", required=True, type=parse_int_list)
    return parser.parse_args()


def waveguide_vertical_bounds(fem_data: FEMFieldData) -> tuple[float, float]:
    y_min = float(np.min(fem_data.y))
    y_max = float(np.max(fem_data.y))
    height = y_max - y_min
    if not np.isfinite(height) or height <= 0.0:
        raise ValueError(
            f"Invalid FEM vertical bounds y=[{y_min}, {y_max}] from {fem_data.filepath}"
        )
    return y_min, height


def incident_wave(
    x: np.ndarray,
    y: np.ndarray,
    *,
    k0: float,
    mode: int,
    y_min: float,
    height: float,
) -> np.ndarray:
    amplitude = np.sqrt(1.0 / height) if mode == 0 else np.sqrt(2.0 / height)
    y_shifted = y - y_min
    transverse_wavenumber = mode * np.pi / height
    beta = np.sqrt(k0**2 - transverse_wavenumber**2 + 0j)
    mode_shape = amplitude * np.cos(transverse_wavenumber * y_shifted)
    return mode_shape * np.exp(1j * beta * x)


def select_cases(
    fem_data: FEMFieldData,
    frequencies: list[float],
    modes: list[int],
) -> list[tuple[float, int]]:
    selected = []
    missing = []
    for frequency in frequencies:
        for mode in modes:
            if fem_data.has_case(frequency, mode):
                case = fem_data.case(frequency, mode)
                selected.append((case.frequency, case.mode))
            else:
                missing.append((frequency, mode))

    if missing:
        raise ValueError(
            f"Requested FEM cases are unavailable: {missing}; "
            f"available cases: {fem_data.available_cases}"
        )
    return selected


def prepare_scattered_fields(
    fem_data: FEMFieldData,
    cases: list[tuple[float, int]],
) -> list[ScatteredField]:
    y_min, height = waveguide_vertical_bounds(fem_data)
    fields = []
    for frequency, mode in cases:
        fem_case = fem_data.case(frequency, mode)
        incident = incident_wave(
            fem_case.x,
            fem_case.y,
            k0=fem_case.k0,
            mode=mode,
            y_min=y_min,
            height=height,
        )
        fields.append(
            ScatteredField(
                frequency=fem_case.frequency,
                mode=fem_case.mode,
                values=np.asarray(fem_case.values - incident, dtype=np.complex128),
            )
        )
    return fields


def _symmetric_limit(*arrays: np.ndarray) -> float:
    limit = max(float(np.max(np.abs(array))) for array in arrays)
    return limit if limit > 0.0 else 1.0


def create_scattered_figure(
    triangulation: mtri.Triangulation,
    field: ScatteredField,
) -> plt.Figure:
    real = field.values.real
    imaginary = field.values.imag
    magnitude = np.abs(field.values)
    signed_limit = _symmetric_limit(real, imaginary)
    magnitude_limit = float(np.max(magnitude))
    if magnitude_limit <= 0.0:
        magnitude_limit = 1.0

    figure, axes = plt.subplots(1, 3, figsize=(16.0, 4.6), sharex=True, sharey=True)
    panels = (
        (real, "Re(Us FEM)", "RdBu_r", -signed_limit, signed_limit),
        (imaginary, "Im(Us FEM)", "RdBu_r", -signed_limit, signed_limit),
        (magnitude, "|Us FEM|", "viridis", 0.0, magnitude_limit),
    )
    for axis, (values, title, cmap, vmin, vmax) in zip(axes, panels):
        image = axis.tripcolor(
            triangulation,
            values,
            shading="gouraud",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_aspect("equal", adjustable="box")
        figure.colorbar(image, ax=axis, shrink=0.85)
    axes[0].set_ylabel("y")

    figure.suptitle(
        f"FEM scattered wavefield - f={field.frequency:.8g} Hz, mode={field.mode}"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return figure


def main() -> None:
    args = parse_args()
    fem_data = FEMFieldData(args.fem_field)
    cases = select_cases(fem_data, args.freqs, args.modes)
    fields = prepare_scattered_fields(fem_data, cases)

    triangulation = mtri.Triangulation(fem_data.x, fem_data.y)
    for field in fields:
        print(f"Plotting f={field.frequency:.8g} Hz, mode={field.mode}")
        create_scattered_figure(triangulation, field)
    plt.show()


if __name__ == "__main__":
    main()
