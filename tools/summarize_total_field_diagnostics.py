#!/usr/bin/env python3
"""Summarize FEM-field and measured-boundary errors for total-field checkpoints."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.compare_boundary_fields import incident_wave, scattered_sides  # noqa: E402
from tools.compare_pinn_fem import compute_misfit_metrics, validate_geometry  # noqa: E402
from tools.data_loader import (  # noqa: E402
    FEMFieldData,
    WaveguideBoundaryData,
    load_symmetric_coo_matrix,
)
from tools.uv_checkpoint import load_uv_checkpoint  # noqa: E402


CSV_FIELDS = (
    "label",
    "checkpoint",
    "frequency",
    "mode",
    "fem_l2_relative",
    "fem_h1_relative",
    "boundary_total_relative",
    "boundary_scattered_relative",
)


def checkpoint_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=CHECKPOINT")
    label, path = value.split("=", maxsplit=1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("expected non-empty LABEL=CHECKPOINT")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate total-field PINN checkpoints against the FEM field and the "
            "complex boundary observations, then write one reproducible CSV table."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        action="append",
        type=checkpoint_spec,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--mass-matrix", required=True, type=Path)
    parser.add_argument("--stiffness-matrix", required=True, type=Path)
    parser.add_argument("--fem-field", required=True, type=Path)
    parser.add_argument("--boundary-left", required=True, type=Path)
    parser.add_argument("--boundary-right", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frequency", type=float)
    parser.add_argument("--mode", type=int, action="append")
    return parser.parse_args()


def _complex_total_sides(pair) -> tuple[np.ndarray, np.ndarray]:
    left = pair.u_re_left.astype(np.float64) + 1j * pair.u_im_left.astype(np.float64)
    right = pair.u_re_right.astype(np.float64) + 1j * pair.u_im_right.astype(np.float64)
    return left, right


def _relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    reference_norm = float(np.linalg.norm(reference))
    if reference_norm == 0.0:
        raise ValueError("A relative boundary error is undefined for a zero reference")
    return float(np.linalg.norm(prediction - reference) / reference_norm)


def _boundary_metrics(checkpoint, pair) -> tuple[float, float]:
    if not np.isclose(pair.k0, 2.0 * np.pi * pair.frequency / checkpoint.c0):
        raise ValueError(
            f"Boundary/checkpoint c0 mismatch for f={pair.frequency}, mode={pair.mode}"
        )

    observed_left, observed_right = _complex_total_sides(pair)
    predicted_left = checkpoint.predict_physical(
        pair.frequency,
        pair.mode,
        np.full_like(pair.y_left, pair.x_left, dtype=np.float64),
        pair.y_left,
    )
    predicted_right = checkpoint.predict_physical(
        pair.frequency,
        pair.mode,
        np.full_like(pair.y_right, pair.x_right, dtype=np.float64),
        pair.y_right,
    )
    observed_total = np.concatenate((observed_left, observed_right))
    predicted_total = np.concatenate((predicted_left, predicted_right))

    observed_scattered_left, observed_scattered_right = scattered_sides(pair)
    incident_left = incident_wave(
        pair.x_left,
        pair.y_left,
        k0=pair.k0,
        mode=pair.mode,
        y_min=0.0,
        height=checkpoint.height,
    )
    incident_right = incident_wave(
        pair.x_right,
        pair.y_right,
        k0=pair.k0,
        mode=pair.mode,
        y_min=0.0,
        height=checkpoint.height,
    )
    observed_scattered = np.concatenate(
        (observed_scattered_left, observed_scattered_right)
    )
    predicted_scattered = np.concatenate(
        (predicted_left - incident_left, predicted_right - incident_right)
    )
    return (
        _relative_l2(predicted_total, observed_total),
        _relative_l2(predicted_scattered, observed_scattered),
    )


def main() -> None:
    args = parse_args()
    fem_data = FEMFieldData(args.fem_field)
    boundary_data = WaveguideBoundaryData(
        str(args.boundary_left), str(args.boundary_right)
    )
    mass_matrix = load_symmetric_coo_matrix(
        args.mass_matrix, expected_size=fem_data.size
    )
    stiffness_matrix = load_symmetric_coo_matrix(
        args.stiffness_matrix, expected_size=fem_data.size
    )

    rows: list[dict[str, object]] = []
    labels: set[str] = set()
    for label, checkpoint_path in args.checkpoint:
        if label in labels:
            raise ValueError(f"Duplicate checkpoint label: {label!r}")
        labels.add(label)
        checkpoint = load_uv_checkpoint(checkpoint_path)
        validate_geometry(checkpoint, fem_data)

        cases = checkpoint.available_cases()
        if args.frequency is not None:
            cases = [case for case in cases if np.isclose(case[0], args.frequency)]
        if args.mode is not None:
            cases = [case for case in cases if case[1] in args.mode]
        if not cases:
            raise ValueError(f"No selected case is available in {checkpoint_path}")

        for frequency, mode in cases:
            if not fem_data.has_case(frequency, mode):
                raise ValueError(f"FEM field is missing f={frequency}, mode={mode}")
            if not boundary_data.has_pair(mode, frequency):
                raise ValueError(f"Boundary data are missing f={frequency}, mode={mode}")

            fem_case = fem_data.case(frequency, mode)
            prediction = checkpoint.predict_physical(
                frequency, mode, fem_data.x, fem_data.y
            )
            field_metrics = compute_misfit_metrics(
                fem_case.values, prediction, mass_matrix, stiffness_matrix
            )
            total_error, scattered_error = _boundary_metrics(
                checkpoint, boundary_data.get_pair(mode, frequency)
            )
            rows.append(
                {
                    "label": label,
                    "checkpoint": str(checkpoint_path),
                    "frequency": frequency,
                    "mode": mode,
                    "fem_l2_relative": field_metrics.l2_relative,
                    "fem_h1_relative": field_metrics.h1_relative,
                    "boundary_total_relative": total_error,
                    "boundary_scattered_relative": scattered_error,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['label']}: f={row['frequency']:.8g}, mode={row['mode']}, "
            f"field L2/H1={100.0 * row['fem_l2_relative']:.3f}%/"
            f"{100.0 * row['fem_h1_relative']:.3f}%, boundary total/scattered="
            f"{100.0 * row['boundary_total_relative']:.3f}%/"
            f"{100.0 * row['boundary_scattered_relative']:.3f}%"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
