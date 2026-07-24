#!/usr/bin/env python3
"""Re-solve a checkpoint's material map with FEM and measure the soft-physics gap."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.compare_pinn_fem import compute_misfit_metrics, validate_geometry  # noqa: E402
from tools.data_loader import (  # noqa: E402
    FEMFieldData,
    WaveguideBoundaryData,
    load_symmetric_coo_matrix,
)
from tools.us_checkpoint import load_us_checkpoint  # noqa: E402
from tools.uv_checkpoint import load_uv_checkpoint  # noqa: E402


CSV_FIELDS = (
    "frequency",
    "mode",
    "incidence",
    "pinn_vs_resolved_l2",
    "pinn_vs_resolved_h1",
    "resolved_vs_truth_l2",
    "resolved_vs_truth_h1",
    "pinn_vs_truth_l2",
    "pinn_vs_truth_h1",
    "pinn_vs_observed_trace",
    "resolved_vs_observed_trace",
    "pinn_vs_resolved_trace",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--formulation", choices=("auto", "total", "scattered"), default="auto"
    )
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--fem-field-truth", type=Path, required=True)
    parser.add_argument("--mass-matrix", type=Path, required=True)
    parser.add_argument("--stiffness-matrix", type=Path, required=True)
    parser.add_argument("--boundary-left-truth", type=Path, required=True)
    parser.add_argument("--boundary-right-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frequency", type=float, action="append")
    parser.add_argument("--mode", type=int, action="append")
    parser.add_argument("--incidence", type=int, choices=[-1, 1], action="append")
    parser.add_argument(
        "--generator",
        type=Path,
        default=REPOSITORY_ROOT / "FEM/generate_pinn_data.x",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_error(prediction: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        raise ValueError("Relative error is undefined for a zero reference")
    return float(np.linalg.norm(prediction - reference) / denominator)


def _trace(pair: Any) -> np.ndarray:
    return np.concatenate(
        (
            pair.u_re_left.astype(np.float64) + 1j * pair.u_im_left.astype(np.float64),
            pair.u_re_right.astype(np.float64) + 1j * pair.u_im_right.astype(np.float64),
        )
    ).astype(np.complex128)


def _validate_boundary_sampling(observed: Any, resolved: Any) -> None:
    """Reject a silent comparison of traces sampled at different coordinates."""
    for name in ("x_left", "x_right"):
        if not np.isclose(float(getattr(observed, name)), float(getattr(resolved, name))):
            raise ValueError(
                f"Boundary coordinate mismatch for {name}: "
                f"observed={getattr(observed, name)}, resolved={getattr(resolved, name)}"
            )
    for name in ("y_left", "y_right"):
        observed_values = np.asarray(getattr(observed, name), dtype=np.float64)
        resolved_values = np.asarray(getattr(resolved, name), dtype=np.float64)
        if observed_values.shape != resolved_values.shape or not np.allclose(
            observed_values, resolved_values, rtol=0.0, atol=1e-10
        ):
            raise ValueError(
                f"Boundary sampling mismatch for {name}: "
                f"observed shape={observed_values.shape}, "
                f"resolved shape={resolved_values.shape}"
            )


def _checkpoint_kind(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    return "scattered" if path.name.startswith("scattered_") else "total"


def _select_cases(
    available: list[tuple[float, int, int]],
    frequencies: list[float] | None,
    modes: list[int] | None,
    incidences: list[int] | None = None,
) -> list[tuple[float, int, int]]:
    cases = available
    if frequencies is not None:
        cases = [
            case
            for case in cases
            if any(np.isclose(case[0], frequency) for frequency in frequencies)
        ]
    if modes is not None:
        cases = [case for case in cases if case[1] in modes]
    if incidences is not None:
        cases = [case for case in cases if case[2] in incidences]
    if not cases:
        raise ValueError("No checkpoint case matches the requested filters")
    return cases


def _load_checkpoint(
    path: Path, formulation: str
) -> tuple[Any, Callable[..., np.ndarray]]:
    if formulation == "total":
        checkpoint = load_uv_checkpoint(path)
        if checkpoint.sound_speed is None:
            raise ValueError("A format-v3 total-field checkpoint is required")
        predictor = checkpoint.predict_physical
    else:
        checkpoint = load_us_checkpoint(path)
        predictor = checkpoint.predict_total_physical
    return checkpoint, predictor


def _write_nodal_material(
    path: Path, checkpoint: Any, fem_truth: FEMFieldData
) -> np.ndarray:
    speed = checkpoint.predict_sound_speed_physical(fem_truth.x, fem_truth.y)
    speed = np.asarray(speed, dtype=np.float64)
    if speed.shape != fem_truth.x.shape or not np.isfinite(speed).all() or (speed <= 0.0).any():
        raise ValueError("Checkpoint produced an invalid nodal sound-speed field")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("node_id", "x", "y", "c"))
        writer.writerows(
            (node_id, x, y, value)
            for node_id, (x, y, value) in enumerate(zip(fem_truth.x, fem_truth.y, speed))
        )
    return speed


def _case_strings(cases: list[tuple[float, int, int]]) -> tuple[str, str, str]:
    frequencies = sorted({frequency for frequency, _, _ in cases})
    modes = sorted({mode for _, mode, _ in cases})
    incidences = sorted({incidence for _, _, incidence in cases})
    expected = {
        (frequency, mode, incidence)
        for frequency in frequencies
        for mode in modes
        for incidence in incidences
    }
    if set(cases) != expected:
        raise ValueError(
            "The FEM generator currently accepts a Cartesian frequency/mode/incidence product; "
            "select cases that form a complete product"
        )
    return (
        ",".join(f"{frequency:.17g}" for frequency in frequencies),
        ",".join(str(mode) for mode in modes),
        ",".join(str(incidence) for incidence in incidences),
    )


def main() -> None:
    args = parse_args()
    for path in (
        args.checkpoint,
        args.mesh,
        args.fem_field_truth,
        args.mass_matrix,
        args.stiffness_matrix,
        args.boundary_left_truth,
        args.boundary_right_truth,
        args.generator,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    formulation = _checkpoint_kind(args.checkpoint, args.formulation)
    checkpoint, predict_total = _load_checkpoint(args.checkpoint, formulation)
    fem_truth = FEMFieldData(args.fem_field_truth)
    validate_geometry(checkpoint, fem_truth)
    cases = _select_cases(
        checkpoint.available_cases(), args.frequency, args.mode, args.incidence
    )
    for frequency, mode, incidence in cases:
        if not fem_truth.has_case(frequency, mode, incidence):
            raise ValueError(
                f"Truth FEM field lacks f={frequency}, mode={mode}, incidence={incidence}"
            )

    mass = load_symmetric_coo_matrix(args.mass_matrix, expected_size=fem_truth.size)
    stiffness = load_symmetric_coo_matrix(
        args.stiffness_matrix, expected_size=fem_truth.size
    )
    observed_boundary = WaveguideBoundaryData(
        args.boundary_left_truth, args.boundary_right_truth
    )
    frequency_text, mode_text, incidence_text = _case_strings(cases)

    # Delay the only persistent write until every input and the requested case
    # selection has been validated. The directory is intentionally exclusive so
    # a previous diagnostic can never be overwritten silently.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    nodal_path = args.output_dir / "nodal_sound_speed.csv"
    speed = _write_nodal_material(nodal_path, checkpoint, fem_truth)
    command = (
        str(args.generator),
        "--mesh",
        str(args.mesh),
        "--defectname",
        "resolved_checkpoint",
        "--freqs",
        frequency_text,
        "--modes",
        mode_text,
        "--incidence",
        incidence_text,
        "--outputdir",
        str(args.output_dir),
        "--c0",
        f"{checkpoint.c0:.17g}",
        "--nodal-sound-speed",
        str(nodal_path),
        "--numberofdatapoints",
        "31",
    )
    completed = subprocess.run(command, check=True, capture_output=True, text=True)

    resolved_field = FEMFieldData(args.output_dir / "fem_field_resolved_checkpoint.csv")
    validate_geometry(checkpoint, resolved_field)
    resolved_boundary = WaveguideBoundaryData(
        args.output_dir / "pinn_boundary_left_resolved_checkpoint.csv",
        args.output_dir / "pinn_boundary_right_resolved_checkpoint.csv",
    )

    rows: list[dict[str, float | int]] = []
    for frequency, mode, incidence in cases:
        if not resolved_field.has_case(frequency, mode, incidence):
            raise ValueError(
                f"Resolved FEM field lacks f={frequency}, mode={mode}, incidence={incidence}"
            )
        truth = fem_truth.case(frequency, mode, incidence).values
        resolved = resolved_field.case(frequency, mode, incidence).values
        pinn = np.asarray(
            predict_total(frequency, mode, fem_truth.x, fem_truth.y, incidence=incidence),
            dtype=np.complex128,
        )
        pinn_vs_resolved = compute_misfit_metrics(resolved, pinn, mass, stiffness)
        resolved_vs_truth = compute_misfit_metrics(truth, resolved, mass, stiffness)
        pinn_vs_truth = compute_misfit_metrics(truth, pinn, mass, stiffness)

        observed_pair = observed_boundary.get_pair(mode, frequency, incidence)
        resolved_pair = resolved_boundary.get_pair(mode, frequency, incidence)
        _validate_boundary_sampling(observed_pair, resolved_pair)
        observed_trace = _trace(observed_pair)
        resolved_trace = _trace(resolved_pair)
        pinn_left = predict_total(
            frequency,
            mode,
            np.full_like(observed_pair.y_left, observed_pair.x_left),
            observed_pair.y_left,
            incidence=incidence,
        )
        pinn_right = predict_total(
            frequency,
            mode,
            np.full_like(observed_pair.y_right, observed_pair.x_right),
            observed_pair.y_right,
            incidence=incidence,
        )
        pinn_trace = np.concatenate((pinn_left, pinn_right)).astype(np.complex128)
        rows.append(
            {
                "frequency": frequency,
                "mode": mode,
                "incidence": incidence,
                "pinn_vs_resolved_l2": pinn_vs_resolved.l2_relative,
                "pinn_vs_resolved_h1": pinn_vs_resolved.h1_relative,
                "resolved_vs_truth_l2": resolved_vs_truth.l2_relative,
                "resolved_vs_truth_h1": resolved_vs_truth.h1_relative,
                "pinn_vs_truth_l2": pinn_vs_truth.l2_relative,
                "pinn_vs_truth_h1": pinn_vs_truth.h1_relative,
                "pinn_vs_observed_trace": _relative_error(pinn_trace, observed_trace),
                "resolved_vs_observed_trace": _relative_error(
                    resolved_trace, observed_trace
                ),
                "pinn_vs_resolved_trace": _relative_error(pinn_trace, resolved_trace),
            }
        )

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "formulation": formulation,
        "cases": cases,
        "mesh": str(args.mesh),
        "mesh_sha256": _sha256(args.mesh),
        "fem_field_truth": str(args.fem_field_truth),
        "fem_field_truth_sha256": _sha256(args.fem_field_truth),
        "mass_matrix": str(args.mass_matrix),
        "mass_matrix_sha256": _sha256(args.mass_matrix),
        "stiffness_matrix": str(args.stiffness_matrix),
        "stiffness_matrix_sha256": _sha256(args.stiffness_matrix),
        "boundary_left_truth": str(args.boundary_left_truth),
        "boundary_left_truth_sha256": _sha256(args.boundary_left_truth),
        "boundary_right_truth": str(args.boundary_right_truth),
        "boundary_right_truth_sha256": _sha256(args.boundary_right_truth),
        "generator_command": list(command),
        "generator_stdout": completed.stdout,
        "generator_stderr": completed.stderr,
        "nodal_sound_speed": {
            "minimum": float(np.min(speed)),
            "maximum": float(np.max(speed)),
            "mean": float(np.mean(speed)),
        },
    }
    with (args.output_dir / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)

    for row in rows:
        print(
            f"f={row['frequency']:.8g}, mode={row['mode']}, "
            f"incidence={row['incidence']}: "
            f"PINN/resolved FEM L2={100.0 * row['pinn_vs_resolved_l2']:.2f}%, "
            f"resolved/truth L2={100.0 * row['resolved_vs_truth_l2']:.2f}%, "
            f"trace resolved/data={100.0 * row['resolved_vs_observed_trace']:.2f}%"
        )
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
