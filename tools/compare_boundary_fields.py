#!/usr/bin/env python3
"""Compare scattered boundary traces from two FEM data exports."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.data_loader import BoundaryPair, WaveguideBoundaryData  # noqa: E402


@dataclass(frozen=True)
class BoundarySideComparison:
    y: np.ndarray
    scattered_a: np.ndarray
    scattered_b: np.ndarray
    difference: np.ndarray


@dataclass(frozen=True)
class BoundaryComparisonMetrics:
    relative_l2: float
    symmetric_relative_l2: float
    rms_absolute: float
    max_absolute: float
    left_relative_l2: float
    right_relative_l2: float


@dataclass(frozen=True)
class BoundaryComparisonResult:
    frequency: float
    mode: int
    left: BoundarySideComparison
    right: BoundarySideComparison
    metrics: BoundaryComparisonMetrics


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
            "Compare scattered boundary fields Us from two FEM boundary CSV pairs."
        )
    )
    parser.add_argument("--left-a", required=True, type=Path)
    parser.add_argument("--right-a", required=True, type=Path)
    parser.add_argument("--left-b", required=True, type=Path)
    parser.add_argument("--right-b", required=True, type=Path)
    parser.add_argument("--freqs", required=True, type=parse_float_list)
    parser.add_argument("--modes", required=True, type=parse_int_list)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for a metrics CSV and one PDF per case.",
    )
    return parser.parse_args()


def _complex_left(pair: BoundaryPair) -> np.ndarray:
    return pair.u_re_left.astype(np.float64) + 1j * pair.u_im_left.astype(np.float64)


def _complex_right(pair: BoundaryPair) -> np.ndarray:
    return pair.u_re_right.astype(np.float64) + 1j * pair.u_im_right.astype(np.float64)


def _vertical_bounds(pair: BoundaryPair) -> tuple[float, float]:
    y_values = np.concatenate(
        (
            np.asarray(pair.y_left, dtype=np.float64),
            np.asarray(pair.y_right, dtype=np.float64),
        )
    )
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    height = y_max - y_min
    if not np.isfinite(height) or height <= 0.0:
        raise ValueError(
            f"Invalid boundary vertical bounds y=[{y_min}, {y_max}] for "
            f"f={pair.frequency}, mode={pair.mode}"
        )
    return y_min, height


def incident_wave(
    x: float,
    y: np.ndarray,
    *,
    k0: float,
    mode: int,
    y_min: float,
    height: float,
) -> np.ndarray:
    amplitude = np.sqrt(1.0 / height) if mode == 0 else np.sqrt(2.0 / height)
    y_shifted = np.asarray(y, dtype=np.float64) - y_min
    transverse_wavenumber = mode * np.pi / height
    beta = np.sqrt(k0**2 - transverse_wavenumber**2 + 0j)
    mode_shape = amplitude * np.cos(transverse_wavenumber * y_shifted)
    return mode_shape * np.exp(1j * beta * float(x))


def scattered_sides(pair: BoundaryPair) -> tuple[np.ndarray, np.ndarray]:
    y_min, height = _vertical_bounds(pair)
    left_incident = incident_wave(
        pair.x_left,
        pair.y_left,
        k0=pair.k0,
        mode=pair.mode,
        y_min=y_min,
        height=height,
    )
    right_incident = incident_wave(
        pair.x_right,
        pair.y_right,
        k0=pair.k0,
        mode=pair.mode,
        y_min=y_min,
        height=height,
    )
    return _complex_left(pair) - left_incident, _complex_right(pair) - right_incident


def _validate_side_grid(
    side: str,
    pair_a: BoundaryPair,
    pair_b: BoundaryPair,
    *,
    atol: float = 1e-7,
) -> None:
    x_a = pair_a.x_left if side == "left" else pair_a.x_right
    x_b = pair_b.x_left if side == "left" else pair_b.x_right
    y_a = pair_a.y_left if side == "left" else pair_a.y_right
    y_b = pair_b.y_left if side == "left" else pair_b.y_right

    if not np.isclose(x_a, x_b, rtol=0.0, atol=atol):
        raise ValueError(
            f"{side} boundary x mismatch for f={pair_a.frequency}, mode={pair_a.mode}: "
            f"{x_a} versus {x_b}"
        )
    if y_a.shape != y_b.shape or not np.allclose(y_a, y_b, rtol=0.0, atol=atol):
        raise ValueError(
            f"{side} boundary y-grid mismatch for f={pair_a.frequency}, "
            f"mode={pair_a.mode}"
        )


def validate_pair_compatibility(pair_a: BoundaryPair, pair_b: BoundaryPair) -> None:
    if pair_a.mode != pair_b.mode or not np.isclose(pair_a.frequency, pair_b.frequency):
        raise ValueError(
            f"Boundary case mismatch: A has (f={pair_a.frequency}, mode={pair_a.mode}), "
            f"B has (f={pair_b.frequency}, mode={pair_b.mode})"
        )
    if not np.isclose(pair_a.k0, pair_b.k0, rtol=1e-7, atol=1e-9):
        raise ValueError(
            f"k0 mismatch for f={pair_a.frequency}, mode={pair_a.mode}: "
            f"{pair_a.k0} versus {pair_b.k0}"
        )
    _validate_side_grid("left", pair_a, pair_b)
    _validate_side_grid("right", pair_a, pair_b)


def _relative_l2(error: np.ndarray, reference: np.ndarray) -> float:
    error_norm = float(np.linalg.norm(error))
    reference_norm = float(np.linalg.norm(reference))
    if reference_norm == 0.0:
        return 0.0 if error_norm == 0.0 else float("inf")
    return error_norm / reference_norm


def compute_metrics(
    left_a: np.ndarray,
    right_a: np.ndarray,
    left_b: np.ndarray,
    right_b: np.ndarray,
    left_difference: np.ndarray,
    right_difference: np.ndarray,
) -> BoundaryComparisonMetrics:
    reference = np.concatenate((left_a, right_a))
    comparison = np.concatenate((left_b, right_b))
    difference = np.concatenate((left_difference, right_difference))
    abs_difference = np.abs(difference)
    symmetric_denominator = float(np.linalg.norm(reference) + np.linalg.norm(comparison))
    symmetric_relative_l2 = (
        2.0 * float(np.linalg.norm(difference)) / symmetric_denominator
        if symmetric_denominator > 0.0
        else 0.0
    )
    return BoundaryComparisonMetrics(
        relative_l2=_relative_l2(difference, reference),
        symmetric_relative_l2=symmetric_relative_l2,
        rms_absolute=float(np.sqrt(np.mean(abs_difference**2))),
        max_absolute=float(np.max(abs_difference)),
        left_relative_l2=_relative_l2(left_difference, left_a),
        right_relative_l2=_relative_l2(right_difference, right_a),
    )


def select_cases(
    data_a: WaveguideBoundaryData,
    data_b: WaveguideBoundaryData,
    frequencies: list[float],
    modes: list[int],
) -> list[tuple[int, float]]:
    selected = []
    missing = []
    for frequency in frequencies:
        for mode in modes:
            has_a = data_a.has_pair(mode, frequency)
            has_b = data_b.has_pair(mode, frequency)
            if has_a and has_b:
                pair_a = data_a.get_pair(mode, frequency)
                selected.append((pair_a.mode, pair_a.frequency))
            else:
                missing.append((mode, frequency, has_a, has_b))

    if missing:
        details = [
            f"(mode={mode}, f={frequency}, in_a={has_a}, in_b={has_b})"
            for mode, frequency, has_a, has_b in missing
        ]
        raise ValueError(
            "Requested boundary cases are unavailable: "
            + ", ".join(details)
            + f"; available A: {data_a.available_pairs}; available B: {data_b.available_pairs}"
        )
    return selected


def compare_cases(
    data_a: WaveguideBoundaryData,
    data_b: WaveguideBoundaryData,
    frequencies: list[float],
    modes: list[int],
) -> list[BoundaryComparisonResult]:
    results = []
    for mode, frequency in select_cases(data_a, data_b, frequencies, modes):
        pair_a = data_a.get_pair(mode, frequency)
        pair_b = data_b.get_pair(mode, frequency)
        validate_pair_compatibility(pair_a, pair_b)

        left_a, right_a = scattered_sides(pair_a)
        left_b, right_b = scattered_sides(pair_b)
        left_difference = left_b - left_a
        right_difference = right_b - right_a
        metrics = compute_metrics(
            left_a,
            right_a,
            left_b,
            right_b,
            left_difference,
            right_difference,
        )
        results.append(
            BoundaryComparisonResult(
                frequency=pair_a.frequency,
                mode=pair_a.mode,
                left=BoundarySideComparison(
                    y=np.asarray(pair_a.y_left, dtype=np.float64),
                    scattered_a=left_a,
                    scattered_b=left_b,
                    difference=left_difference,
                ),
                right=BoundarySideComparison(
                    y=np.asarray(pair_a.y_right, dtype=np.float64),
                    scattered_a=right_a,
                    scattered_b=right_b,
                    difference=right_difference,
                ),
                metrics=metrics,
            )
        )
    return results


def aggregate_metrics(results: list[BoundaryComparisonResult]) -> BoundaryComparisonMetrics:
    if not results:
        raise ValueError("Cannot aggregate an empty boundary comparison")
    left_a = np.concatenate([result.left.scattered_a for result in results])
    right_a = np.concatenate([result.right.scattered_a for result in results])
    left_b = np.concatenate([result.left.scattered_b for result in results])
    right_b = np.concatenate([result.right.scattered_b for result in results])
    return compute_metrics(
        left_a,
        right_a,
        left_b,
        right_b,
        left_b - left_a,
        right_b - right_a,
    )


def print_metrics(result: BoundaryComparisonResult) -> None:
    metrics = result.metrics
    print(f"f={result.frequency:.8g} Hz, mode={result.mode}")
    print(
        f"  Us boundary relative L2={metrics.relative_l2:.8e}, "
        f"symmetric relative L2={metrics.symmetric_relative_l2:.8e}, "
        f"RMS abs={metrics.rms_absolute:.8e}, "
        f"max abs={metrics.max_absolute:.8e}"
    )
    print(
        f"  left relative L2={metrics.left_relative_l2:.8e}, "
        f"right relative L2={metrics.right_relative_l2:.8e}"
    )


def _plot_side(
    axis: plt.Axes,
    side: BoundarySideComparison,
    *,
    side_label: str,
    label_a: str,
    label_b: str,
) -> None:
    axis.plot(side.y, side.scattered_a.real, label=f"Re Us {label_a}", linewidth=1.5)
    axis.plot(side.y, side.scattered_b.real, label=f"Re Us {label_b}", linewidth=1.5)
    axis.plot(
        side.y,
        side.scattered_a.imag,
        label=f"Im Us {label_a}",
        linewidth=1.2,
        linestyle="--",
    )
    axis.plot(
        side.y,
        side.scattered_b.imag,
        label=f"Im Us {label_b}",
        linewidth=1.2,
        linestyle="--",
    )
    axis.plot(
        side.y,
        np.abs(side.difference),
        color="black",
        linewidth=1.4,
        label="|diff|",
    )
    axis.set_title(side_label)
    axis.set_xlabel("y")
    axis.grid(True, alpha=0.25)


def create_comparison_figure(
    result: BoundaryComparisonResult,
    *,
    label_a: str = "A",
    label_b: str = "B",
) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.4), sharex=False, sharey=True)
    _plot_side(axes[0], result.left, side_label="Left boundary", label_a=label_a, label_b=label_b)
    _plot_side(axes[1], result.right, side_label="Right boundary", label_a=label_a, label_b=label_b)
    axes[0].set_ylabel("scattered pressure")
    axes[1].legend(loc="best", fontsize="small")
    figure.suptitle(
        f"Scattered boundary comparison - f={result.frequency:.8g} Hz, "
        f"mode={result.mode}, relative L2={result.metrics.relative_l2:.4e}"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    return figure


def main() -> None:
    args = parse_args()
    data_a = WaveguideBoundaryData(args.left_a, args.right_a)
    data_b = WaveguideBoundaryData(args.left_b, args.right_b)
    results = compare_cases(data_a, data_b, args.freqs, args.modes)
    combined_metrics = aggregate_metrics(results)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = args.output_dir / "boundary_comparison_metrics.csv"
        with metrics_path.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "frequency",
                    "mode",
                    "relative_l2",
                    "symmetric_relative_l2",
                    "rms_absolute",
                    "max_absolute",
                    "left_relative_l2",
                    "right_relative_l2",
                ),
            )
            writer.writeheader()
            for result in results:
                writer.writerow(
                    {
                        "frequency": result.frequency,
                        "mode": result.mode,
                        **vars(result.metrics),
                    }
                )
            writer.writerow(
                {
                    "frequency": "all",
                    "mode": "all",
                    **vars(combined_metrics),
                }
            )

    for result in results:
        print_metrics(result)
        figure = create_comparison_figure(
            result, label_a=args.label_a, label_b=args.label_b
        )
        if args.output_dir is not None:
            frequency_label = f"{result.frequency:g}".replace(".", "p")
            figure.savefig(
                args.output_dir
                / f"boundary_comparison_f{frequency_label}_m{result.mode}.pdf",
                bbox_inches="tight",
            )
            plt.close(figure)
    if args.output_dir is None:
        plt.show()
    print("all requested cases")
    print(
        f"  Us boundary relative L2={combined_metrics.relative_l2:.8e}, "
        f"symmetric relative L2={combined_metrics.symmetric_relative_l2:.8e}, "
        f"RMS abs={combined_metrics.rms_absolute:.8e}, "
        f"max abs={combined_metrics.max_absolute:.8e}"
    )


if __name__ == "__main__":
    main()
