#!/usr/bin/env python3
"""Build a machine-readable sound-speed inventory for UV and US checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.compare_sound_speed import compute_misfit  # noqa: E402
from tools.ground_truth import build_registered_sound_speed  # noqa: E402
from tools.us_checkpoint import load_us_checkpoint  # noqa: E402
from tools.uv_checkpoint import load_uv_checkpoint  # noqa: E402


FIELDNAMES = (
    "checkpoint",
    "formulation",
    "evaluable",
    "error",
    "format_version",
    "defect_name",
    "contrast_ratio",
    "length",
    "height",
    "c0",
    "c_min",
    "c_max",
    "random_seed",
    "case_count",
    "cases",
    "last_inverse_monitor",
    "anomaly_relative_l1",
    "improvement_over_homogeneous",
    "anomaly_mean_absolute",
    "background_mean_absolute",
    "mean_absolute",
    "relative_l1",
    "rmse",
    "relative_l2",
    "max_absolute",
    "created_at_utc",
    "git_commit",
    "git_dirty",
    "network_config",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every total-field and scattered-field checkpoint on one "
            "regular grid and export a CSV inventory."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPOSITORY_ROOT / "pinn_waveguide_2d" / "checkpoints",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "checkpoint_sound_speed_metrics.csv",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "checkpoint_monitor_vs_reconstruction.pdf",
    )
    parser.add_argument("--nx", type=int, default=201)
    parser.add_argument("--ny", type=int, default=121)
    return parser.parse_args()


def _last_inverse_monitor(best_validation_losses: dict[str, Any]) -> float | None:
    records = best_validation_losses.get("inverse", [])
    if not isinstance(records, list) or not records:
        return None
    value = records[-1].get("weighted_total")
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _base_row(path: Path, formulation: str) -> dict[str, Any]:
    return {
        field: "" for field in FIELDNAMES
    } | {
        "checkpoint": path.name,
        "formulation": formulation,
        "evaluable": False,
    }


def _common_checkpoint_row(path: Path, checkpoint: Any, formulation: str) -> dict[str, Any]:
    metadata = checkpoint.metadata
    provenance = checkpoint.provenance
    return _base_row(path, formulation) | {
        "format_version": checkpoint.format_version,
        "defect_name": metadata.get("defect_name", ""),
        "contrast_ratio": metadata.get("contrast_ratio", ""),
        "length": checkpoint.length,
        "height": checkpoint.height,
        "c0": checkpoint.c0,
        "random_seed": "" if checkpoint.random_seed is None else checkpoint.random_seed,
        "case_count": len(checkpoint.available_cases()),
        "cases": _json(checkpoint.available_cases()),
        "last_inverse_monitor": (
            _last_inverse_monitor(checkpoint.best_validation_losses) or ""
        ),
        "created_at_utc": provenance.get("created_at_utc", ""),
        "git_commit": provenance.get("git_commit", ""),
        "git_dirty": provenance.get("git_dirty", ""),
        "network_config": _json(checkpoint.network_config),
    }


def _metric_row(
    checkpoint: Any,
    reconstructed: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> dict[str, Any]:
    ground_truth = build_registered_sound_speed(
        x_grid,
        y_grid,
        defect_name=str(checkpoint.metadata["defect_name"]),
        c0=checkpoint.c0,
        contrast_ratio=float(checkpoint.metadata["contrast_ratio"]),
    )
    metrics = compute_misfit(reconstructed, ground_truth, checkpoint.c0)
    return {
        "evaluable": True,
        "anomaly_relative_l1": metrics.anomaly_relative_l1,
        "improvement_over_homogeneous": metrics.improvement_over_homogeneous,
        "anomaly_mean_absolute": metrics.anomaly_mean_absolute,
        "background_mean_absolute": metrics.background_mean_absolute,
        "mean_absolute": metrics.mean_absolute,
        "relative_l1": metrics.relative_l1,
        "rmse": metrics.rmse,
        "relative_l2": metrics.relative_l2,
        "max_absolute": metrics.max_absolute,
    }


def summarize_total_checkpoint(path: Path, nx: int, ny: int) -> dict[str, Any]:
    checkpoint = load_uv_checkpoint(path)
    row = _common_checkpoint_row(path, checkpoint, "total")
    if checkpoint.sound_speed is None:
        return row | {
            "error": "format-v1 checkpoint has no stored sound-speed network",
        }

    row["c_min"] = checkpoint.sound_speed.c_min
    row["c_max"] = checkpoint.sound_speed.c_max
    x_axis = np.linspace(-checkpoint.length, checkpoint.length, nx)
    y_axis = np.linspace(0.0, checkpoint.height, ny)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="xy")
    reconstructed = checkpoint.predict_sound_speed_physical(x_grid, y_grid)
    return row | _metric_row(checkpoint, reconstructed, x_grid, y_grid)


def summarize_scattered_checkpoint(path: Path, nx: int, ny: int) -> dict[str, Any]:
    checkpoint = load_us_checkpoint(path)
    row = _common_checkpoint_row(path, checkpoint, "scattered")
    m0 = 1.0 / checkpoint.c0**2
    row["c_min"] = 1.0 / math.sqrt(m0 + checkpoint.ms_model.ms_max)
    row["c_max"] = 1.0 / math.sqrt(m0 + checkpoint.ms_model.ms_min)
    x_axis = np.linspace(-checkpoint.length, checkpoint.length, nx)
    y_axis = np.linspace(0.0, checkpoint.height, ny)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="xy")
    reconstructed = checkpoint.predict_sound_speed_physical(x_grid, y_grid)
    return row | _metric_row(checkpoint, reconstructed, x_grid, y_grid)


def summarize_directory(checkpoint_dir: Path, nx: int, ny: int) -> list[dict[str, Any]]:
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must both be at least 2")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")

    rows = []
    for path in sorted(checkpoint_dir.glob("*.npz")):
        formulation = "scattered" if path.name.startswith("scattered_") else "total"
        try:
            if formulation == "scattered":
                row = summarize_scattered_checkpoint(path, nx, ny)
            else:
                row = summarize_total_checkpoint(path, nx, ny)
        except Exception as error:  # Keep the inventory complete and explicit.
            row = _base_row(path, formulation) | {
                "error": f"{type(error).__name__}: {error}",
            }
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_monitor_figure(rows: list[dict[str, Any]], output: Path) -> None:
    valid_rows = [
        row
        for row in rows
        if row["evaluable"]
        and row["last_inverse_monitor"] != ""
        and row["anomaly_relative_l1"] is not None
        and float(row["last_inverse_monitor"]) > 0.0
    ]
    if not valid_rows:
        raise ValueError("No evaluated checkpoint has a positive inverse monitor")

    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    colors = {"total": "#0072B2", "scattered": "#D55E00"}
    labels = {"total": "Total field", "scattered": "Scattered field"}
    for formulation in ("total", "scattered"):
        subset = [row for row in valid_rows if row["formulation"] == formulation]
        axis.scatter(
            [float(row["last_inverse_monitor"]) for row in subset],
            [100.0 * float(row["anomaly_relative_l1"]) for row in subset],
            s=38,
            alpha=0.82,
            color=colors[formulation],
            label=labels[formulation],
        )

    axis.axhline(100.0, color="black", linestyle="--", linewidth=1.0)
    axis.text(
        max(float(row["last_inverse_monitor"]) for row in valid_rows),
        104.0,
        r"homogeneous $c=c_0$ baseline",
        ha="right",
        va="bottom",
        fontsize=8,
    )

    annotations = []
    for formulation in ("total", "scattered"):
        subset = [row for row in valid_rows if row["formulation"] == formulation]
        annotations.append(
            (min(subset, key=lambda row: float(row["anomaly_relative_l1"])), f"best {formulation}")
        )
    worse_than_baseline = [
        row for row in valid_rows if float(row["anomaly_relative_l1"]) > 1.0
    ]
    if worse_than_baseline:
        low_monitor_poor_map = min(
            worse_than_baseline,
            key=lambda row: float(row["last_inverse_monitor"]),
        )
        annotations.append((low_monitor_poor_map, "low monitor, poor map"))
    seen = set()
    for row, label in annotations:
        key = row["checkpoint"]
        if key in seen:
            continue
        seen.add(key)
        axis.annotate(
            label,
            (
                float(row["last_inverse_monitor"]),
                100.0 * float(row["anomaly_relative_l1"]),
            ),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8,
        )

    axis.set_xscale("log")
    axis.set_xlabel("Last stored fixed-collocation optimization monitor")
    axis.set_ylabel("Anomaly-relative L1 error [%]")
    axis.set_title("Archived checkpoints: internal monitor vs coefficient error")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.text(
        0.5,
        0.01,
        "Exploratory archive: seed 0 and non-paired protocols; no causal method comparison.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows = summarize_directory(args.checkpoint_dir, args.nx, args.ny)
    write_csv(rows, args.output)
    write_monitor_figure(rows, args.figure)
    evaluated = sum(str(row["evaluable"]).lower() == "true" for row in rows)
    print(
        f"Wrote {len(rows)} checkpoints ({evaluated} evaluated) to {args.output} "
        f"and {args.figure}"
    )


if __name__ == "__main__":
    main()
