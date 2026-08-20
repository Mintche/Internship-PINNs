"""Offline PDF post-processing for inverse-PINN runs and campaigns."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .checkpoints import load_material_checkpoint, load_pressure_checkpoint
from .config import Case, InverseConfig
from .data import load_inverse_dataset, truth_sound_speed
from .losses import build_case_context, build_physics_context, physical_pressure_prediction
from .models import material_sound_speed
from .runtime import configure_jax_compilation_cache
from .variants import parse_variant


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _run_directories(root: Path) -> list[Path]:
    candidates = []
    for manifest_path in root.rglob("manifest.json"):
        manifest = _load_json(manifest_path)
        if "variant" in manifest and "config" in manifest:
            candidates.append(manifest_path.parent)
    if not candidates:
        raise ValueError(f"No inverse run found below {root}")
    return sorted(set(candidates))


def _batch_pressure(params, physics, context, variant, x, y, batch_size):
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    function = jax.jit(
        lambda candidate, b_base, xv, yv: physical_pressure_prediction(
            candidate, physics, replace(context, b_base=b_base), variant, xv, yv
        )
    )
    chunks = []
    for start in range(0, x.size, batch_size):
        chunks.append(
            np.asarray(
                function(
                    params,
                    context.b_base,
                    jnp.asarray(x[start : start + batch_size], dtype=jnp.float32),
                    jnp.asarray(y[start : start + batch_size], dtype=jnp.float32),
                )
            )
        )
    return np.concatenate(chunks)


def _batch_celerity(params, config, x, y):
    shape = np.asarray(x).shape
    xn = np.asarray(x).ravel() / config.geometry.half_length
    yn = 2.0 * np.asarray(y).ravel() / config.geometry.height - 1.0
    function = jax.jit(
        lambda candidate, xv, yv: jax.vmap(
            lambda x_value, y_value: material_sound_speed(
                candidate, config.geometry, x_value, y_value
            )
        )(xv, yv)
    )
    chunks = []
    size = config.logging.prediction_batch_size
    for start in range(0, xn.size, size):
        chunks.append(
            np.asarray(
                function(
                    params,
                    jnp.asarray(xn[start : start + size], dtype=jnp.float32),
                    jnp.asarray(yn[start : start + size], dtype=jnp.float32),
                )
            )
        )
    return np.concatenate(chunks).reshape(shape)


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def _tripcolor(axis, mesh, values, *, title: str, cmap: str, vmin=None, vmax=None):
    artist = axis.tripcolor(
        mesh.x, mesh.y, mesh.subtriangles(), np.asarray(values),
        shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True,
    )
    axis.set_title(title)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_aspect("equal")
    return artist


def _save_pressure_figures(
    figures: Path, case: Case, fem, prediction, mesh, metric: Mapping[str, Any]
) -> None:
    plt = _style()
    reference_real = np.real(fem.values)
    prediction_real = np.real(prediction)
    lower = min(reference_real.min(), prediction_real.min())
    upper = max(reference_real.max(), prediction_real.max())

    figure, axis = plt.subplots(figsize=(8.0, 3.1), constrained_layout=True)
    artist = _tripcolor(
        axis, mesh, prediction_real, title=f"PINN Re(u) — {case.id}",
        cmap="RdBu_r", vmin=lower, vmax=upper,
    )
    figure.colorbar(artist, ax=axis)
    figure.savefig(figures / f"pressure_real_{case.id}.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(8.0, 5.8), constrained_layout=True)
    artists = [
        _tripcolor(axes[0], mesh, reference_real, title="FEM Re(u)", cmap="RdBu_r", vmin=lower, vmax=upper),
        _tripcolor(axes[1], mesh, prediction_real, title="PINN Re(u)", cmap="RdBu_r", vmin=lower, vmax=upper),
    ]
    figure.colorbar(artists[-1], ax=axes, shrink=0.8)
    figure.savefig(figures / f"pressure_comparison_{case.id}.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.0, 3.1), constrained_layout=True)
    artist = _tripcolor(
        axis, mesh, np.abs(prediction - fem.values),
        title=(
            f"|u PINN − u FEM| — L2 rel. {float(metric['l2_relative']):.3e}, "
            f"H1 rel. {float(metric['h1_relative']):.3e}"
        ), cmap="magma",
    )
    figure.colorbar(artist, ax=axis)
    figure.savefig(figures / f"pressure_misfit_{case.id}.pdf")
    plt.close(figure)


def _pcolormesh(axis, x, y, values, *, title, cmap, vmin=None, vmax=None):
    artist = axis.pcolormesh(
        x, y, values, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax,
        rasterized=True,
    )
    axis.set_title(title)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_aspect("equal")
    return artist


def _save_pressure_common_loss_figure(figures: Path, history_path: Path) -> None:
    if not history_path.is_file() or history_path.stat().st_size == 0:
        return
    frame = pd.read_csv(history_path)
    if frame.empty:
        return
    plt = _style()
    figure, axis = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    x = pd.to_numeric(frame["evaluation_index"], errors="coerce").to_numpy()
    styles = {
        "current_objective": ("common objective", 2.0),
        "static_objective": ("static monitor objective", 2.0),
        "pde": ("mean PDE", 1.2),
        "neumann": ("mean Neumann", 1.2),
        "dtn": ("mean DtN", 1.2),
        "data": ("mean data", 1.2),
    }
    for column, (label, width) in styles.items():
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        values = np.where(values > 0.0, values, np.nan)
        if np.isfinite(values).any():
            axis.semilogy(
                x, values, label=label, linewidth=width, rasterized=True
            )
    phases = frame["phase"].astype(str).to_numpy()
    for index in np.flatnonzero(phases[1:] != phases[:-1]) + 1:
        axis.axvline(x[index], color="0.75", linewidth=0.8, linestyle="--")
    axis.set_title("Common pressure loss (mean over active acquisitions)")
    axis.set_xlabel("monitor evaluation")
    axis.set_ylabel("loss")
    axis.grid(True, which="both", alpha=0.2)
    axis.legend(fontsize=8, ncol=2)
    figure.savefig(figures / "pressure_common_loss.pdf")
    plt.close(figure)


def _save_material_figures(figures, config, params, metrics, snapshots_path, cosine_path, cosines):
    plt = _style()
    with np.load(snapshots_path, allow_pickle=False) as archive:
        x = archive["x"]
        y = archive["y"]
        snapshots = archive["celerity"]
        steps = archive["steps"]
        fractions = archive["fractions"]
    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
    prediction = _batch_celerity(params, config, x_grid, y_grid)
    truth = truth_sound_speed(config.geometry, x_grid, y_grid)
    lower = min(truth.min(), prediction.min())
    upper = max(truth.max(), prediction.max())

    figure, axis = plt.subplots(figsize=(8.0, 3.1), constrained_layout=True)
    artist = _pcolormesh(axis, x, y, prediction, title="Reconstructed sound speed", cmap="viridis", vmin=lower, vmax=upper)
    figure.colorbar(artist, ax=axis, label="c (m/s)")
    figure.savefig(figures / "celerity_reconstructed.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(8.0, 5.8), constrained_layout=True)
    _pcolormesh(axes[0], x, y, truth, title="Ground-truth sound speed", cmap="viridis", vmin=lower, vmax=upper)
    artist = _pcolormesh(axes[1], x, y, prediction, title="Reconstructed sound speed", cmap="viridis", vmin=lower, vmax=upper)
    figure.colorbar(artist, ax=axes, shrink=0.8, label="c (m/s)")
    figure.savefig(figures / "celerity_comparison.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.0, 3.1), constrained_layout=True)
    annotation = metrics.get("anomaly_relative_l1")
    label = "undefined" if annotation is None else f"{float(annotation):.3e}"
    artist = _pcolormesh(
        axis, x, y, np.abs(prediction - truth),
        title=f"|c PINN − c truth| — anomaly-relative L1 {label}", cmap="magma",
    )
    figure.colorbar(artist, ax=axis, label="error (m/s)")
    figure.savefig(figures / "celerity_misfit.pdf")
    plt.close(figure)

    if len(snapshots):
        columns = min(3, len(snapshots))
        rows = int(np.ceil(len(snapshots) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(5.0 * columns, 2.8 * rows), squeeze=False, constrained_layout=True)
        for index, axis in enumerate(axes.ravel()):
            if index >= len(snapshots):
                axis.set_visible(False)
                continue
            artist = _pcolormesh(
                axis, x, y, snapshots[index],
                title=f"{100 * fractions[index]:.0f}% — step {steps[index]}",
                cmap="viridis", vmin=lower, vmax=upper,
            )
        figure.colorbar(artist, ax=list(axes.ravel()), shrink=0.8, label="c (m/s)")
        figure.savefig(figures / "celerity_snapshots.pdf")
        plt.close(figure)

    if cosines and cosine_path.is_file() and cosine_path.stat().st_size:
        frame = pd.read_csv(cosine_path)
        if not frame.empty:
            grouped = list(frame.groupby(["step", "fraction"], sort=True))
            figure, axes = plt.subplots(1, len(grouped), figsize=(4.2 * len(grouped), 3.7), squeeze=False, constrained_layout=True)
            for axis, ((step, fraction), group) in zip(axes.ravel(), grouped):
                size = int(max(group["row_index"].max(), group["column_index"].max())) + 1
                matrix = np.full((size, size), np.nan)
                matrix[group["row_index"].astype(int), group["column_index"].astype(int)] = group["cosine"]
                artist = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm", rasterized=True)
                labels = group.sort_values("row_index").drop_duplicates("row_index")["row_case_id"].tolist()
                axis.set_xticks(range(size), labels, rotation=90, fontsize=7)
                axis.set_yticks(range(size), labels, fontsize=7)
                axis.set_title(f"{100 * fraction:.0f}% — step {int(step)}")
            figure.colorbar(artist, ax=list(axes.ravel()), shrink=0.7, label="cosine")
            figure.savefig(figures / "material_gradient_cosines.pdf")
            plt.close(figure)


def plot_campaign(root: Path, *, cosines: bool = False) -> list[Path]:
    configure_jax_compilation_cache()
    outputs = []
    for run_directory in _run_directories(Path(root)):
        manifest = _load_json(run_directory / "manifest.json")
        config = InverseConfig.from_json(manifest["config"]["source"])
        variant = parse_variant(manifest["variant"]["name"])
        dataset = load_inverse_dataset(config)
        physics = build_physics_context(config.geometry, config.all_cases)
        for package_directory in sorted((run_directory / "packages").glob("pkg*")):
            summary = _load_json(package_directory / "summary.json")
            active = tuple(
                Case(float(item["frequency"]), int(item["mode"]), int(item["incidence"]))
                for item in summary["active_cases"]
            )
            field_models, scales, _ = load_pressure_checkpoint(
                package_directory / "pressure_weights_best.npz"
            )
            material_params, _ = load_material_checkpoint(
                package_directory / "slowness_weights_best.npz"
            )
            contexts = {
                case: build_case_context(
                    physics, case, field_models[case].b_base,
                    dataset.boundaries[case], variant,
                )
                for case in active
            }
            for case in active:
                if not np.isclose(contexts[case].field_scale, scales[case]):
                    raise ValueError(f"Boundary normalization changed for {case.id}")
            metric_frame = pd.read_csv(package_directory / "pressure_metrics.csv").set_index("case_id")
            figures = package_directory / "figures"
            figures.mkdir(exist_ok=True)
            _save_pressure_common_loss_figure(
                figures, package_directory / "pressure_common_loss_history.csv"
            )
            for case in active:
                fem = dataset.fem_cases[case]
                prediction = _batch_pressure(
                    field_models[case].params, physics, contexts[case], variant,
                    fem.x, fem.y, config.logging.prediction_batch_size,
                )
                _save_pressure_figures(
                    figures, case, fem, prediction, dataset.mesh,
                    metric_frame.loc[case.id].to_dict(),
                )
            _save_material_figures(
                figures, config, material_params,
                _load_json(package_directory / "celerity_metrics.json"),
                package_directory / "celerity_snapshots.npz",
                package_directory / "material_gradient_cosines.csv",
                cosines,
            )
            outputs.append(figures)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--cosines", action="store_true")
    arguments = parser.parse_args(argv)
    for directory in plot_campaign(arguments.campaign_root, cosines=arguments.cosines):
        print(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
