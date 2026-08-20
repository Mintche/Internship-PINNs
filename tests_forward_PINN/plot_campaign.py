#!/usr/bin/env python3
"""Aggregate completed forward runs and create report-ready figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tests_forward_PINN.model_variants import VARIANTS, uses_adweights, variant_label


VARIANT_ORDER = VARIANTS
VARIANT_COLORS = {
    variant: plt.get_cmap("tab20")(index)
    for index, variant in enumerate(VARIANT_ORDER)
}


def _label(variant: str) -> str:
    return variant_label(variant) if variant in VARIANT_ORDER else variant


def discover_runs(campaign_root: str | Path) -> list[Path]:
    root = Path(campaign_root)
    runs = sorted(path.parent for path in root.rglob("summary.json"))
    if not runs:
        raise FileNotFoundError(f"No completed run summaries found under {root}")
    return runs


def load_campaign_frames(
    campaign_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    campaign_root = Path(campaign_root)
    summaries = []
    loss_frames = []
    fem_frames = []
    gradient_frames = []
    adweights_frames = []
    missing_gradient_directories = []
    identities: set[tuple[str, int]] = set()
    for directory in discover_runs(campaign_root):
        with (directory / "summary.json").open(encoding="utf-8") as stream:
            summary = json.load(stream)
        identity = (str(summary["variant"]), int(summary["seed"]))
        if identity in identities:
            raise ValueError(f"Duplicate run identity {identity}")
        identities.add(identity)
        summaries.append(summary)

        loss = pd.read_csv(directory / "loss_history.csv")
        fem = pd.read_csv(directory / "fem_metrics.csv")
        gradient_path = directory / "gradient_history.csv"
        gradient = pd.read_csv(gradient_path) if gradient_path.is_file() else None
        adweights_path = directory / "adweights_history.csv"
        adweights = (
            pd.read_csv(adweights_path) if adweights_path.is_file() else None
        )
        for frame, expected in (
            (loss, "loss_monitor"),
            (fem, "fem_metrics"),
        ):
            if frame.empty or set(frame["record_type"]) != {expected}:
                raise ValueError(f"Invalid {expected} history in {directory}")
            if set(zip(frame["variant"], frame["seed"])) != {identity}:
                raise ValueError(f"History identity mismatch in {directory}")
        if gradient is None:
            missing_gradient_directories.append(directory)
        else:
            if (
                gradient.empty
                or set(gradient["record_type"]) != {"gradient_monitor"}
            ):
                raise ValueError(f"Invalid gradient_monitor history in {directory}")
            if set(zip(gradient["variant"], gradient["seed"])) != {identity}:
                raise ValueError(f"Gradient history identity mismatch in {directory}")
            gradient_frames.append(gradient)
        if uses_adweights(identity[0]):
            if adweights is None:
                raise ValueError(f"Missing adweights history in {directory}")
            if (
                adweights.empty
                or set(adweights["record_type"]) != {"adweights_state"}
            ):
                raise ValueError(f"Invalid adweights history in {directory}")
            if set(zip(adweights["variant"], adweights["seed"])) != {identity}:
                raise ValueError(f"Adweights history identity mismatch in {directory}")
            adweights_frames.append(adweights)
        loss_frames.append(loss)
        fem_frames.append(fem)

    if gradient_frames and missing_gradient_directories:
        raise ValueError(
            "Incomplete gradient histories; missing="
            f"{missing_gradient_directories}"
        )

    campaign_summary_path = campaign_root / "campaign_summary.json"
    if campaign_summary_path.is_file():
        with campaign_summary_path.open(encoding="utf-8") as stream:
            campaign_summary = json.load(stream)
        failed = [
            item for item in campaign_summary.get("statuses", [])
            if item.get("status") != "complete"
        ]
        if failed:
            raise ValueError(f"Campaign contains failed runs: {failed}")

    manifest_path = campaign_root / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        configuration = manifest.get("configuration", {})
        variants = configuration.get("variants")
        seeds = configuration.get("seeds")
        if variants is not None and seeds is not None:
            expected = {
                (str(variant), int(seed)) for variant in variants for seed in seeds
            }
            missing = sorted(expected - identities)
            unexpected = sorted(identities - expected)
            if missing or unexpected:
                raise ValueError(
                    f"Incomplete campaign; missing={missing}, unexpected={unexpected}"
                )

    runs = pd.DataFrame(summaries)
    if "adam_steps" in runs and runs["adam_steps"].nunique() != 1:
        raise ValueError("Runs use inconsistent adam_steps values")
    losses = pd.concat(loss_frames, ignore_index=True)
    fem_metrics = pd.concat(fem_frames, ignore_index=True)
    gradients = (
        pd.concat(gradient_frames, ignore_index=True)
        if gradient_frames
        else pd.DataFrame()
    )
    adweights_history = (
        pd.concat(adweights_frames, ignore_index=True)
        if adweights_frames
        else pd.DataFrame()
    )
    return runs, losses, fem_metrics, gradients, adweights_history


def aggregate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    runs = runs.copy()
    if "adweights_seconds" not in runs:
        runs["adweights_seconds"] = 0.0
    metrics = [
        "optimizer_seconds",
        "rad_resampling_seconds",
        "adweights_seconds",
        "training_seconds",
        "final_l2_relative",
        "final_h1_relative",
        "final_l2_absolute",
        "final_h1_absolute",
    ]
    missing = [column for column in ("variant", "seed", *metrics) if column not in runs]
    if missing:
        raise ValueError(f"Run table is missing columns: {missing}")
    grouped = runs.groupby("variant", sort=False)[metrics].agg(
        ["mean", "std", "min", "max"]
    )
    grouped.columns = [f"{metric}_{statistic}" for metric, statistic in grouped.columns]
    grouped = grouped.reset_index()
    counts = runs.groupby("variant")["seed"].nunique().rename("seed_count")
    grouped = grouped.merge(counts, on="variant", validate="one_to_one")
    std_columns = [column for column in grouped if column.endswith("_std")]
    grouped[std_columns] = grouped[std_columns].fillna(0.0)
    return grouped


def _mean_std(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(["variant", "global_step"], sort=True)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped


def _positive_plot_floor(frame: pd.DataFrame, metric: str) -> float:
    """Return a data-scale floor for mean +/- std bands on logarithmic axes.

    An arithmetic ``mean - std`` can legitimately be non-positive even though
    every loss or relative error is non-negative.  Clipping such a bound to the
    smallest representable float makes Matplotlib expand the logarithmic axis
    towards 1e-308 and hides all useful curves.  Use half of the smallest
    strictly positive observation instead; zeros themselves remain absent from
    a logarithmic plot.
    """
    values = frame[metric].to_numpy(dtype=float)
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        raise ValueError(f"{metric} has no finite, strictly positive value to plot")
    return max(0.5 * float(positive.min()), np.finfo(float).tiny)


def _plot_positive_history(
    axis: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    ylabel: str,
) -> None:
    plot_floor = _positive_plot_floor(frame, metric)
    for variant in VARIANT_ORDER:
        subset = frame.loc[frame["variant"] == variant]
        if subset.empty:
            continue
        color = VARIANT_COLORS[variant]
        for _, individual in subset.groupby("seed"):
            individual = individual.sort_values("global_step")
            axis.semilogy(
                individual["global_step"],
                individual[metric],
                color=color,
                alpha=0.16,
                linewidth=0.8,
            )
        aggregate = _mean_std(subset, metric)
        x = aggregate["global_step"].to_numpy(dtype=float)
        mean = aggregate["mean"].to_numpy(dtype=float)
        std = aggregate["std"].to_numpy(dtype=float)
        # The requested statistics remain the arithmetic mean +/- sample
        # standard deviation.  Only the non-representable lower part of the
        # band is clipped to a floor derived from the plotted observations.
        lower = np.maximum(mean - std, plot_floor)
        axis.semilogy(
            x,
            mean,
            color=color,
            linewidth=2.0,
            label=_label(variant),
        )
        axis.fill_between(x, lower, mean + std, color=color, alpha=0.2)
    axis.set_xlabel("Steps")
    axis.set_ylabel(ylabel)
    axis.grid(True, which="both", alpha=0.25)


def _all_finite_values_are_positive(frame: pd.DataFrame, metrics: Sequence[str]) -> bool:
    values = frame.loc[:, list(metrics)].to_numpy(dtype=float).ravel()
    finite = values[np.isfinite(values)]
    return finite.size > 0 and bool(np.all(finite > 0.0))


def _plot_gradient_norm_history(axis: plt.Axes, gradients: pd.DataFrame) -> None:
    metrics = (
        ("pde_gradient_l2_norm", "PDE", "-"),
        ("neumann_gradient_l2_norm", "Neumann", "--"),
        ("dtn_gradient_l2_norm", "DtN", ":"),
    )
    use_log = _all_finite_values_are_positive(
        gradients, [metric for metric, _, _ in metrics]
    )
    plot_floor = (
        min(_positive_plot_floor(gradients, metric) for metric, _, _ in metrics)
        if use_log
        else None
    )
    for metric, metric_label, linestyle in metrics:
        for variant in VARIANT_ORDER:
            subset = gradients.loc[gradients["variant"] == variant]
            if subset.empty:
                continue
            color = VARIANT_COLORS[variant]
            for _, individual in subset.groupby("seed"):
                individual = individual.sort_values("global_step")
                plotter = axis.semilogy if use_log else axis.plot
                plotter(
                    individual["global_step"],
                    individual[metric],
                    color=color,
                    alpha=0.13,
                    linewidth=0.8,
                    linestyle=linestyle,
                )
            aggregate = _mean_std(subset, metric)
            x = aggregate["global_step"].to_numpy(dtype=float)
            mean = aggregate["mean"].to_numpy(dtype=float)
            std = aggregate["std"].to_numpy(dtype=float)
            if use_log:
                assert plot_floor is not None
                lower = np.maximum(mean - std, plot_floor)
                axis.semilogy(
                    x,
                    mean,
                    color=color,
                    linewidth=2.0,
                    linestyle=linestyle,
                    label=f"{_label(variant)} {metric_label}",
                )
                axis.fill_between(x, lower, mean + std, color=color, alpha=0.12)
            else:
                axis.plot(
                    x,
                    mean,
                    color=color,
                    linewidth=2.0,
                    linestyle=linestyle,
                    label=f"{_label(variant)} {metric_label}",
                )
                axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)
    axis.set_xlabel("Global optimizer iteration")
    axis.set_ylabel("Gradient L2 norm")
    axis.grid(True, which="both", alpha=0.25)


def _plot_linear_history(
    axis: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    ylabel: str,
) -> None:
    for variant in VARIANT_ORDER:
        subset = frame.loc[frame["variant"] == variant]
        if subset.empty:
            continue
        color = VARIANT_COLORS[variant]
        for _, individual in subset.groupby("seed"):
            individual = individual.sort_values("global_step")
            axis.plot(
                individual["global_step"],
                individual[metric],
                color=color,
                alpha=0.16,
                linewidth=0.8,
            )
        aggregate = _mean_std(subset, metric)
        x = aggregate["global_step"].to_numpy(dtype=float)
        mean = aggregate["mean"].to_numpy(dtype=float)
        std = aggregate["std"].to_numpy(dtype=float)
        axis.plot(
            x,
            mean,
            color=color,
            linewidth=2.0,
            label=_label(variant),
        )
        axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    axis.set_xlabel("Global optimizer iteration")
    axis.set_ylabel(ylabel)
    axis.grid(True, which="both", alpha=0.25)


def create_loss_figure(losses: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 4, figsize=(19, 4.7), sharex=True)
    panels = (
        ("unweighted_total", "Unweighted PDE + BC"),
        ("pde_loss", "Raw PDE loss"),
        ("neumann_loss", "Raw Neumann loss"),
        ("dtn_loss", "Raw DtN loss"),
    )
    for axis, (metric, label) in zip(axes, panels):
        _plot_positive_history(axis, losses, metric, label)
        axis.set_title(label)
    axes[0].legend(loc="best")
    figure.tight_layout()
    return figure


def create_error_figure(fem_metrics: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharex=True)
    for axis, metric, label in (
        (axes[0], "l2_relative", "Relative L2 error"),
        (axes[1], "h1_relative", "Relative H1 error"),
    ):
        _plot_positive_history(axis, fem_metrics, metric, label)
        axis.set_title(label)
    axes[0].legend(loc="best")
    figure.tight_layout()
    return figure


def create_gradient_figure(gradients: pd.DataFrame) -> plt.Figure:
    required = {
        "variant",
        "seed",
        "global_step",
        "pde_gradient_l2_norm",
        "neumann_gradient_l2_norm",
        "dtn_gradient_l2_norm",
        "bc_gradient_l2_norm",
        "pde_bc_gradient_cosine",
    }
    missing = sorted(required - set(gradients.columns))
    if missing:
        raise ValueError(f"Gradient table is missing columns: {missing}")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharex=True)
    _plot_gradient_norm_history(axes[0], gradients)
    axes[0].set_title("Raw PDE and BC gradient norms")
    _plot_linear_history(
        axes[1],
        gradients,
        "pde_bc_gradient_cosine",
        "PDE/BC gradient cosine",
    )
    axes[1].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_title("PDE/BC gradient cosine")
    axes[0].legend(loc="best", fontsize="small")
    figure.tight_layout()
    return figure


def create_adweights_figure(adweights: pd.DataFrame) -> plt.Figure:
    required = {
        "variant",
        "seed",
        "global_step",
        "component",
        "inverse_weight",
        "effective_weight",
    }
    missing = sorted(required - set(adweights.columns))
    if missing:
        raise ValueError(f"Adweights table is missing columns: {missing}")
    components = (
        ("pde", "PDE", "-"),
        ("neumann", "Neumann", "--"),
        ("dtn", "DtN", ":"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.9), sharex=True)
    for axis, metric, title in (
        (axes[0], "inverse_weight", "Raw inverse weights"),
        (axes[1], "effective_weight", "Custom-scaled effective weights"),
    ):
        use_log = _all_finite_values_are_positive(adweights, [metric])
        for component, component_label, linestyle in components:
            component_rows = adweights.loc[adweights["component"] == component]
            for variant in VARIANT_ORDER:
                subset = component_rows.loc[component_rows["variant"] == variant]
                if subset.empty:
                    continue
                color = VARIANT_COLORS[variant]
                for _, individual in subset.groupby("seed"):
                    individual = individual.sort_values("global_step")
                    plotter = axis.semilogy if use_log else axis.plot
                    plotter(
                        individual["global_step"],
                        individual[metric],
                        color=color,
                        alpha=0.14,
                        linewidth=0.8,
                        linestyle=linestyle,
                    )
                aggregate = _mean_std(subset, metric)
                plotter = axis.semilogy if use_log else axis.plot
                plotter(
                    aggregate["global_step"],
                    aggregate["mean"],
                    color=color,
                    linewidth=2.0,
                    linestyle=linestyle,
                    label=f"{_label(variant)} {component_label}",
                )
        axis.set_xlabel("Global optimizer iteration")
        axis.set_ylabel("Loss weight")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend(loc="best", fontsize="x-small")
    figure.tight_layout()
    return figure


def create_error_time_figure(fem_metrics: pd.DataFrame) -> plt.Figure:
    """Plot reference accuracy against measured optimizer-plus-RAD time."""
    required = {"variant", "seed", "training_seconds"}
    missing = sorted(required - set(fem_metrics.columns))
    if missing:
        raise ValueError(f"FEM metric table is missing columns: {missing}")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharex=True)
    for axis, metric, label in (
        (axes[0], "l2_relative", "Relative L2 error"),
        (axes[1], "h1_relative", "Relative H1 error"),
    ):
        for variant in VARIANT_ORDER:
            subset = fem_metrics.loc[fem_metrics["variant"] == variant]
            if subset.empty:
                continue
            color = VARIANT_COLORS[variant]
            for seed_index, (_, individual) in enumerate(subset.groupby("seed")):
                individual = individual.sort_values("training_seconds")
                axis.semilogy(
                    individual["training_seconds"],
                    individual[metric],
                    color=color,
                    alpha=0.8,
                    linewidth=1.5,
                    label=(_label(variant) if seed_index == 0 else None),
                )
        axis.set_xlabel("Measured training time [s]")
        axis.set_ylabel(label)
        axis.set_title(label)
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend(loc="best")
    figure.tight_layout()
    return figure


def create_timing_figure(runs: pd.DataFrame) -> plt.Figure:
    if "adweights_seconds" not in runs:
        runs = runs.assign(adweights_seconds=0.0)
    panels = [("optimizer_seconds", "Adam optimizer")]
    has_rad_time = (
        "rad_resampling_seconds" in runs
        and bool(np.any(runs["rad_resampling_seconds"].to_numpy(dtype=float) > 0.0))
    )
    if has_rad_time:
        panels.append(("rad_resampling_seconds", "RAD resampling"))
    has_adweights_time = bool(
        np.any(runs["adweights_seconds"].to_numpy(dtype=float) > 0.0)
    )
    if has_adweights_time:
        panels.append(("adweights_seconds", "Adaptive-weight updates"))
    if has_rad_time or has_adweights_time:
        panels.append(("training_seconds", "Total training"))
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(max(5.5, 4.8 * len(panels)), 4.7),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    variants = [
        variant for variant in VARIANT_ORDER if not runs.loc[runs["variant"] == variant].empty
    ]
    random = np.random.default_rng(0)
    for axis, (metric, title) in zip(axes, panels):
        for index, variant in enumerate(variants):
            values = runs.loc[runs["variant"] == variant, metric].to_numpy(dtype=float)
            if values.size == 0:
                continue
            jitter = random.uniform(-0.07, 0.07, size=values.size)
            color = VARIANT_COLORS[variant]
            axis.scatter(
                np.full(values.size, index) + jitter,
                values,
                color=color,
                alpha=0.65,
                s=28,
            )
            axis.errorbar(
                index,
                values.mean(),
                yerr=values.std(ddof=1) if values.size > 1 else 0.0,
                fmt="o",
                color="black",
                capsize=4,
                linewidth=1.4,
            )
        axis.set_xticks(range(len(variants)))
        axis.set_xticklabels(
            [_label(value) for value in variants], rotation=20, ha="right"
        )
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Measured training time [s]")
    figure.tight_layout()
    return figure


def _save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")


def write_campaign_outputs(
    campaign_root: str | Path,
    output_directory: str | Path,
    *,
    show: bool = False,
) -> Path:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    runs, losses, fem_metrics, gradients, adweights = load_campaign_frames(
        campaign_root
    )
    aggregate = aggregate_runs(runs)
    runs.sort_values(["variant", "seed"]).to_csv(output / "runs.csv", index=False)
    aggregate.to_csv(output / "aggregate.csv", index=False)
    losses.sort_values(["variant", "seed", "global_step"]).to_csv(
        output / "loss_history_all.csv", index=False
    )
    fem_metrics.sort_values(["variant", "seed", "global_step"]).to_csv(
        output / "fem_metrics_all.csv", index=False
    )
    if not gradients.empty:
        gradients.sort_values(["variant", "seed", "global_step"]).to_csv(
            output / "gradient_history_all.csv", index=False
        )
    if not adweights.empty:
        adweights.sort_values(
            ["variant", "seed", "global_step", "component"]
        ).to_csv(output / "adweights_history_all.csv", index=False)

    figures = {
        "losses": create_loss_figure(losses),
        "fem_errors": create_error_figure(fem_metrics),
        "fem_errors_vs_time": create_error_time_figure(fem_metrics),
        "optimizer_times": create_timing_figure(runs),
    }
    if not gradients.empty:
        figures["gradient_stats"] = create_gradient_figure(gradients)
    if not adweights.empty:
        figures["adweights_weights"] = create_adweights_figure(adweights)
    for name, figure in figures.items():
        _save_figure(figure, output / name)
    if show:
        plt.show()
    for figure in figures.values():
        plt.close(figure)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--show", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = write_campaign_outputs(
        args.campaign_root, args.output_dir, show=args.show
    )
    print(f"Wrote campaign analysis to: {output}")


if __name__ == "__main__":
    main()
