#!/usr/bin/env python3
"""Directly fit the m-network to a prescribed sound-speed map.

This is a small diagnostic for the inverse PINN: it removes the wavefield,
PDE, boundary, and FEM data losses, and tests whether the m-network
architecture can learn the target c(x, y) map from uniformly sampled points.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", False)


# ==============================================================================
# Geometry and m-network configuration copied from pinn_waveguide_multi_modes.py
# ==============================================================================

H = 0.6
L = 1.0

c0 = 340.0
contrast_max = 0.4
cmin = c0 * (1.0 - contrast_max)
cmax = c0 * (1.0 + 0.01)
m0 = 1.0 / c0**2
m_min = 1.0 / cmax**2
m_max = 1.0 / cmin**2

n_input = 2
n_layers_m = [n_input, 128, 64, 1]


# ==============================================================================
# USER-EDITABLE LOSS WEIGHTS AND TV REGULARIZATION
# ==============================================================================

USE_TV_LOSS = False

LOSS_WEIGHTS = {
    "l2_c": 1.0,
    "tv_m": 1e1,
}

TV_EPSILON_SQUARED = 1e-12


# ==============================================================================
# USER-EDITABLE GROUND TRUTH
# ==============================================================================

DEFAULT_DEFECT_NAME = "circlebottomright"
DEFAULT_CONTRAST_RATIO = 0.8


def normalized_to_physical(x, y):
    """Convert network coordinates x,y in [-1, 1] to physical coordinates."""
    x_phys = x * L
    y_phys = (y + 1.0) * H / 2.0
    return x_phys, y_phys


def ground_truth_c_map(
    x,
    y,
    defect_name: str = DEFAULT_DEFECT_NAME,
    contrast_ratio: float = DEFAULT_CONTRAST_RATIO,
):
    """Return the target sound-speed map at normalized network coordinates.

    Edit this function to test another geometry. The inputs x and y are the
    normalized coordinates used by the m-network in pinn_waveguide_multi_modes.py.
    """
    x_phys, y_phys = normalized_to_physical(x, y)
    c_background = jnp.full_like(x_phys, c0, dtype=jnp.float32)
    c_defect = jnp.asarray(c0 * contrast_ratio, dtype=jnp.float32)

    if defect_name == "circlebottomright":
        mask = (x_phys - 0.2) ** 2 + (y_phys - 0.2) ** 2 <= 0.1**2
    elif defect_name == "barhalf":
        mask = (
            (x_phys >= -0.2)
            & (x_phys <= 0.2)
            & (y_phys >= 0.0)
            & (y_phys <= 0.3)
        )
    elif defect_name == "homogeneous":
        mask = jnp.zeros_like(x_phys, dtype=bool)
    else:
        raise ValueError(
            f"Unknown defect {defect_name!r}. Edit ground_truth_c_map to add it."
        )

    return jnp.where(mask, c_defect, c_background)


# ==============================================================================
# m-network
# ==============================================================================


def init_layers(key, n_layers):
    layers = []
    keys = jax.random.split(key, len(n_layers) - 1)
    for i, k in enumerate(keys):
        width_in = n_layers[i]
        width_out = n_layers[i + 1]
        scale = jnp.sqrt(2.0 / (width_in + width_out))
        W = jax.random.normal(k, (width_in, width_out)) * scale
        b = jnp.zeros(width_out)
        layers.append({"W": W, "b": b})
    return layers


def init_m_network(key):
    """Initialize layers_m exactly as in the multi-mode inverse script."""
    layers_m = init_layers(key, n_layers_m)
    initial_output_bias = -jnp.log((m_max - m_min) / (m0 - m_min) - 1.0)
    layers_m[-1]["b"] = jnp.full_like(layers_m[-1]["b"], initial_output_bias)
    layers_m[-1]["W"] = layers_m[-1]["W"] / 10.0
    return layers_m


def forward_params(layers, X):
    """Network output: bounded slowness squared m=1/c^2."""
    Z = X
    for layer in layers[:-1]:
        Z = jax.nn.tanh(Z @ layer["W"] + layer["b"])
    raw = Z @ layers[-1]["W"] + layers[-1]["b"]
    m_pred = m_min + (m_max - m_min) * jax.nn.sigmoid(raw)
    return jnp.squeeze(m_pred, axis=-1)


def predict_c(layers_m, x, y):
    X = jnp.stack([x, y], axis=-1)
    return 1.0 / jnp.sqrt(forward_params(layers_m, X))


def predict_m_scalar(layers_m, x, y):
    return forward_params(layers_m, jnp.stack([x, y]))


# ==============================================================================
# Sampling, loss, and optimization
# ==============================================================================


def sample_uniform_points(key, n_points):
    """Uniform points matching the PDE sampling in pinn_waveguide_multi_modes.py."""
    key_x, key_y = jax.random.split(key)
    x = jax.random.uniform(key_x, (n_points,), minval=-1.0, maxval=1.0)
    y = jax.random.uniform(key_y, (n_points,), minval=-1.0, maxval=1.0)
    return x, y


def l2_c_loss(layers_m, x, y, defect_name, contrast_ratio):
    c_pred = predict_c(layers_m, x, y)
    c_true = ground_truth_c_map(x, y, defect_name, contrast_ratio)
    return jnp.mean((c_pred - c_true) ** 2)


def scaled_m_gradient(layers_m, x, y):
    """Return c0**2-scaled physical gradients of m=1/c**2."""

    def m_at_point(x_point, y_point):
        return predict_m_scalar(layers_m, x_point, y_point)

    dm_dx_norm, dm_dy_norm = jax.grad(m_at_point, argnums=(0, 1))(x, y)
    dm_dx_phys = dm_dx_norm / L
    dm_dy_phys = dm_dy_norm * (2.0 / H)
    return c0**2 * dm_dx_phys, c0**2 * dm_dy_phys


def tv_m_loss(layers_m, x, y, epsilon_squared):
    def point_tv(x_point, y_point):
        dm_dx, dm_dy = scaled_m_gradient(layers_m, x_point, y_point)
        return jnp.sqrt(dm_dx**2 + dm_dy**2 + epsilon_squared)

    return jnp.mean(jax.vmap(point_tv)(x, y))


def total_m_loss(
    layers_m,
    x,
    y,
    defect_name,
    contrast_ratio,
    loss_weights,
    use_tv_loss,
    tv_epsilon_squared,
):
    l2_loss = l2_c_loss(layers_m, x, y, defect_name, contrast_ratio)
    if use_tv_loss:
        tv_loss = tv_m_loss(layers_m, x, y, tv_epsilon_squared)
    else:
        tv_loss = jnp.zeros((), dtype=l2_loss.dtype)
    total_loss = loss_weights["l2_c"] * l2_loss + loss_weights["tv_m"] * tv_loss
    return total_loss, (l2_loss, tv_loss)


def make_train_step(
    optimizer,
    batch_size,
    defect_name,
    contrast_ratio,
    loss_weights,
    use_tv_loss,
    tv_epsilon_squared,
):
    @jax.jit
    def train_step(layers_m, opt_state, key):
        x, y = sample_uniform_points(key, batch_size)

        def batch_loss(candidate_layers):
            return total_m_loss(
                candidate_layers,
                x,
                y,
                defect_name,
                contrast_ratio,
                loss_weights,
                use_tv_loss,
                tv_epsilon_squared,
            )

        (loss, aux), grads = jax.value_and_grad(batch_loss, has_aux=True)(layers_m)
        updates, opt_state = optimizer.update(grads, opt_state, layers_m)
        layers_m = optax.apply_updates(layers_m, updates)
        return layers_m, opt_state, loss, aux

    return train_step


def make_eval_loss(
    defect_name,
    contrast_ratio,
    loss_weights,
    use_tv_loss,
    tv_epsilon_squared,
):
    @jax.jit
    def eval_loss(layers_m, x, y):
        return total_m_loss(
            layers_m,
            x,
            y,
            defect_name,
            contrast_ratio,
            loss_weights,
            use_tv_loss,
            tv_epsilon_squared,
        )

    return eval_loss


def build_eval_grid(nx, ny):
    x_axis = jnp.linspace(-1.0, 1.0, nx)
    y_axis = jnp.linspace(-1.0, 1.0, ny)
    x_grid, y_grid = jnp.meshgrid(x_axis, y_axis, indexing="xy")
    return x_grid, y_grid


def evaluate_maps(layers_m, nx, ny, defect_name, contrast_ratio):
    x_grid, y_grid = build_eval_grid(nx, ny)
    c_pred = predict_c(layers_m, x_grid, y_grid)
    c_true = ground_truth_c_map(x_grid, y_grid, defect_name, contrast_ratio)
    x_phys, y_phys = normalized_to_physical(x_grid, y_grid)
    return (
        np.asarray(x_phys),
        np.asarray(y_phys),
        np.asarray(c_true),
        np.asarray(c_pred),
    )


def compute_metrics(c_true, c_pred):
    error = c_pred - c_true
    abs_error = np.abs(error)
    anomaly = np.abs(c_true - c0)
    anomaly_mask = anomaly > 1e-9 * c0
    background_mask = ~anomaly_mask
    anomaly_l1 = float(np.sum(anomaly))

    metrics = {
        "l2": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(abs_error)),
        "max_abs": float(np.max(abs_error)),
    }
    if anomaly_mask.any():
        metrics["defect_mae"] = float(np.mean(abs_error[anomaly_mask]))
        metrics["anomaly_relative_l1"] = float(np.sum(abs_error) / anomaly_l1)
    else:
        metrics["defect_mae"] = None
        metrics["anomaly_relative_l1"] = None
    if background_mask.any():
        metrics["background_mae"] = float(np.mean(abs_error[background_mask]))
    else:
        metrics["background_mae"] = None
    return metrics


# ==============================================================================
# Plotting and CLI
# ==============================================================================


def handle_figures(
    output_dir: Path,
    defect_name: str,
    x_grid,
    y_grid,
    c_true,
    c_pred,
    loss_steps,
    train_losses,
    val_losses,
    save=False,
    show=False,
):
    if not save and not show:
        return None, None

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
    speed_min = float(min(np.min(c_true), np.min(c_pred)))
    speed_max = float(max(np.max(c_true), np.max(c_pred)))
    abs_error = np.abs(c_pred - c_true)
    figures = []
    map_path = None
    loss_path = None

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.6), sharex=True, sharey=True)
    panels = (
        (axes[0], c_true, "Ground truth", "viridis", speed_min, speed_max),
        (axes[1], c_pred, "m-network prediction", "viridis", speed_min, speed_max),
        (axes[2], abs_error, "Absolute error", "magma", 0.0, float(np.max(abs_error))),
    )
    for axis, values, title, cmap, vmin, vmax in panels:
        image = axis.pcolormesh(
            x_grid,
            y_grid,
            values,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax if vmax > vmin else vmin + 1.0,
            rasterized=True,
        )
        axis.set_title(title)
        axis.set_xlabel("x [m]")
        axis.set_aspect("equal", adjustable="box")
        fig.colorbar(image, ax=axis, label="c [m/s]")
    axes[0].set_ylabel("y [m]")
    fig.tight_layout()
    if save:
        map_path = output_dir / f"m_ground_truth_fit_{defect_name}.png"
        fig.savefig(map_path, dpi=200)
    figures.append(fig)

    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.semilogy(loss_steps, train_losses, label="sampled train weighted total")
    axis.semilogy(loss_steps, val_losses, label="grid validation weighted total")
    axis.set_xlabel("Adam step")
    axis.set_ylabel("weighted loss")
    axis.grid(True)
    axis.legend()
    fig.tight_layout()
    if save:
        loss_path = output_dir / f"m_ground_truth_fit_{defect_name}_loss.png"
        fig.savefig(loss_path, dpi=200)
    figures.append(fig)
    if show:
        plt.show()
    for figure in figures:
        plt.close(figure)
    return map_path, loss_path


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Fit only the m-network to a known c(x,y) map."
    )
    parser.add_argument(
        "--defect",
        default=DEFAULT_DEFECT_NAME,
        choices=("circlebottomright", "barhalf", "homogeneous"),
        help="Ground-truth geometry to fit.",
    )
    parser.add_argument(
        "--contrast-ratio",
        type=float,
        default=DEFAULT_CONTRAST_RATIO,
        help="Defect speed divided by c0.",
    )
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--l2-weight", type=float, default=LOSS_WEIGHTS["l2_c"])
    parser.add_argument("--tv-weight", type=float, default=LOSS_WEIGHTS["tv_m"])
    tv_group = parser.add_mutually_exclusive_group()
    tv_group.add_argument(
        "--use-tv-loss",
        dest="use_tv_loss",
        action="store_true",
        help="Enable TV regularization on c0**2 * grad(m).",
    )
    tv_group.add_argument(
        "--no-tv-loss",
        dest="use_tv_loss",
        action="store_false",
        help="Disable TV regularization.",
    )
    parser.set_defaults(use_tv_loss=USE_TV_LOSS)
    parser.add_argument(
        "--tv-epsilon-squared",
        type=float,
        default=TV_EPSILON_SQUARED,
        help="Epsilon squared inside sqrt(|c0**2 grad(m)|^2 + eps^2).",
    )
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nx", type=int, default=201)
    parser.add_argument("--ny", type=int, default=121)
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Save map and loss figures.",
    )
    parser.add_argument("--out-dir", type=Path, default=script_dir / "fig")
    parser.add_argument("--show", action="store_true", help="Show figures at the end.")
    return parser.parse_args()


def validate_args(args):
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.l2_weight < 0.0:
        raise ValueError("--l2-weight must be non-negative")
    if args.tv_weight < 0.0:
        raise ValueError("--tv-weight must be non-negative")
    if args.tv_epsilon_squared < 0.0:
        raise ValueError("--tv-epsilon-squared must be non-negative")
    if args.eval_interval < 1:
        raise ValueError("--eval-interval must be at least 1")
    if args.nx < 2 or args.ny < 2:
        raise ValueError("--nx and --ny must be at least 2")
    if args.contrast_ratio <= 0.0:
        raise ValueError("--contrast-ratio must be positive")


def main():
    args = parse_args()
    validate_args(args)

    print("JAX devices:", jax.devices())
    print("m-network layers:", n_layers_m)
    print(
        "Target:",
        f"defect={args.defect}",
        f"contrast_ratio={args.contrast_ratio}",
    )
    loss_weights = {
        "l2_c": args.l2_weight,
        "tv_m": args.tv_weight,
    }
    print(
        "Loss:",
        f"use_tv_loss={args.use_tv_loss}",
        f"l2_weight={loss_weights['l2_c']}",
        f"tv_weight={loss_weights['tv_m']}",
        f"tv_epsilon_squared={args.tv_epsilon_squared}",
    )

    key = jax.random.key(args.seed)
    key, init_key = jax.random.split(key)
    layers_m = init_m_network(init_key)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(args.learning_rate),
    )
    opt_state = optimizer.init(layers_m)
    train_step = make_train_step(
        optimizer,
        args.batch_size,
        args.defect,
        args.contrast_ratio,
        loss_weights,
        args.use_tv_loss,
        args.tv_epsilon_squared,
    )
    eval_loss = make_eval_loss(
        args.defect,
        args.contrast_ratio,
        loss_weights,
        args.use_tv_loss,
        args.tv_epsilon_squared,
    )

    x_val, y_val = build_eval_grid(args.nx, args.ny)
    x_val_flat = x_val.reshape(-1)
    y_val_flat = y_val.reshape(-1)

    loss_steps = []
    train_losses = []
    val_losses = []

    initial_train_loss, initial_aux = eval_loss(layers_m, x_val_flat, y_val_flat)
    initial_l2_loss, initial_tv_loss = initial_aux
    loss_steps.append(0)
    train_losses.append(float(initial_train_loss))
    val_losses.append(float(initial_train_loss))
    print(
        f"step {0:6d} | train total {float(initial_train_loss):.6e} "
        f"(L2 {float(initial_l2_loss):.6e}, TV {float(initial_tv_loss):.6e}) "
        f"| val total {float(initial_train_loss):.6e} "
        f"(L2 {float(initial_l2_loss):.6e}, TV {float(initial_tv_loss):.6e})"
    )

    for step in range(1, args.steps + 1):
        key, step_key = jax.random.split(key)
        layers_m, opt_state, train_loss, train_aux = train_step(
            layers_m,
            opt_state,
            step_key,
        )

        if step % args.eval_interval == 0 or step == args.steps:
            train_l2_loss, train_tv_loss = train_aux
            val_loss, val_aux = eval_loss(layers_m, x_val_flat, y_val_flat)
            val_l2_loss, val_tv_loss = val_aux
            loss_steps.append(step)
            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))
            print(
                f"step {step:6d} | train total {float(train_loss):.6e} "
                f"(L2 {float(train_l2_loss):.6e}, TV {float(train_tv_loss):.6e}) "
                f"| val total {float(val_loss):.6e} "
                f"(L2 {float(val_l2_loss):.6e}, TV {float(val_tv_loss):.6e})"
            )

    x_phys, y_phys, c_true, c_pred = evaluate_maps(
        layers_m,
        args.nx,
        args.ny,
        args.defect,
        args.contrast_ratio,
    )
    metrics = compute_metrics(c_true, c_pred)

    print("\nFinal grid metrics:")
    print(f"  L2(c): {metrics['l2']:.8e} (m/s)^2")
    print(f"  RMSE: {metrics['rmse']:.8e} m/s")
    print(f"  MAE: {metrics['mae']:.8e} m/s")
    print(f"  Max abs error: {metrics['max_abs']:.8e} m/s")
    if metrics["defect_mae"] is not None:
        print(f"  Defect MAE: {metrics['defect_mae']:.8e} m/s")
        print(
            "  Anomaly-relative L1: "
            f"{metrics['anomaly_relative_l1']:.8e}"
        )
    if metrics["background_mae"] is not None:
        print(f"  Background MAE: {metrics['background_mae']:.8e} m/s")

    map_path, loss_path = handle_figures(
        args.out_dir,
        args.defect,
        x_phys,
        y_phys,
        c_true,
        c_pred,
        loss_steps,
        train_losses,
        val_losses,
        save=args.save_figures,
        show=args.show,
    )
    if args.save_figures:
        print(f"\nSaved map figure: {map_path}")
        print(f"Saved loss figure: {loss_path}")
    elif args.show:
        print("\nDisplayed figures without saving them.")


if __name__ == "__main__":
    main()
