from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax


jax.config.update("jax_enable_x64", False)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.data_loader import (
    FEMFieldData,
    SymmetricCOOMatrix,
    load_symmetric_coo_matrix,
)
from tools.experiment_manifest import (
    canonical_manifest,
    configuration_id,
    write_manifest_exclusive,
)
from tests_forward_PINN.model_variants import (
    BASE_VARIANTS,
    VARIANTS,
    _validate_variant,
    initialize_model,
    is_rad_variant,
    is_scattered_variant,
    model_value,
    uses_adweights,
    uses_fourier_features,
    value_gradient_laplacian,
    variant_parameter_count,
)


RAD_REFERENCE = "https://arxiv.org/abs/2207.10289"
RAD_SAMPLING_PROTOCOL = "uniform_plus_candidate_bootstrap_with_replacement_v2"


@dataclass(frozen=True)
class Circle:
    center: tuple[float, float]
    radius: float
    speed_ratio: float


@dataclass(frozen=True)
class AdWeightsConfig:
    epsilon: float
    alpha: float
    initial_lambdas: tuple[float, float, float]
    update_interval_adam: int
    custom_weights: tuple[float, float, float]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AdWeightsConfig":
        expected = {
            "epsilon",
            "alpha",
            "initial_lambdas",
            "update_interval_adam",
            "custom_weights",
        }
        missing = sorted(expected - set(raw))
        unknown = sorted(set(raw) - expected)
        if missing or unknown:
            raise ValueError(
                f"Invalid adweights configuration keys; missing={missing}, "
                f"unknown={unknown}"
            )

        def triplet(name: str) -> tuple[float, float, float]:
            values = tuple(float(value) for value in raw[name])
            if len(values) != 3:
                raise ValueError(f"adweights.{name} must contain three values")
            return values

        config = cls(
            epsilon=float(raw["epsilon"]),
            alpha=float(raw["alpha"]),
            initial_lambdas=triplet("initial_lambdas"),
            update_interval_adam=int(raw["update_interval_adam"]),
            custom_weights=triplet("custom_weights"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        scalars = (self.epsilon, self.alpha, *self.initial_lambdas, *self.custom_weights)
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("All adweights values must be finite")
        if self.epsilon <= 0.0:
            raise ValueError("adweights.epsilon must be positive")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("adweights.alpha must lie in [0, 1]")
        if min(self.initial_lambdas) < 0.0:
            raise ValueError("adweights.initial_lambdas must be non-negative")
        if min(self.custom_weights) < 0.0 or not any(
            value > 0.0 for value in self.custom_weights
        ):
            raise ValueError(
                "adweights.custom_weights must be non-negative and not all zero"
            )
        if self.update_interval_adam <= 0:
            raise ValueError("adweights.update_interval_adam must be positive")


@dataclass(frozen=True)
class ForwardConfig:
    frequency: float
    mode: int
    incidence: int
    height: float
    half_length: float
    c0: float
    circles: tuple[Circle, ...]
    hidden_layers: tuple[int, ...]
    fourier_features: int
    classical_learning_rate: float
    fourier_field_learning_rate: float
    fourier_sigma_learning_rate: float
    fourier_sigma_decay_fraction: float
    fourier_sigma_cosine_alpha: float
    classical_loss_weights: tuple[float, float, float]
    fourier_loss_weights: tuple[float, float, float]
    collocation_adam: tuple[int, int, int]
    collocation_monitor: tuple[int, int, int]
    rad_k: float
    rad_c: float
    rad_points: int
    rad_resample_interval_adam: int
    rad_candidate_points: int
    rad_candidate_batch_size: int
    loss_eval_interval_adam: int
    fem_eval_interval_adam: int
    gradient_eval_interval_adam: int
    fem_prediction_batch_size: int
    reference_kind: str
    analytic_triangulation: tuple[int, int] | None
    fem_field: Path | None
    mass_matrix: Path | None
    stiffness_matrix: Path | None
    output_root: Path
    adweights: AdWeightsConfig | None

    @classmethod
    def from_json(cls, path: str | Path) -> "ForwardConfig":
        path = Path(path).resolve()
        with path.open(encoding="utf-8") as stream:
            raw = json.load(stream)
        if not isinstance(raw, dict):
            raise ValueError("The forward configuration must be a JSON object")

        optional = {
            "reference_kind",
            "analytic_triangulation",
            "fem_field",
            "mass_matrix",
            "stiffness_matrix",
            "adweights",
        }
        required = set(cls.__dataclass_fields__) | {"circles"}
        required -= optional
        reference_kind = str(raw.get("reference_kind", "fem"))
        if reference_kind == "fem":
            required |= {"fem_field", "mass_matrix", "stiffness_matrix"}
        elif reference_kind == "analytic_mode":
            required |= {"analytic_triangulation"}
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(f"Invalid configuration keys; missing={missing}")

        circles = tuple(
            Circle(
                center=(float(item["center"][0]), float(item["center"][1])),
                radius=float(item["radius"]),
                speed_ratio=float(item["speed_ratio"]),
            )
            for item in raw["circles"]
        )

        def triple(name: str) -> tuple[int, int, int]:
            values = tuple(int(value) for value in raw[name])
            if len(values) != 3 or min(values) <= 0:
                raise ValueError(f"{name} must contain three positive integers")
            return values

        def loss_weight_triplet(name: str) -> tuple[float, float, float]:
            values = tuple(float(value) for value in raw[name])
            if len(values) != 3 or min(values) <= 0.0:
                raise ValueError(f"{name} must contain three positive values")
            return values

        def integer_pair(name: str) -> tuple[int, int]:
            values = tuple(int(value) for value in raw[name])
            if len(values) != 2 or min(values) <= 0:
                raise ValueError(f"{name} must contain two positive integers")
            return values

        def repository_path(name: str) -> Path:
            value = Path(str(raw[name])).expanduser()
            return value if value.is_absolute() else REPOSITORY_ROOT / value

        config = cls(
            frequency=float(raw["frequency"]),
            mode=int(raw["mode"]),
            incidence=int(raw["incidence"]),
            height=float(raw["height"]),
            half_length=float(raw["half_length"]),
            c0=float(raw["c0"]),
            circles=circles,
            hidden_layers=tuple(int(value) for value in raw["hidden_layers"]),
            fourier_features=int(raw["fourier_features"]),
            classical_learning_rate=float(raw["classical_learning_rate"]),
            fourier_field_learning_rate=float(raw["fourier_field_learning_rate"]),
            fourier_sigma_learning_rate=float(raw["fourier_sigma_learning_rate"]),
            fourier_sigma_decay_fraction=float(raw["fourier_sigma_decay_fraction"]),
            fourier_sigma_cosine_alpha=float(raw["fourier_sigma_cosine_alpha"]),
            classical_loss_weights=loss_weight_triplet("classical_loss_weights"),
            fourier_loss_weights=loss_weight_triplet("fourier_loss_weights"),
            collocation_adam=triple("collocation_adam"),
            collocation_monitor=triple("collocation_monitor"),
            rad_k=float(raw["rad_k"]),
            rad_c=float(raw["rad_c"]),
            rad_points=int(raw["rad_points"]),
            rad_resample_interval_adam=int(raw["rad_resample_interval_adam"]),
            rad_candidate_points=int(raw["rad_candidate_points"]),
            rad_candidate_batch_size=int(raw["rad_candidate_batch_size"]),
            loss_eval_interval_adam=int(raw["loss_eval_interval_adam"]),
            fem_eval_interval_adam=int(raw["fem_eval_interval_adam"]),
            gradient_eval_interval_adam=int(raw["gradient_eval_interval_adam"]),
            fem_prediction_batch_size=int(raw["fem_prediction_batch_size"]),
            reference_kind=reference_kind,
            analytic_triangulation=(
                integer_pair("analytic_triangulation")
                if "analytic_triangulation" in raw
                else None
            ),
            fem_field=(repository_path("fem_field") if "fem_field" in raw else None),
            mass_matrix=(
                repository_path("mass_matrix") if "mass_matrix" in raw else None
            ),
            stiffness_matrix=(
                repository_path("stiffness_matrix")
                if "stiffness_matrix" in raw
                else None
            ),
            output_root=repository_path("output_root"),
            adweights=(
                AdWeightsConfig.from_mapping(raw["adweights"])
                if isinstance(raw.get("adweights"), dict)
                else None
            ),
        )
        if "adweights" in raw and not isinstance(raw["adweights"], dict):
            raise ValueError("adweights must be a JSON object")
        config.validate()
        return config

    def validate(self) -> None:
        if self.adweights is not None:
            self.adweights.validate()
        if self.frequency <= 0.0 or self.height <= 0.0 or self.half_length <= 0.0:
            raise ValueError("frequency and domain dimensions must be positive")
        if self.mode < 0 or self.c0 <= 0.0:
            raise ValueError("mode must be non-negative and c0 positive")
        if self.incidence not in {-1, 1}:
            raise ValueError("incidence must be either -1 or 1")
        if self.reference_kind not in {"fem", "analytic_mode"}:
            raise ValueError("reference_kind must be 'fem' or 'analytic_mode'")
        if self.reference_kind == "fem":
            if any(
                value is None
                for value in (self.fem_field, self.mass_matrix, self.stiffness_matrix)
            ):
                raise ValueError("A FEM reference requires field, mass, and stiffness files")
            if self.analytic_triangulation is not None:
                raise ValueError("analytic_triangulation is only valid for analytic_mode")
        else:
            if self.circles:
                raise ValueError("analytic_mode is exact only for a homogeneous guide")
            if self.analytic_triangulation is None:
                raise ValueError("analytic_mode requires analytic_triangulation")
            if any(
                value is not None
                for value in (self.fem_field, self.mass_matrix, self.stiffness_matrix)
            ):
                raise ValueError("analytic_mode must not load FEM reference files")
        if not self.hidden_layers or min(self.hidden_layers) <= 0:
            raise ValueError("hidden_layers must contain positive widths")
        if self.fourier_features <= 0 or self.fem_prediction_batch_size <= 0:
            raise ValueError("feature and reference batch counts must be positive")
        positive_scalars = (
            self.classical_learning_rate,
            self.fourier_field_learning_rate,
            self.fourier_sigma_learning_rate,
            self.fourier_sigma_decay_fraction,
        )
        if min(positive_scalars) <= 0.0:
            raise ValueError("learning rates and decay fraction must be positive")
        if not 0.0 <= self.fourier_sigma_cosine_alpha <= 1.0:
            raise ValueError("fourier_sigma_cosine_alpha must lie in [0, 1]")
        if not 0.0 < self.fourier_sigma_decay_fraction <= 1.0:
            raise ValueError("fourier_sigma_decay_fraction must lie in (0, 1]")
        if self.rad_k < 0.0 or self.rad_c < 0.0:
            raise ValueError("RAD k and c must be non-negative")
        if self.rad_points <= 0:
            raise ValueError("rad_points must be positive")
        if self.rad_points >= self.collocation_adam[0]:
            raise ValueError(
                "rad_points must be smaller than the Adam PDE counts"
            )
        if min(
            self.rad_resample_interval_adam,
            self.rad_candidate_points,
            self.rad_candidate_batch_size,
        ) <= 0:
            raise ValueError("RAD interval and candidate counts must be positive")
        intervals = (
            self.loss_eval_interval_adam,
            self.fem_eval_interval_adam,
            self.gradient_eval_interval_adam,
        )
        if min(intervals) <= 0:
            raise ValueError("All evaluation intervals must be positive")
        if (self.fem_eval_interval_adam < self.loss_eval_interval_adam):
            raise ValueError(
                "FEM evaluation intervals must not be shorter than loss intervals"
            )
        for circle in self.circles:
            cx, cy = circle.center
            if circle.radius <= 0.0 or circle.speed_ratio <= 0.0:
                raise ValueError("Circle radii and speed ratios must be positive")
            if not (
                -self.half_length <= cx - circle.radius
                and cx + circle.radius <= self.half_length
                and 0.0 <= cy - circle.radius
                and cy + circle.radius <= self.height
            ):
                raise ValueError(f"Circle {circle} lies outside the waveguide")

    def manifest(self) -> dict[str, Any]:
        values = asdict(self)
        for name in ("fem_field", "mass_matrix", "stiffness_matrix"):
            value = getattr(self, name)
            values[name] = str(value) if value is not None else None
        values["output_root"] = str(self.output_root)
        return canonical_manifest(values)


def require_adweights_config(
    config: ForwardConfig, variant: str
) -> AdWeightsConfig:
    _validate_variant(variant)
    if not uses_adweights(variant):
        raise ValueError(f"Variant {variant!r} does not use adaptive weights")
    if config.adweights is None:
        raise ValueError(
            f"Variant {variant!r} requires an 'adweights' JSON object with "
            "epsilon, alpha, initial_lambdas, update_interval_adam, and "
            "custom_weights"
        )
    return config.adweights


@dataclass(frozen=True)
class ForwardContext:
    config: ForwardConfig
    b_base: jax.Array
    n_modes: int
    n_indices: jax.Array
    modal_scales: jax.Array
    beta: jax.Array
    y_quadrature_normalized: jax.Array
    quadrature_weights: jax.Array
    modal_basis_quadrature: jax.Array
    field_scale: float


@dataclass(frozen=True)
class FEMReference:
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    mass: SymmetricCOOMatrix
    stiffness: SymmetricCOOMatrix


@dataclass(frozen=True)
class MisfitMetrics:
    l2_absolute: float
    l2_relative: float
    h1_absolute: float
    h1_relative: float


def _build_context(config: ForwardConfig, b_base: jax.Array) -> ForwardContext:
    n_modes = int(round(2.0 * config.height * config.frequency / config.c0)) + 5
    nodes, weights = np.polynomial.legendre.leggauss(3 * n_modes)
    y_physical = (nodes + 1.0) * config.height / 2.0
    weights_physical = weights * config.height / 2.0
    indices = np.arange(n_modes, dtype=np.float32)
    modal_scales = np.sqrt(2.0 / config.height) * np.ones(n_modes, dtype=np.float32)
    modal_scales[0] = np.sqrt(1.0 / config.height)
    basis = modal_scales[:, None] * np.cos(
        indices[:, None] * np.pi * y_physical[None, :] / config.height
    )
    k0 = 2.0 * np.pi * config.frequency / config.c0
    beta = np.sqrt(k0**2 - (indices * np.pi / config.height) ** 2 + 0j)
    if config.mode >= n_modes:
        raise ValueError(f"Mode {config.mode} exceeds the retained modal basis")
    return ForwardContext(
        config=config,
        b_base=jnp.asarray(b_base, dtype=jnp.float32),
        n_modes=n_modes,
        n_indices=jnp.asarray(indices),
        modal_scales=jnp.asarray(modal_scales),
        beta=jnp.asarray(beta, dtype=jnp.complex64),
        y_quadrature_normalized=jnp.asarray(nodes, dtype=jnp.float32),
        quadrature_weights=jnp.asarray(weights_physical, dtype=jnp.float32),
        modal_basis_quadrature=jnp.asarray(basis, dtype=jnp.float32),
        field_scale=float(modal_scales[config.mode]),
    )

def true_squared_slowness(
    context: ForwardContext, x: jax.Array, y: jax.Array
) -> jax.Array:
    """Known piecewise-constant squared slowness at normalized coordinates."""
    x_physical = x * context.config.half_length
    y_physical = (y + 1.0) * context.config.height / 2.0
    value = jnp.asarray(1.0 / context.config.c0**2, dtype=jnp.float32)
    for circle in context.config.circles:
        inside = (
            (x_physical - circle.center[0]) ** 2
            + (y_physical - circle.center[1]) ** 2
            <= circle.radius**2
        )
        circle_value = 1.0 / (context.config.c0 * circle.speed_ratio) ** 2
        value = jnp.where(inside, circle_value, value)
    return value


def incident_wave_complex(
    context: ForwardContext, x: jax.Array, y: jax.Array, *, normalized: bool
) -> jax.Array:
    mode = context.config.mode
    x_physical = x * context.config.half_length
    y_physical = (y + 1.0) * context.config.height / 2.0
    shape = context.modal_scales[mode] * jnp.cos(
        mode * jnp.pi * y_physical / context.config.height
    )
    value = shape * jnp.exp(
        -1j * context.config.incidence * context.beta[mode] * x_physical
    )
    return value / context.field_scale if normalized else value


def _real_pair_to_complex(value: jax.Array) -> jax.Array:
    return value[..., 0] + 1j * value[..., 1]


def _complex_to_real_pair(value: jax.Array) -> jax.Array:
    return jnp.stack((jnp.real(value), jnp.imag(value)), axis=-1)


def regular_collocation_points(
    sizes: tuple[int, int, int], context: ForwardContext
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    n_pde, n_neumann, n_dtn = sizes
    aspect = 2.0 * context.config.half_length / context.config.height
    factor_pairs = [
        (n_pde // ny, ny)
        for ny in range(1, int(math.sqrt(n_pde)) + 1)
        if n_pde % ny == 0
    ]
    nx, ny = min(
        factor_pairs,
        key=lambda shape: abs(math.log((shape[0] / shape[1]) / aspect)),
    )
    x_axis = jnp.linspace(-1.0 + 1.0 / nx, 1.0 - 1.0 / nx, nx)
    y_axis = jnp.linspace(-1.0 + 1.0 / ny, 1.0 - 1.0 / ny, ny)
    x_grid, y_grid = jnp.meshgrid(x_axis, y_axis, indexing="xy")

    def edge(count: int) -> jax.Array:
        return jnp.linspace(-1.0 + 1.0 / count, 1.0 - 1.0 / count, count)

    return x_grid.reshape(-1), y_grid.reshape(-1), edge(n_neumann), edge(n_dtn)


def sample_collocation_points(
    key: jax.Array, sizes: tuple[int, int, int]
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    n_pde, n_neumann, n_dtn = sizes
    keys = jax.random.split(key, 4)
    return (
        jax.random.uniform(keys[0], (n_pde,), minval=-1.0, maxval=1.0),
        jax.random.uniform(keys[1], (n_pde,), minval=-1.0, maxval=1.0),
        jax.random.uniform(keys[2], (n_neumann,), minval=-1.0, maxval=1.0),
        jax.random.uniform(keys[3], (n_dtn,), minval=-1.0, maxval=1.0),
    )


def rad_probabilities(
    residual_magnitudes: np.ndarray, *, k: float, c: float
) -> np.ndarray:
    """Discretize ``p(x) ∝ eps(x)^k / E[eps^k] + c`` on a candidate pool."""
    residuals = np.asarray(residual_magnitudes, dtype=np.float64)
    if residuals.ndim != 1 or residuals.size == 0:
        raise ValueError("RAD residual magnitudes must be a non-empty vector")
    if k < 0.0 or c < 0.0:
        raise ValueError("RAD k and c must be non-negative")
    if not np.isfinite(residuals).all() or np.any(residuals < 0.0):
        raise ValueError("RAD residual magnitudes must be finite and non-negative")

    powered = np.power(residuals, k)
    mean_powered = float(powered.mean())
    if not np.isfinite(mean_powered) or mean_powered <= 0.0:
        weights = np.ones_like(powered)
    else:
        weights = powered / mean_powered + c
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        weights = np.ones_like(powered)
        total = float(weights.size)
    return weights / total


def make_rad_residual_evaluator(context: ForwardContext, variant: str):
    """Create the batched complex-residual magnitude evaluator used by RAD."""
    _validate_variant(variant)

    @jax.jit
    def evaluate(params, x_values, y_values):
        residual = jax.vmap(
            lambda x_value, y_value: pointwise_pde_residual(
                params, context, variant, x_value, y_value
            )
        )(x_values, y_values)
        return jnp.linalg.norm(residual, axis=-1)

    return evaluate


def uniform_rad_candidate_distribution(
    rng: np.random.Generator, candidate_count: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Create a finite uniform pool and its inverse-CDF bootstrap table."""
    if candidate_count <= 0:
        raise ValueError("RAD candidate count must be positive")
    candidates = rng.uniform(-1.0, 1.0, size=(candidate_count, 2)).astype(
        np.float32
    )
    cdf = np.arange(1, candidate_count + 1, dtype=np.float64) / candidate_count
    cdf[-1] = 1.0
    return (
        jnp.asarray(candidates[:, 0]),
        jnp.asarray(candidates[:, 1]),
        jnp.asarray(cdf, dtype=jnp.float32),
    )


def build_rad_candidate_distribution(
    params: Mapping[str, Any],
    residual_evaluator,
    rng: np.random.Generator,
    *,
    candidate_count: int,
    candidate_batch_size: int,
    k: float,
    c: float,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], dict[str, float | int]]:
    """Evaluate RAD probabilities on a dense uniform candidate pool."""
    if candidate_count <= 0 or candidate_batch_size <= 0:
        raise ValueError("Invalid RAD candidate counts")
    candidates = rng.uniform(-1.0, 1.0, size=(candidate_count, 2)).astype(
        np.float32
    )
    magnitudes = np.empty(candidate_count, dtype=np.float64)
    for start in range(0, candidate_count, candidate_batch_size):
        stop = min(start + candidate_batch_size, candidate_count)
        batch = candidates[start:stop]
        batch_count = stop - start
        if batch_count < candidate_batch_size:
            batch = np.pad(
                batch,
                ((0, candidate_batch_size - batch_count), (0, 0)),
                mode="edge",
            )
        values = np.asarray(
            jax.device_get(
                residual_evaluator(
                    params,
                    jnp.asarray(batch[:, 0]),
                    jnp.asarray(batch[:, 1]),
                )
            ),
            dtype=np.float64,
        )
        magnitudes[start:stop] = values[:batch_count]

    probabilities = rad_probabilities(magnitudes, k=k, c=c)
    cdf = np.cumsum(probabilities, dtype=np.float64)
    cdf[-1] = 1.0
    diagnostics: dict[str, float | int] = {
        "candidate_count": candidate_count,
        "candidate_residual_mean": float(magnitudes.mean()),
        "candidate_residual_max": float(magnitudes.max()),
        "bootstrap_expected_residual_mean": float(
            np.dot(probabilities, magnitudes)
        ),
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "effective_candidate_count": float(1.0 / np.sum(probabilities**2)),
    }
    return (
        jnp.asarray(candidates[:, 0]),
        jnp.asarray(candidates[:, 1]),
        jnp.asarray(cdf, dtype=jnp.float32),
    ), diagnostics


def bootstrap_rad_pde_points(
    key: jax.Array,
    distribution: tuple[jax.Array, jax.Array, jax.Array],
    count: int,
) -> tuple[jax.Array, jax.Array]:
    """Draw a fresh with-replacement PDE batch from a cached RAD distribution."""
    if count <= 0:
        raise ValueError("RAD bootstrap batch size must be positive")
    x_candidates, y_candidates, cdf = distribution
    uniforms = jax.random.uniform(key, (count,), minval=0.0, maxval=1.0)
    indices = jnp.searchsorted(cdf, uniforms, side="right")
    indices = jnp.minimum(indices, cdf.size - 1)
    return x_candidates[indices], y_candidates[indices]


def sample_mixed_rad_collocation_points(
    key: jax.Array,
    sizes: tuple[int, int, int],
    candidate_distribution: tuple[jax.Array, jax.Array, jax.Array],
    rad_points: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Keep the total PDE count fixed while replacing its tail by RAD points."""
    n_pde, n_neumann, n_dtn = sizes
    if rad_points <= 0 or rad_points >= n_pde:
        raise ValueError("rad_points must lie strictly between zero and n_pde")
    uniform_points = sample_collocation_points(
        key, (n_pde - rad_points, n_neumann, n_dtn)
    )
    rad_key = jax.random.fold_in(key, 220710289)
    rad_x, rad_y = bootstrap_rad_pde_points(
        rad_key, candidate_distribution, rad_points
    )
    return (
        jnp.concatenate((uniform_points[0], rad_x)),
        jnp.concatenate((uniform_points[1], rad_y)),
        uniform_points[2],
        uniform_points[3],
    )


ADWEIGHTS_COMPONENTS = ("pde", "neumann", "dtn")


def adweights_from_lambdas(
    config: AdWeightsConfig, lambdas: jax.Array | Sequence[float]
) -> tuple[jax.Array, jax.Array]:
    """Return raw inverse weights and custom-scaled effective weights."""
    lambdas_array = jnp.asarray(lambdas, dtype=jnp.float32)
    if lambdas_array.shape != (3,):
        raise ValueError("Adaptive lambdas must have shape (3,)")
    inverse_weights = 1.0 / (config.epsilon + lambdas_array)
    effective_weights = (
        jnp.asarray(config.custom_weights, dtype=jnp.float32) * inverse_weights
    )
    return inverse_weights, effective_weights


def update_adweights_lambdas(
    config: AdWeightsConfig,
    lambdas: jax.Array | Sequence[float],
    gradient_norms: jax.Array | Sequence[float],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply the configured exponential update and derive both weight forms."""
    lambdas_array = jnp.asarray(lambdas, dtype=jnp.float32)
    norms_array = jnp.asarray(gradient_norms, dtype=jnp.float32)
    if lambdas_array.shape != (3,) or norms_array.shape != (3,):
        raise ValueError("Adaptive lambdas and gradient norms must have shape (3,)")
    updated = (1.0 - config.alpha) * lambdas_array + config.alpha * norms_array
    inverse_weights, effective_weights = adweights_from_lambdas(config, updated)
    return updated, inverse_weights, effective_weights


def loss_weights(config: ForwardConfig, variant: str) -> jax.Array:
    if uses_adweights(variant):
        adaptive = require_adweights_config(config, variant)
        return adweights_from_lambdas(adaptive, adaptive.initial_lambdas)[1]
    values = (
        config.fourier_loss_weights
        if uses_fourier_features(variant)
        else config.classical_loss_weights
    )
    return jnp.asarray(values, dtype=jnp.float32)


def pointwise_pde_residual(
    params: Mapping[str, Any],
    context: ForwardContext,
    variant: str,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Return the real/imaginary Helmholtz residual at one normalized point."""
    _validate_variant(variant)
    value, _, laplacian = value_gradient_laplacian(
        params, context, variant, x, y
    )
    omega_squared = (2.0 * jnp.pi * context.config.frequency) ** 2
    material = true_squared_slowness(context, x, y)
    if is_scattered_variant(variant):
        incident = _complex_to_real_pair(
            incident_wave_complex(context, x, y, normalized=True)
        )
        scattering = material * value + (
            material - 1.0 / context.config.c0**2
        ) * incident
        return laplacian + omega_squared * scattering
    return laplacian + omega_squared * material * value


def forward_loss(
    params: Mapping[str, Any],
    context: ForwardContext,
    variant: str,
    points: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    weights: jax.Array | Sequence[float] | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:

    k0 = 2.0 * jnp.pi * context.config.frequency / context.config.c0

    _validate_variant(variant)
    x_pde, y_pde, x_neumann, y_dtn = points
    scattered = is_scattered_variant(variant)

    def field_value(x: jax.Array, y: jax.Array) -> jax.Array:
        return model_value(params, context, variant, x, y)

    def field_terms(x: jax.Array, y: jax.Array):
        return value_gradient_laplacian(params, context, variant, x, y)

    def physical_derivative(
        x: jax.Array, y: jax.Array, axis: int
    ) -> jax.Array:
        _, gradient, _ = field_terms(x, y)
        return gradient[axis]

    def dtn_loss(x_boundary: float, sign: float) -> jax.Array:
        field_on_quadrature = jax.vmap(field_value, in_axes=(None, 0))(
            x_boundary, context.y_quadrature_normalized
        )
        modal_coefficients = context.modal_basis_quadrature @ (
            context.quadrature_weights * _real_pair_to_complex(field_on_quadrature)
        )
        expected_modes = sign * 1j * context.beta * modal_coefficients
        if (not scattered) and int(x_boundary) == context.config.incidence:
            incident_coefficients = jnp.zeros(
                (context.n_modes,), dtype=jnp.complex64
            )
            x_physical_boundary = x_boundary * context.config.half_length
            incident_at_boundary = (
                jnp.exp(
                    -1j
                    * context.config.incidence
                    * context.beta[context.config.mode]
                    * x_physical_boundary
                )
                / context.field_scale
            )
            incident_coefficients = incident_coefficients.at[context.config.mode].set(
                incident_at_boundary
            )
            expected_modes = (
                expected_modes
                - context.config.incidence * 2j * context.beta * incident_coefficients
            )

        def expected_value(y_value: jax.Array) -> jax.Array:
            y_physical = (y_value + 1.0) * context.config.height / 2.0
            basis = context.modal_scales * jnp.cos(
                context.n_indices * jnp.pi * y_physical / context.config.height
            )
            return jnp.dot(basis, expected_modes)

        expected = jax.vmap(expected_value)(y_dtn)
        actual_pairs = jax.vmap(
            lambda y_value: physical_derivative(x_boundary, y_value, 0)
        )(y_dtn)
        actual = _real_pair_to_complex(actual_pairs)
        return jnp.mean(jnp.abs(actual - expected) ** 2)

    residual = jax.vmap(
        lambda x_value, y_value: pointwise_pde_residual(
            params, context, variant, x_value, y_value
        )
    )(x_pde, y_pde)/k0**2
    pde_loss = jnp.mean(residual**2)
    top = jax.vmap(lambda x_value: physical_derivative(x_value, 1.0, 1))(x_neumann)
    bottom = jax.vmap(lambda x_value: physical_derivative(x_value, -1.0, 1))(
        x_neumann
    )
    neumann_loss = jnp.mean(top**2 + bottom**2)/k0**2
    left = dtn_loss(-1.0, -1.0)
    right = dtn_loss(1.0, 1.0)
    dtn_total = (left + right)/k0**2
    boundary_loss = neumann_loss + dtn_total
    weights = (
        loss_weights(context.config, variant)
        if weights is None
        else jnp.asarray(weights, dtype=jnp.float32)
    )
    if weights.shape != (3,):
        raise ValueError("Loss weights must have shape (3,)")
    objective = (
        weights[0] * pde_loss
        + weights[1] * neumann_loss
        + weights[2] * dtn_total
    )
    return objective, {
        "pde": pde_loss,
        "bc": boundary_loss,
        "unweighted_total": pde_loss + boundary_loss,
        "neumann": neumann_loss,
        "dtn": dtn_total,
        "dtn_left": left,
        "dtn_right": right,
    }


def _tree_dot(left: Mapping[str, Any], right: Mapping[str, Any]) -> jax.Array:
    products = [
        jnp.real(jnp.vdot(left_leaf, right_leaf))
        for left_leaf, right_leaf in zip(
            jax.tree_util.tree_leaves(left),
            jax.tree_util.tree_leaves(right),
        )
    ]
    if not products:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return sum(products[1:], products[0])


def gradient_alignment_stats(
    params: Mapping[str, Any],
    context: ForwardContext,
    variant: str,
    points: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
) -> dict[str, jax.Array]:
    """Return loss-component gradient norms and the PDE/BC cosine similarity."""

    def pde_objective(model_params):
        return forward_loss(model_params, context, variant, points)[1]["pde"]

    def neumann_objective(model_params):
        return forward_loss(model_params, context, variant, points)[1]["neumann"]

    def dtn_objective(model_params):
        return forward_loss(model_params, context, variant, points)[1]["dtn"]

    def bc_objective(model_params):
        return forward_loss(model_params, context, variant, points)[1]["bc"]

    pde_gradients = jax.grad(pde_objective)(params)
    neumann_gradients = jax.grad(neumann_objective)(params)
    dtn_gradients = jax.grad(dtn_objective)(params)
    bc_gradients = jax.grad(bc_objective)(params)
    pde_norm = jnp.sqrt(jnp.maximum(_tree_dot(pde_gradients, pde_gradients), 0.0))
    neumann_norm = jnp.sqrt(
        jnp.maximum(_tree_dot(neumann_gradients, neumann_gradients), 0.0)
    )
    dtn_norm = jnp.sqrt(jnp.maximum(_tree_dot(dtn_gradients, dtn_gradients), 0.0))
    bc_norm = jnp.sqrt(jnp.maximum(_tree_dot(bc_gradients, bc_gradients), 0.0))
    denominator = pde_norm * bc_norm
    raw_cosine = _tree_dot(pde_gradients, bc_gradients) / denominator
    cosine = jnp.where(
        denominator > 0.0,
        jnp.clip(raw_cosine, -1.0, 1.0),
        jnp.nan,
    )
    return {
        "pde_gradient_l2_norm": pde_norm,
        "neumann_gradient_l2_norm": neumann_norm,
        "dtn_gradient_l2_norm": dtn_norm,
        "bc_gradient_l2_norm": bc_norm,
        "pde_bc_gradient_cosine": cosine,
    }


def make_adam_optimizer(
    params: Mapping[str, Any], config: ForwardConfig, variant: str, adam_steps: int
) -> optax.GradientTransformation:
    if not uses_fourier_features(variant):
        return optax.adam(config.classical_learning_rate)

    sigma_steps = sigma_train_step_count(config, variant, adam_steps)
    sigma_schedule = optax.cosine_decay_schedule(
        config.fourier_sigma_learning_rate,
        decay_steps=max(sigma_steps - 1, 1),
        alpha=config.fourier_sigma_cosine_alpha,
    )
    field_optimizer = optax.adam(config.fourier_field_learning_rate)
    sigma_optimizer = optax.adam(sigma_schedule)
    labels = jax.tree_util.tree_map(lambda _: "field", params)
    labels["sigma"] = jax.tree_util.tree_map(lambda _: "sigma", params["sigma"])
    return optax.multi_transform(
        {"field": field_optimizer, "sigma": sigma_optimizer}, labels
    )


def sigma_train_step_count(config: ForwardConfig, variant: str, adam_steps: int) -> int:
    if not uses_fourier_features(variant):
        return 0
    return max(1, int(config.fourier_sigma_decay_fraction * adam_steps))


def _stop_sigma_gradient(params: Mapping[str, Any]) -> dict[str, Any]:
    if "sigma" not in params:
        return dict(params)
    return {**params, "sigma": jax.lax.stop_gradient(params["sigma"])}


def _zero_sigma_update(updates: Mapping[str, Any]) -> dict[str, Any]:
    if "sigma" not in updates:
        return dict(updates)
    return {
        **updates,
        "sigma": jax.tree_util.tree_map(jnp.zeros_like, updates["sigma"]),
    }


def adweights_gradient_l2_norms(
    params: Mapping[str, Any],
    context: ForwardContext,
    variant: str,
    points: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    *,
    freeze_sigma: bool = False,
) -> jax.Array:
    """Return raw component-gradient norms over currently active parameters."""

    def component_norm(component: str) -> jax.Array:
        def objective(model_params):
            if freeze_sigma:
                model_params = _stop_sigma_gradient(model_params)
            return forward_loss(model_params, context, variant, points)[1][component]

        gradients = jax.grad(objective)(params)
        return jnp.sqrt(jnp.maximum(_tree_dot(gradients, gradients), 0.0))

    return jnp.stack(tuple(component_norm(name) for name in ADWEIGHTS_COMPONENTS))


def make_adam_step(
    optimizer: optax.GradientTransformation,
    context: ForwardContext,
    variant: str,
    sizes: tuple[int, int, int],
    *,
    freeze_sigma: bool = False,
):
    @jax.jit
    def compiled_step(params, state, collocation_key, weights):
        points = sample_collocation_points(collocation_key, sizes)

        def objective(model_params):
            if freeze_sigma:
                model_params = _stop_sigma_gradient(model_params)
            return forward_loss(model_params, context, variant, points, weights)

        (value, aux), gradients = jax.value_and_grad(objective, has_aux=True)(params)
        updates, state = optimizer.update(gradients, state, params)
        if freeze_sigma:
            updates = _zero_sigma_update(updates)
        params = optax.apply_updates(params, updates)
        return params, state, value, aux

    def step(params, state, collocation_key, weights=None):
        if weights is None:
            weights = loss_weights(context.config, variant)
        return compiled_step(params, state, collocation_key, weights)

    return step


def make_rad_adam_step(
    optimizer: optax.GradientTransformation,
    context: ForwardContext,
    variant: str,
    sizes: tuple[int, int, int],
    *,
    freeze_sigma: bool = False,
):
    """Create an Adam step mixing uniform and RAD points at fixed batch size."""
    if not is_rad_variant(variant):
        raise ValueError("make_rad_adam_step requires a RAD variant")

    @jax.jit
    def compiled_step(params, state, collocation_key, candidate_distribution, weights):
        points = sample_mixed_rad_collocation_points(
            collocation_key,
            sizes,
            candidate_distribution,
            context.config.rad_points,
        )

        def objective(model_params):
            if freeze_sigma:
                model_params = _stop_sigma_gradient(model_params)
            return forward_loss(model_params, context, variant, points, weights)

        (value, aux), gradients = jax.value_and_grad(objective, has_aux=True)(params)
        updates, state = optimizer.update(gradients, state, params)
        if freeze_sigma:
            updates = _zero_sigma_update(updates)
        params = optax.apply_updates(params, updates)
        return params, state, value, aux

    def step(params, state, collocation_key, candidate_distribution, weights=None):
        if weights is None:
            weights = loss_weights(context.config, variant)
        return compiled_step(
            params, state, collocation_key, candidate_distribution, weights
        )

    return step


def analytic_modal_complex(
    config: ForwardConfig, x_physical: np.ndarray, y_physical: np.ndarray
) -> np.ndarray:
    """Return the exact incident mode in a homogeneous waveguide."""
    x_physical = np.asarray(x_physical, dtype=np.float64)
    y_physical = np.asarray(y_physical, dtype=np.float64)
    transverse_wavenumber = config.mode * np.pi / config.height
    longitudinal_wavenumber = np.sqrt(
        (2.0 * np.pi * config.frequency / config.c0) ** 2
        - transverse_wavenumber**2
        + 0j
    )
    modal_scale = np.sqrt((1.0 if config.mode == 0 else 2.0) / config.height)
    return (
        modal_scale
        * np.cos(transverse_wavenumber * y_physical)
        * np.exp(-1j * config.incidence * longitudinal_wavenumber * x_physical)
    )


def _coalesced_symmetric_matrix(
    size: int,
    row_chunks: Sequence[np.ndarray],
    column_chunks: Sequence[np.ndarray],
    value_chunks: Sequence[np.ndarray],
) -> SymmetricCOOMatrix:
    rows = np.concatenate(row_chunks).astype(np.int64, copy=False)
    columns = np.concatenate(column_chunks).astype(np.int64, copy=False)
    values = np.concatenate(value_chunks).astype(np.float64, copy=False)
    keys = rows * np.int64(size) + columns
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    values = values[order]
    starts = np.concatenate(([0], np.flatnonzero(keys[1:] != keys[:-1]) + 1))
    summed = np.add.reduceat(values, starts)
    unique_keys = keys[starts]
    tolerance = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(summed))))
    keep = np.abs(summed) > tolerance
    unique_keys = unique_keys[keep]
    summed = summed[keep]
    return SymmetricCOOMatrix(
        size=size,
        rows=unique_keys // size,
        columns=unique_keys % size,
        values=summed,
    )


def build_analytic_modal_reference(config: ForwardConfig) -> FEMReference:
    """Build an independent P1 triangular mesh and its exact modal field.

    ``analytic_triangulation=(nx, ny)`` denotes the number of rectangular
    cells.  Every cell is divided along the lower-left to upper-right diagonal,
    producing two triangles.  The assembled P1 mass and stiffness matrices are
    then used for the same discrete L2/H1 norms as a FEM reference.
    """
    if config.reference_kind != "analytic_mode":
        raise ValueError("An analytic modal reference requires analytic_mode")
    if config.analytic_triangulation is None:
        raise ValueError("Missing analytic_triangulation")
    nx, ny = config.analytic_triangulation
    x_axis = np.linspace(-config.half_length, config.half_length, nx + 1)
    y_axis = np.linspace(0.0, config.height, ny + 1)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="xy")
    x = x_grid.reshape(-1)
    y = y_grid.reshape(-1)

    node_ids = np.arange((nx + 1) * (ny + 1), dtype=np.int64).reshape(
        ny + 1, nx + 1
    )
    lower_left = node_ids[:-1, :-1].reshape(-1)
    lower_right = node_ids[:-1, 1:].reshape(-1)
    upper_left = node_ids[1:, :-1].reshape(-1)
    upper_right = node_ids[1:, 1:].reshape(-1)
    triangles = np.concatenate(
        (
            np.stack((lower_left, lower_right, upper_right), axis=1),
            np.stack((lower_left, upper_right, upper_left), axis=1),
        ),
        axis=0,
    )

    triangle_x = x[triangles]
    triangle_y = y[triangles]
    twice_area = (
        (triangle_x[:, 1] - triangle_x[:, 0])
        * (triangle_y[:, 2] - triangle_y[:, 0])
        - (triangle_x[:, 2] - triangle_x[:, 0])
        * (triangle_y[:, 1] - triangle_y[:, 0])
    )
    if np.any(twice_area <= 0.0):
        raise RuntimeError("Analytic reference triangulation is not counter-clockwise")
    areas = 0.5 * twice_area
    gradient_x = np.stack(
        (
            triangle_y[:, 1] - triangle_y[:, 2],
            triangle_y[:, 2] - triangle_y[:, 0],
            triangle_y[:, 0] - triangle_y[:, 1],
        ),
        axis=1,
    ) / twice_area[:, None]
    gradient_y = np.stack(
        (
            triangle_x[:, 2] - triangle_x[:, 1],
            triangle_x[:, 0] - triangle_x[:, 2],
            triangle_x[:, 1] - triangle_x[:, 0],
        ),
        axis=1,
    ) / twice_area[:, None]

    row_chunks: list[np.ndarray] = []
    column_chunks: list[np.ndarray] = []
    mass_chunks: list[np.ndarray] = []
    stiffness_chunks: list[np.ndarray] = []
    for local_row in range(3):
        for local_column in range(local_row + 1):
            first = triangles[:, local_row]
            second = triangles[:, local_column]
            row_chunks.append(np.maximum(first, second))
            column_chunks.append(np.minimum(first, second))
            mass_chunks.append(
                areas * (2.0 if local_row == local_column else 1.0) / 12.0
            )
            stiffness_chunks.append(
                areas
                * (
                    gradient_x[:, local_row] * gradient_x[:, local_column]
                    + gradient_y[:, local_row] * gradient_y[:, local_column]
                )
            )

    size = x.size
    mass = _coalesced_symmetric_matrix(
        size, row_chunks, column_chunks, mass_chunks
    )
    stiffness = _coalesced_symmetric_matrix(
        size, row_chunks, column_chunks, stiffness_chunks
    )
    return FEMReference(
        x=x,
        y=y,
        values=analytic_modal_complex(config, x, y),
        mass=mass,
        stiffness=stiffness,
    )


def load_fem_reference(config: ForwardConfig) -> FEMReference:
    if any(
        value is None
        for value in (config.fem_field, config.mass_matrix, config.stiffness_matrix)
    ):
        raise ValueError("FEM reference paths are missing")
    assert config.fem_field is not None
    assert config.mass_matrix is not None
    assert config.stiffness_matrix is not None
    data = FEMFieldData(config.fem_field)
    case = data.case(config.frequency, config.mode, config.incidence)
    expected_bounds = np.asarray(
        [-config.half_length, config.half_length, 0.0, config.height]
    )
    observed_bounds = np.asarray(
        [data.x.min(), data.x.max(), data.y.min(), data.y.max()]
    )
    if not np.allclose(expected_bounds, observed_bounds, rtol=0.0, atol=1e-7):
        raise ValueError(
            f"FEM geometry {observed_bounds.tolist()} does not match {expected_bounds.tolist()}"
        )
    expected_k0 = 2.0 * np.pi * config.frequency / config.c0
    if not np.isclose(case.k0, expected_k0, rtol=1e-7, atol=1e-9):
        raise ValueError(f"FEM k0={case.k0} does not match expected k0={expected_k0}")
    mass = load_symmetric_coo_matrix(config.mass_matrix, expected_size=data.size)
    stiffness = load_symmetric_coo_matrix(
        config.stiffness_matrix, expected_size=data.size
    )
    return FEMReference(data.x, data.y, case.values, mass, stiffness)


def load_evaluation_reference(config: ForwardConfig) -> FEMReference:
    if config.reference_kind == "analytic_mode":
        return build_analytic_modal_reference(config)
    return load_fem_reference(config)


def compute_misfit_metrics(
    reference_values: np.ndarray,
    pinn_values: np.ndarray,
    mass: SymmetricCOOMatrix,
    stiffness: SymmetricCOOMatrix,
) -> MisfitMetrics:
    reference_values = np.asarray(reference_values, dtype=np.complex128)
    pinn_values = np.asarray(pinn_values, dtype=np.complex128)
    if (
        reference_values.shape != pinn_values.shape
        or reference_values.shape != (mass.size,)
    ):
        raise ValueError("Reference and PINN vectors must match the matrix size")
    if stiffness.size != mass.size:
        raise ValueError("Mass and stiffness matrices must have the same size")
    if not np.isfinite(reference_values).all() or not np.isfinite(pinn_values).all():
        raise ValueError("Reference/PINN vectors contain NaN or infinite values")
    error = pinn_values - reference_values
    reference_l2_squared = mass.quadratic_form(reference_values)
    error_l2_squared = mass.quadratic_form(error)
    reference_h1_squared = reference_l2_squared + stiffness.quadratic_form(
        reference_values
    )
    error_h1_squared = error_l2_squared + stiffness.quadratic_form(error)
    if reference_l2_squared <= 0.0 or reference_h1_squared <= 0.0:
        raise ValueError("Relative norms require a nonzero reference field")
    return MisfitMetrics(
        l2_absolute=float(np.sqrt(error_l2_squared)),
        l2_relative=float(np.sqrt(error_l2_squared / reference_l2_squared)),
        h1_absolute=float(np.sqrt(error_h1_squared)),
        h1_relative=float(np.sqrt(error_h1_squared / reference_h1_squared)),
    )


def make_physical_predictor(context: ForwardContext, variant: str):
    @jax.jit
    def predict(params, x_physical, y_physical):
        x_normalized = x_physical / context.config.half_length
        y_normalized = 2.0 * y_physical / context.config.height - 1.0

        def one(x_value, y_value):
            learned = _real_pair_to_complex(
                model_value(params, context, variant, x_value, y_value)
            )
            if is_scattered_variant(variant):
                learned = learned + incident_wave_complex(
                    context, x_value, y_value, normalized=True
                )
            return context.field_scale * learned

        return jax.vmap(one)(x_normalized, y_normalized)

    return predict


def predict_reference_nodes(
    params: Mapping[str, Any],
    reference: FEMReference,
    predictor,
    batch_size: int,
) -> np.ndarray:
    output = np.empty(reference.x.size, dtype=np.complex128)
    for start in range(0, reference.x.size, batch_size):
        stop = min(start + batch_size, reference.x.size)
        x_batch = reference.x[start:stop]
        y_batch = reference.y[start:stop]
        count = stop - start
        if count < batch_size:
            x_batch = np.pad(x_batch, (0, batch_size - count), mode="edge")
            y_batch = np.pad(y_batch, (0, batch_size - count), mode="edge")
        values = np.asarray(
            jax.device_get(
                predictor(
                    params,
                    jnp.asarray(x_batch, dtype=jnp.float32),
                    jnp.asarray(y_batch, dtype=jnp.float32),
                )
            )
        )
        output[start:stop] = values[:count]
    return output


def _copy_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return jax.tree_util.tree_map(lambda value: jnp.array(value, copy=True), params)


def save_checkpoint(
    path: str | Path,
    params: Mapping[str, Any],
    context: ForwardContext,
    metadata: Mapping[str, Any],
) -> Path:
    path = Path(path)
    arrays: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(canonical_manifest(metadata), sort_keys=True)),
        "b_base": np.asarray(jax.device_get(context.b_base)),
    }
    for index, layer in enumerate(params["layers"]):
        arrays[f"layer_{index}_W"] = np.asarray(jax.device_get(layer["W"]))
        arrays[f"layer_{index}_b"] = np.asarray(jax.device_get(layer["b"]))
    if "sigma" in params:
        arrays["sigma"] = np.asarray(jax.device_get(params["sigma"]))
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return path


def load_checkpoint(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        b_base = np.asarray(archive["b_base"]).copy()
        layer_indices = sorted(
            int(name.split("_")[1])
            for name in archive.files
            if name.startswith("layer_") and name.endswith("_W")
        )
        layers = [
            {
                "W": jnp.asarray(archive[f"layer_{index}_W"]),
                "b": jnp.asarray(archive[f"layer_{index}_b"]),
            }
            for index in layer_indices
        ]
        params: dict[str, Any] = {"layers": layers}
        if "sigma" in archive:
            params["sigma"] = jnp.asarray(archive["sigma"])
    return params, metadata, b_base


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty history: {path}")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(canonical_manifest(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _block(value: Any) -> None:
    leaves = jax.tree_util.tree_leaves(value)
    if leaves:
        jax.block_until_ready(leaves[0])


def _evaluation_due(local_step: int, interval: int, final_step: int) -> bool:
    return local_step == 0 or local_step == final_step or local_step % interval == 0


def _optimizer_timed_call(function, *args):
    _block(args)
    start = time.perf_counter()
    result = function(*args)
    _block(result)
    return result, time.perf_counter() - start


def _environment_manifest() -> dict[str, Any]:
    devices = jax.devices()
    return {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "optax": optax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
    }


def run_training(
    config: ForwardConfig,
    *,
    variant: str,
    seed: int,
    adam_steps: int,
    output_root: Path | None = None,
) -> Path:
    """Execute one isolated run. Scientific campaigns call this in subprocesses."""
    _validate_variant(variant)
    if seed < 0 or adam_steps < 0:
        raise ValueError("seed and optimizer step counts must be non-negative")
    if adam_steps <= 0:
        raise ValueError("At least one Adam step is required")
    adweights_enabled = uses_adweights(variant)
    adweights_config = (
        require_adweights_config(config, variant) if adweights_enabled else None
    )

    effective = {
        "forward_config": config.manifest(),
        "variant": variant,
        "sampling_reference": RAD_REFERENCE if is_rad_variant(variant) else None,
        "sampling_protocol": (
            RAD_SAMPLING_PROTOCOL if is_rad_variant(variant) else None
        ),
        "seed": int(seed),
        "adam_steps": int(adam_steps),
        "sigma_train_steps": sigma_train_step_count(config, variant, adam_steps),
    }
    run_id = configuration_id(effective)
    parent = Path(output_root) if output_root is not None else config.output_root / "runs"
    run_directory = parent / f"{variant}_seed{seed}_cfg{run_id}"
    if run_directory.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_directory}")

    # Validate all expensive external inputs before claiming the output name.
    reference = load_evaluation_reference(config)
    reference_sha256: dict[str, str] = {}
    if config.reference_kind == "fem":
        assert config.fem_field is not None
        assert config.mass_matrix is not None
        assert config.stiffness_matrix is not None
        reference_sha256 = {
            "fem_field": _file_sha256(config.fem_field),
            "mass_matrix": _file_sha256(config.mass_matrix),
            "stiffness_matrix": _file_sha256(config.stiffness_matrix),
        }
    run_directory.mkdir(parents=True, exist_ok=False)
    write_manifest_exclusive(
        run_directory / "manifest.json",
        {
            **effective,
            "environment": _environment_manifest(),
            "reference_kind": config.reference_kind,
            "reference_sha256": reference_sha256,
        },
        experiment_id=run_id,
    )
    _write_json_exclusive(run_directory / "effective_config.json", effective)

    initialization_key = jax.random.fold_in(jax.random.key(seed), 101)
    params, b_base = initialize_model(initialization_key, config, variant)
    context = _build_context(config, b_base)
    monitor_points = regular_collocation_points(config.collocation_monitor, context)
    monitor = jax.jit(
        lambda model, weights: forward_loss(
            model, context, variant, monitor_points, weights
        )
    )
    gradient_monitor = jax.jit(
        lambda model, batch_points: gradient_alignment_stats(
            model, context, variant, batch_points
        )
    )
    predictor = make_physical_predictor(context, variant)
    rad_enabled = is_rad_variant(variant)
    sigma_train_steps = sigma_train_step_count(config, variant, adam_steps)
    if adweights_config is None:
        current_lambdas = None
        current_inverse_weights = None
        current_effective_weights = loss_weights(config, variant)
        adweights_gradient_evaluator = None
        frozen_adweights_gradient_evaluator = None
    else:
        current_lambdas = jnp.asarray(
            adweights_config.initial_lambdas, dtype=jnp.float32
        )
        current_inverse_weights, current_effective_weights = adweights_from_lambdas(
            adweights_config, current_lambdas
        )
        adweights_gradient_evaluator = jax.jit(
            lambda model, batch_points: adweights_gradient_l2_norms(
                model, context, variant, batch_points
            )
        )
        frozen_adweights_gradient_evaluator = jax.jit(
            lambda model, batch_points: adweights_gradient_l2_norms(
                model, context, variant, batch_points, freeze_sigma=True
            )
        )

    adam_optimizer = make_adam_optimizer(params, config, variant, adam_steps)
    adam_state = adam_optimizer.init(params)
    if rad_enabled:
        adam_step = make_rad_adam_step(
            adam_optimizer, context, variant, config.collocation_adam
        )
        frozen_adam_step = make_rad_adam_step(
            adam_optimizer,
            context,
            variant,
            config.collocation_adam,
            freeze_sigma=True,
        )
        rad_rng = np.random.default_rng(np.random.SeedSequence([seed, 220710289]))
        rad_candidate_distribution = uniform_rad_candidate_distribution(
            rad_rng, config.rad_candidate_points
        )
        rad_residual_evaluator = make_rad_residual_evaluator(context, variant)
        # Candidate residual compilation is excluded just like optimizer-step
        # compilation.  The fixed batch shape is also used for the padded tail.
        dummy_candidates = jnp.zeros(
            (config.rad_candidate_batch_size,), dtype=jnp.float32
        )
        _block(
            rad_residual_evaluator(
                params, dummy_candidates, dummy_candidates
            )
        )
    else:
        adam_step = make_adam_step(
            adam_optimizer, context, variant, config.collocation_adam
        )
        frozen_adam_step = make_adam_step(
            adam_optimizer,
            context,
            variant,
            config.collocation_adam,
            freeze_sigma=True,
        )
        rad_candidate_distribution = None
        rad_rng = None
        rad_residual_evaluator = None

    # Compile optimizer steps with discarded outputs.  Compilation is excluded
    # from the measured optimizer time by construction.
    compile_key = jax.random.fold_in(jax.random.key(seed), 202)
    if adam_steps:
        if rad_enabled:
            assert rad_candidate_distribution is not None
            _block(
                adam_step(
                    params,
                    adam_state,
                    compile_key,
                    rad_candidate_distribution,
                    current_effective_weights,
                )
            )
            if uses_fourier_features(variant) and sigma_train_steps < adam_steps:
                _block(
                    frozen_adam_step(
                        params,
                        adam_state,
                        jax.random.fold_in(compile_key, 1),
                        rad_candidate_distribution,
                        current_effective_weights,
                    )
                )
        else:
            _block(
                adam_step(
                    params, adam_state, compile_key, current_effective_weights
                )
            )
            if uses_fourier_features(variant) and sigma_train_steps < adam_steps:
                _block(
                    frozen_adam_step(
                        params,
                        adam_state,
                        jax.random.fold_in(compile_key, 1),
                        current_effective_weights,
                    )
                )

    if adweights_gradient_evaluator is not None:
        if rad_enabled:
            assert rad_candidate_distribution is not None
            compile_points = sample_mixed_rad_collocation_points(
                compile_key,
                config.collocation_adam,
                rad_candidate_distribution,
                config.rad_points,
            )
        else:
            compile_points = sample_collocation_points(
                compile_key, config.collocation_adam
            )
        _block(adweights_gradient_evaluator(params, compile_points))
        if uses_fourier_features(variant) and sigma_train_steps < adam_steps:
            assert frozen_adweights_gradient_evaluator is not None
            _block(frozen_adweights_gradient_evaluator(params, compile_points))

    loss_rows: list[dict[str, Any]] = []
    fem_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    rad_rows: list[dict[str, Any]] = []
    adweights_rows: list[dict[str, Any]] = []
    optimizer_seconds = 0.0
    rad_resampling_seconds = 0.0
    adweights_seconds = 0.0
    adweights_update_count = 0
    best_monitor = float("inf")
    best_monitor_metric = "unweighted_total" if adweights_enabled else "objective"
    best_params = _copy_params(params)
    best_location = {"phase": "initial", "local_step": 0, "global_step": 0}

    def measured_training_seconds() -> float:
        return optimizer_seconds + rad_resampling_seconds + adweights_seconds

    def record_adweights_state(
        *,
        local_step: int,
        update_index: int,
        gradient_norms: Sequence[float] | None,
        update_seconds: float,
    ) -> None:
        if (
            adweights_config is None
            or current_lambdas is None
            or current_inverse_weights is None
        ):
            raise RuntimeError("Adaptive state requested for a static-weight run")
        lambdas = np.asarray(jax.device_get(current_lambdas), dtype=float)
        inverse = np.asarray(jax.device_get(current_inverse_weights), dtype=float)
        effective = np.asarray(jax.device_get(current_effective_weights), dtype=float)
        norms = None if gradient_norms is None else np.asarray(gradient_norms, dtype=float)
        for component_index, component in enumerate(ADWEIGHTS_COMPONENTS):
            adweights_rows.append(
                {
                    "record_type": "adweights_state",
                    "variant": variant,
                    "seed": seed,
                    "phase": "initial" if update_index == 0 else "adam",
                    "local_step": local_step,
                    "global_step": local_step,
                    "update_index": update_index,
                    "component": component,
                    "alpha": adweights_config.alpha,
                    "epsilon": adweights_config.epsilon,
                    "gradient_l2_norm": (
                        None if norms is None else float(norms[component_index])
                    ),
                    "lambda": float(lambdas[component_index]),
                    "inverse_weight": float(inverse[component_index]),
                    "custom_weight": adweights_config.custom_weights[component_index],
                    "effective_weight": float(effective[component_index]),
                    "update_seconds": update_seconds,
                    "cumulative_adweights_seconds": adweights_seconds,
                    "optimizer_seconds": optimizer_seconds,
                    "rad_resampling_seconds": rad_resampling_seconds,
                    "training_seconds": measured_training_seconds(),
                }
            )

    def refresh_rad_distribution(
        phase: str,
        local_step: int,
        global_step: int,
        bootstrap_batch_size: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        nonlocal rad_resampling_seconds
        if rad_rng is None or rad_residual_evaluator is None:
            raise RuntimeError("RAD sampler requested for a non-RAD run")
        _block(params)
        start = time.perf_counter()
        distribution, diagnostics = build_rad_candidate_distribution(
            params,
            rad_residual_evaluator,
            rad_rng,
            candidate_count=config.rad_candidate_points,
            candidate_batch_size=config.rad_candidate_batch_size,
            k=config.rad_k,
            c=config.rad_c,
        )
        _block(distribution)
        elapsed = time.perf_counter() - start
        rad_resampling_seconds += elapsed
        rad_rows.append(
            {
                "record_type": "rad_distribution_refresh",
                "variant": variant,
                "seed": seed,
                "phase": phase,
                "local_step": local_step,
                "global_step": global_step,
                "refresh_index": len(rad_rows) + 1,
                "bootstrap_batch_size": bootstrap_batch_size,
                "bootstrap_with_replacement": True,
                "k": config.rad_k,
                "c": config.rad_c,
                "resampling_seconds": elapsed,
                "cumulative_resampling_seconds": rad_resampling_seconds,
                **diagnostics,
            }
        )
        return distribution

    def record_loss(phase: str, local_step: int, global_step: int) -> None:
        nonlocal best_monitor, best_params, best_location
        objective, components = monitor(params, current_effective_weights)
        _block(objective)
        row = {
            "record_type": "loss_monitor",
            "variant": variant,
            "seed": seed,
            "phase": phase,
            "local_step": local_step,
            "global_step": global_step,
            "optimizer_seconds": optimizer_seconds,
            "rad_resampling_seconds": rad_resampling_seconds,
            "adweights_seconds": adweights_seconds,
            "training_seconds": measured_training_seconds(),
            "objective": float(objective),
            "pde_loss": float(components["pde"]),
            "bc_loss": float(components["bc"]),
            "unweighted_total": float(components["unweighted_total"]),
            "neumann_loss": float(components["neumann"]),
            "dtn_loss": float(components["dtn"]),
            "dtn_left_loss": float(components["dtn_left"]),
            "dtn_right_loss": float(components["dtn_right"]),
        }
        finite_values = (
            np.isfinite(float(value))
            for value in row.values()
            if isinstance(value, float)
        )
        if not all(finite_values):
            raise FloatingPointError(f"Non-finite fixed-monitor loss at {phase} {local_step}")
        loss_rows.append(row)
        monitor_value = float(row[best_monitor_metric])
        if monitor_value < best_monitor:
            best_monitor = monitor_value
            best_params = _copy_params(params)
            best_location = {
                "phase": phase,
                "local_step": local_step,
                "global_step": global_step,
            }

    def record_fem(phase: str, local_step: int, global_step: int) -> None:
        prediction = predict_reference_nodes(
            params,
            reference,
            predictor,
            config.fem_prediction_batch_size,
        )
        metrics = compute_misfit_metrics(
            reference.values, prediction, reference.mass, reference.stiffness
        )
        fem_rows.append(
            {
                "record_type": "fem_metrics",
                "reference_kind": config.reference_kind,
                "variant": variant,
                "seed": seed,
                "phase": phase,
                "local_step": local_step,
                "global_step": global_step,
                "optimizer_seconds": optimizer_seconds,
                "rad_resampling_seconds": rad_resampling_seconds,
                "adweights_seconds": adweights_seconds,
                "training_seconds": measured_training_seconds(),
                **asdict(metrics),
            }
        )

    def adam_gradient_points(
        collocation_key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        if rad_enabled:
            assert rad_candidate_distribution is not None
            return sample_mixed_rad_collocation_points(
                collocation_key,
                config.collocation_adam,
                rad_candidate_distribution,
                config.rad_points,
            )
        return sample_collocation_points(collocation_key, config.collocation_adam)

    def record_gradient(
        phase: str,
        local_step: int,
        global_step: int,
        points: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> None:
        stats = jax.device_get(gradient_monitor(params, points))
        pde_norm = float(stats["pde_gradient_l2_norm"])
        neumann_norm = float(stats["neumann_gradient_l2_norm"])
        dtn_norm = float(stats["dtn_gradient_l2_norm"])
        bc_norm = float(stats["bc_gradient_l2_norm"])
        cosine = float(stats["pde_bc_gradient_cosine"])
        if (
            not np.isfinite(pde_norm)
            or not np.isfinite(neumann_norm)
            or not np.isfinite(dtn_norm)
            or not np.isfinite(bc_norm)
            or (np.isnan(cosine) and pde_norm > 0.0 and bc_norm > 0.0)
            or (not np.isnan(cosine) and not np.isfinite(cosine))
        ):
            raise FloatingPointError(
                f"Non-finite gradient statistics at {phase} {local_step}"
            )
        gradient_rows.append(
            {
                "record_type": "gradient_monitor",
                "variant": variant,
                "seed": seed,
                "phase": phase,
                "local_step": local_step,
                "global_step": global_step,
                "optimizer_seconds": optimizer_seconds,
                "rad_resampling_seconds": rad_resampling_seconds,
                "adweights_seconds": adweights_seconds,
                "training_seconds": measured_training_seconds(),
                "pde_gradient_l2_norm": pde_norm,
                "neumann_gradient_l2_norm": neumann_norm,
                "dtn_gradient_l2_norm": dtn_norm,
                "bc_gradient_l2_norm": bc_norm,
                "pde_bc_gradient_cosine": cosine,
            }
        )

    if adweights_enabled:
        record_adweights_state(
            local_step=0,
            update_index=0,
            gradient_norms=None,
            update_seconds=0.0,
        )
    record_loss("initial", 0, 0)
    record_fem("initial", 0, 0)

    for local_step in range(1, adam_steps + 1):
        if (
            rad_enabled
            and local_step > 1
            and (local_step - 1) % config.rad_resample_interval_adam == 0
        ):
            rad_candidate_distribution = refresh_rad_distribution(
                "adam",
                local_step - 1,
                local_step - 1,
                config.rad_points,
            )
        # This key depends only on (seed, global step), not on model initialization.
        # RAD variants keep the same boundary cloud and the uniform prefix of the
        # paired baseline, then replace exactly config.rad_points PDE points.
        collocation_key = jax.random.fold_in(jax.random.key(seed), local_step)
        freeze_sigma = (
            uses_fourier_features(variant) and local_step > sigma_train_steps
        )
        step_function = (
            adam_step
            if not freeze_sigma
            else frozen_adam_step
        )
        if (
            adweights_config is not None
            and (local_step - 1) % adweights_config.update_interval_adam == 0
        ):
            if adweights_gradient_evaluator is None:
                raise RuntimeError("Missing adaptive gradient evaluator")
            points_for_update = adam_gradient_points(collocation_key)
            evaluator = (
                frozen_adweights_gradient_evaluator
                if freeze_sigma
                else adweights_gradient_evaluator
            )
            if evaluator is None:
                raise RuntimeError("Missing frozen adaptive gradient evaluator")
            _block(params)
            update_start = time.perf_counter()
            gradient_norms_array = evaluator(params, points_for_update)
            _block(gradient_norms_array)
            gradient_norms = np.asarray(
                jax.device_get(gradient_norms_array), dtype=float
            )
            if not np.all(np.isfinite(gradient_norms)) or np.any(gradient_norms < 0.0):
                raise FloatingPointError(
                    f"Non-finite adaptive gradient norms before Adam {local_step}"
                )
            if current_lambdas is None:
                raise RuntimeError("Missing adaptive lambda state")
            if local_step > 1:
                (
                    current_lambdas,
                    current_inverse_weights,
                    current_effective_weights,
                ) = update_adweights_lambdas(
                    adweights_config, current_lambdas, gradient_norms
                )
            _block(current_effective_weights)
            update_elapsed = time.perf_counter() - update_start
            adweights_seconds += update_elapsed
            adweights_update_count += 1
            record_adweights_state(
                local_step=local_step,
                update_index=adweights_update_count,
                gradient_norms=gradient_norms,
                update_seconds=update_elapsed,
            )
        if rad_enabled:
            assert rad_candidate_distribution is not None
            (params, adam_state, _, _), elapsed = _optimizer_timed_call(
                step_function,
                params,
                adam_state,
                collocation_key,
                rad_candidate_distribution,
                current_effective_weights,
            )
        else:
            (params, adam_state, _, _), elapsed = _optimizer_timed_call(
                step_function,
                params,
                adam_state,
                collocation_key,
                current_effective_weights,
            )
        optimizer_seconds += elapsed
        if _evaluation_due(
            local_step, config.gradient_eval_interval_adam, adam_steps
        ):
            record_gradient(
                "adam",
                local_step,
                local_step,
                adam_gradient_points(collocation_key),
            )
        if _evaluation_due(
            local_step, config.loss_eval_interval_adam, adam_steps
        ):
            record_loss("adam", local_step, local_step)
        if _evaluation_due(
            local_step, config.fem_eval_interval_adam, adam_steps
        ):
            record_fem("adam", local_step, local_step)

    if current_lambdas is None or current_inverse_weights is None:
        final_adweights_lambdas = None
        final_adweights_inverse_weights = None
        final_adweights_effective_weights = None
    else:
        final_adweights_lambdas = dict(
            zip(
                ADWEIGHTS_COMPONENTS,
                np.asarray(jax.device_get(current_lambdas), dtype=float).tolist(),
            )
        )
        final_adweights_inverse_weights = dict(
            zip(
                ADWEIGHTS_COMPONENTS,
                np.asarray(
                    jax.device_get(current_inverse_weights), dtype=float
                ).tolist(),
            )
        )
        final_adweights_effective_weights = dict(
            zip(
                ADWEIGHTS_COMPONENTS,
                np.asarray(
                    jax.device_get(current_effective_weights), dtype=float
                ).tolist(),
            )
        )

    checkpoint_metadata = {
        "run_id": run_id,
        "variant": variant,
        "seed": seed,
        "frequency": config.frequency,
        "mode": config.mode,
        "incidence": config.incidence,
        "field_scale": context.field_scale,
        "reference_kind": config.reference_kind,
        "parameter_count": variant_parameter_count(config, variant),
        "sampling_method": (
            "uniform_plus_RAD_bootstrap" if rad_enabled else "uniform_random"
        ),
        "sampling_reference": RAD_REFERENCE if rad_enabled else None,
        "sampling_protocol": RAD_SAMPLING_PROTOCOL if rad_enabled else None,
        "best_monitor": best_monitor,
        "best_monitor_metric": best_monitor_metric,
        "best_location": best_location,
        "adweights_update_count": adweights_update_count,
        "final_adweights_lambdas": final_adweights_lambdas,
        "final_adweights_inverse_weights": final_adweights_inverse_weights,
        "final_adweights_effective_weights": final_adweights_effective_weights,
    }
    save_checkpoint(
        run_directory / "checkpoint_final.npz", params, context, checkpoint_metadata
    )
    save_checkpoint(
        run_directory / "checkpoint_best_monitor.npz",
        best_params,
        context,
        checkpoint_metadata,
    )
    _write_csv(run_directory / "loss_history.csv", loss_rows)
    _write_csv(run_directory / "fem_metrics.csv", fem_rows)
    _write_csv(run_directory / "gradient_history.csv", gradient_rows)
    if rad_rows:
        _write_csv(run_directory / "rad_resampling_history.csv", rad_rows)
    if adweights_rows:
        _write_csv(run_directory / "adweights_history.csv", adweights_rows)
    final_fem = fem_rows[-1]
    summary = {
        **checkpoint_metadata,
        "status": "complete",
        "adam_steps": adam_steps,
        "sigma_train_steps": sigma_train_steps,
        "optimizer_seconds": optimizer_seconds,
        "rad_resampling_seconds": rad_resampling_seconds,
        "adweights_seconds": adweights_seconds,
        "training_seconds": measured_training_seconds(),
        "rad_resample_count": len(rad_rows),
        "rad_refresh_count": len(rad_rows),
        "final_l2_absolute": final_fem["l2_absolute"],
        "final_l2_relative": final_fem["l2_relative"],
        "final_h1_absolute": final_fem["h1_absolute"],
        "final_h1_relative": final_fem["h1_relative"],
        "loss_records": len(loss_rows),
        "fem_records": len(fem_rows),
        "gradient_records": len(gradient_rows),
        "adweights_records": len(adweights_rows),
    }
    _write_json_exclusive(run_directory / "summary.json", summary)
    print(f"Completed run: {run_directory}")
    return run_directory


def _parse_seeds(text: str) -> list[int]:
    seeds = [int(token.strip()) for token in text.split(",") if token.strip()]
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be a non-empty comma-separated list of unique integers")
    return seeds


def _parse_variants(text: str) -> list[str]:
    variants = [token.strip() for token in text.split(",") if token.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if not variants or unknown or len(set(variants)) != len(variants):
        raise ValueError(
            "Variants must be a non-empty, unique subset of "
            f"{VARIANTS}; unknown={unknown}"
        )
    return variants


def run_campaign(
    config_path: Path,
    config: ForwardConfig,
    *,
    variants: Sequence[str],
    seeds: Sequence[int],
    adam_steps: int,
    output_root: Path | None,
) -> Path:
    if adam_steps < 0 or adam_steps <= 0:
        raise ValueError("Campaign optimizer budgets must be non-negative and nonzero")
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError("Campaign seeds must be unique non-negative integers")
    if not variants or any(variant not in VARIANTS for variant in variants):
        raise ValueError("Campaign variants must be a non-empty subset of VARIANTS")
    if len(set(variants)) != len(variants):
        raise ValueError("Campaign variants must be unique")
    adaptive_variants = [variant for variant in variants if uses_adweights(variant)]
    if adaptive_variants and config.adweights is None:
        raise ValueError(
            "Adaptive campaign variants require an 'adweights' JSON object; "
            f"variants={adaptive_variants}"
        )
    rad_enabled = any(is_rad_variant(variant) for variant in variants)
    campaign_config = {
        "forward_config": config.manifest(),
        "variants": list(variants),
        "rad_sampling_reference": RAD_REFERENCE if rad_enabled else None,
        "rad_sampling_protocol": (
            RAD_SAMPLING_PROTOCOL if rad_enabled else None
        ),
        "seeds": list(seeds),
        "adam_steps": adam_steps,
    }
    campaign_id = configuration_id(campaign_config)
    parent = Path(output_root) if output_root is not None else config.output_root
    directory = parent / f"campaign_cfg{campaign_id}"
    directory.mkdir(parents=True, exist_ok=False)
    write_manifest_exclusive(
        directory / "manifest.json", campaign_config, experiment_id=campaign_id
    )
    runs_root = directory / "runs"
    statuses = []
    for variant in variants:
        for seed in seeds:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "run",
                "--config",
                str(config_path.resolve()),
                "--variant",
                variant,
                "--seed",
                str(seed),
                "--adam-steps",
                str(adam_steps),
                "--output-root",
                str(runs_root),
            ]
            print(f"Launching {variant}, seed={seed}", flush=True)
            completed = subprocess.run(command, check=False)
            statuses.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "returncode": completed.returncode,
                    "status": "complete" if completed.returncode == 0 else "failed",
                }
            )
    _write_json_exclusive(
        directory / "campaign_summary.json",
        {
            "campaign_id": campaign_id,
            "statuses": statuses,
            "complete": all(item["returncode"] == 0 for item in statuses),
        },
    )
    failures = [item for item in statuses if item["returncode"] != 0]
    if failures:
        raise RuntimeError(f"Campaign completed with {len(failures)} failed run(s)")
    print(f"Completed campaign: {directory}")
    return directory


def build_parser() -> argparse.ArgumentParser:
    default_config = Path(__file__).with_name(
        "forward_circlebottomright_1200_m0.json"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run one variant and one seed")
    run.add_argument("--config", type=Path, default=default_config)
    run.add_argument("--variant", choices=VARIANTS, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--adam-steps", type=int, required=True)
    run.add_argument("--output-root", type=Path)

    campaign = subparsers.add_parser(
        "campaign", help="run all variants sequentially in isolated subprocesses"
    )
    campaign.add_argument("--config", type=Path, default=default_config)
    campaign.add_argument("--variants", default=",".join(BASE_VARIANTS))
    campaign.add_argument("--seeds", default="0")
    campaign.add_argument("--adam-steps", type=int, required=True)
    campaign.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = ForwardConfig.from_json(args.config)
    if args.command == "run":
        run_training(
            config,
            variant=args.variant,
            seed=args.seed,
            adam_steps=args.adam_steps,
            output_root=args.output_root,
        )
    else:
        run_campaign(
            args.config,
            config,
            variants=_parse_variants(args.variants),
            seeds=_parse_seeds(args.seeds),
            adam_steps=args.adam_steps,
            output_root=args.output_root,
        )


if __name__ == "__main__":
    main()
