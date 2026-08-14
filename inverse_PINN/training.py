"""Curriculum trainer for modular inverse waveguide PINNs.

The module intentionally contains no plotting import. Monitoring, material
snapshots, FEM inference, checkpointing and I/O are timed outside the comparable
training duration.
"""

from __future__ import annotations

import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .adaptive import AdaptiveState, initialize_state, update_state
from .artifacts import (
    create_directory,
    environment_manifest,
    package_name,
    short_digest,
    write_csv,
    write_json,
)
from .checkpoints import save_material_checkpoint, save_pressure_checkpoint
from .config import Case, InverseConfig
from .data import InverseDataset, load_inverse_dataset, truth_sound_speed
from .losses import (
    FIELD_COMPONENTS,
    CaseContext,
    build_case_context,
    build_physics_context,
    celerity_metrics,
    material_case_gradient_norms,
    material_objective,
    material_snapshot_statistics,
    packed_pressure_objective,
    physical_pressure_prediction,
    pressure_gradient_statistics,
    pressure_metrics,
)
from .models import (
    FieldModel,
    initialize_field_model,
    initialize_material_model,
    material_sound_speed,
    pack_field_parameters,
    unpack_field_parameters,
)
from .optimizers import (
    make_field_adam,
    make_inverse_adam_step,
    make_material_adam,
    make_material_lbfgs_step,
    make_pressure_lbfgs_step,
    make_warmup_adam_step,
    sigma_train_steps,
)
from .sampling import (
    CollocationPoints,
    regular_collocation_points,
    snapshot_steps,
    sobol_collocation_points,
    stable_seed,
    uniform_collocation_points,
)
from .runtime import configure_jax_compilation_cache
from .variants import VariantSpec, parse_variant


PRESSURE_COMMON_LOSS_COLUMNS = (
    "evaluation_index", "phase", "optimizer", "step", "case_count",
    "pde", "neumann", "dtn", "data", "current_objective",
    "static_objective", "unweighted_total", "data_factor",
)
MATERIAL_LOSS_COLUMNS = (
    "phase", "optimizer", "step", "case_count", "pde_objective", "tv",
    "weighted_tv", "objective",
)
PRESSURE_GRADIENT_COLUMNS = (
    "phase", "step", "case_id", "incidence", "frequency", "mode",
    "pde_gradient_l2_norm", "neumann_gradient_l2_norm", "dtn_gradient_l2_norm",
    "data_gradient_l2_norm", "objective_gradient_l2_norm", "sigma_frozen",
)
MATERIAL_DIAGNOSTIC_COLUMNS = (
    "step", "fraction", "kind", "case_id", "pde", "pde_gradient_l2_norm",
    "tv", "tv_gradient_l2_norm", "aggregate", "aggregate_gradient_l2_norm",
    "negative_pair_fraction", "cancellation_ratio",
)
MATERIAL_COSINE_COLUMNS = (
    "step", "fraction", "row_index", "column_index", "row_case_id",
    "column_case_id", "cosine",
)
ADWEIGHT_COLUMNS = (
    "phase", "step", "scope", "case_id", "component", "lambda",
    "inverse_weight", "effective_weight", "update_index",
)
SIGMA_COLUMNS = (
    "phase", "step", "case_id", "incidence", "frequency", "mode",
    "sigma_x", "sigma_y", "frozen",
)
TIMING_COLUMNS = ("name", "seconds")
PRESSURE_METRIC_COLUMNS = (
    "case_id", "incidence", "frequency", "mode", "l2_absolute", "l2_relative",
    "h1_absolute", "h1_relative",
)


def _copy_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: jnp.array(value, copy=True), tree)


def _ready(tree: Any) -> None:
    leaves = jax.tree_util.tree_leaves(tree)
    if leaves:
        jax.block_until_ready(leaves[0])


def _float(value: Any) -> float:
    return float(np.asarray(value))


def _field_weights(
    config: InverseConfig,
    variant: VariantSpec,
    state: AdaptiveState | None,
    data_factor: float,
) -> jax.Array:
    if variant.field_adweights:
        assert state is not None
        weights = state.effective_weights
    else:
        weights = jnp.asarray(config.loss.field_weights, dtype=jnp.float32)
    return weights.at[3].multiply(float(data_factor))


def _warmup_weights(
    config: InverseConfig, variant: VariantSpec, state: AdaptiveState | None
) -> jax.Array:
    weights = _field_weights(config, variant, state, 0.0)
    return weights.at[3].set(0.0)


def _material_weights(
    count: int, variant: VariantSpec, state: AdaptiveState | None
) -> jax.Array:
    if variant.material_adweights:
        assert state is not None
        return state.effective_weights
    return jnp.ones((count,), dtype=jnp.float32)


def _data_factor(config: InverseConfig, step: int) -> float:
    initial = config.optimization.data_initial_factor
    transition = config.optimization.data_transition_steps
    if transition == 1:
        progress = 1.0
    else:
        progress = min(max((step - 1) / (transition - 1), 0.0), 1.0)
    return initial + (1.0 - initial) * progress


def _pressure_loss_text(components: Mapping[str, float]) -> str:
    boundary = float(components["neumann"]) + float(components["dtn"])
    return (
        f"global={float(components['current_objective']):.3e} "
        f"pde={float(components['pde']):.3e} "
        f"boundary={boundary:.3e} "
        f"data={float(components['data']):.3e}"
    )


def _initial_field_adweight(config: InverseConfig) -> AdaptiveState:
    options = config.loss.field_adweights
    return initialize_state(
        options.initial_lambdas,
        epsilon=options.epsilon,
        custom_weights=options.custom_weights,
    )


def _initial_material_adweight(config: InverseConfig, count: int) -> AdaptiveState:
    options = config.loss.material_adweights
    return initialize_state(
        [options.initial_lambda] * count,
        epsilon=options.epsilon,
        custom_weights=[options.custom_weight] * count,
    )


def _append_adweight_rows(
    rows: list[dict[str, Any]],
    *,
    phase: str,
    step: int,
    scope: str,
    state: AdaptiveState,
    labels: Sequence[str],
    case_id: str,
) -> None:
    for index, label in enumerate(labels):
        rows.append(
            {
                "phase": phase,
                "step": step,
                "scope": scope,
                "case_id": case_id,
                "component": label,
                "lambda": _float(state.lambdas[index]),
                "inverse_weight": _float(state.inverse_weights[index]),
                "effective_weight": _float(state.effective_weights[index]),
                "update_index": state.update_index,
            }
        )


def _append_packed_sigma_rows(
    rows: list[dict[str, Any]],
    phase: str,
    step: int,
    cases: Sequence[Case],
    packed_params: Mapping[str, Any],
    frozen: bool,
) -> None:
    sigma_values = np.asarray(packed_params["sigma"])
    for case, sigma in zip(cases, sigma_values):
        rows.append(
            {
                "phase": phase,
                "step": step,
                "case_id": case.id,
                "incidence": case.incidence,
                "frequency": case.frequency,
                "mode": case.mode,
                "sigma_x": float(sigma[0]),
                "sigma_y": float(sigma[1]),
                "frozen": int(frozen),
            }
        )


def _make_pressure_monitor(
    physics,
    contexts: Sequence[CaseContext],
    variant: VariantSpec,
    points: CollocationPoints,
    static_weights: jax.Array,
    *,
    include_data: bool = True,
    homogeneous_material: bool = False,
):
    contexts = tuple(contexts)
    static_weights = jnp.asarray(static_weights, dtype=jnp.float32)

    @jax.jit
    def monitor(
        packed_field_params, material_params, current_weights, b_bases
    ):
        current_objective, components = packed_pressure_objective(
            packed_field_params,
            material_params,
            physics,
            contexts,
            b_bases,
            variant,
            points,
            current_weights,
            include_data=include_data,
            homogeneous_material=homogeneous_material,
        )
        static_objective = sum(
            static_weights[index] * components[name]
            for index, name in enumerate(FIELD_COMPONENTS)
        )
        return {
            **components,
            "current_objective": current_objective,
            "static_objective": static_objective,
        }

    return monitor


def _evaluate_pressure_monitor(
    monitor,
    packed_field_params,
    material_params,
    current_weights,
    b_bases,
) -> dict[str, float]:
    values = monitor(
        packed_field_params,
        material_params,
        jnp.stack(tuple(current_weights)),
        b_bases,
    )
    _ready(values)
    return {name: _float(value) for name, value in values.items()}


def _make_material_monitor(
    physics,
    contexts: Sequence[CaseContext],
    variant: VariantSpec,
    points: CollocationPoints,
    *,
    tv_weight: float,
    tv_epsilon_squared: float,
):
    contexts = tuple(contexts)

    @jax.jit
    def monitor(material_params, packed_field_params, case_weights, b_bases):
        dynamic_contexts = tuple(
            replace(context, b_base=b_base)
            for context, b_base in zip(contexts, b_bases)
        )
        _, components = material_objective(
            material_params,
            unpack_field_parameters(packed_field_params),
            physics,
            dynamic_contexts,
            variant,
            points,
            case_weights,
            tv_weight=tv_weight,
            tv_epsilon_squared=tv_epsilon_squared,
        )
        return {
            name: components[name]
            for name in ("pde_objective", "tv", "weighted_tv", "objective")
        }

    return monitor


def _append_pressure_common_loss_row(
    rows: list[dict[str, Any]],
    *,
    phase: str,
    optimizer: str,
    step: int,
    case_count: int,
    components: Mapping[str, float],
    data_factor: float,
) -> None:
    rows.append(
        {
            "evaluation_index": len(rows),
            "phase": phase,
            "optimizer": optimizer,
            "step": step,
            "case_count": case_count,
            **{
                name: float(components[name])
                for name in (
                    "pde", "neumann", "dtn", "data", "unweighted_total",
                    "current_objective", "static_objective",
                )
            },
            "data_factor": float(data_factor),
        }
    )


def _append_material_loss_row(
    rows: list[dict[str, Any]],
    *,
    phase: str,
    optimizer: str,
    step: int,
    case_count: int,
    components: Mapping[str, Any],
    tv_enabled: bool,
) -> None:
    rows.append(
        {
            "phase": phase,
            "optimizer": optimizer,
            "step": step,
            "case_count": case_count,
            "pde_objective": _float(components["pde_objective"]),
            "tv": _float(components["tv"]) if tv_enabled else "",
            "weighted_tv": _float(components["weighted_tv"]) if tv_enabled else "",
            "objective": _float(components["objective"]),
        }
    )


def _snapshot_grid(config: InverseConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nx, ny = config.logging.snapshot_grid
    x = np.linspace(-config.geometry.half_length, config.geometry.half_length, nx)
    y = np.linspace(0.0, config.geometry.height, ny)
    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
    return x, y, x_grid, y_grid


def _predict_celerity(
    material_params: Mapping[str, Any], config: InverseConfig, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    batch_size = config.logging.prediction_batch_size
    x_normalized = np.asarray(x).ravel() / config.geometry.half_length
    y_normalized = 2.0 * np.asarray(y).ravel() / config.geometry.height - 1.0
    chunks = []
    function = jax.jit(
        lambda candidate, xv, yv: jax.vmap(
            lambda x_value, y_value: material_sound_speed(
                candidate, config.geometry, x_value, y_value
            )
        )(xv, yv)
    )
    for start in range(0, x_normalized.size, batch_size):
        values = function(
            material_params,
            jnp.asarray(x_normalized[start : start + batch_size], dtype=jnp.float32),
            jnp.asarray(y_normalized[start : start + batch_size], dtype=jnp.float32),
        )
        chunks.append(np.asarray(values))
    return np.concatenate(chunks).reshape(np.asarray(x).shape)


def _predict_pressure(
    params: Mapping[str, Any],
    physics,
    context: CaseContext,
    variant: VariantSpec,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    function = jax.jit(
        lambda candidate, b_base, xv, yv: physical_pressure_prediction(
            candidate, physics, replace(context, b_base=b_base), variant, xv, yv
        )
    )
    chunks = []
    for start in range(0, x.size, batch_size):
        values = function(
            params,
            context.b_base,
            jnp.asarray(x[start : start + batch_size], dtype=jnp.float32),
            jnp.asarray(y[start : start + batch_size], dtype=jnp.float32),
        )
        chunks.append(np.asarray(values))
    return np.concatenate(chunks)


def _run_warmup(
    *,
    cases: Sequence[Case],
    field_models: Mapping[Case, FieldModel],
    material_params: Mapping[str, Any],
    physics,
    contexts: Sequence[CaseContext],
    config: InverseConfig,
    variant: VariantSpec,
    package_index: int,
    homogeneous_material: bool,
    pressure_common_rows: list[dict[str, Any]],
    pressure_gradient_rows: list[dict[str, Any]],
    adweight_rows: list[dict[str, Any]],
    sigma_rows: list[dict[str, Any]],
) -> tuple[dict[Case, FieldModel], float, float]:
    cases = tuple(cases)
    contexts = tuple(contexts)
    if not cases or len(cases) != len(contexts):
        raise ValueError("Packed warmup requires aligned non-empty cases and contexts")
    budget = config.training_packages[package_index].warmup
    packed_params = pack_field_parameters(
        tuple(field_models[case].params for case in cases)
    )
    b_bases = jnp.stack(tuple(field_models[case].b_base for case in cases))
    training_seconds = 0.0
    compilation_seconds = 0.0
    adaptive = {
        case: _initial_field_adweight(config) for case in cases
    } if variant.field_adweights else {}
    final_static = jnp.asarray(config.loss.field_weights, dtype=jnp.float32).at[3].set(0.0)
    monitor_points = regular_collocation_points(
        config.sampling.monitor,
        half_length=config.geometry.half_length,
        height=config.geometry.height,
    )

    def current_weights() -> tuple[jax.Array, ...]:
        return tuple(
            _warmup_weights(config, variant, adaptive.get(case)) for case in cases
        )

    pressure_monitor = _make_pressure_monitor(
        physics,
        contexts,
        variant,
        monitor_points,
        final_static,
        include_data=False,
        homogeneous_material=homogeneous_material,
    )

    if budget.adam_steps:
        optimizer = make_field_adam(
            packed_params, config.optimization, budget.adam_steps
        )
        state = optimizer.init(packed_params)
        train_steps = sigma_train_steps(config.optimization, budget.adam_steps)
        unfrozen_step = make_warmup_adam_step(
            optimizer, physics, contexts, variant, config.sampling.adam,
            homogeneous_material=homogeneous_material, freeze_sigma=False,
        )
        frozen_step = make_warmup_adam_step(
            optimizer, physics, contexts, variant, config.sampling.adam,
            homogeneous_material=homogeneous_material, freeze_sigma=True,
        )
        compile_key = jax.random.key(
            stable_seed("compile", package_index, "warmup", len(cases))
        )
        compile_started = time.perf_counter()
        compiled = unfrozen_step(
            packed_params, material_params, state, compile_key,
            jnp.stack(current_weights()), b_bases,
        )
        _ready(compiled)
        if train_steps < budget.adam_steps:
            compiled = frozen_step(
                packed_params, material_params, state, compile_key,
                jnp.stack(current_weights()), b_bases,
        )
        _ready(compiled)
        compilation_seconds += time.perf_counter() - compile_started

        training_block_started = time.perf_counter()
        for step in range(1, budget.adam_steps + 1):
            collocation_key = jax.random.key(
                stable_seed(package_index, "packed_warmup", step)
            )
            points = None
            frozen = step > train_steps
            if (
                variant.field_adweights
                and (step - 1) % config.loss.field_adweights.update_interval_adam == 0
            ):
                points = uniform_collocation_points(
                    collocation_key, config.sampling.adam
                )
                unpacked_params = unpack_field_parameters(packed_params)
                for case, context, params in zip(cases, contexts, unpacked_params):
                    weights = _warmup_weights(config, variant, adaptive[case])
                    statistics = pressure_gradient_statistics(
                        params, material_params, physics, context, variant,
                        points, weights, freeze_sigma=frozen,
                    )
                    norms = jnp.asarray(
                        [statistics[f"{name}_gradient_l2_norm"] for name in FIELD_COMPONENTS]
                    )
                    if step > 1:
                        options = config.loss.field_adweights
                        adaptive[case] = update_state(
                            adaptive[case], norms, epsilon=options.epsilon,
                            alpha=options.alpha,
                            custom_weights=options.custom_weights,
                        )
                    _append_adweight_rows(
                        adweight_rows, phase="warmup_adam", step=step,
                        scope="pressure", state=adaptive[case],
                        labels=FIELD_COMPONENTS, case_id=case.id,
                    )
            weights = current_weights()
            function = frozen_step if frozen else unfrozen_step
            packed_params, state, _ = function(
                packed_params, material_params, state, collocation_key,
                jnp.stack(weights), b_bases,
            )
            print_event = step % config.logging.print_interval_adam == 0 or step == budget.adam_steps
            monitor_event = (
                step == 1 or step == budget.adam_steps
                or step % config.logging.loss_interval_adam == 0 or print_event
            )
            sigma_event = (
                step == 1 or step == budget.adam_steps
                or step % config.logging.sigma_interval_adam == 0
            )
            gradient_event = (
                step % config.logging.pressure_gradient_interval_adam == 0
                or step == budget.adam_steps
            )
            if monitor_event or sigma_event or gradient_event:
                _ready(packed_params)
                training_seconds += time.perf_counter() - training_block_started
            if monitor_event:
                values = _evaluate_pressure_monitor(
                    pressure_monitor, packed_params, material_params,
                    weights, b_bases,
                )
                _append_pressure_common_loss_row(
                    pressure_common_rows, phase="warmup_adam", optimizer="adam",
                    step=step, case_count=len(cases), components=values,
                    data_factor=0.0,
                )
                if print_event:
                    print(
                        f"[{variant.name} pkg{package_index + 1:02d} warmup "
                        f"packed Adam {step}/{budget.adam_steps}] "
                        f"{_pressure_loss_text(values)}",
                        flush=True,
                    )
            if sigma_event:
                _append_packed_sigma_rows(
                    sigma_rows, "warmup_adam", step, cases, packed_params, frozen
                )
            if gradient_event:
                if points is None:
                    points = uniform_collocation_points(
                        collocation_key, config.sampling.adam
                    )
                unpacked_params = unpack_field_parameters(packed_params)
                for case, context, params, case_weights in zip(
                    cases, contexts, unpacked_params, weights
                ):
                    statistics = pressure_gradient_statistics(
                        params, material_params, physics, context, variant,
                        points, case_weights, freeze_sigma=frozen,
                    )
                    pressure_gradient_rows.append(
                        {
                            "phase": "warmup_adam", "step": step,
                            "case_id": case.id, "incidence": case.incidence,
                            "frequency": case.frequency, "mode": case.mode,
                            **{name: _float(value) for name, value in statistics.items()},
                            "sigma_frozen": int(frozen),
                        }
                    )
            if monitor_event or sigma_event or gradient_event:
                training_block_started = time.perf_counter()

    if budget.lbfgs_steps:
        points = sobol_collocation_points(
            config.sampling.lbfgs,
            scramble=config.sampling.sobol_scramble,
            seed=stable_seed(
                package_index, "packed_warmup_lbfgs",
                config.sampling.sobol_seed_offset,
            ),
        )
        frozen_weights = current_weights()
        optimizer = optax.lbfgs()
        state = optimizer.init(packed_params)
        step_function = make_pressure_lbfgs_step(
            optimizer, physics, contexts, variant, points,
            frozen_weights, include_data=False,
            homogeneous_material=homogeneous_material,
        )
        pressure_monitor = _make_pressure_monitor(
            physics,
            contexts,
            variant,
            monitor_points,
            final_static,
            include_data=False,
            homogeneous_material=homogeneous_material,
        )
        compile_started = time.perf_counter()
        compiled = step_function(
            packed_params, state, material_params, b_bases
        )
        _ready(compiled)
        compilation_seconds += time.perf_counter() - compile_started
        training_block_started = time.perf_counter()
        for step in range(1, budget.lbfgs_steps + 1):
            packed_params, state, _ = step_function(
                packed_params, state, material_params, b_bases
            )
            monitor_event = (
                step == 1 or step == budget.lbfgs_steps
                or step % config.logging.loss_interval_lbfgs == 0
            )
            if monitor_event:
                _ready(packed_params)
                training_seconds += time.perf_counter() - training_block_started
                values = _evaluate_pressure_monitor(
                    pressure_monitor, packed_params, material_params,
                    frozen_weights, b_bases,
                )
                _append_pressure_common_loss_row(
                    pressure_common_rows, phase="warmup_lbfgs",
                    optimizer="lbfgs", step=step, components=values,
                    case_count=len(cases), data_factor=0.0,
                )
                print(
                    f"[{variant.name} pkg{package_index + 1:02d} warmup "
                    f"packed L-BFGS {step}/{budget.lbfgs_steps}] "
                    f"{_pressure_loss_text(values)}",
                    flush=True,
                )
                training_block_started = time.perf_counter()
    unpacked_params = unpack_field_parameters(packed_params)
    result = {
        case: FieldModel(params, field_models[case].b_base)
        for case, params in zip(cases, unpacked_params)
    }
    return result, training_seconds, compilation_seconds


def _final_metrics(
    package_directory: Path,
    active: Sequence[Case],
    models: Mapping[Case, FieldModel],
    contexts: Mapping[Case, CaseContext],
    material_params: Mapping[str, Any],
    physics,
    variant: VariantSpec,
    config: InverseConfig,
    dataset: InverseDataset,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pressure_rows = []
    for case in active:
        fem = dataset.fem_cases[case]
        prediction = _predict_pressure(
            models[case].params, physics, contexts[case], variant, fem.x, fem.y,
            config.logging.prediction_batch_size,
        )
        metrics = pressure_metrics(fem.values, prediction, dataset.mass, dataset.stiffness)
        pressure_rows.append(
            {
                "case_id": case.id,
                "incidence": case.incidence,
                "frequency": case.frequency,
                "mode": case.mode,
                **asdict(metrics),
            }
        )
    _, _, x_grid, y_grid = _snapshot_grid(config)
    prediction_c = _predict_celerity(material_params, config, x_grid, y_grid)
    truth_c = truth_sound_speed(config.geometry, x_grid, y_grid)
    material_metrics = asdict(celerity_metrics(truth_c, prediction_c, config.geometry.c0))
    write_csv(package_directory / "pressure_metrics.csv", pressure_rows, PRESSURE_METRIC_COLUMNS)
    write_json(package_directory / "celerity_metrics.json", material_metrics)
    return pressure_rows, material_metrics


def run_training(
    config: InverseConfig,
    variant_name: str,
    seed: int,
    *,
    output_parent: Path | None = None,
) -> Path:
    """Run one variant/seed and return its newly-created directory."""
    cache_directory = configure_jax_compilation_cache()
    variant = parse_variant(variant_name)
    seed = int(seed)
    dataset = load_inverse_dataset(config)
    physics = build_physics_context(config.geometry, config.all_cases)
    parent = Path(output_parent) if output_parent is not None else config.output_root / config.source.stem
    parent.mkdir(parents=True, exist_ok=True)
    run_directory = create_directory(parent / f"{variant.name}_seed{seed}")
    packages_directory = run_directory / "packages"
    packages_directory.mkdir()

    warning_ratios = [
        region.speed_ratio for region in config.geometry.truth_regions
        if not config.geometry.celerity_ratio_bounds[0] <= region.speed_ratio <= config.geometry.celerity_ratio_bounds[1]
    ]
    manifest = {
        "format_version": 2,
        "config": config.manifest(),
        "config_digest": short_digest(config.manifest()),
        "variant": asdict(variant),
        "seed": seed,
        "environment": environment_manifest(),
        "jax_compilation_cache": {
            "directory": str(cache_directory),
            "persistent": True,
        },
        "warnings": (
            [f"Truth speed ratios outside admissible bounds: {warning_ratios}"]
            if warning_ratios else []
        ),
    }
    write_json(run_directory / "manifest.json", manifest)

    models: dict[Case, FieldModel] = {}
    for case in config.all_cases:
        models[case] = initialize_field_model(
            jax.random.key(stable_seed(seed, "pressure", case.id)),
            config.models, config.geometry, case, variant,
        )
    material_params = initialize_material_model(
        jax.random.key(stable_seed(seed, "material")), config.models, config.geometry
    )
    contexts = {
        case: build_case_context(
            physics, case, models[case].b_base, dataset.boundaries[case], variant
        )
        for case in config.all_cases
    }
    field_scales = {case: contexts[case].field_scale for case in config.all_cases}
    monitor_points = regular_collocation_points(
        config.sampling.monitor,
        half_length=config.geometry.half_length,
        height=config.geometry.height,
    )
    final_static_weights = jnp.asarray(config.loss.field_weights, dtype=jnp.float32)
    seen: set[Case] = set()
    run_summaries: list[dict[str, Any]] = []

    for package_index, package in enumerate(config.training_packages):
        package_wall_started = time.perf_counter()
        package_directory = create_directory(
            packages_directory / package_name(package_index, package.label)
        )
        active = tuple(package.cases)
        active_contexts = tuple(contexts[case] for case in active)
        new_cases = tuple(case for case in active if case not in seen)
        pressure_common_rows: list[dict[str, Any]] = []
        material_rows: list[dict[str, Any]] = []
        pressure_gradient_rows: list[dict[str, Any]] = []
        material_diagnostic_rows: list[dict[str, Any]] = []
        cosine_rows: list[dict[str, Any]] = []
        adweight_rows: list[dict[str, Any]] = []
        sigma_rows: list[dict[str, Any]] = []
        snapshots: list[np.ndarray] = []
        snapshot_step_values: list[int] = []
        snapshot_fraction_values: list[float] = []
        training_seconds = 0.0
        compilation_seconds = 0.0

        do_first_total_warmup = package_index == 0 and not variant.scattered
        do_previous_map_pretrain = package_index > 0
        if new_cases and (do_first_total_warmup or do_previous_map_pretrain):
            warmed_models, elapsed, compiled = _run_warmup(
                cases=new_cases,
                field_models={case: models[case] for case in new_cases},
                material_params=material_params,
                physics=physics,
                contexts=tuple(contexts[case] for case in new_cases),
                config=config,
                variant=variant,
                package_index=package_index,
                homogeneous_material=do_first_total_warmup,
                pressure_common_rows=pressure_common_rows,
                pressure_gradient_rows=pressure_gradient_rows,
                adweight_rows=adweight_rows,
                sigma_rows=sigma_rows,
            )
            models.update(warmed_models)
            training_seconds += elapsed
            compilation_seconds += compiled

        packed_field_params = pack_field_parameters(
            tuple(models[case].params for case in active)
        )
        b_bases = jnp.stack(tuple(models[case].b_base for case in active))
        field_adaptive = {
            case: _initial_field_adweight(config) for case in active
        } if variant.field_adweights else {}
        material_adaptive = (
            _initial_material_adweight(config, len(active))
            if variant.material_adweights else None
        )
        adam_steps = package.inverse.adam_steps
        initial_data_factor = (
            config.optimization.data_initial_factor if adam_steps else 1.0
        )
        initial_field_weights = tuple(
            _field_weights(
                config, variant, field_adaptive.get(case), initial_data_factor
            )
            for case in active
        )
        pressure_monitor = _make_pressure_monitor(
            physics, active_contexts, variant, monitor_points,
            final_static_weights,
        )
        best_components = _evaluate_pressure_monitor(
            pressure_monitor, packed_field_params, material_params,
            initial_field_weights, b_bases,
        )
        best_selection = best_components["static_objective"]
        _append_pressure_common_loss_row(
            pressure_common_rows, phase="inverse_initial", optimizer="none",
            step=0, case_count=len(active), components=best_components,
            data_factor=initial_data_factor,
        )
        _append_packed_sigma_rows(
            sigma_rows, "inverse_initial", 0, active, packed_field_params,
            frozen=adam_steps == 0,
        )
        for case in active:
            if variant.field_adweights:
                _append_adweight_rows(
                    adweight_rows, phase="inverse_initial", step=0,
                    scope="pressure", state=field_adaptive[case],
                    labels=FIELD_COMPONENTS, case_id=case.id,
                )
        if variant.material_adweights:
            assert material_adaptive is not None
            _append_adweight_rows(
                adweight_rows, phase="inverse_initial", step=0,
                scope="material", state=material_adaptive,
                labels=[case.id for case in active], case_id="",
            )
        best_fields = _copy_tree(packed_field_params)
        best_material = _copy_tree(material_params)

        tv_weight = config.loss.tv.weight if variant.total_variation else 0.0
        material_monitor = _make_material_monitor(
            physics, active_contexts, variant, monitor_points,
            tv_weight=tv_weight,
            tv_epsilon_squared=config.loss.tv.epsilon_squared,
        )
        if adam_steps:
            field_optimizer = make_field_adam(
                packed_field_params, config.optimization, adam_steps
            )
            field_state = field_optimizer.init(packed_field_params)
            material_optimizer = make_material_adam(config.optimization)
            material_state = material_optimizer.init(material_params)
            sigma_steps = sigma_train_steps(config.optimization, adam_steps)
            unfrozen_step = make_inverse_adam_step(
                field_optimizer, material_optimizer, physics, active_contexts,
                variant, config.sampling.adam,
                freeze_sigma=False, tv_weight=tv_weight,
                tv_epsilon_squared=config.loss.tv.epsilon_squared,
            )
            frozen_step = make_inverse_adam_step(
                field_optimizer, material_optimizer, physics, active_contexts,
                variant, config.sampling.adam,
                freeze_sigma=True, tv_weight=tv_weight,
                tv_epsilon_squared=config.loss.tv.epsilon_squared,
            )
            compile_key = jax.random.key(
                stable_seed("compile", package_index, "inverse")
            )
            compile_field_weights = tuple(
                _field_weights(config, variant, field_adaptive.get(case), config.optimization.data_initial_factor)
                for case in active
            )
            compile_material_weights = _material_weights(
                len(active), variant, material_adaptive
            )
            compile_started = time.perf_counter()
            compiled = unfrozen_step(
                packed_field_params, material_params, field_state, material_state,
                compile_key, jnp.stack(compile_field_weights),
                compile_material_weights,
                config.optimization.data_initial_factor, b_bases,
            )
            _ready(compiled)
            if sigma_steps < adam_steps:
                compiled = frozen_step(
                    packed_field_params, material_params, field_state,
                    material_state, compile_key,
                    jnp.stack(compile_field_weights), compile_material_weights,
                    config.optimization.data_initial_factor, b_bases,
                )
                _ready(compiled)
            compilation_seconds += time.perf_counter() - compile_started
            configured_snapshots = snapshot_steps(
                adam_steps, config.logging.material_snapshot_fractions
            )

            training_block_started = time.perf_counter()
            for step in range(1, adam_steps + 1):
                collocation_key = jax.random.key(
                    stable_seed(seed, package_index, "inverse", step)
                )
                points = None
                frozen = step > sigma_steps
                data_factor = _data_factor(config, step)
                field_adweight_event = (
                    variant.field_adweights
                    and (step - 1)
                    % config.loss.field_adweights.update_interval_adam == 0
                )
                material_adweight_event = (
                    variant.material_adweights
                    and (step - 1)
                    % config.loss.material_adweights.update_interval_adam == 0
                )
                field_params = None
                if field_adweight_event or material_adweight_event:
                    field_params = unpack_field_parameters(packed_field_params)

                if field_adweight_event:
                    assert field_params is not None
                    points = uniform_collocation_points(
                        collocation_key, config.sampling.adam
                    )
                    for index, (case, context) in enumerate(zip(active, active_contexts)):
                        current_weights = _field_weights(
                            config, variant, field_adaptive[case], data_factor
                        )
                        statistics = pressure_gradient_statistics(
                            field_params[index], material_params, physics, context,
                            variant, points, current_weights, freeze_sigma=frozen,
                        )
                        norms = jnp.asarray(
                            [statistics[f"{name}_gradient_l2_norm"] for name in FIELD_COMPONENTS]
                        )
                        options = config.loss.field_adweights
                        if step > 1:
                            field_adaptive[case] = update_state(
                                field_adaptive[case], norms, epsilon=options.epsilon,
                                alpha=options.alpha, custom_weights=options.custom_weights,
                            )
                        _append_adweight_rows(
                            adweight_rows, phase="inverse_adam", step=step,
                            scope="pressure", state=field_adaptive[case],
                            labels=FIELD_COMPONENTS, case_id=case.id,
                        )

                if material_adweight_event:
                    assert material_adaptive is not None
                    assert field_params is not None
                    if points is None:
                        points = uniform_collocation_points(
                            collocation_key, config.sampling.adam
                        )
                    norms = material_case_gradient_norms(
                        material_params, field_params, physics, active_contexts,
                        variant, points,
                    )
                    options = config.loss.material_adweights
                    if step > 1:
                        material_adaptive = update_state(
                            material_adaptive, norms, epsilon=options.epsilon,
                            alpha=options.alpha,
                            custom_weights=[options.custom_weight] * len(active),
                        )
                    _append_adweight_rows(
                        adweight_rows, phase="inverse_adam", step=step,
                        scope="material", state=material_adaptive,
                        labels=[case.id for case in active], case_id="",
                    )

                field_weights = tuple(
                    _field_weights(config, variant, field_adaptive.get(case), data_factor)
                    for case in active
                )
                material_weights = _material_weights(
                    len(active), variant, material_adaptive
                )
                function = frozen_step if frozen else unfrozen_step
                (
                    packed_field_params, material_params, field_state, material_state,
                    _, _,
                ) = function(
                    packed_field_params, material_params, field_state,
                    material_state, collocation_key, jnp.stack(field_weights),
                    material_weights, data_factor, b_bases,
                )

                print_event = step % config.logging.print_interval_adam == 0 or step == adam_steps
                monitor_event = (
                    step == 1 or step == adam_steps
                    or step % config.logging.loss_interval_adam == 0 or print_event
                )
                sigma_event = (
                    step == 1 or step == adam_steps
                    or step % config.logging.sigma_interval_adam == 0
                )
                gradient_event = (
                    step % config.logging.pressure_gradient_interval_adam == 0
                    or step == adam_steps
                )
                snapshot_event = step in configured_snapshots
                if monitor_event or sigma_event or gradient_event or snapshot_event:
                    _ready((packed_field_params, material_params))
                    training_seconds += (
                        time.perf_counter() - training_block_started
                    )
                if monitor_event:
                    components = _evaluate_pressure_monitor(
                        pressure_monitor, packed_field_params, material_params,
                        field_weights, b_bases,
                    )
                    _append_pressure_common_loss_row(
                        pressure_common_rows, phase="inverse_adam",
                        optimizer="adam", step=step, components=components,
                        case_count=len(active), data_factor=data_factor,
                    )
                    material_components = material_monitor(
                        material_params, packed_field_params,
                        material_weights, b_bases,
                    )
                    _ready(material_components)
                    _append_material_loss_row(
                        material_rows, phase="inverse_adam", optimizer="adam",
                        step=step, case_count=len(active),
                        components=material_components,
                        tv_enabled=tv_weight != 0.0,
                    )
                    if components["static_objective"] < best_selection:
                        best_selection = components["static_objective"]
                        best_fields = _copy_tree(packed_field_params)
                        best_material = _copy_tree(material_params)

                if sigma_event:
                    _append_packed_sigma_rows(
                        sigma_rows, "inverse_adam", step, active,
                        packed_field_params, frozen,
                    )

                if gradient_event:
                    if points is None:
                        points = uniform_collocation_points(
                            collocation_key, config.sampling.adam
                        )
                    field_params = unpack_field_parameters(packed_field_params)
                    for case, context, params, weights in zip(
                        active, active_contexts, field_params, field_weights
                    ):
                        statistics = pressure_gradient_statistics(
                            params, material_params, physics, context, variant,
                            points, weights, freeze_sigma=frozen,
                        )
                        pressure_gradient_rows.append(
                            {
                                "phase": "inverse_adam", "step": step,
                                "case_id": case.id, "incidence": case.incidence,
                                "frequency": case.frequency, "mode": case.mode,
                                **{name: _float(value) for name, value in statistics.items()},
                                "sigma_frozen": int(frozen),
                            }
                        )

                if snapshot_event:
                    fraction = configured_snapshots[step]
                    _, _, x_grid, y_grid = _snapshot_grid(config)
                    snapshots.append(_predict_celerity(material_params, config, x_grid, y_grid))
                    snapshot_step_values.append(step)
                    snapshot_fraction_values.append(fraction)
                    field_params = unpack_field_parameters(packed_field_params)
                    diagnostics = material_snapshot_statistics(
                        material_params, field_params, physics, active_contexts,
                        variant, monitor_points, material_weights,
                        tv_weight=tv_weight,
                        tv_epsilon_squared=config.loss.tv.epsilon_squared,
                    )
                    pdes = np.asarray(diagnostics["pde_values"])
                    norms = np.asarray(diagnostics["pde_gradient_norms"])
                    for case, pde, norm in zip(active, pdes, norms):
                        material_diagnostic_rows.append(
                            {
                                "step": step, "fraction": fraction, "kind": "pde",
                                "case_id": case.id, "pde": float(pde),
                                "pde_gradient_l2_norm": float(norm), "tv": "",
                                "tv_gradient_l2_norm": "", "aggregate": "",
                                "aggregate_gradient_l2_norm": "",
                                "negative_pair_fraction": "", "cancellation_ratio": "",
                            }
                        )
                    material_diagnostic_rows.append(
                        {
                            "step": step, "fraction": fraction, "kind": "aggregate",
                            "case_id": "", "pde": "", "pde_gradient_l2_norm": "",
                            "tv": (
                                _float(diagnostics["tv"])
                                if diagnostics["tv"] is not None else ""
                            ),
                            "tv_gradient_l2_norm": (
                                _float(diagnostics["tv_gradient_norm"])
                                if diagnostics["tv_gradient_norm"] is not None else ""
                            ),
                            "aggregate": _float(diagnostics["aggregate"]),
                            "aggregate_gradient_l2_norm": _float(diagnostics["aggregate_gradient_norm"]),
                            "negative_pair_fraction": _float(diagnostics["negative_pair_fraction"]),
                            "cancellation_ratio": _float(diagnostics["cancellation_ratio"]),
                        }
                    )
                    cosines = np.asarray(diagnostics["cosines"])
                    for row_index, row_case in enumerate(active):
                        for column_index, column_case in enumerate(active):
                            cosine_rows.append(
                                {
                                    "step": step, "fraction": fraction,
                                    "row_index": row_index, "column_index": column_index,
                                    "row_case_id": row_case.id,
                                    "column_case_id": column_case.id,
                                    "cosine": float(cosines[row_index, column_index]),
                                }
                            )

                if print_event:
                    print(
                        f"[{variant.name} pkg{package_index + 1:02d} "
                        f"packed Adam {step}/{adam_steps}] "
                        f"{_pressure_loss_text(components)} "
                        f"monitor={components['static_objective']:.3e} "
                        f"material={_float(material_components['objective']):.3e} "
                        f"best={best_selection:.3e} data_factor={data_factor:.3f}",
                        flush=True,
                    )
                if monitor_event or sigma_event or gradient_event or snapshot_event:
                    training_block_started = time.perf_counter()

        frozen_field_weights = tuple(
            _field_weights(config, variant, field_adaptive.get(case), 1.0)
            for case in active
        )
        frozen_material_weights = _material_weights(
            len(active), variant, material_adaptive
        )
        lbfgs_points = sobol_collocation_points(
            config.sampling.lbfgs,
            scramble=config.sampling.sobol_scramble,
            seed=stable_seed(seed, package_index, "inverse_lbfgs", config.sampling.sobol_seed_offset),
        )
        global_lbfgs_step = 0
        for cycle in range(1, package.inverse.lbfgs_cycles + 1):
            if package.inverse.lbfgs_field_steps:
                optimizer = optax.lbfgs()
                state = optimizer.init(packed_field_params)
                function = make_pressure_lbfgs_step(
                    optimizer, physics, active_contexts, variant,
                    lbfgs_points, frozen_field_weights, include_data=True,
                )
                compile_started = time.perf_counter()
                compiled = function(
                    packed_field_params, state, material_params, b_bases
                )
                _ready(compiled)
                compilation_seconds += time.perf_counter() - compile_started
                training_block_started = time.perf_counter()
                for block_step in range(1, package.inverse.lbfgs_field_steps + 1):
                    global_lbfgs_step += 1
                    packed_field_params, state, _ = function(
                        packed_field_params, state, material_params, b_bases
                    )
                    monitor_event = (
                        block_step == package.inverse.lbfgs_field_steps
                        or block_step % config.logging.loss_interval_lbfgs == 0
                    )
                    if monitor_event:
                        _ready(packed_field_params)
                        training_seconds += (
                            time.perf_counter() - training_block_started
                        )
                        components = _evaluate_pressure_monitor(
                            pressure_monitor, packed_field_params,
                            material_params, frozen_field_weights, b_bases,
                        )
                        _append_pressure_common_loss_row(
                            pressure_common_rows,
                            phase=f"inverse_lbfgs_field_cycle{cycle}",
                            optimizer="lbfgs", step=global_lbfgs_step,
                            components=components, case_count=len(active),
                            data_factor=1.0,
                        )
                        if components["static_objective"] < best_selection:
                            best_selection = components["static_objective"]
                            best_fields = _copy_tree(packed_field_params)
                            best_material = _copy_tree(material_params)
                        print(
                            f"[{variant.name} pkg{package_index + 1:02d} "
                            f"pressure L-BFGS cycle {cycle} "
                            f"{block_step}/{package.inverse.lbfgs_field_steps}] "
                            f"{_pressure_loss_text(components)} "
                            f"best={best_selection:.3e}",
                            flush=True,
                        )
                        training_block_started = time.perf_counter()

            if package.inverse.lbfgs_material_steps:
                optimizer = optax.lbfgs()
                state = optimizer.init(material_params)
                function = make_material_lbfgs_step(
                    optimizer, physics, active_contexts, variant,
                    lbfgs_points, frozen_material_weights,
                    tv_weight=tv_weight,
                    tv_epsilon_squared=config.loss.tv.epsilon_squared,
                )
                compile_started = time.perf_counter()
                compiled = function(
                    material_params, state, packed_field_params, b_bases
                )
                _ready(compiled)
                compilation_seconds += time.perf_counter() - compile_started
                training_block_started = time.perf_counter()
                for block_step in range(1, package.inverse.lbfgs_material_steps + 1):
                    global_lbfgs_step += 1
                    material_params, state, _ = function(
                        material_params, state, packed_field_params, b_bases
                    )
                    monitor_event = (
                        block_step == package.inverse.lbfgs_material_steps
                        or block_step % config.logging.loss_interval_lbfgs == 0
                    )
                    if monitor_event:
                        _ready(material_params)
                        training_seconds += (
                            time.perf_counter() - training_block_started
                        )
                        components = _evaluate_pressure_monitor(
                            pressure_monitor, packed_field_params,
                            material_params, frozen_field_weights, b_bases,
                        )
                        _append_pressure_common_loss_row(
                            pressure_common_rows,
                            phase=f"inverse_lbfgs_material_cycle{cycle}",
                            optimizer="lbfgs", step=global_lbfgs_step,
                            components=components, case_count=len(active),
                            data_factor=1.0,
                        )
                        material_components = material_monitor(
                            material_params, packed_field_params,
                            frozen_material_weights, b_bases,
                        )
                        _ready(material_components)
                        _append_material_loss_row(
                            material_rows,
                            phase=f"inverse_lbfgs_material_cycle{cycle}",
                            optimizer="lbfgs", step=global_lbfgs_step,
                            case_count=len(active), components=material_components,
                            tv_enabled=tv_weight != 0.0,
                        )
                        if components["static_objective"] < best_selection:
                            best_selection = components["static_objective"]
                            best_fields = _copy_tree(packed_field_params)
                            best_material = _copy_tree(material_params)
                        print(
                            f"[{variant.name} pkg{package_index + 1:02d} "
                            f"material L-BFGS cycle {cycle} "
                            f"{block_step}/{package.inverse.lbfgs_material_steps}] "
                            f"{_pressure_loss_text(components)} "
                            f"material={_float(material_components['objective']):.3e} "
                            f"best={best_selection:.3e}",
                            flush=True,
                        )
                        training_block_started = time.perf_counter()

        packed_field_params = best_fields
        field_params = unpack_field_parameters(packed_field_params)
        material_params = best_material
        for case, params in zip(active, field_params):
            models[case] = FieldModel(params, models[case].b_base)
        seen.update(active)
        saved_models = {case: models[case] for case in sorted(seen)}
        saved_scales = {case: field_scales[case] for case in sorted(seen)}
        save_pressure_checkpoint(
            package_directory / "pressure_weights_best.npz", saved_models,
            saved_scales, variant=variant.name, package_index=package_index,
            monitor_loss=best_selection,
        )
        save_material_checkpoint(
            package_directory / "slowness_weights_best.npz", material_params,
            variant=variant.name, package_index=package_index,
            monitor_loss=best_selection,
        )
        x, y, _, _ = _snapshot_grid(config)
        maps = (
            np.stack(snapshots)
            if snapshots
            else np.empty((0, y.size, x.size), dtype=np.float32)
        )
        np.savez_compressed(
            package_directory / "celerity_snapshots.npz",
            x=x, y=y, celerity=maps,
            steps=np.asarray(snapshot_step_values, dtype=np.int64),
            fractions=np.asarray(snapshot_fraction_values, dtype=np.float64),
        )

        pressure_metric_rows, material_metric_values = _final_metrics(
            package_directory, active, models, contexts, material_params, physics,
            variant, config, dataset,
        )
        write_csv(
            package_directory / "pressure_common_loss_history.csv",
            pressure_common_rows,
            PRESSURE_COMMON_LOSS_COLUMNS,
        )
        write_csv(package_directory / "material_loss_history.csv", material_rows, MATERIAL_LOSS_COLUMNS)
        write_csv(package_directory / "pressure_gradient_history.csv", pressure_gradient_rows, PRESSURE_GRADIENT_COLUMNS)
        write_csv(package_directory / "material_snapshot_diagnostics.csv", material_diagnostic_rows, MATERIAL_DIAGNOSTIC_COLUMNS)
        write_csv(package_directory / "material_gradient_cosines.csv", cosine_rows, MATERIAL_COSINE_COLUMNS)
        write_csv(package_directory / "adweights_history.csv", adweight_rows, ADWEIGHT_COLUMNS)
        write_csv(package_directory / "sigma_history.csv", sigma_rows, SIGMA_COLUMNS)
        package_wall_seconds = time.perf_counter() - package_wall_started
        timing_rows = [
            {"name": "training_comparable", "seconds": training_seconds},
            {"name": "jit_compilation", "seconds": compilation_seconds},
            {"name": "wall_complete", "seconds": package_wall_seconds},
        ]
        write_csv(package_directory / "timing.csv", timing_rows, TIMING_COLUMNS)
        summary = {
            "package_index": package_index,
            "label": package.label,
            "active_cases": [case.manifest() for case in active],
            "new_cases": [case.manifest() for case in new_cases],
            "seen_cases": [case.manifest() for case in sorted(seen)],
            "best_pressure_monitor_loss": best_selection,
            "pressure_common_evaluations": len(pressure_common_rows),
            "snapshot_steps": snapshot_step_values,
            "snapshot_fractions": snapshot_fraction_values,
            "training_seconds": training_seconds,
            "jit_compilation_seconds": compilation_seconds,
            "wall_seconds": package_wall_seconds,
            "pressure_metrics": pressure_metric_rows,
            "celerity_metrics": material_metric_values,
        }
        write_json(package_directory / "summary.json", summary)
        run_summaries.append(summary)

    write_json(
        run_directory / "summary.json",
        {"variant": variant.name, "seed": seed, "packages": run_summaries},
    )
    return run_directory
