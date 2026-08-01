"""Curriculum trainer for modular inverse waveguide PINNs.

The module intentionally contains no plotting import. Monitoring, material
snapshots, FEM inference, checkpointing and I/O are timed outside the comparable
training duration.
"""

from __future__ import annotations

import time
from dataclasses import asdict
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
    physical_pressure_prediction,
    pressure_gradient_statistics,
    pressure_metrics,
    pressure_objective,
)
from .models import (
    FieldModel,
    initialize_field_model,
    initialize_material_model,
    material_sound_speed,
)
from .optimizers import (
    make_field_adam,
    make_inverse_adam_step,
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
from .variants import VariantSpec, parse_variant


PRESSURE_LOSS_COLUMNS = (
    "phase", "optimizer", "step", "case_id", "incidence", "frequency", "mode",
    "pde", "neumann", "dtn", "dtn_left", "dtn_right", "data",
    "objective", "unweighted_total", "selection_objective",
)
MATERIAL_LOSS_COLUMNS = (
    "phase", "optimizer", "step", "case_id", "incidence", "frequency", "mode",
    "pde", "pde_objective", "tv", "weighted_tv", "objective",
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


def _data_factor(config: InverseConfig, step: int, adam_steps: int) -> float:
    initial = config.optimization.data_initial_factor
    transition = max(1, int(config.optimization.data_transition_fraction * adam_steps))
    if transition == 1:
        progress = 1.0
    else:
        progress = min(max((step - 1) / (transition - 1), 0.0), 1.0)
    return initial + (1.0 - initial) * progress


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


def _append_sigma_row(
    rows: list[dict[str, Any]], phase: str, step: int, case: Case, params: Mapping[str, Any], frozen: bool
) -> None:
    sigma = np.asarray(params["sigma"])
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


def _evaluate_pressure(
    field_params: Sequence[Mapping[str, Any]],
    material_params: Mapping[str, Any],
    physics,
    contexts: Sequence[CaseContext],
    variant: VariantSpec,
    points: CollocationPoints,
    weights: jax.Array,
    *,
    include_data: bool = True,
    homogeneous_material: bool = False,
) -> tuple[float, list[dict[str, float]]]:
    evaluated = []
    for params, context in zip(field_params, contexts):
        objective, components = pressure_objective(
            params,
            material_params,
            physics,
            context,
            variant,
            points,
            weights,
            include_data=include_data,
            homogeneous_material=homogeneous_material,
        )
        _ready(objective)
        evaluated.append({name: _float(value) for name, value in components.items()})
    selection = float(np.mean([item["objective"] for item in evaluated]))
    return selection, evaluated


def _append_pressure_loss_rows(
    rows: list[dict[str, Any]],
    *,
    phase: str,
    optimizer: str,
    step: int,
    contexts: Sequence[CaseContext],
    components: Sequence[Mapping[str, float]],
    selection: float,
) -> None:
    for context, values in zip(contexts, components):
        case = context.case
        rows.append(
            {
                "phase": phase,
                "optimizer": optimizer,
                "step": step,
                "case_id": case.id,
                "incidence": case.incidence,
                "frequency": case.frequency,
                "mode": case.mode,
                **{name: values[name] for name in (
                    "pde", "neumann", "dtn", "dtn_left", "dtn_right", "data",
                    "objective", "unweighted_total",
                )},
                "selection_objective": selection,
            }
        )


def _append_material_loss_rows(
    rows: list[dict[str, Any]],
    *,
    phase: str,
    optimizer: str,
    step: int,
    contexts: Sequence[CaseContext],
    components: Mapping[str, Any],
    tv_enabled: bool,
) -> None:
    pdes = np.asarray(components["case_pdes"])
    common = {
        "pde_objective": _float(components["pde_objective"]),
        "tv": _float(components["tv"]) if tv_enabled else "",
        "weighted_tv": _float(components["weighted_tv"]) if tv_enabled else "",
        "objective": _float(components["objective"]),
    }
    for context, pde in zip(contexts, pdes):
        case = context.case
        rows.append(
            {
                "phase": phase,
                "optimizer": optimizer,
                "step": step,
                "case_id": case.id,
                "incidence": case.incidence,
                "frequency": case.frequency,
                "mode": case.mode,
                "pde": float(pde),
                **common,
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
        jax.vmap(lambda xv, yv: material_sound_speed(material_params, config.geometry, xv, yv))
    )
    for start in range(0, x_normalized.size, batch_size):
        values = function(
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
        lambda xv, yv: physical_pressure_prediction(
            params, physics, context, variant, xv, yv
        )
    )
    chunks = []
    for start in range(0, x.size, batch_size):
        values = function(
            jnp.asarray(x[start : start + batch_size], dtype=jnp.float32),
            jnp.asarray(y[start : start + batch_size], dtype=jnp.float32),
        )
        chunks.append(np.asarray(values))
    return np.concatenate(chunks)


def _run_warmup(
    *,
    case: Case,
    field_model: FieldModel,
    material_params: Mapping[str, Any],
    physics,
    context: CaseContext,
    config: InverseConfig,
    variant: VariantSpec,
    package_index: int,
    homogeneous_material: bool,
    pressure_rows: list[dict[str, Any]],
    pressure_gradient_rows: list[dict[str, Any]],
    adweight_rows: list[dict[str, Any]],
    sigma_rows: list[dict[str, Any]],
) -> tuple[FieldModel, float, float]:
    budget = config.training_packages[package_index].warmup
    params = field_model.params
    training_seconds = 0.0
    compilation_seconds = 0.0
    adaptive = _initial_field_adweight(config) if variant.field_adweights else None
    final_static = jnp.asarray(config.loss.field_weights, dtype=jnp.float32).at[3].set(0.0)

    if budget.adam_steps:
        optimizer = make_field_adam(params, config.optimization, budget.adam_steps)
        state = optimizer.init(params)
        train_steps = sigma_train_steps(config.optimization, budget.adam_steps)
        unfrozen_step = make_warmup_adam_step(
            optimizer, physics, context, variant,
            homogeneous_material=homogeneous_material, freeze_sigma=False,
        )
        frozen_step = make_warmup_adam_step(
            optimizer, physics, context, variant,
            homogeneous_material=homogeneous_material, freeze_sigma=True,
        )
        compile_points = uniform_collocation_points(
            jax.random.key(stable_seed("compile", package_index, case.id)), config.sampling.adam
        )
        compile_started = time.perf_counter()
        compiled = unfrozen_step(
            params, material_params, state, compile_points,
            _warmup_weights(config, variant, adaptive),
        )
        _ready(compiled)
        if train_steps < budget.adam_steps:
            compiled = frozen_step(
                params, material_params, state, compile_points,
                _warmup_weights(config, variant, adaptive),
            )
            _ready(compiled)
        compilation_seconds += time.perf_counter() - compile_started

        for step in range(1, budget.adam_steps + 1):
            iteration_started = time.perf_counter()
            points = uniform_collocation_points(
                jax.random.key(stable_seed(package_index, case.id, "warmup", step)),
                config.sampling.adam,
            )
            frozen = step > train_steps
            weights = _warmup_weights(config, variant, adaptive)
            if (
                variant.field_adweights
                and (step - 1) % config.loss.field_adweights.update_interval_adam == 0
            ):
                statistics = pressure_gradient_statistics(
                    params, material_params, physics, context, variant, points, weights,
                    freeze_sigma=frozen,
                )
                norms = jnp.asarray(
                    [statistics[f"{name}_gradient_l2_norm"] for name in FIELD_COMPONENTS]
                )
                assert adaptive is not None
                if step > 1:
                    adaptive = update_state(
                        adaptive, norms,
                        epsilon=config.loss.field_adweights.epsilon,
                        alpha=config.loss.field_adweights.alpha,
                        custom_weights=config.loss.field_adweights.custom_weights,
                    )
                weights = _warmup_weights(config, variant, adaptive)
                _append_adweight_rows(
                    adweight_rows, phase="warmup_adam", step=step, scope="pressure",
                    state=adaptive, labels=FIELD_COMPONENTS, case_id=case.id,
                )
            function = frozen_step if frozen else unfrozen_step
            params, state, _, _ = function(params, material_params, state, points, weights)
            _ready(params)
            training_seconds += time.perf_counter() - iteration_started

            print_event = step % config.logging.print_interval_adam == 0 or step == budget.adam_steps
            monitor_event = (
                step == 1 or step == budget.adam_steps
                or step % config.logging.loss_interval_adam == 0 or print_event
            )
            if monitor_event:
                selection, values = _evaluate_pressure(
                    (params,), material_params, physics, (context,), variant,
                    regular_collocation_points(
                        config.sampling.monitor,
                        half_length=config.geometry.half_length,
                        height=config.geometry.height,
                    ),
                    final_static, include_data=False,
                    homogeneous_material=homogeneous_material,
                )
                _append_pressure_loss_rows(
                    pressure_rows, phase="warmup_adam", optimizer="adam", step=step,
                    contexts=(context,), components=values, selection=selection,
                )
                if print_event:
                    print(
                        f"[{variant.name} pkg{package_index + 1:02d} warmup "
                        f"{case.id} Adam {step}/{budget.adam_steps}] "
                        f"loss={selection:.3e} weights={np.asarray(weights).tolist()} "
                        f"sigma={np.asarray(params['sigma']).tolist()}",
                        flush=True,
                    )
            if step == 1 or step == budget.adam_steps or step % config.logging.sigma_interval_adam == 0:
                _append_sigma_row(sigma_rows, "warmup_adam", step, case, params, frozen)
            if step % config.logging.pressure_gradient_interval_adam == 0 or step == budget.adam_steps:
                statistics = pressure_gradient_statistics(
                    params, material_params, physics, context, variant, points, weights,
                    freeze_sigma=frozen,
                )
                pressure_gradient_rows.append(
                    {
                        "phase": "warmup_adam", "step": step, "case_id": case.id,
                        "incidence": case.incidence, "frequency": case.frequency,
                        "mode": case.mode,
                        **{name: _float(value) for name, value in statistics.items()},
                        "sigma_frozen": int(frozen),
                    }
                )

    if budget.lbfgs_steps:
        points = sobol_collocation_points(
            config.sampling.lbfgs,
            scramble=config.sampling.sobol_scramble,
            seed=stable_seed(package_index, case.id, "warmup_lbfgs", config.sampling.sobol_seed_offset),
        )
        frozen_weights = _warmup_weights(config, variant, adaptive)
        optimizer = optax.lbfgs()
        packed = (params,)
        state = optimizer.init(packed)
        step_function = make_pressure_lbfgs_step(
            optimizer, material_params, physics, (context,), variant, points,
            (frozen_weights,), include_data=False,
            homogeneous_material=homogeneous_material,
        )
        compile_started = time.perf_counter()
        compiled = step_function(packed, state)
        _ready(compiled)
        compilation_seconds += time.perf_counter() - compile_started
        for step in range(1, budget.lbfgs_steps + 1):
            started = time.perf_counter()
            packed, state, _ = step_function(packed, state)
            _ready(packed)
            training_seconds += time.perf_counter() - started
            if step == 1 or step == budget.lbfgs_steps or step % config.logging.loss_interval_lbfgs == 0:
                selection, values = _evaluate_pressure(
                    packed, material_params, physics, (context,), variant,
                    regular_collocation_points(
                        config.sampling.monitor,
                        half_length=config.geometry.half_length,
                        height=config.geometry.height,
                    ),
                    final_static, include_data=False,
                    homogeneous_material=homogeneous_material,
                )
                _append_pressure_loss_rows(
                    pressure_rows, phase="warmup_lbfgs", optimizer="lbfgs", step=step,
                    contexts=(context,), components=values, selection=selection,
                )
        params = packed[0]
    return FieldModel(params, field_model.b_base), training_seconds, compilation_seconds


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
        "format_version": 1,
        "config": config.manifest(),
        "config_digest": short_digest(config.manifest()),
        "variant": asdict(variant),
        "seed": seed,
        "environment": environment_manifest(),
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
        pressure_rows: list[dict[str, Any]] = []
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
        if do_first_total_warmup or do_previous_map_pretrain:
            for case in new_cases:
                models[case], elapsed, compiled = _run_warmup(
                    case=case,
                    field_model=models[case],
                    material_params=material_params,
                    physics=physics,
                    context=contexts[case],
                    config=config,
                    variant=variant,
                    package_index=package_index,
                    homogeneous_material=do_first_total_warmup,
                    pressure_rows=pressure_rows,
                    pressure_gradient_rows=pressure_gradient_rows,
                    adweight_rows=adweight_rows,
                    sigma_rows=sigma_rows,
                )
                training_seconds += elapsed
                compilation_seconds += compiled

        field_params = tuple(models[case].params for case in active)
        field_adaptive = {
            case: _initial_field_adweight(config) for case in active
        } if variant.field_adweights else {}
        material_adaptive = (
            _initial_material_adweight(config, len(active))
            if variant.material_adweights else None
        )
        adam_steps = package.inverse.adam_steps
        best_selection, best_components = _evaluate_pressure(
            field_params, material_params, physics, active_contexts, variant,
            monitor_points, final_static_weights,
        )
        _append_pressure_loss_rows(
            pressure_rows, phase="inverse_initial", optimizer="none", step=0,
            contexts=active_contexts, components=best_components,
            selection=best_selection,
        )
        for case, params in zip(active, field_params):
            _append_sigma_row(
                sigma_rows, "inverse_initial", 0, case, params,
                frozen=adam_steps == 0,
            )
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
        best_fields = _copy_tree(field_params)
        best_material = _copy_tree(material_params)

        tv_weight = config.loss.tv.weight if variant.total_variation else 0.0
        if adam_steps:
            field_optimizers = tuple(
                make_field_adam(params, config.optimization, adam_steps)
                for params in field_params
            )
            field_states = tuple(
                optimizer.init(params)
                for optimizer, params in zip(field_optimizers, field_params)
            )
            material_optimizer = optax.adam(config.optimization.material_learning_rate)
            material_state = material_optimizer.init(material_params)
            sigma_steps = sigma_train_steps(config.optimization, adam_steps)
            unfrozen_step = make_inverse_adam_step(
                field_optimizers, material_optimizer, physics, active_contexts, variant,
                freeze_sigma=False, tv_weight=tv_weight,
                tv_epsilon_squared=config.loss.tv.epsilon_squared,
            )
            frozen_step = make_inverse_adam_step(
                field_optimizers, material_optimizer, physics, active_contexts, variant,
                freeze_sigma=True, tv_weight=tv_weight,
                tv_epsilon_squared=config.loss.tv.epsilon_squared,
            )
            compile_points = uniform_collocation_points(
                jax.random.key(stable_seed("compile", package_index, "inverse")),
                config.sampling.adam,
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
                field_params, material_params, field_states, material_state,
                compile_points, compile_field_weights, compile_material_weights,
                config.optimization.data_initial_factor,
            )
            _ready(compiled)
            if sigma_steps < adam_steps:
                compiled = frozen_step(
                    field_params, material_params, field_states, material_state,
                    compile_points, compile_field_weights, compile_material_weights,
                    config.optimization.data_initial_factor,
                )
                _ready(compiled)
            compilation_seconds += time.perf_counter() - compile_started
            configured_snapshots = snapshot_steps(
                adam_steps, config.logging.material_snapshot_fractions
            )

            for step in range(1, adam_steps + 1):
                iteration_started = time.perf_counter()
                points = uniform_collocation_points(
                    jax.random.key(stable_seed(seed, package_index, "inverse", step)),
                    config.sampling.adam,
                )
                frozen = step > sigma_steps
                data_factor = _data_factor(config, step, adam_steps)

                if (
                    variant.field_adweights
                    and (step - 1) % config.loss.field_adweights.update_interval_adam == 0
                ):
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

                if (
                    variant.material_adweights
                    and (step - 1) % config.loss.material_adweights.update_interval_adam == 0
                ):
                    assert material_adaptive is not None
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
                    field_params, material_params, field_states, material_state,
                    _, _, _, _,
                ) = function(
                    field_params, material_params, field_states, material_state,
                    points, field_weights, material_weights, data_factor,
                )
                _ready((field_params, material_params))
                training_seconds += time.perf_counter() - iteration_started

                print_event = step % config.logging.print_interval_adam == 0 or step == adam_steps
                monitor_event = (
                    step == 1 or step == adam_steps
                    or step % config.logging.loss_interval_adam == 0 or print_event
                )
                if monitor_event:
                    selection, components = _evaluate_pressure(
                        field_params, material_params, physics, active_contexts,
                        variant, monitor_points, final_static_weights,
                    )
                    _append_pressure_loss_rows(
                        pressure_rows, phase="inverse_adam", optimizer="adam",
                        step=step, contexts=active_contexts, components=components,
                        selection=selection,
                    )
                    _, material_components = material_objective(
                        material_params, field_params, physics, active_contexts,
                        variant, monitor_points, material_weights,
                        tv_weight=tv_weight,
                        tv_epsilon_squared=config.loss.tv.epsilon_squared,
                    )
                    _append_material_loss_rows(
                        material_rows, phase="inverse_adam", optimizer="adam",
                        step=step, contexts=active_contexts,
                        components=material_components,
                        tv_enabled=tv_weight != 0.0,
                    )
                    if selection < best_selection:
                        best_selection = selection
                        best_fields = _copy_tree(field_params)
                        best_material = _copy_tree(material_params)

                if step == 1 or step == adam_steps or step % config.logging.sigma_interval_adam == 0:
                    for case, params in zip(active, field_params):
                        _append_sigma_row(sigma_rows, "inverse_adam", step, case, params, frozen)

                if step % config.logging.pressure_gradient_interval_adam == 0 or step == adam_steps:
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

                if step in configured_snapshots:
                    fraction = configured_snapshots[step]
                    _, _, x_grid, y_grid = _snapshot_grid(config)
                    snapshots.append(_predict_celerity(material_params, config, x_grid, y_grid))
                    snapshot_step_values.append(step)
                    snapshot_fraction_values.append(fraction)
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
                    current = pressure_rows[-len(active):] if monitor_event else []
                    if current:
                        losses = ", ".join(
                            f"{row['case_id']}={row['objective']:.3e}" for row in current
                        )
                        field_weight_text = "; ".join(
                            f"{case.id}:{np.asarray(weights).tolist()}"
                            for case, weights in zip(active, field_weights)
                        )
                        sigma_text = "; ".join(
                            f"{case.id}:{np.asarray(params['sigma']).tolist()}"
                            for case, params in zip(active, field_params)
                        )
                        print(
                            f"[{variant.name} pkg{package_index + 1:02d} Adam {step}/{adam_steps}] "
                            f"pressure({losses}) best={best_selection:.3e} data={data_factor:.3f} "
                            f"field_weights=({field_weight_text}) "
                            f"material_weights={np.asarray(material_weights).tolist()} "
                            f"sigma=({sigma_text})",
                            flush=True,
                        )

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
                state = optimizer.init(field_params)
                function = make_pressure_lbfgs_step(
                    optimizer, material_params, physics, active_contexts, variant,
                    lbfgs_points, frozen_field_weights, include_data=True,
                )
                compile_started = time.perf_counter()
                compiled = function(field_params, state)
                _ready(compiled)
                compilation_seconds += time.perf_counter() - compile_started
                for block_step in range(1, package.inverse.lbfgs_field_steps + 1):
                    global_lbfgs_step += 1
                    started = time.perf_counter()
                    field_params, state, _ = function(field_params, state)
                    _ready(field_params)
                    training_seconds += time.perf_counter() - started
                    if block_step == package.inverse.lbfgs_field_steps or block_step % config.logging.loss_interval_lbfgs == 0:
                        selection, components = _evaluate_pressure(
                            field_params, material_params, physics, active_contexts,
                            variant, monitor_points, final_static_weights,
                        )
                        _append_pressure_loss_rows(
                            pressure_rows, phase=f"inverse_lbfgs_field_cycle{cycle}",
                            optimizer="lbfgs", step=global_lbfgs_step,
                            contexts=active_contexts, components=components,
                            selection=selection,
                        )
                        if selection < best_selection:
                            best_selection = selection
                            best_fields = _copy_tree(field_params)
                            best_material = _copy_tree(material_params)

            if package.inverse.lbfgs_material_steps:
                optimizer = optax.lbfgs()
                state = optimizer.init(material_params)
                function = make_material_lbfgs_step(
                    optimizer, field_params, physics, active_contexts, variant,
                    lbfgs_points, frozen_material_weights,
                    tv_weight=tv_weight,
                    tv_epsilon_squared=config.loss.tv.epsilon_squared,
                )
                compile_started = time.perf_counter()
                compiled = function(material_params, state)
                _ready(compiled)
                compilation_seconds += time.perf_counter() - compile_started
                for block_step in range(1, package.inverse.lbfgs_material_steps + 1):
                    global_lbfgs_step += 1
                    started = time.perf_counter()
                    material_params, state, _ = function(material_params, state)
                    _ready(material_params)
                    training_seconds += time.perf_counter() - started
                    if block_step == package.inverse.lbfgs_material_steps or block_step % config.logging.loss_interval_lbfgs == 0:
                        selection, components = _evaluate_pressure(
                            field_params, material_params, physics, active_contexts,
                            variant, monitor_points, final_static_weights,
                        )
                        _append_pressure_loss_rows(
                            pressure_rows, phase=f"inverse_lbfgs_material_cycle{cycle}",
                            optimizer="lbfgs", step=global_lbfgs_step,
                            contexts=active_contexts, components=components,
                            selection=selection,
                        )
                        _, material_components = material_objective(
                            material_params, field_params, physics, active_contexts,
                            variant, monitor_points, frozen_material_weights,
                            tv_weight=tv_weight,
                            tv_epsilon_squared=config.loss.tv.epsilon_squared,
                        )
                        _append_material_loss_rows(
                            material_rows,
                            phase=f"inverse_lbfgs_material_cycle{cycle}",
                            optimizer="lbfgs", step=global_lbfgs_step,
                            contexts=active_contexts, components=material_components,
                            tv_enabled=tv_weight != 0.0,
                        )
                        if selection < best_selection:
                            best_selection = selection
                            best_fields = _copy_tree(field_params)
                            best_material = _copy_tree(material_params)

        field_params = best_fields
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
        write_csv(package_directory / "pressure_loss_history.csv", pressure_rows, PRESSURE_LOSS_COLUMNS)
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
