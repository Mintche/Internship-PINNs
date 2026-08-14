"""Constant-rate Adam steps and deterministic block L-BFGS steps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import optax

from .adaptive import zero_sigma_update
from .config import OptimizationConfig
from .losses import (
    CaseContext,
    PhysicsContext,
    material_objective,
    packed_pressure_objective,
)
from .sampling import CollocationPoints, uniform_collocation_points
from .models import unpack_field_parameters
from .variants import VariantSpec


def sigma_train_steps(optimization: OptimizationConfig, adam_steps: int) -> int:
    if adam_steps <= 0:
        return 0
    return max(1, int(optimization.sigma_decay_fraction * adam_steps))


def learning_rate_schedule(
    initial_learning_rate: float, optimization: OptimizationConfig
) -> optax.Schedule:
    """Keep Adam's rate constant, then decay it to an alpha-scaled plateau."""
    decay = optax.cosine_decay_schedule(
        initial_learning_rate,
        decay_steps=(
            optimization.consine_decay_stop - optimization.cosine_decay_start
        ),
        alpha=optimization.cosine_decay_alpha,
    )
    return optax.join_schedules(
        schedules=(optax.constant_schedule(initial_learning_rate), decay),
        boundaries=(optimization.cosine_decay_start,),
    )


def make_field_adam(
    params: Mapping[str, Any], optimization: OptimizationConfig, adam_steps: int
) -> optax.GradientTransformation:
    sigma_steps = sigma_train_steps(optimization, adam_steps)
    sigma_schedule = optax.cosine_decay_schedule(
        optimization.sigma_learning_rate,
        decay_steps=max(sigma_steps - 1, 1),
        alpha=optimization.sigma_cosine_alpha,
    )
    labels = jax.tree_util.tree_map(lambda _: "field", params)
    labels["sigma"] = jax.tree_util.tree_map(lambda _: "sigma", params["sigma"])
    return optax.multi_transform(
        {
            "field": optax.adam(
                learning_rate_schedule(
                    optimization.field_learning_rate, optimization
                )
            ),
            "sigma": optax.adam(sigma_schedule),
        },
        labels,
    )


def make_material_adam(
    optimization: OptimizationConfig,
) -> optax.GradientTransformation:
    return optax.adam(
        learning_rate_schedule(optimization.material_learning_rate, optimization)
    )


def make_warmup_adam_step(
    optimizer: optax.GradientTransformation,
    physics: PhysicsContext,
    contexts: Sequence[CaseContext],
    variant: VariantSpec,
    adam_sizes: Sequence[int],
    *,
    homogeneous_material: bool,
    freeze_sigma: bool,
):
    contexts = tuple(contexts)

    @jax.jit
    def step(packed_field_params, material_params, state, key, weights, b_bases):
        points = uniform_collocation_points(key, adam_sizes)

        def objective(candidate):
            return packed_pressure_objective(
                candidate, material_params, physics, contexts, b_bases,
                variant, points, weights, include_data=False,
                homogeneous_material=homogeneous_material,
                freeze_sigma=freeze_sigma,
            )[0]

        value, gradient = jax.value_and_grad(objective)(packed_field_params)
        updates, state = optimizer.update(
            gradient, state, packed_field_params
        )
        if freeze_sigma:
            updates = zero_sigma_update(updates)
        return optax.apply_updates(packed_field_params, updates), state, value

    return step


def make_inverse_adam_step(
    field_optimizer: optax.GradientTransformation,
    material_optimizer: optax.GradientTransformation,
    physics: PhysicsContext,
    contexts: Sequence[CaseContext],
    variant: VariantSpec,
    adam_sizes: Sequence[int],
    *,
    freeze_sigma: bool,
    tv_weight: float,
    tv_epsilon_squared: float,
):
    contexts = tuple(contexts)

    @jax.jit
    def step(
        packed_field_params,
        material_params,
        field_state,
        material_state,
        key,
        field_weights,
        material_weights,
        material_update_scale,
        b_bases,
    ):
        points = uniform_collocation_points(key, adam_sizes)

        def pressure_loss(candidate):
            return packed_pressure_objective(
                candidate, material_params, physics, contexts, b_bases,
                variant, points, field_weights, freeze_sigma=freeze_sigma,
            )[0]

        pressure_value, field_gradient = jax.value_and_grad(pressure_loss)(
            packed_field_params
        )
        field_updates, field_state = field_optimizer.update(
            field_gradient, field_state, packed_field_params
        )
        if freeze_sigma:
            field_updates = zero_sigma_update(field_updates)
        new_field_params = optax.apply_updates(
            packed_field_params, field_updates
        )

        unpacked_field_params = unpack_field_parameters(packed_field_params)

        def material_loss(candidate):
            return material_objective(
                candidate,
                unpacked_field_params,
                physics,
                tuple(
                    replace(context, b_base=b_base)
                    for context, b_base in zip(contexts, b_bases)
                ),
                variant,
                points,
                material_weights,
                tv_weight=tv_weight,
                tv_epsilon_squared=tv_epsilon_squared,
            )

        material_value, material_gradient = jax.value_and_grad(
            lambda candidate: material_loss(candidate)[0]
        )(material_params)
        material_updates, material_state = material_optimizer.update(
            material_gradient, material_state, material_params
        )
        material_updates = jax.tree_util.tree_map(
            lambda update: material_update_scale * update, material_updates
        )
        material_params = optax.apply_updates(material_params, material_updates)
        return (
            new_field_params,
            material_params,
            field_state,
            material_state,
            pressure_value,
            material_value,
        )

    return step


def make_pressure_lbfgs_step(
    optimizer: optax.GradientTransformation,
    physics: PhysicsContext,
    contexts: Sequence[CaseContext],
    variant: VariantSpec,
    points: CollocationPoints,
    weights: Sequence[jax.Array],
    *,
    include_data: bool,
    homogeneous_material: bool = False,
):
    contexts = tuple(contexts)
    weights = jnp.stack(tuple(weights))

    @jax.jit
    def step(packed_field_params, state, material_params, b_bases):
        def objective(candidate):
            return packed_pressure_objective(
                candidate, material_params, physics, contexts, b_bases,
                variant, points, weights, include_data=include_data,
                homogeneous_material=homogeneous_material, freeze_sigma=True,
            )[0]

        value, gradient = jax.value_and_grad(objective)(packed_field_params)
        updates, state = optimizer.update(
            gradient,
            state,
            packed_field_params,
            value=value,
            grad=gradient,
            value_fn=objective,
        )
        updates = zero_sigma_update(updates)
        return optax.apply_updates(packed_field_params, updates), state, value

    return step


def make_material_lbfgs_step(
    optimizer: optax.GradientTransformation,
    physics: PhysicsContext,
    contexts: Sequence[CaseContext],
    variant: VariantSpec,
    points: CollocationPoints,
    case_weights: jax.Array,
    *,
    tv_weight: float,
    tv_epsilon_squared: float,
):
    contexts = tuple(contexts)

    @jax.jit
    def step(material_params, state, packed_field_params, b_bases):
        dynamic_contexts = tuple(
            replace(context, b_base=b_base)
            for context, b_base in zip(contexts, b_bases)
        )
        field_params = unpack_field_parameters(packed_field_params)

        def objective(candidate):
            return material_objective(
                candidate,
                field_params,
                physics,
                dynamic_contexts,
                variant,
                points,
                case_weights,
                tv_weight=tv_weight,
                tv_epsilon_squared=tv_epsilon_squared,
            )[0]

        value, gradient = jax.value_and_grad(objective)(material_params)
        updates, state = optimizer.update(
            gradient,
            state,
            material_params,
            value=value,
            grad=gradient,
            value_fn=objective,
        )
        return optax.apply_updates(material_params, updates), state, value

    return step
