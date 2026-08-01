"""Constant-rate Adam steps and deterministic block L-BFGS steps."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import optax

from .adaptive import stop_sigma, zero_sigma_update
from .config import OptimizationConfig
from .losses import (
    CaseContext,
    PhysicsContext,
    material_objective,
    pressure_objective,
)
from .sampling import CollocationPoints
from .variants import VariantSpec


def sigma_train_steps(optimization: OptimizationConfig, adam_steps: int) -> int:
    if adam_steps <= 0:
        return 0
    return max(1, int(optimization.sigma_decay_fraction * adam_steps))


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
            "field": optax.adam(optimization.field_learning_rate),
            "sigma": optax.adam(sigma_schedule),
        },
        labels,
    )


def make_warmup_adam_step(
    optimizer: optax.GradientTransformation,
    physics: PhysicsContext,
    context: CaseContext,
    variant: VariantSpec,
    *,
    homogeneous_material: bool,
    freeze_sigma: bool,
):
    @jax.jit
    def step(field_params, material_params, state, points, weights):
        def objective(candidate):
            evaluated = stop_sigma(candidate) if freeze_sigma else candidate
            return pressure_objective(
                evaluated,
                material_params,
                physics,
                context,
                variant,
                points,
                weights,
                include_data=False,
                homogeneous_material=homogeneous_material,
            )

        (value, components), gradient = jax.value_and_grad(
            objective, has_aux=True
        )(field_params)
        updates, state = optimizer.update(gradient, state, field_params)
        if freeze_sigma:
            updates = zero_sigma_update(updates)
        return optax.apply_updates(field_params, updates), state, value, components

    return step


def make_inverse_adam_step(
    field_optimizers: Sequence[optax.GradientTransformation],
    material_optimizer: optax.GradientTransformation,
    physics: PhysicsContext,
    contexts: Sequence[CaseContext],
    variant: VariantSpec,
    *,
    freeze_sigma: bool,
    tv_weight: float,
    tv_epsilon_squared: float,
):
    contexts = tuple(contexts)
    field_optimizers = tuple(field_optimizers)

    @jax.jit
    def step(
        field_params,
        material_params,
        field_states,
        material_state,
        points,
        field_weights,
        material_weights,
        material_update_scale,
    ):
        new_field_params = []
        new_field_states = []
        pressure_values = []
        pressure_components = []
        # Every gradient is evaluated at the same pre-update material state.
        for params, state, optimizer, context, weights in zip(
            field_params, field_states, field_optimizers, contexts, field_weights
        ):
            def field_loss(candidate):
                evaluated = stop_sigma(candidate) if freeze_sigma else candidate
                return pressure_objective(
                    evaluated,
                    material_params,
                    physics,
                    context,
                    variant,
                    points,
                    weights,
                )

            (value, components), gradient = jax.value_and_grad(
                field_loss, has_aux=True
            )(params)
            updates, next_state = optimizer.update(gradient, state, params)
            if freeze_sigma:
                updates = zero_sigma_update(updates)
            new_field_params.append(optax.apply_updates(params, updates))
            new_field_states.append(next_state)
            pressure_values.append(value)
            pressure_components.append(components)

        def material_loss(candidate):
            return material_objective(
                candidate,
                field_params,
                physics,
                contexts,
                variant,
                points,
                material_weights,
                tv_weight=tv_weight,
                tv_epsilon_squared=tv_epsilon_squared,
            )

        (material_value, material_components), material_gradient = jax.value_and_grad(
            material_loss, has_aux=True
        )(material_params)
        material_updates, material_state = material_optimizer.update(
            material_gradient, material_state, material_params
        )
        material_updates = jax.tree_util.tree_map(
            lambda update: material_update_scale * update, material_updates
        )
        material_params = optax.apply_updates(material_params, material_updates)
        return (
            tuple(new_field_params),
            material_params,
            tuple(new_field_states),
            material_state,
            jnp.stack(tuple(pressure_values)),
            tuple(pressure_components),
            material_value,
            material_components,
        )

    return step


def make_pressure_lbfgs_step(
    optimizer: optax.GradientTransformation,
    material_params: Mapping[str, Any],
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
    weights = tuple(weights)

    def objective(field_params):
        values = tuple(
            pressure_objective(
                stop_sigma(params),
                material_params,
                physics,
                context,
                variant,
                points,
                case_weights,
                include_data=include_data,
                homogeneous_material=homogeneous_material,
            )[0]
            for params, context, case_weights in zip(field_params, contexts, weights)
        )
        return sum(values[1:], values[0]) / len(values)

    value_and_grad = jax.jit(jax.value_and_grad(objective))

    @jax.jit
    def step(field_params, state):
        value, gradient = value_and_grad(field_params)
        updates, state = optimizer.update(
            gradient,
            state,
            field_params,
            value=value,
            grad=gradient,
            value_fn=objective,
        )
        updates = tuple(zero_sigma_update(update) for update in updates)
        return optax.apply_updates(field_params, updates), state, value

    return step


def make_material_lbfgs_step(
    optimizer: optax.GradientTransformation,
    field_params: Sequence[Mapping[str, Any]],
    physics: PhysicsContext,
    contexts: Sequence[CaseContext],
    variant: VariantSpec,
    points: CollocationPoints,
    case_weights: jax.Array,
    *,
    tv_weight: float,
    tv_epsilon_squared: float,
):
    field_params = tuple(field_params)
    contexts = tuple(contexts)

    def objective(material_params):
        return material_objective(
            material_params,
            field_params,
            physics,
            contexts,
            variant,
            points,
            case_weights,
            tv_weight=tv_weight,
            tv_epsilon_squared=tv_epsilon_squared,
        )[0]

    value_and_grad = jax.jit(jax.value_and_grad(objective))

    @jax.jit
    def step(material_params, state):
        value, gradient = value_and_grad(material_params)
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
