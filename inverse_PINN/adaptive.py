"""Inverse-gradient-norm adaptive loss weights shared by both optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class AdaptiveState:
    lambdas: jax.Array
    inverse_weights: jax.Array
    effective_weights: jax.Array
    update_index: int = 0


def weights_from_lambdas(
    lambdas: jax.Array | Sequence[float],
    *,
    epsilon: float,
    custom_weights: jax.Array | Sequence[float],
) -> tuple[jax.Array, jax.Array]:
    lambdas = jnp.asarray(lambdas, dtype=jnp.float32)
    custom = jnp.asarray(custom_weights, dtype=jnp.float32)
    if lambdas.shape != custom.shape:
        raise ValueError("Adaptive lambdas and custom weights must have the same shape")
    inverse = 1.0 / (float(epsilon) + lambdas)
    return inverse, custom * inverse


def initialize_state(
    lambdas: Sequence[float] | jax.Array,
    *,
    epsilon: float,
    custom_weights: Sequence[float] | jax.Array,
) -> AdaptiveState:
    lambdas_array = jnp.asarray(lambdas, dtype=jnp.float32)
    inverse, effective = weights_from_lambdas(
        lambdas_array, epsilon=epsilon, custom_weights=custom_weights
    )
    return AdaptiveState(lambdas_array, inverse, effective, 0)


def update_state(
    state: AdaptiveState,
    gradient_norms: Sequence[float] | jax.Array,
    *,
    epsilon: float,
    alpha: float,
    custom_weights: Sequence[float] | jax.Array,
) -> AdaptiveState:
    norms = jnp.asarray(gradient_norms, dtype=jnp.float32)
    if norms.shape != state.lambdas.shape:
        raise ValueError("Adaptive gradient norms have an invalid shape")
    lambdas = (1.0 - float(alpha)) * state.lambdas + float(alpha) * norms
    inverse, effective = weights_from_lambdas(
        lambdas, epsilon=epsilon, custom_weights=custom_weights
    )
    return AdaptiveState(lambdas, inverse, effective, state.update_index + 1)


def tree_dot(left: object, right: object) -> jax.Array:
    products = [
        jnp.real(jnp.vdot(a, b))
        for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right))
    ]
    if not products:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return sum(products[1:], products[0])


def tree_l2_norm(tree: object) -> jax.Array:
    return jnp.sqrt(jnp.maximum(tree_dot(tree, tree), 0.0))


def stop_sigma(params: dict[str, object]) -> dict[str, object]:
    if "sigma" not in params:
        return params
    return {**params, "sigma": jax.lax.stop_gradient(params["sigma"])}


def zero_sigma_update(updates: dict[str, object]) -> dict[str, object]:
    if "sigma" not in updates:
        return updates
    return {**updates, "sigma": jax.tree_util.tree_map(jnp.zeros_like, updates["sigma"])}
