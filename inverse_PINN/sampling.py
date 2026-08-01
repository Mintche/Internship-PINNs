"""Collocation protocols: stochastic uniform Adam, fixed monitor, fixed Sobol."""

from __future__ import annotations

import hashlib
import math
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc


class CollocationPoints(NamedTuple):
    x_pde: jax.Array
    y_pde: jax.Array
    x_neumann: jax.Array
    y_dtn: jax.Array


def regular_collocation_points(
    sizes: Sequence[int], *, half_length: float, height: float
) -> CollocationPoints:
    """Return a fixed cell-centred grid with exactly the requested counts."""
    n_pde, n_neumann, n_dtn = (int(value) for value in sizes)
    if min(n_pde, n_neumann, n_dtn) <= 0:
        raise ValueError("Collocation counts must be positive")
    aspect = 2.0 * half_length / height
    factors = [
        (n_pde // ny, ny)
        for ny in range(1, int(math.sqrt(n_pde)) + 1)
        if n_pde % ny == 0
    ]
    nx, ny = min(factors, key=lambda pair: abs(math.log((pair[0] / pair[1]) / aspect)))

    def axis(count: int) -> jax.Array:
        return jnp.linspace(
            -1.0 + 1.0 / count,
            1.0 - 1.0 / count,
            count,
            dtype=jnp.float32,
        )

    x_axis, y_axis = axis(nx), axis(ny)
    x_grid, y_grid = jnp.meshgrid(x_axis, y_axis, indexing="xy")
    return CollocationPoints(
        x_grid.reshape(-1), y_grid.reshape(-1), axis(n_neumann), axis(n_dtn)
    )


def uniform_collocation_points(
    key: jax.Array, sizes: Sequence[int]
) -> CollocationPoints:
    n_pde, n_neumann, n_dtn = (int(value) for value in sizes)
    keys = jax.random.split(key, 4)
    return CollocationPoints(
        jax.random.uniform(keys[0], (n_pde,), minval=-1.0, maxval=1.0),
        jax.random.uniform(keys[1], (n_pde,), minval=-1.0, maxval=1.0),
        jax.random.uniform(keys[2], (n_neumann,), minval=-1.0, maxval=1.0),
        jax.random.uniform(keys[3], (n_dtn,), minval=-1.0, maxval=1.0),
    )


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _sobol_block(dimension: int, count: int, *, scramble: bool, seed: int) -> np.ndarray:
    if count <= 0 or count & (count - 1):
        raise ValueError("Sobol counts must be positive powers of two")
    engine = qmc.Sobol(d=dimension, scramble=scramble, seed=seed)
    values = engine.random_base2(int(math.log2(count)))
    return 2.0 * values - 1.0


def sobol_collocation_points(
    sizes: Sequence[int], *, scramble: bool, seed: int
) -> CollocationPoints:
    """Build one immutable Sobol objective cloud for an L-BFGS phase."""
    n_pde, n_neumann, n_dtn = (int(value) for value in sizes)
    pde = _sobol_block(2, n_pde, scramble=scramble, seed=stable_seed(seed, "pde"))
    neumann = _sobol_block(1, n_neumann, scramble=scramble, seed=stable_seed(seed, "neumann"))
    dtn = _sobol_block(1, n_dtn, scramble=scramble, seed=stable_seed(seed, "dtn"))
    return CollocationPoints(
        jnp.asarray(pde[:, 0], dtype=jnp.float32),
        jnp.asarray(pde[:, 1], dtype=jnp.float32),
        jnp.asarray(neumann[:, 0], dtype=jnp.float32),
        jnp.asarray(dtn[:, 0], dtype=jnp.float32),
    )


def snapshot_steps(adam_steps: int, fractions: Sequence[float]) -> dict[int, float]:
    """Map one-based Adam steps to requested fractions, coalescing collisions."""
    if adam_steps <= 0:
        return {}
    result: dict[int, float] = {}
    for fraction in fractions:
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError("Snapshot fractions must lie in (0, 1]")
        step = min(adam_steps, max(1, int(round(float(fraction) * adam_steps))))
        result[step] = float(fraction)
    return result
