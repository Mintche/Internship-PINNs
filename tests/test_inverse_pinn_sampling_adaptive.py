from __future__ import annotations

import jax
import numpy as np
import pytest

from inverse_PINN.adaptive import initialize_state, update_state
from inverse_PINN.sampling import (
    regular_collocation_points,
    snapshot_steps,
    sobol_collocation_points,
    uniform_collocation_points,
)


def test_sampling_protocols_are_independent_and_deterministic():
    first = uniform_collocation_points(jax.random.key(1), (16, 8, 8))
    second = uniform_collocation_points(jax.random.key(2), (16, 8, 8))
    assert not np.array_equal(first.x_pde, second.x_pde)

    monitor_a = regular_collocation_points((16, 8, 8), half_length=1.0, height=0.6)
    monitor_b = regular_collocation_points((16, 8, 8), half_length=1.0, height=0.6)
    np.testing.assert_array_equal(monitor_a.x_pde, monitor_b.x_pde)
    assert not np.any(np.isin(np.asarray(monitor_a.x_pde), [-1.0, 1.0]))

    sobol_a = sobol_collocation_points((16, 8, 8), scramble=True, seed=4)
    sobol_b = sobol_collocation_points((16, 8, 8), scramble=True, seed=4)
    np.testing.assert_array_equal(sobol_a.x_pde, sobol_b.x_pde)
    with pytest.raises(ValueError, match="powers of two"):
        sobol_collocation_points((12, 8, 8), scramble=True, seed=4)


def test_adweights_use_unclipped_inverse_ema_formula():
    state = initialize_state([1.0, 2.0], epsilon=0.5, custom_weights=[2.0, 3.0])
    updated = update_state(
        state, [3.0, 6.0], epsilon=0.5, alpha=0.25,
        custom_weights=[2.0, 3.0],
    )
    expected_lambdas = np.asarray([1.5, 3.0])
    np.testing.assert_allclose(updated.lambdas, expected_lambdas)
    np.testing.assert_allclose(updated.effective_weights, [1.0, 3.0 / 3.5])
    assert updated.update_index == 1


def test_snapshot_fractions_trigger_one_based_steps_only():
    assert snapshot_steps(10, (0.2, 0.5, 0.8)) == {2: 0.2, 5: 0.5, 8: 0.8}
    assert snapshot_steps(2, (0.2, 0.5, 0.8)) == {1: 0.5, 2: 0.8}
    assert snapshot_steps(0, (0.5,)) == {}

