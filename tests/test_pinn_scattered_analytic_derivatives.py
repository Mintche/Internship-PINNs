import unittest
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import numpy as np

from pinn_waveguide_2d import pinn_scattered_waveguide as pinn


# XLA evaluates the explicit recurrence and nested AD in a different floating-
# point order.  They agree to machine precision in float64; on CUDA float32 the
# observed relative differences are about 2e-4 for values and 1e-3 for the
# differentiated loss.  These bounds still catch material formula errors while
# matching the precision used by production training.
FLOAT32_GPU_VALUE_RTOL = 3e-4
FLOAT32_GPU_GRADIENT_RTOL = 2e-3


class ScatteredPinnAnalyticDerivativeTests(unittest.TestCase):
    def test_analytic_spatial_derivatives_match_jacfwd(self):
        params = pinn.init_layers_us(
            jax.random.key(123), pinn.n_layers_us, 1000.0, 1
        )
        x = jnp.asarray(-0.41, dtype=jnp.float32)
        y = jnp.asarray(0.17, dtype=jnp.float32)

        def us(x_value, y_value):
            return pinn.us_apply(params, x_value, y_value)

        value = us(x, y)
        expected_derivatives = (
            jax.jacfwd(us, argnums=0)(x, y) / pinn.L,
            jax.jacfwd(us, argnums=1)(x, y) * (2.0 / pinn.H),
        )
        for axis, expected in enumerate(expected_derivatives):
            actual_value, actual = pinn.us_value_and_physical_derivative(
                params, x, y, axis
            )
            np.testing.assert_allclose(actual_value, value, rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

        expected_laplacian = (
            jax.jacfwd(jax.jacfwd(us, argnums=0), argnums=0)(x, y)
            / pinn.L**2
            + jax.jacfwd(jax.jacfwd(us, argnums=1), argnums=1)(x, y)
            * (2.0 / pinn.H) ** 2
        )
        actual_value, actual_laplacian = pinn.us_value_and_physical_laplacian(
            params, x, y
        )
        np.testing.assert_allclose(actual_value, value, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(
            actual_laplacian,
            expected_laplacian,
            rtol=FLOAT32_GPU_VALUE_RTOL,
            atol=3e-4,
        )

    def test_analytic_package_loss_and_gradient_match_jacfwd(self):
        key = jax.random.key(7)
        params_us = pinn.init_layers_us(key, pinn.n_layers_us, 600.0, 1)
        params_us = jax.tree_util.tree_map(lambda value: value[None], params_us)
        layers_ms = pinn.init_layers(jax.random.fold_in(key, 1), pinn.n_layers_ms)
        collocation = pinn.regular_collocation_points((8, 4, 3))
        y_boundary = jnp.linspace(-1.0, 1.0, 5, dtype=jnp.float32)
        package = (
            jnp.asarray([600.0], dtype=jnp.float32),
            jnp.asarray([1], dtype=jnp.int32),
            jnp.zeros((1, 5, 2), dtype=jnp.float32),
            jnp.zeros((1, 5, 2), dtype=jnp.float32),
            y_boundary[None],
            y_boundary[None],
            jnp.ones(1, dtype=jnp.float32),
        )
        weights = jnp.asarray([1.0, 3.0, 10.0], dtype=jnp.float32)

        def evaluate():
            return jax.jit(
                jax.value_and_grad(
                    lambda p_us, p_ms: pinn.package_scattered_loss_fn(
                        p_us, p_ms, *collocation, *package, weights
                    ),
                    argnums=(0, 1),
                    has_aux=True,
                )
            )(params_us, layers_ms)

        analytic, analytic_grads = evaluate()

        def reference_derivative(params, x, y, axis):
            def us(x_value, y_value):
                return pinn.us_apply(params, x_value, y_value)

            scales = (1.0 / pinn.L, 2.0 / pinn.H)
            return us(x, y), jax.jacfwd(us, argnums=axis)(x, y) * scales[axis]

        def reference_laplacian(params, x, y):
            def us(x_value, y_value):
                return pinn.us_apply(params, x_value, y_value)

            value = us(x, y)
            laplacian = (
                jax.jacfwd(jax.jacfwd(us, argnums=0), argnums=0)(x, y)
                / pinn.L**2
                + jax.jacfwd(jax.jacfwd(us, argnums=1), argnums=1)(x, y)
                * (2.0 / pinn.H) ** 2
            )
            return value, laplacian

        with (
            mock.patch.object(
                pinn, "us_value_and_physical_derivative", reference_derivative
            ),
            mock.patch.object(
                pinn, "us_value_and_physical_laplacian", reference_laplacian
            ),
        ):
            reference, reference_grads = evaluate()

        np.testing.assert_allclose(
            analytic[0],
            reference[0],
            rtol=FLOAT32_GPU_VALUE_RTOL,
            atol=3e-4,
        )
        for expected, actual in zip(reference[1], analytic[1]):
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=FLOAT32_GPU_VALUE_RTOL,
                atol=3e-4,
            )

        flatten = lambda tree: jnp.concatenate(
            [jnp.ravel(value) for value in jax.tree_util.tree_leaves(tree)]
        )
        reference_flat = flatten(reference_grads)
        relative_error = jnp.linalg.norm(
            flatten(analytic_grads) - reference_flat
        ) / jnp.linalg.norm(reference_flat)
        self.assertLess(float(relative_error), FLOAT32_GPU_GRADIENT_RTOL)


if __name__ == "__main__":
    unittest.main()
