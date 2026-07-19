import unittest
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import numpy as np

from pinn_waveguide_2d import pinn_fullwave_waveguide as pinn


FLOAT32_VALUE_RTOL = 3e-4
FLOAT32_GRADIENT_RTOL = 2e-3


class FullwavePinnAnalyticDerivativeTests(unittest.TestCase):
    def test_analytic_spatial_derivatives_match_jacfwd(self):
        params = pinn.init_layers_uv(
            jax.random.key(123), pinn.n_layers_uv, 1000.0, 1
        )
        x = jnp.asarray(-0.41, dtype=jnp.float32)
        y = jnp.asarray(0.17, dtype=jnp.float32)

        def uv(x_value, y_value):
            return pinn.uv_apply(params, x_value, y_value)

        value = uv(x, y)
        expected_derivatives = (
            jax.jacfwd(uv, argnums=0)(x, y) / pinn.L,
            jax.jacfwd(uv, argnums=1)(x, y) * (2.0 / pinn.H),
        )
        for axis, expected in enumerate(expected_derivatives):
            actual_value, actual = pinn.uv_value_and_physical_derivative(
                params, x, y, axis
            )
            np.testing.assert_allclose(actual_value, value, rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

        expected_laplacian = (
            jax.jacfwd(jax.jacfwd(uv, argnums=0), argnums=0)(x, y)
            / pinn.L**2
            + jax.jacfwd(jax.jacfwd(uv, argnums=1), argnums=1)(x, y)
            * (2.0 / pinn.H) ** 2
        )
        actual_value, actual_laplacian = pinn.uv_value_and_physical_laplacian(
            params, x, y
        )
        np.testing.assert_allclose(actual_value, value, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(
            actual_laplacian,
            expected_laplacian,
            rtol=FLOAT32_VALUE_RTOL,
            atol=3e-4,
        )

    def test_analytic_loss_and_gradient_match_jacfwd(self):
        key = jax.random.key(7)
        params_uv = pinn.init_layers_uv(key, pinn.n_layers_uv, 600.0, 1)
        layers_m = pinn.init_layers(jax.random.fold_in(key, 1), pinn.n_layers_m)
        collocation = pinn.regular_collocation_points((8, 4, 3))
        y_boundary = jnp.linspace(-1.0, 1.0, 5, dtype=jnp.float32)
        target = jnp.zeros((5, 2), dtype=jnp.float32)
        frequency = jnp.asarray(600.0, dtype=jnp.float32)
        mode_index = jnp.asarray(1, dtype=jnp.int32)
        weights = jnp.asarray([1.0, 1.0, 10.0], dtype=jnp.float32)
        k0 = 2.0 * jnp.pi * frequency / pinn.c0
        beta_n = jnp.sqrt(
            k0**2 - (pinn.n_modes * jnp.pi / pinn.H) ** 2 + 0j
        )

        def evaluate():
            return jax.jit(
                jax.value_and_grad(
                    lambda p_uv, p_m: pinn.loss_fn(
                        p_uv,
                        p_m,
                        *collocation,
                        frequency,
                        mode_index,
                        target,
                        target,
                        y_boundary,
                        y_boundary,
                        jnp.asarray(1.0, dtype=jnp.float32),
                        weights,
                        beta_n,
                        is_warmup=False,
                        use_healthy_guide=False,
                    ),
                    argnums=(0, 1),
                    has_aux=True,
                )
            )(params_uv, layers_m)

        analytic, analytic_grads = evaluate()

        def reference_derivative(params, x, y, axis):
            def uv(x_value, y_value):
                return pinn.uv_apply(params, x_value, y_value)

            scales = (1.0 / pinn.L, 2.0 / pinn.H)
            return uv(x, y), jax.jacfwd(uv, argnums=axis)(x, y) * scales[axis]

        def reference_laplacian(params, x, y):
            def uv(x_value, y_value):
                return pinn.uv_apply(params, x_value, y_value)

            value = uv(x, y)
            laplacian = (
                jax.jacfwd(jax.jacfwd(uv, argnums=0), argnums=0)(x, y)
                / pinn.L**2
                + jax.jacfwd(jax.jacfwd(uv, argnums=1), argnums=1)(x, y)
                * (2.0 / pinn.H) ** 2
            )
            return value, laplacian

        with (
            mock.patch.object(
                pinn, "uv_value_and_physical_derivative", reference_derivative
            ),
            mock.patch.object(
                pinn, "uv_value_and_physical_laplacian", reference_laplacian
            ),
        ):
            reference, reference_grads = evaluate()

        np.testing.assert_allclose(
            analytic[0], reference[0], rtol=FLOAT32_VALUE_RTOL, atol=3e-4
        )
        for expected, actual in zip(reference[1], analytic[1]):
            np.testing.assert_allclose(
                actual, expected, rtol=FLOAT32_VALUE_RTOL, atol=3e-4
            )

        flatten = lambda tree: jnp.concatenate(
            [jnp.ravel(value) for value in jax.tree_util.tree_leaves(tree)]
        )
        reference_flat = flatten(reference_grads)
        relative_error = jnp.linalg.norm(
            flatten(analytic_grads) - reference_flat
        ) / jnp.linalg.norm(reference_flat)
        self.assertLess(float(relative_error), FLOAT32_GRADIENT_RTOL)


if __name__ == "__main__":
    unittest.main()
