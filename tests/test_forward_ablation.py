import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.forward_ablation import (
    ForwardConfig,
    _build_context,
    adapted_value_gradient_laplacian,
    forward_loss,
    initialize_model,
    load_checkpoint,
    model_value,
    regular_collocation_points,
    save_checkpoint,
    true_squared_slowness,
    variant_parameter_count,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "experiments/configs/forward_halfbar_600_m0.json"


class ForwardAblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ForwardConfig.from_json(CONFIG_PATH)

    def test_classical_and_adapted_parameter_counts_are_matched(self):
        classical = variant_parameter_count(self.config, "classical_total")
        adapted = variant_parameter_count(self.config, "adapted_total")
        self.assertLess(abs(classical - adapted) / adapted, 0.005)

    def test_fixed_halfbar_coefficient(self):
        _, b_base = initialize_model(jax.random.key(0), self.config, "adapted_total")
        context = _build_context(self.config, b_base)
        inside = float(true_squared_slowness(context, 0.0, -0.5))
        outside = float(true_squared_slowness(context, 0.8, 0.5))
        self.assertAlmostEqual(inside, 1.0 / (self.config.c0 * self.config.contrast_ratio) ** 2)
        self.assertAlmostEqual(outside, 1.0 / self.config.c0**2)

    def test_adapted_analytic_derivatives_match_autodiff(self):
        params, b_base = initialize_model(jax.random.key(4), self.config, "adapted_total")
        context = _build_context(self.config, b_base)
        x = jnp.asarray(0.17, dtype=jnp.float32)
        y = jnp.asarray(-0.23, dtype=jnp.float32)
        value, gradient, laplacian = adapted_value_gradient_laplacian(
            params, context, x, y
        )

        def value_function(x_value, y_value):
            return model_value(params, context, "adapted_total", x_value, y_value)

        dx = jax.jacfwd(value_function, argnums=0)(x, y) / context.length
        dy = jax.jacfwd(value_function, argnums=1)(x, y) * 2.0 / context.height
        dxx = (
            jax.jacfwd(jax.jacfwd(value_function, argnums=0), argnums=0)(x, y)
            / context.length**2
        )
        dyy = (
            jax.jacfwd(jax.jacfwd(value_function, argnums=1), argnums=1)(x, y)
            * (2.0 / context.height) ** 2
        )
        np.testing.assert_allclose(value, value_function(x, y), rtol=3e-4, atol=2e-4)
        np.testing.assert_allclose(gradient, jnp.stack((dx, dy)), rtol=8e-4, atol=4e-4)
        np.testing.assert_allclose(laplacian, dxx + dyy, rtol=2e-3, atol=2e-3)

    def test_all_three_variant_losses_are_finite(self):
        small = replace(self.config, validation_collocation=(16, 8, 8))
        for variant in ("classical_total", "adapted_total", "adapted_scattered"):
            with self.subTest(variant=variant):
                params, b_base = initialize_model(jax.random.key(8), small, variant)
                context = _build_context(small, b_base)
                points = regular_collocation_points(small.validation_collocation, context)
                value, aux = forward_loss(params, context, variant, points)
                self.assertTrue(np.isfinite(float(value)))
                self.assertTrue(all(np.isfinite(float(component)) for component in aux.values()))

    def test_checkpoint_round_trip(self):
        params, b_base = initialize_model(jax.random.key(11), self.config, "adapted_scattered")
        context = _build_context(self.config, b_base)
        manifest = {
            "number_of_layers": len(params["layers"]),
            "variant": "adapted_scattered",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            save_checkpoint(path, params, context, manifest)
            loaded, loaded_manifest, loaded_b_base = load_checkpoint(path)
        self.assertEqual(loaded_manifest, manifest)
        np.testing.assert_array_equal(loaded_b_base, b_base)
        np.testing.assert_array_equal(loaded["sigma"], params["sigma"])
        for expected, actual in zip(params["layers"], loaded["layers"]):
            np.testing.assert_array_equal(actual["W"], expected["W"])
            np.testing.assert_array_equal(actual["b"], expected["b"])


if __name__ == "__main__":
    unittest.main()
