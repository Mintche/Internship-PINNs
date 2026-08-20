import csv
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")

from tests_forward_PINN.forward_ablation import (
    ADWEIGHTS_COMPONENTS,
    AdWeightsConfig,
    ForwardConfig,
    _build_context,
    adweights_from_lambdas,
    adweights_gradient_l2_norms,
    build_parser,
    forward_loss,
    load_checkpoint,
    make_adam_optimizer,
    make_adam_step,
    pointwise_pde_residual,
    rad_probabilities,
    regular_collocation_points,
    run_training,
    sample_collocation_points,
    save_checkpoint,
    sigma_train_step_count,
    update_adweights_lambdas,
)
from tests_forward_PINN.model_variants import (
    ADWEIGHTS_VARIANTS,
    BASE_VARIANTS,
    RAD_ADWEIGHTS_VARIANTS,
    RAD_VARIANTS,
    VARIANTS,
    base_variant,
    initialize_model,
    is_rad_variant,
    is_scattered_variant,
    model_value,
    uses_adweights,
    uses_fourier_features,
    uses_modified_mlp,
    value_gradient_laplacian,
    variant_parameter_count,
)
from tests_forward_PINN.plot_campaign import VARIANT_COLORS, write_campaign_outputs


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPOSITORY_ROOT / "tests_forward_PINN"


class PlotCampaignTest(unittest.TestCase):
    def test_assigns_a_distinct_color_to_every_variant(self):
        self.assertEqual(set(VARIANT_COLORS), set(VARIANTS))
        self.assertEqual(len(set(VARIANT_COLORS.values())), len(VARIANTS))


class ForwardAblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = ForwardConfig.from_json(
            CONFIG_DIR / "forward_homogeneous_1200_m0_manual.json"
        )
        cls.config = replace(
            base,
            hidden_layers=(4, 4),
            fourier_features=3,
            collocation_adam=(4, 2, 2),
            collocation_monitor=(4, 2, 2),
            rad_points=1,
            rad_candidate_points=8,
            rad_candidate_batch_size=4,
            loss_eval_interval_adam=1,
            fem_eval_interval_adam=1,
            gradient_eval_interval_adam=1,
            fem_prediction_batch_size=8,
            analytic_triangulation=(2, 2),
        )

    def test_all_forward_json_configs_load(self):
        paths = sorted(CONFIG_DIR.glob("forward_*_1200_m0.json"))
        self.assertGreater(len(paths), 0)
        for path in paths:
            with self.subTest(path=path.name):
                config = ForwardConfig.from_json(path)
                self.assertGreater(config.fourier_field_learning_rate, 0.0)
                self.assertGreater(config.fourier_sigma_learning_rate, 0.0)
                self.assertEqual(len(config.classical_loss_weights), 3)
                self.assertEqual(len(config.fourier_loss_weights), 3)
                self.assertIsNotNone(config.adweights)

    def test_default_cli_config_exists(self):
        args = build_parser().parse_args(
            [
                "run",
                "--variant",
                "classical_total",
                "--seed",
                "0",
                "--adam-steps",
                "1",
            ]
        )
        self.assertEqual(
            args.config.name, "forward_circlebottomright_1200_m0.json"
        )
        self.assertTrue(args.config.is_file())

    def test_adweights_formula_is_exact(self):
        adaptive = AdWeightsConfig(
            epsilon=0.25,
            alpha=0.2,
            initial_lambdas=(1.0, 2.0, 3.0),
            update_interval_adam=4,
            custom_weights=(2.0, 3.0, 4.0),
        )
        norms = np.asarray([6.0, 7.0, 8.0])
        expected_lambdas = 0.8 * np.asarray(adaptive.initial_lambdas) + 0.2 * norms
        expected_inverse = 1.0 / (adaptive.epsilon + expected_lambdas)
        lambdas, inverse, effective = update_adweights_lambdas(
            adaptive, adaptive.initial_lambdas, norms
        )
        np.testing.assert_allclose(lambdas, expected_lambdas)
        np.testing.assert_allclose(inverse, expected_inverse)
        np.testing.assert_allclose(
            effective, expected_inverse * np.asarray(adaptive.custom_weights)
        )
        initial_inverse, initial_effective = adweights_from_lambdas(
            adaptive, adaptive.initial_lambdas
        )
        np.testing.assert_allclose(
            initial_effective,
            initial_inverse * np.asarray(adaptive.custom_weights),
        )

    def test_adweights_configuration_validation_and_missing_block(self):
        invalid_values = (
            replace(self.config.adweights, epsilon=0.0),
            replace(self.config.adweights, alpha=1.1),
            replace(self.config.adweights, initial_lambdas=(-1.0, 1.0, 1.0)),
            replace(self.config.adweights, update_interval_adam=0),
            replace(self.config.adweights, custom_weights=(0.0, 0.0, 0.0)),
        )
        for adaptive in invalid_values:
            with self.subTest(adaptive=adaptive), self.assertRaises(ValueError):
                adaptive.validate()

        with tempfile.TemporaryDirectory() as directory:
            source = CONFIG_DIR / "forward_homogeneous_1200_m0_manual.json"
            with source.open(encoding="utf-8") as stream:
                payload = json.load(stream)
            payload.pop("adweights")
            path = Path(directory) / "without_adweights.json"
            self._write_json(path, payload)
            config = ForwardConfig.from_json(path)
            self.assertIsNone(config.adweights)
            with self.assertRaisesRegex(ValueError, "requires an 'adweights'"):
                run_training(
                    config,
                    variant="classical_total_adweights",
                    seed=0,
                    adam_steps=1,
                    output_root=Path(directory) / "runs",
                )

    def test_adweights_variants_match_their_bases(self):
        self.assertEqual(len(ADWEIGHTS_VARIANTS), len(BASE_VARIANTS))
        for base, adaptive in zip(BASE_VARIANTS, ADWEIGHTS_VARIANTS):
            with self.subTest(adaptive=adaptive):
                self.assertTrue(uses_adweights(adaptive))
                self.assertEqual(base_variant(adaptive), base)
                self.assertEqual(
                    variant_parameter_count(self.config, adaptive),
                    variant_parameter_count(self.config, base),
                )
                self.assertEqual(
                    uses_fourier_features(adaptive), uses_fourier_features(base)
                )
                self.assertEqual(
                    is_scattered_variant(adaptive), is_scattered_variant(base)
                )

    def test_rad_variants_match_their_bases(self):
        self.assertEqual(len(RAD_VARIANTS), len(BASE_VARIANTS))
        self.assertEqual(len(RAD_ADWEIGHTS_VARIANTS), len(BASE_VARIANTS))
        for base, rad, combined in zip(
            BASE_VARIANTS, RAD_VARIANTS, RAD_ADWEIGHTS_VARIANTS
        ):
            with self.subTest(rad=rad, combined=combined):
                self.assertTrue(is_rad_variant(rad))
                self.assertTrue(is_rad_variant(combined))
                self.assertFalse(uses_adweights(rad))
                self.assertTrue(uses_adweights(combined))
                self.assertEqual(base_variant(rad), base)
                self.assertEqual(base_variant(combined), base)
                self.assertEqual(
                    variant_parameter_count(self.config, rad),
                    variant_parameter_count(self.config, base),
                )
                self.assertEqual(
                    variant_parameter_count(self.config, combined),
                    variant_parameter_count(self.config, base),
                )

    def test_variant_tokens_and_losses_are_finite(self):
        expected_fourier = {
            "fourier_total",
            "fourier_modified_total",
            "fourier_scattered",
            "fourier_modified_scattered",
        }
        self.assertTrue(expected_fourier.issubset(VARIANTS))
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                self.assertEqual(uses_fourier_features(variant), "fourier" in variant)
                self.assertEqual(uses_modified_mlp(variant), "modified" in variant)
                self.assertEqual(is_scattered_variant(variant), "scattered" in variant)
                params, b_base = initialize_model(
                    jax.random.key(8), self.config, variant
                )
                context = _build_context(self.config, b_base)
                points = regular_collocation_points(
                    self.config.collocation_monitor, context
                )
                value, aux = forward_loss(params, context, variant, points)
                self.assertTrue(np.isfinite(float(value)))
                self.assertTrue(
                    all(np.isfinite(float(component)) for component in aux.values())
                )

    def test_pde_and_neumann_normalizations_match_report_definition(self):
        variant = "classical_total"
        params, b_base = initialize_model(jax.random.key(17), self.config, variant)
        context = _build_context(self.config, b_base)
        points = regular_collocation_points(
            self.config.collocation_monitor, context
        )
        _, aux = forward_loss(params, context, variant, points)
        x_pde, y_pde, x_neumann, _ = points
        k0 = 2.0 * jnp.pi * self.config.frequency / self.config.c0

        raw_pde = jax.vmap(
            lambda x, y: pointwise_pde_residual(
                params, context, variant, x, y
            )
        )(x_pde, y_pde)
        expected_pde = jnp.mean((raw_pde / k0**2) ** 2)

        def dy(x, y):
            return value_gradient_laplacian(
                params, context, variant, x, y
            )[1][1]

        top = jax.vmap(lambda x: dy(x, 1.0))(x_neumann)
        bottom = jax.vmap(lambda x: dy(x, -1.0))(x_neumann)
        expected_neumann = jnp.mean(top**2 + bottom**2) / k0**2

        np.testing.assert_allclose(aux["pde"], expected_pde, rtol=2e-6)
        np.testing.assert_allclose(
            aux["neumann"], expected_neumann, rtol=2e-6
        )

    def test_fourier_analytic_derivatives_match_autodiff(self):
        for variant in ("fourier_total", "fourier_modified_total"):
            with self.subTest(variant=variant):
                self._assert_fourier_derivatives_match_autodiff(variant)

    def test_sigma_freezes_after_decay_threshold(self):
        variant = "fourier_total"
        adam_steps = 4
        self.assertEqual(sigma_train_step_count(self.config, variant, adam_steps), 2)
        params, b_base = initialize_model(jax.random.key(11), self.config, variant)
        context = _build_context(self.config, b_base)
        optimizer = make_adam_optimizer(params, self.config, variant, adam_steps)
        state = optimizer.init(params)
        train_step = make_adam_step(
            optimizer, context, variant, self.config.collocation_adam
        )
        frozen_step = make_adam_step(
            optimizer,
            context,
            variant,
            self.config.collocation_adam,
            freeze_sigma=True,
        )
        params_after_train, state, _, _ = train_step(
            params, state, jax.random.key(21)
        )
        self.assertFalse(
            np.array_equal(
                np.asarray(params["sigma"]), np.asarray(params_after_train["sigma"])
            )
        )
        field_before = np.asarray(params_after_train["layers"][0]["W"])
        params_after_freeze, _, _, _ = frozen_step(
            params_after_train, state, jax.random.key(22)
        )
        np.testing.assert_array_equal(
            np.asarray(params_after_train["sigma"]),
            np.asarray(params_after_freeze["sigma"]),
        )
        self.assertFalse(
            np.array_equal(
                field_before, np.asarray(params_after_freeze["layers"][0]["W"])
            )
        )

    def test_adweights_norm_excludes_frozen_sigma(self):
        variant = "fourier_total_adweights"
        params, b_base = initialize_model(jax.random.key(31), self.config, variant)
        context = _build_context(self.config, b_base)
        points = regular_collocation_points(self.config.collocation_monitor, context)
        frozen_norms = adweights_gradient_l2_norms(
            params, context, variant, points, freeze_sigma=True
        )

        def pde_objective(model_params):
            return forward_loss(model_params, context, variant, points)[1]["pde"]

        gradients = jax.grad(pde_objective)(params)
        field_squared_norm = sum(
            float(jnp.real(jnp.vdot(leaf, leaf)))
            for leaf in jax.tree_util.tree_leaves(gradients["layers"])
        )
        self.assertAlmostEqual(
            float(frozen_norms[0]), np.sqrt(field_squared_norm), places=5
        )

    def test_modified_checkpoint_round_trip(self):
        variant = "fourier_modified_scattered"
        params, b_base = initialize_model(jax.random.key(12), self.config, variant)
        context = _build_context(self.config, b_base)
        manifest = {
            "number_of_layers": len(params["layers"]),
            "variant": variant,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            save_checkpoint(path, params, context, manifest)
            loaded, loaded_manifest, loaded_b_base = load_checkpoint(path)
        self.assertEqual(
            loaded_manifest["number_of_layers"], manifest["number_of_layers"]
        )
        self.assertEqual(loaded_manifest["variant"], manifest["variant"])
        np.testing.assert_array_equal(loaded_b_base, b_base)
        np.testing.assert_array_equal(loaded["sigma"], params["sigma"])
        for expected, actual in zip(params["layers"], loaded["layers"]):
            np.testing.assert_array_equal(actual["W"], expected["W"])
            np.testing.assert_array_equal(actual["b"], expected["b"])

    def test_tiny_adam_run_on_homogeneous_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            run_directory = run_training(
                replace(self.config, output_root=Path(directory)),
                variant="fourier_total",
                seed=0,
                adam_steps=1,
                output_root=Path(directory) / "runs",
            )
            with (run_directory / "summary.json").open(encoding="utf-8") as stream:
                summary = json.load(stream)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["sigma_train_steps"], 1)
        self.assertIn("optimizer_seconds", summary)
        self.assertNotIn("lbfgs_optimizer_seconds", summary)

    def test_tiny_rad_run_refreshes_the_sampling_distribution(self):
        run_config = replace(
            self.config,
            rad_resample_interval_adam=1,
            loss_eval_interval_adam=2,
            fem_eval_interval_adam=2,
            gradient_eval_interval_adam=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            run_directory = run_training(
                replace(run_config, output_root=Path(directory)),
                variant="classical_total_rad",
                seed=2,
                adam_steps=2,
                output_root=Path(directory) / "runs",
            )
            with (run_directory / "summary.json").open(encoding="utf-8") as stream:
                summary = json.load(stream)
            rad_history_exists = (
                run_directory / "rad_resampling_history.csv"
            ).is_file()
        self.assertEqual(summary["sampling_method"], "uniform_plus_RAD_bootstrap")
        self.assertEqual(summary["rad_refresh_count"], 1)
        self.assertGreater(summary["rad_resampling_seconds"], 0.0)
        self.assertTrue(rad_history_exists)

    def test_tiny_rad_adweights_run_combines_both_strategies(self):
        adaptive = replace(self.config.adweights, update_interval_adam=1)
        run_config = replace(self.config, adweights=adaptive)
        with tempfile.TemporaryDirectory() as directory:
            run_directory = run_training(
                replace(run_config, output_root=Path(directory)),
                variant="classical_total_rad_adweights",
                seed=3,
                adam_steps=1,
                output_root=Path(directory) / "runs",
            )
            with (run_directory / "summary.json").open(encoding="utf-8") as stream:
                summary = json.load(stream)
            adweights_history_exists = (
                run_directory / "adweights_history.csv"
            ).is_file()
        self.assertEqual(summary["sampling_method"], "uniform_plus_RAD_bootstrap")
        self.assertEqual(summary["adweights_update_count"], 1)
        self.assertEqual(summary["best_monitor_metric"], "unweighted_total")
        self.assertTrue(adweights_history_exists)

    def test_tiny_adweights_run_tracks_updates_and_uses_step_batch(self):
        adaptive = replace(self.config.adweights, update_interval_adam=2)
        run_config = replace(
            self.config,
            adweights=adaptive,
            loss_eval_interval_adam=3,
            fem_eval_interval_adam=3,
            gradient_eval_interval_adam=3,
        )
        seed = 4
        initialization_key = jax.random.fold_in(jax.random.key(seed), 101)
        initial_params, b_base = initialize_model(
            initialization_key, run_config, "classical_total_adweights"
        )
        context = _build_context(run_config, b_base)
        step_one_key = jax.random.fold_in(jax.random.key(seed), 1)
        expected_norms = np.asarray(
            adweights_gradient_l2_norms(
                initial_params,
                context,
                "classical_total_adweights",
                sample_collocation_points(step_one_key, run_config.collocation_adam),
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            run_directory = run_training(
                replace(run_config, output_root=Path(directory)),
                variant="classical_total_adweights",
                seed=seed,
                adam_steps=3,
                output_root=Path(directory) / "runs",
            )
            with (run_directory / "summary.json").open(encoding="utf-8") as stream:
                summary = json.load(stream)
            with (run_directory / "adweights_history.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(summary["adweights_update_count"], 2)
        self.assertEqual(summary["best_monitor_metric"], "unweighted_total")
        self.assertGreater(summary["adweights_seconds"], 0.0)
        self.assertAlmostEqual(
            summary["training_seconds"],
            summary["optimizer_seconds"]
            + summary["rad_resampling_seconds"]
            + summary["adweights_seconds"],
        )
        self.assertEqual(
            sorted({(int(row["update_index"]), int(row["global_step"])) for row in rows}),
            [(0, 0), (1, 1), (2, 3)],
        )
        first_update = {
            row["component"]: row
            for row in rows
            if int(row["update_index"]) == 1
        }
        np.testing.assert_allclose(
            [float(first_update[name]["gradient_l2_norm"]) for name in ADWEIGHTS_COMPONENTS],
            expected_norms,
            rtol=2e-5,
            atol=2e-6,
        )
        for row in rows:
            inverse = 1.0 / (adaptive.epsilon + float(row["lambda"]))
            np.testing.assert_allclose(
                float(row["inverse_weight"]), inverse, rtol=1e-6
            )
            np.testing.assert_allclose(
                float(row["effective_weight"]),
                float(row["custom_weight"]) * inverse,
                rtol=1e-6,
            )

    def test_plot_campaign_accepts_static_rad_and_adweights_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            plotted_variants = (
                *BASE_VARIANTS,
                ADWEIGHTS_VARIANTS[0],
                RAD_VARIANTS[0],
                RAD_ADWEIGHTS_VARIANTS[0],
            )
            for index, variant in enumerate(plotted_variants):
                run = root / "runs" / f"{variant}_seed0"
                run.mkdir(parents=True)
                seconds = 0.1 + 0.01 * index
                adaptive_seconds = 0.02 if uses_adweights(variant) else 0.0
                self._write_json(
                    run / "summary.json",
                    {
                        "variant": variant,
                        "seed": 0,
                        "adam_steps": 2,
                        "sigma_train_steps": 1 if "fourier" in variant else 0,
                        "optimizer_seconds": seconds,
                        "rad_resampling_seconds": 0.0,
                        "adweights_seconds": adaptive_seconds,
                        "training_seconds": seconds + adaptive_seconds,
                        "final_l2_relative": 0.2 + 0.01 * index,
                        "final_h1_relative": 0.3 + 0.01 * index,
                        "final_l2_absolute": 0.4 + 0.01 * index,
                        "final_h1_absolute": 0.5 + 0.01 * index,
                    },
                )
                self._write_csv(
                    run / "loss_history.csv",
                    [
                        {
                            "record_type": "loss_monitor",
                            "variant": variant,
                            "seed": 0,
                            "phase": "adam",
                            "local_step": step,
                            "global_step": step,
                            "optimizer_seconds": seconds * step,
                            "rad_resampling_seconds": 0.0,
                            "training_seconds": seconds * step,
                            "objective": 1.0 / (step + 1 + index),
                            "pde_loss": 0.7 / (step + 1 + index),
                            "bc_loss": 0.3 / (step + 1 + index),
                            "unweighted_total": 1.0 / (step + 1 + index),
                            "neumann_loss": 0.1 / (step + 1 + index),
                            "dtn_loss": 0.2 / (step + 1 + index),
                            "dtn_left_loss": 0.1 / (step + 1 + index),
                            "dtn_right_loss": 0.1 / (step + 1 + index),
                        }
                        for step in (0, 2)
                    ],
                )
                self._write_csv(
                    run / "fem_metrics.csv",
                    [
                        {
                            "record_type": "fem_metrics",
                            "reference_kind": "analytic_mode",
                            "variant": variant,
                            "seed": 0,
                            "phase": "adam",
                            "local_step": step,
                            "global_step": step,
                            "optimizer_seconds": seconds * step,
                            "rad_resampling_seconds": 0.0,
                            "training_seconds": seconds * step,
                            "l2_absolute": 0.4 / (step + 1 + index),
                            "l2_relative": 0.2 / (step + 1 + index),
                            "h1_absolute": 0.5 / (step + 1 + index),
                            "h1_relative": 0.3 / (step + 1 + index),
                        }
                        for step in (0, 2)
                    ],
                )
                self._write_csv(
                    run / "gradient_history.csv",
                    [
                        {
                            "record_type": "gradient_monitor",
                            "variant": variant,
                            "seed": 0,
                            "phase": "adam",
                            "local_step": step,
                            "global_step": step,
                            "optimizer_seconds": seconds * step,
                            "rad_resampling_seconds": 0.0,
                            "training_seconds": seconds * step,
                            "pde_gradient_l2_norm": 1.0 + index,
                            "neumann_gradient_l2_norm": 1.5 + index,
                            "dtn_gradient_l2_norm": 1.8 + index,
                            "bc_gradient_l2_norm": 2.0 + index,
                            "pde_bc_gradient_cosine": 0.1,
                        }
                        for step in (0, 2)
                    ],
                )
                if uses_adweights(variant):
                    self._write_csv(
                        run / "adweights_history.csv",
                        [
                            {
                                "record_type": "adweights_state",
                                "variant": variant,
                                "seed": 0,
                                "phase": "initial" if step == 0 else "adam",
                                "local_step": step,
                                "global_step": step,
                                "update_index": step // 2,
                                "component": component,
                                "alpha": 0.1,
                                "epsilon": 1e-8,
                                "gradient_l2_norm": "" if step == 0 else 0.5,
                                "lambda": 1.0 + component_index,
                                "inverse_weight": 1.0 / (1.0 + component_index),
                                "custom_weight": 1.0,
                                "effective_weight": 1.0 / (1.0 + component_index),
                                "update_seconds": adaptive_seconds,
                                "cumulative_adweights_seconds": adaptive_seconds,
                                "optimizer_seconds": seconds * step,
                                "rad_resampling_seconds": 0.0,
                                "training_seconds": seconds * step + adaptive_seconds,
                            }
                            for step in (0, 2)
                            for component_index, component in enumerate(
                                ADWEIGHTS_COMPONENTS
                            )
                        ],
                    )
            output = write_campaign_outputs(root, Path(directory) / "figures")
            aggregate = (output / "aggregate.csv").read_text(encoding="utf-8")
            adweights_outputs_exist = all(
                (output / name).is_file()
                for name in (
                    "adweights_history_all.csv",
                    "adweights_weights.pdf",
                    "adweights_weights.png",
                )
            )
        self.assertIn("fourier_modified_scattered", aggregate)
        self.assertNotIn("lbfgs", aggregate.lower())
        self.assertTrue(adweights_outputs_exist)

    def test_rad_probabilities_are_deterministic(self):
        residuals = np.asarray([0.0, 1.0, 3.0])
        probabilities = rad_probabilities(residuals, k=1.0, c=0.1)
        weights = residuals / residuals.mean() + 0.1
        np.testing.assert_allclose(probabilities, weights / weights.sum())

    def _assert_fourier_derivatives_match_autodiff(self, variant: str):
        params, b_base = initialize_model(jax.random.key(4), self.config, variant)
        context = _build_context(self.config, b_base)
        x = jnp.asarray(0.17, dtype=jnp.float32)
        y = jnp.asarray(-0.23, dtype=jnp.float32)
        value, gradient, laplacian = value_gradient_laplacian(
            params, context, variant, x, y
        )

        def value_function(x_value, y_value):
            return model_value(params, context, variant, x_value, y_value)

        dx = jax.jacfwd(value_function, argnums=0)(x, y) / context.config.half_length
        dy = jax.jacfwd(value_function, argnums=1)(x, y) * 2.0 / context.config.height
        dxx = (
            jax.jacfwd(jax.jacfwd(value_function, argnums=0), argnums=0)(x, y)
            / context.config.half_length**2
        )
        dyy = (
            jax.jacfwd(jax.jacfwd(value_function, argnums=1), argnums=1)(x, y)
            * (2.0 / context.config.height) ** 2
        )
        np.testing.assert_allclose(value, value_function(x, y), rtol=3e-4, atol=2e-4)
        np.testing.assert_allclose(
            gradient, jnp.stack((dx, dy)), rtol=1e-3, atol=7e-4
        )
        np.testing.assert_allclose(laplacian, dxx + dyy, rtol=4e-3, atol=3e-3)

    @staticmethod
    def _write_json(path: Path, payload: dict):
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]):
        with path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
