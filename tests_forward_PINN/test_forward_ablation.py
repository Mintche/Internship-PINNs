from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from tools.data_loader import SymmetricCOOMatrix
from tests_forward_PINN.forward_ablation import (
    RAD_VARIANTS,
    VARIANTS,
    ForwardConfig,
    _build_context,
    _evaluation_due,
    adapted_value_gradient_laplacian,
    analytic_modal_complex,
    bootstrap_rad_pde_points,
    build_rad_candidate_distribution,
    build_analytic_modal_reference,
    compute_misfit_metrics,
    forward_loss,
    gradient_alignment_stats,
    incident_wave_complex,
    initialize_model,
    load_checkpoint,
    make_adam_optimizer,
    make_adam_step,
    make_lbfgs_step,
    make_rad_adam_step,
    model_value,
    rad_probabilities,
    regular_collocation_points,
    run_training,
    sample_collocation_points,
    sample_mixed_rad_collocation_points,
    save_checkpoint,
    true_squared_slowness,
    uniform_rad_candidate_distribution,
    variant_parameter_count,
)
from tests_forward_PINN.plot_campaign import (
    _positive_plot_floor,
    aggregate_runs,
    create_error_figure,
    create_error_time_figure,
    create_gradient_figure,
    create_loss_figure,
    create_timing_figure,
    write_campaign_outputs,
)


CONFIG_PATH = Path(__file__).with_name("forward_2circles_1200_m0.json")
HOMOGENEOUS_CONFIG_PATH = Path(__file__).with_name(
    "forward_homogeneous_1200_m0.json"
)


def _diagonal_matrix(size: int, value: float) -> SymmetricCOOMatrix:
    indices = np.arange(size, dtype=np.int64)
    return SymmetricCOOMatrix(
        size=size,
        rows=indices,
        columns=indices,
        values=np.full(size, value, dtype=np.float64),
    )


class ForwardAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ForwardConfig.from_json(CONFIG_PATH)
        cls.homogeneous_config = ForwardConfig.from_json(HOMOGENEOUS_CONFIG_PATH)

    def test_homogeneous_analytic_reference_uses_independent_triangulation(self):
        config = replace(
            self.homogeneous_config,
            analytic_triangulation=(4, 2),
        )
        reference = build_analytic_modal_reference(config)
        self.assertEqual(reference.x.size, 15)
        self.assertEqual(config.reference_kind, "analytic_mode")
        self.assertIsNone(config.fem_field)

        constant = np.ones(reference.x.size)
        area = 2.0 * config.half_length * config.height
        self.assertAlmostEqual(
            reference.mass.quadratic_form(constant), area, places=12
        )
        self.assertAlmostEqual(
            reference.stiffness.quadratic_form(constant), 0.0, places=12
        )
        self.assertAlmostEqual(
            reference.stiffness.quadratic_form(reference.x), area, places=12
        )
        expected = analytic_modal_complex(config, reference.x, reference.y)
        np.testing.assert_allclose(reference.values, expected, rtol=0.0, atol=0.0)
        exact = compute_misfit_metrics(
            reference.values,
            expected,
            reference.mass,
            reference.stiffness,
        )
        self.assertEqual(exact.l2_relative, 0.0)
        self.assertEqual(exact.h1_relative, 0.0)
        scaled = compute_misfit_metrics(
            reference.values,
            1.1 * expected,
            reference.mass,
            reference.stiffness,
        )
        self.assertAlmostEqual(scaled.l2_relative, 0.1, places=12)
        self.assertAlmostEqual(scaled.h1_relative, 0.1, places=12)

    def test_zero_scattered_field_exactly_solves_homogeneous_problem(self):
        small = replace(
            self.homogeneous_config,
            fourier_features=2,
            hidden_layers=(4,),
            collocation_monitor=(16, 4, 4),
        )
        for variant in ("adapted_scattered", "adapted_scattered_RAD"):
            params, basis = initialize_model(jax.random.key(20), small, variant)
            params = jax.tree_util.tree_map(jnp.zeros_like, params)
            context = _build_context(small, basis)
            points = regular_collocation_points(small.collocation_monitor, context)
            objective, components = forward_loss(params, context, variant, points)
            self.assertEqual(float(objective), 0.0)
            self.assertEqual(float(components["pde"]), 0.0)
            self.assertEqual(float(components["bc"]), 0.0)

    def test_one_step_homogeneous_run_never_requires_fem_files(self):
        with tempfile.TemporaryDirectory() as directory:
            small = replace(
                self.homogeneous_config,
                fourier_features=2,
                hidden_layers=(4,),
                collocation_adam=(4, 2, 2),
                collocation_lbfgs=(4, 2, 2),
                collocation_monitor=(4, 2, 2),
                loss_eval_interval_adam=1,
                fem_eval_interval_adam=1,
                gradient_eval_interval_adam=50,
                gradient_eval_interval_lbfgs=50,
                analytic_triangulation=(4, 2),
                fem_prediction_batch_size=8,
            )
            output = run_training(
                small,
                variant="adapted_total",
                seed=0,
                adam_steps=1,
                lbfgs_steps=0,
                output_root=Path(directory),
            )
            metrics = pd.read_csv(output / "fem_metrics.csv")
            gradients = pd.read_csv(output / "gradient_history.csv")
            with (output / "summary.json").open(encoding="utf-8") as stream:
                summary = json.load(stream)
            self.assertEqual(set(metrics["reference_kind"]), {"analytic_mode"})
            self.assertEqual(metrics["global_step"].tolist(), [0, 1])
            self.assertEqual(gradients["record_type"].tolist(), ["gradient_monitor"])
            self.assertEqual(gradients["global_step"].tolist(), [1])
            self.assertEqual(summary["gradient_records"], 1)
            self.assertTrue(
                {
                    "pde_gradient_l2_norm",
                    "bc_gradient_l2_norm",
                    "pde_bc_gradient_cosine",
                }.issubset(gradients.columns)
            )

    def test_tiny_rad_run_records_resampling_cost_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            small = replace(
                self.homogeneous_config,
                fourier_features=2,
                hidden_layers=(4,),
                collocation_adam=(4, 2, 2),
                collocation_lbfgs=(4, 2, 2),
                collocation_monitor=(4, 2, 2),
                rad_points=1,
                rad_resample_interval_adam=1,
                rad_candidate_points=8,
                rad_candidate_batch_size=4,
                loss_eval_interval_adam=1,
                fem_eval_interval_adam=1,
                analytic_triangulation=(4, 2),
                fem_prediction_batch_size=8,
            )
            output = run_training(
                small,
                variant="adapted_total_RAD",
                seed=0,
                adam_steps=2,
                lbfgs_steps=1,
                output_root=Path(directory),
            )
            with (output / "summary.json").open(encoding="utf-8") as stream:
                summary = json.load(stream)
            history = pd.read_csv(output / "rad_resampling_history.csv")
            self.assertEqual(
                summary["sampling_method"], "uniform_plus_RAD_bootstrap"
            )
            self.assertEqual(summary["rad_resample_count"], 2)
            self.assertEqual(summary["rad_refresh_count"], 2)
            self.assertGreaterEqual(summary["rad_resampling_seconds"], 0.0)
            self.assertAlmostEqual(
                summary["training_seconds"],
                summary["optimizer_seconds"] + summary["rad_resampling_seconds"],
            )
            self.assertEqual(history["phase"].tolist(), ["adam", "lbfgs"])
            self.assertEqual(history["global_step"].tolist(), [1, 2])
            self.assertEqual(history["candidate_count"].tolist(), [8, 8])
            self.assertEqual(history["bootstrap_batch_size"].tolist(), [1, 1])
            self.assertTrue(history["bootstrap_with_replacement"].all())

    def test_known_two_circle_material(self):
        _, basis = initialize_model(jax.random.key(0), self.config, "adapted_total")
        context = _build_context(self.config, basis)

        def normalized(x_physical, y_physical):
            return (
                x_physical / self.config.half_length,
                2.0 * y_physical / self.config.height - 1.0,
            )

        first = float(true_squared_slowness(context, *normalized(-0.3, 0.4)))
        second = float(true_squared_slowness(context, *normalized(0.3, 0.2)))
        background = float(true_squared_slowness(context, *normalized(0.0, 0.05)))
        self.assertAlmostEqual(first, 1.0 / (0.8 * self.config.c0) ** 2, places=10)
        self.assertAlmostEqual(second, 1.0 / (1.1 * self.config.c0) ** 2, places=10)
        self.assertAlmostEqual(background, 1.0 / self.config.c0**2, places=10)

    def test_parameter_counts_and_paired_adapted_initialization(self):
        self.assertEqual(variant_parameter_count(self.config, "classical_total"), 33666)
        self.assertEqual(variant_parameter_count(self.config, "adapted_total"), 66180)
        self.assertEqual(
            variant_parameter_count(self.config, "adapted_scattered"), 66180
        )
        self.assertEqual(
            variant_parameter_count(self.config, "adapted_total_RAD"), 66180
        )
        self.assertEqual(
            variant_parameter_count(self.config, "adapted_scattered_RAD"), 66180
        )
        key = jax.random.key(9)
        total, total_basis = initialize_model(key, self.config, "adapted_total")
        for variant in ("adapted_scattered", *RAD_VARIANTS):
            paired, paired_basis = initialize_model(key, self.config, variant)
            for expected, actual in zip(
                jax.tree_util.tree_leaves(total), jax.tree_util.tree_leaves(paired)
            ):
                np.testing.assert_array_equal(expected, actual)
            np.testing.assert_array_equal(total_basis, paired_basis)

    def test_adapted_analytic_derivatives_match_autodiff(self):
        params, basis = initialize_model(
            jax.random.key(3), self.config, "adapted_total"
        )
        context = _build_context(self.config, basis)
        x = jnp.asarray(0.17, dtype=jnp.float32)
        y = jnp.asarray(-0.23, dtype=jnp.float32)
        value, gradient, laplacian = adapted_value_gradient_laplacian(
            params, context, x, y
        )

        def function(x_value, y_value):
            return model_value(params, context, "adapted_total", x_value, y_value)

        dx = jax.jacfwd(function, argnums=0)(x, y) / context.config.half_length
        dy = (
            jax.jacfwd(function, argnums=1)(x, y)
            * 2.0
            / context.config.height
        )
        dxx = (
            jax.jacfwd(jax.jacfwd(function, argnums=0), argnums=0)(x, y)
            / context.config.half_length**2
        )
        dyy = (
            jax.jacfwd(jax.jacfwd(function, argnums=1), argnums=1)(x, y)
            * (2.0 / context.config.height) ** 2
        )
        np.testing.assert_allclose(value, function(x, y), rtol=3e-4, atol=2e-4)
        np.testing.assert_allclose(gradient, jnp.stack((dx, dy)), rtol=8e-4, atol=4e-4)
        np.testing.assert_allclose(laplacian, dxx + dyy, rtol=2e-3, atol=2e-3)

    def test_incident_mode_zero_is_neumann_and_analytically_normalized(self):
        _, basis = initialize_model(jax.random.key(4), self.config, "adapted_total")
        context = _build_context(self.config, basis)
        x = jnp.asarray(-0.2, dtype=jnp.float32)
        y = jnp.asarray(0.31, dtype=jnp.float32)
        normalized = incident_wave_complex(context, x, y, normalized=True)
        expected = jnp.exp(
            -1j
            * self.config.incidence
            * context.beta[0]
            * x
            * self.config.half_length
        )
        np.testing.assert_allclose(normalized, expected, rtol=1e-6, atol=1e-6)
        derivative_y = jax.jacfwd(
            lambda y_value: incident_wave_complex(
                context, x, y_value, normalized=True
            )
        )(y)
        np.testing.assert_allclose(derivative_y, 0.0, atol=1e-7)

        right_config = replace(self.config, incidence=1)
        _, right_basis = initialize_model(
            jax.random.key(4), right_config, "adapted_total"
        )
        right_context = _build_context(right_config, right_basis)
        right_value = incident_wave_complex(right_context, x, y, normalized=True)
        right_expected = jnp.exp(
            -1j * right_context.beta[0] * x * right_config.half_length
        )
        np.testing.assert_allclose(right_value, right_expected, rtol=1e-6, atol=1e-6)

    def test_all_formulation_losses_are_finite_and_do_not_accept_fem_data(self):
        small = replace(
            self.config,
            fourier_features=4,
            hidden_layers=(8, 8),
            collocation_monitor=(16, 4, 4),
        )
        self.assertEqual(
            tuple(inspect.signature(forward_loss).parameters),
            ("params", "context", "variant", "points"),
        )
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                params, basis = initialize_model(jax.random.key(5), small, variant)
                context = _build_context(small, basis)
                points = regular_collocation_points(
                    small.collocation_monitor, context
                )
                objective, components = forward_loss(
                    params, context, variant, points
                )
                self.assertTrue(np.isfinite(float(objective)))
                self.assertTrue(
                    all(np.isfinite(float(value)) for value in components.values())
                )
                self.assertAlmostEqual(
                    float(components["unweighted_total"]),
                    float(components["pde"] + components["bc"]),
                    places=5,
                )

    def test_gradient_alignment_stats_are_finite_or_nan(self):
        small = replace(
            self.config,
            fourier_features=2,
            hidden_layers=(4,),
            collocation_monitor=(4, 2, 2),
        )
        params, basis = initialize_model(jax.random.key(22), small, "adapted_total")
        context = _build_context(small, basis)
        points = regular_collocation_points(small.collocation_monitor, context)
        stats = jax.device_get(
            gradient_alignment_stats(params, context, "adapted_total", points)
        )
        pde_norm = float(stats["pde_gradient_l2_norm"])
        bc_norm = float(stats["bc_gradient_l2_norm"])
        cosine = float(stats["pde_bc_gradient_cosine"])
        self.assertTrue(np.isfinite(pde_norm))
        self.assertTrue(np.isfinite(bc_norm))
        self.assertGreaterEqual(pde_norm, 0.0)
        self.assertGreaterEqual(bc_norm, 0.0)
        self.assertTrue(np.isnan(cosine) or -1.0 <= cosine <= 1.0)

    def test_seeded_collocation_is_reproducible(self):
        sizes = (16, 4, 4)
        first = sample_collocation_points(jax.random.fold_in(jax.random.key(2), 7), sizes)
        second = sample_collocation_points(jax.random.fold_in(jax.random.key(2), 7), sizes)
        third = sample_collocation_points(jax.random.fold_in(jax.random.key(3), 7), sizes)
        for expected, actual in zip(first, second):
            np.testing.assert_array_equal(expected, actual)
        self.assertFalse(np.array_equal(first[0], third[0]))

    def test_rad_probability_matches_paper_equation(self):
        residuals = np.asarray([0.0, 1.0, 3.0])
        probabilities = rad_probabilities(residuals, k=1.0, c=1.0)
        expected_weights = residuals / residuals.mean() + 1.0
        np.testing.assert_allclose(
            probabilities, expected_weights / expected_weights.sum()
        )
        np.testing.assert_array_equal(
            rad_probabilities(np.zeros(4), k=1.0, c=1.0),
            np.full(4, 0.25),
        )

    def test_rad_bootstrap_is_reproducible_and_biases_large_residuals(self):
        def residual_evaluator(_params, x_values, _y_values):
            return x_values + 1.0

        arguments = {
            "candidate_count": 1000,
            "candidate_batch_size": 128,
            "k": 1.0,
            "c": 1.0,
        }
        first, first_diagnostics = build_rad_candidate_distribution(
            {}, residual_evaluator, np.random.default_rng(42), **arguments
        )
        second, second_diagnostics = build_rad_candidate_distribution(
            {}, residual_evaluator, np.random.default_rng(42), **arguments
        )
        for expected, actual in zip(first, second):
            np.testing.assert_array_equal(expected, actual)
        self.assertEqual(first_diagnostics, second_diagnostics)
        first_sample = bootstrap_rad_pde_points(jax.random.key(91), first, 200)
        repeated_sample = bootstrap_rad_pde_points(
            jax.random.key(91), first, 200
        )
        fresh_sample = bootstrap_rad_pde_points(jax.random.key(92), first, 200)
        for expected, actual in zip(first_sample, repeated_sample):
            np.testing.assert_array_equal(expected, actual)
        self.assertFalse(np.array_equal(first_sample[0], fresh_sample[0]))
        self.assertLess(np.unique(np.asarray(first_sample[0])).size, 200)
        self.assertGreater(
            float(jnp.mean(first_sample[0] + 1.0)),
            first_diagnostics["candidate_residual_mean"],
        )

    def test_initial_rad_bootstrap_pool_is_uniform(self):
        distribution = uniform_rad_candidate_distribution(
            np.random.default_rng(7), 1000
        )
        sample = bootstrap_rad_pde_points(jax.random.key(8), distribution, 5000)
        self.assertLess(abs(float(jnp.mean(sample[0]))), 0.05)
        self.assertLess(abs(float(jnp.mean(sample[1]))), 0.05)

    def test_mixed_rad_points_replace_a_fixed_uniform_tail(self):
        key = jax.random.key(18)
        sizes = (8, 4, 4)
        baseline = sample_collocation_points(key, sizes)
        distribution = (
            jnp.asarray([2.0, 3.0]),
            jnp.asarray([4.0, 5.0]),
            jnp.asarray([0.5, 1.0]),
        )
        mixed = sample_mixed_rad_collocation_points(
            key, sizes, distribution, rad_points=2
        )

        self.assertEqual(mixed[0].shape, (8,))
        self.assertEqual(mixed[1].shape, (8,))
        np.testing.assert_array_equal(mixed[0][:6], baseline[0][:6])
        np.testing.assert_array_equal(mixed[1][:6], baseline[1][:6])
        np.testing.assert_array_equal(mixed[2], baseline[2])
        np.testing.assert_array_equal(mixed[3], baseline[3])
        self.assertTrue(np.isin(np.asarray(mixed[0][6:]), [2.0, 3.0]).all())
        self.assertTrue(np.isin(np.asarray(mixed[1][6:]), [4.0, 5.0]).all())

    def test_small_adam_update_for_each_variant(self):
        small = replace(
            self.config,
            fourier_features=4,
            hidden_layers=(8, 8),
            collocation_adam=(8, 4, 4),
            rad_points=2,
        )
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                params, basis = initialize_model(jax.random.key(6), small, variant)
                context = _build_context(small, basis)
                optimizer = make_adam_optimizer(params, small, variant, adam_steps=2)
                state = optimizer.init(params)
                if variant in RAD_VARIANTS:
                    step = make_rad_adam_step(
                        optimizer, context, variant, small.collocation_adam
                    )
                    distribution = uniform_rad_candidate_distribution(
                        np.random.default_rng(13), 32
                    )
                    updated, _, objective, _ = step(
                        params,
                        state,
                        jax.random.key(7),
                        distribution,
                    )
                else:
                    step = make_adam_step(
                        optimizer, context, variant, small.collocation_adam
                    )
                    updated, _, objective, _ = step(
                        params, state, jax.random.key(7)
                    )
                jax.block_until_ready(objective)
                self.assertTrue(np.isfinite(float(objective)))
                self.assertTrue(
                    any(
                        not np.array_equal(before, after)
                        for before, after in zip(
                            jax.tree_util.tree_leaves(params),
                            jax.tree_util.tree_leaves(updated),
                        )
                    )
                )

    def test_small_lbfgs_update(self):
        small = replace(
            self.config,
            fourier_features=2,
            hidden_layers=(4,),
            collocation_lbfgs=(4, 2, 2),
        )
        params, basis = initialize_model(
            jax.random.key(12), small, "adapted_scattered"
        )
        context = _build_context(small, basis)
        points = regular_collocation_points(small.collocation_lbfgs, context)
        optimizer, step = make_lbfgs_step(
            context, "adapted_scattered", points
        )
        updated, _, objective = step(params, optimizer.init(params))
        jax.block_until_ready(objective)
        self.assertTrue(np.isfinite(float(objective)))
        self.assertTrue(
            any(
                not np.array_equal(before, after)
                for before, after in zip(
                    jax.tree_util.tree_leaves(params),
                    jax.tree_util.tree_leaves(updated),
                )
            )
        )

    def test_complex_l2_h1_metrics(self):
        fem = np.asarray([1.0 + 1j, 2.0 - 1j, 0.5j])
        error = np.asarray([0.1, 0.1, 0.1])
        mass = _diagonal_matrix(3, 1.0)
        stiffness = _diagonal_matrix(3, 2.0)
        metrics = compute_misfit_metrics(fem, fem + error, mass, stiffness)
        expected = np.linalg.norm(error) / np.linalg.norm(fem)
        self.assertAlmostEqual(metrics.l2_relative, expected)
        self.assertAlmostEqual(metrics.h1_relative, expected)

    def test_checkpoint_round_trip(self):
        params, basis = initialize_model(
            jax.random.key(8), self.config, "adapted_scattered"
        )
        context = _build_context(self.config, basis)
        metadata = {"variant": "adapted_scattered", "seed": 8}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            save_checkpoint(path, params, context, metadata)
            loaded, loaded_metadata, loaded_basis = load_checkpoint(path)
        self.assertEqual(loaded_metadata["variant"], metadata["variant"])
        self.assertEqual(loaded_metadata["seed"], metadata["seed"])
        self.assertEqual(loaded_metadata["manifest_schema_version"], 1)
        np.testing.assert_array_equal(loaded_basis, basis)
        for expected, actual in zip(
            jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(loaded)
        ):
            np.testing.assert_array_equal(expected, actual)

    def test_independent_evaluation_cadences(self):
        self.assertTrue(_evaluation_due(200, 200, 1000))
        self.assertFalse(_evaluation_due(200, 1000, 5000))
        self.assertTrue(_evaluation_due(5000, 1000, 5000))

    def test_run_aggregation_uses_sample_statistics(self):
        rows = []
        for variant in ("classical_total", "adapted_total"):
            for seed, value in enumerate((1.0, 2.0, 3.0)):
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "optimizer_seconds": value,
                        "adam_optimizer_seconds": value * 0.75,
                        "lbfgs_optimizer_seconds": value * 0.25,
                        "rad_resampling_seconds": 0.0,
                        "training_seconds": value,
                        "final_l2_relative": value / 10.0,
                        "final_h1_relative": value / 8.0,
                        "final_l2_absolute": value / 5.0,
                        "final_h1_absolute": value / 4.0,
                    }
                )
        aggregate = aggregate_runs(pd.DataFrame(rows)).set_index("variant")
        self.assertEqual(int(aggregate.loc["classical_total", "seed_count"]), 3)
        self.assertAlmostEqual(
            aggregate.loc["classical_total", "optimizer_seconds_mean"], 2.0
        )
        self.assertAlmostEqual(
            aggregate.loc["classical_total", "optimizer_seconds_std"], 1.0
        )

    def test_report_figures_accept_native_cadences(self):
        run_rows = []
        loss_rows = []
        fem_rows = []
        gradient_rows = []
        for variant_index, variant in enumerate(
            ("classical_total", "adapted_total", "adapted_scattered")
        ):
            for seed in (0, 1):
                base = 1.0 + variant_index + 0.1 * seed
                run_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "optimizer_seconds": 3.0 * base,
                        "adam_optimizer_seconds": 2.0 * base,
                        "lbfgs_optimizer_seconds": base,
                        "rad_resampling_seconds": 0.0,
                        "training_seconds": 3.0 * base,
                    }
                )
                for step in (0, 200, 400):
                    loss_rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "global_step": step,
                            "unweighted_total": base / (step + 1),
                            "pde_loss": 0.7 * base / (step + 1),
                            "bc_loss": 0.3 * base / (step + 1),
                        }
                    )
                for step in (0, 1000):
                    fem_rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "global_step": step,
                            "training_seconds": base * step / 100.0,
                            "l2_relative": base / (step + 2),
                            "h1_relative": 1.2 * base / (step + 2),
                        }
                    )
                for step in (500, 1000):
                    gradient_rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "phase": "adam",
                            "local_step": step,
                            "global_step": step,
                            "pde_gradient_l2_norm": base / step,
                            "bc_gradient_l2_norm": 0.5 * base / step,
                            "pde_bc_gradient_cosine": 0.1 * variant_index,
                        }
                    )
        figures = (
            create_loss_figure(pd.DataFrame(loss_rows)),
            create_error_figure(pd.DataFrame(fem_rows)),
            create_error_time_figure(pd.DataFrame(fem_rows)),
            create_timing_figure(pd.DataFrame(run_rows)),
            create_gradient_figure(pd.DataFrame(gradient_rows)),
        )
        self.assertEqual([len(figure.axes) for figure in figures], [3, 2, 2, 4, 2])
        for figure in figures:
            matplotlib.pyplot.close(figure)

    def test_campaign_outputs_include_gradient_history_when_present(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            run = root / "runs" / "adapted_total_seed0_cfgtest"
            run.mkdir(parents=True)
            summary = {
                "variant": "adapted_total",
                "seed": 0,
                "adam_steps": 1,
                "lbfgs_steps": 0,
                "optimizer_seconds": 1.0,
                "adam_optimizer_seconds": 1.0,
                "lbfgs_optimizer_seconds": 0.0,
                "rad_resampling_seconds": 0.0,
                "training_seconds": 1.0,
                "final_l2_relative": 0.2,
                "final_h1_relative": 0.3,
                "final_l2_absolute": 0.4,
                "final_h1_absolute": 0.5,
            }
            with (run / "summary.json").open("w", encoding="utf-8") as stream:
                json.dump(summary, stream)
            pd.DataFrame(
                [
                    {
                        "record_type": "loss_monitor",
                        "variant": "adapted_total",
                        "seed": 0,
                        "phase": "adam",
                        "local_step": 1,
                        "global_step": 1,
                        "unweighted_total": 1.0,
                        "pde_loss": 0.7,
                        "bc_loss": 0.3,
                    }
                ]
            ).to_csv(run / "loss_history.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "record_type": "fem_metrics",
                        "reference_kind": "analytic_mode",
                        "variant": "adapted_total",
                        "seed": 0,
                        "phase": "adam",
                        "local_step": 1,
                        "global_step": 1,
                        "training_seconds": 1.0,
                        "l2_relative": 0.2,
                        "h1_relative": 0.3,
                    }
                ]
            ).to_csv(run / "fem_metrics.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "record_type": "gradient_monitor",
                        "variant": "adapted_total",
                        "seed": 0,
                        "phase": "adam",
                        "local_step": 1,
                        "global_step": 1,
                        "pde_gradient_l2_norm": 2.0,
                        "bc_gradient_l2_norm": 1.0,
                        "pde_bc_gradient_cosine": 0.25,
                    }
                ]
            ).to_csv(run / "gradient_history.csv", index=False)

            output = write_campaign_outputs(root, Path(directory) / "report")

            self.assertTrue((output / "gradient_history_all.csv").is_file())
            self.assertTrue((output / "gradient_stats.png").is_file())
            gradients = pd.read_csv(output / "gradient_history_all.csv")
            self.assertEqual(gradients["record_type"].tolist(), ["gradient_monitor"])

    def test_log_band_floor_uses_observed_scale(self):
        # mean - sample_std is negative for these two positive observations.
        # The log-band floor must follow the data scale rather than 1e-308.
        frame = pd.DataFrame(
            {
                "variant": ["adapted_scattered", "adapted_scattered"],
                "seed": [0, 1],
                "global_step": [1000, 1000],
                "l2_relative": [1.0e-6, 1.0e-3],
                "h1_relative": [2.0e-6, 2.0e-3],
            }
        )
        self.assertEqual(_positive_plot_floor(frame, "l2_relative"), 5.0e-7)
        figure = create_error_figure(frame)
        for axis in figure.axes:
            self.assertGreater(axis.get_ylim()[0], 1.0e-12)
        matplotlib.pyplot.close(figure)


if __name__ == "__main__":
    unittest.main()
