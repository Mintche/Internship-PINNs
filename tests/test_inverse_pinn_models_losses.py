from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import inverse_PINN.losses as losses_module
from inverse_PINN.config import Case, GeometryConfig, ModelConfig
from inverse_PINN.data import BoundaryTrace
from inverse_PINN.losses import (
    build_case_context,
    build_physics_context,
    material_objective,
    pair_to_complex,
    pointwise_pde_residual,
    pressure_loss_components,
)
from inverse_PINN.models import (
    field_sigma,
    field_value,
    field_value_gradient_laplacian,
    initialize_field_model,
    initialize_material_model,
    material_physical_gradient,
    material_value,
)
from inverse_PINN.sampling import CollocationPoints
from inverse_PINN.variants import parse_variant


GEOMETRY = GeometryConfig(0.6, 1.0, 340.0, (0.7, 1.2), ())
CASE = Case(1200.0, 0, -1)


@pytest.mark.parametrize("name", ["fourier_total", "fourier_modified_total"])
def test_analytic_field_derivatives_match_autodiff(name):
    variant = parse_variant(name)
    model = initialize_field_model(
        jax.random.key(2), ModelConfig(3, (5, 5), (4,)), GEOMETRY, CASE, variant
    )
    x, y = jnp.asarray(0.17), jnp.asarray(-0.31)
    value, gradient, laplacian = field_value_gradient_laplacian(
        model.params, model.b_base, GEOMETRY, variant, x, y
    )
    function = lambda xv, yv: field_value(
        model.params, model.b_base, GEOMETRY, variant, xv, yv
    )
    autodiff_gradient = jnp.stack(
        (jax.jacfwd(function, 0)(x, y) / GEOMETRY.half_length,
         jax.jacfwd(function, 1)(x, y) * 2.0 / GEOMETRY.height)
    )
    autodiff_laplacian = (
        jax.jacfwd(jax.jacfwd(function, 0), 0)(x, y) / GEOMETRY.half_length**2
        + jax.jacfwd(jax.jacfwd(function, 1), 1)(x, y) * 4.0 / GEOMETRY.height**2
    )
    np.testing.assert_allclose(value, function(x, y), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(gradient, autodiff_gradient, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(laplacian, autodiff_laplacian, rtol=5e-4, atol=5e-4)


def test_initialization_is_deterministic_and_formulation_independent():
    models = ModelConfig(3, (5,), (4,))
    total = initialize_field_model(
        jax.random.key(8), models, GEOMETRY, CASE, parse_variant("fourier_total")
    )
    scattered = initialize_field_model(
        jax.random.key(8), models, GEOMETRY, CASE, parse_variant("fourier_scattered")
    )
    for left, right in zip(
        jax.tree_util.tree_leaves((total.params, total.b_base)),
        jax.tree_util.tree_leaves((scattered.params, scattered.b_base)),
    ):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_allclose(
        field_sigma(CASE, GEOMETRY),
        [2 * np.pi * CASE.frequency / GEOMETRY.c0, np.pi / GEOMETRY.height],
    )
    assert np.max(np.abs(np.asarray(total.params["layers"][-1]["W"]))) < 0.2


def test_material_starts_at_background_is_bounded_and_has_physical_gradient():
    params = initialize_material_model(jax.random.key(3), ModelConfig(2, (4,), (5,)), GEOMETRY)
    assert float(material_value(params, GEOMETRY, 0.0, 0.0)) == pytest.approx(
        GEOMETRY.m0, rel=2e-6
    )
    for x, y in ((-1.0, -1.0), (0.2, 0.7), (1.0, 1.0)):
        value = float(material_value(params, GEOMETRY, x, y))
        assert GEOMETRY.m_min < value < GEOMETRY.m_max
    x, y = jnp.asarray(0.1), jnp.asarray(-0.2)
    gradient = material_physical_gradient(params, GEOMETRY, x, y)
    function = lambda xv, yv: material_value(params, GEOMETRY, xv, yv)
    expected = jnp.asarray(
        [jax.grad(function, 0)(x, y) / GEOMETRY.half_length,
         jax.grad(function, 1)(x, y) * 2.0 / GEOMETRY.height]
    )
    np.testing.assert_allclose(gradient, expected, rtol=1e-6, atol=1e-10)


def _physics_fixture(variant_name="fourier_total"):
    variant = parse_variant(variant_name)
    physics = build_physics_context(GEOMETRY, (CASE,))
    model_config = ModelConfig(3, (5,), (4,))
    field = initialize_field_model(
        jax.random.key(5), model_config, GEOMETRY, CASE, variant
    )
    material = initialize_material_model(jax.random.key(6), model_config, GEOMETRY)
    y = np.linspace(0.0, GEOMETRY.height, 5)
    beta = 2.0 * np.pi * CASE.frequency / GEOMETRY.c0
    scale = np.sqrt(1.0 / GEOMETRY.height)
    left_incident = scale * np.exp(1j * beta * -GEOMETRY.half_length)
    right_incident = scale * np.exp(1j * beta * GEOMETRY.half_length)
    boundary = BoundaryTrace(
        CASE, beta, -GEOMETRY.half_length, y,
        np.full_like(y, left_incident + 0.2 + 0.1j, dtype=np.complex128),
        GEOMETRY.half_length, y,
        np.full_like(y, right_incident + 0.2 + 0.1j, dtype=np.complex128),
    )
    context = build_case_context(physics, CASE, field.b_base, boundary, variant)
    points = CollocationPoints(
        jnp.asarray([-0.7, -0.1, 0.3, 0.8]),
        jnp.asarray([-0.6, 0.2, 0.7, -0.1]),
        jnp.asarray([-0.4, 0.5]),
        jnp.asarray([-0.5, 0.6]),
    )
    return variant, physics, field, material, context, points


@pytest.mark.parametrize("variant_name", ["fourier_total", "fourier_scattered"])
def test_pde_neumann_dtn_normalizations_and_scattered_data(variant_name):
    variant, physics, field, material, context, points = _physics_fixture(variant_name)
    components = pressure_loss_components(
        field.params, material, physics, context, variant, points
    )
    residual = jax.vmap(
        lambda x, y: pointwise_pde_residual(
            field.params, material, physics, context, variant, x, y
        )
    )(points.x_pde, points.y_pde)
    k0 = 2.0 * np.pi * CASE.frequency / GEOMETRY.c0
    assert float(components["pde"]) == pytest.approx(
        float(jnp.mean((residual / k0**2) ** 2)), rel=2e-6
    )
    terms = lambda x, y: field_value_gradient_laplacian(
        field.params, field.b_base, GEOMETRY, variant, x, y
    )
    top = jax.vmap(lambda x: terms(x, 1.0)[1][1])(points.x_neumann)
    bottom = jax.vmap(lambda x: terms(x, -1.0)[1][1])(points.x_neumann)
    ky = np.pi / GEOMETRY.height
    expected_neumann = jnp.mean((top / ky) ** 2 + (bottom / ky) ** 2)
    assert float(components["neumann"]) == pytest.approx(float(expected_neumann), rel=2e-6)
    assert float(components["dtn"]) == pytest.approx(
        float(components["dtn_left"] + components["dtn_right"]), rel=2e-6
    )
    assert np.isfinite(float(components["dtn"]))
    if variant.scattered:
        scattered_target = context.field_scale * pair_to_complex(context.target_left)
        np.testing.assert_allclose(
            scattered_target, np.full(scattered_target.shape, 0.2 + 0.1j),
            rtol=2e-5, atol=2e-5,
        )


def test_tv_keeps_a_fixed_weight_in_material_objective():
    variant, physics, field, material, context, points = _physics_fixture()
    objective, components = material_objective(
        material, (field.params,), physics, (context,), variant, points,
        jnp.asarray([7.0]), tv_weight=0.3, tv_epsilon_squared=1e-12,
    )
    assert float(objective) == pytest.approx(
        float(components["pde_objective"] + 0.3 * components["tv"]), rel=2e-6
    )
    assert float(components["weighted_tv"]) == pytest.approx(
        0.3 * float(components["tv"]), rel=2e-6
    )


def test_tv_value_and_gradient_are_not_evaluated_when_disabled(monkeypatch):
    variant, physics, field, material, context, points = _physics_fixture()

    def forbidden_tv(*args, **kwargs):
        raise AssertionError("TV must not be evaluated when its weight is zero")

    monkeypatch.setattr(losses_module, "material_tv", forbidden_tv)
    objective, components = losses_module.material_objective(
        material, (field.params,), physics, (context,), variant, points,
        jnp.asarray([1.0]), tv_weight=0.0, tv_epsilon_squared=1e-12,
    )
    assert float(objective) == pytest.approx(float(components["pde_objective"]))
    assert float(components["tv"]) == 0.0
    diagnostics = losses_module.material_snapshot_statistics(
        material, (field.params,), physics, (context,), variant, points,
        jnp.asarray([1.0]), tv_weight=0.0, tv_epsilon_squared=1e-12,
    )
    assert diagnostics["tv"] is None
    assert diagnostics["tv_gradient_norm"] is None
