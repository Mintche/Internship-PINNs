"""Waveguide physics, normalized pressure losses, material loss, and metrics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .adaptive import tree_dot, tree_l2_norm
from .config import Case, GeometryConfig
from .data import BoundaryTrace, SymmetricCOOMatrix
from .models import (
    field_value,
    field_value_gradient_laplacian,
    material_physical_gradient,
    material_value,
    unpack_field_parameters,
)
from .sampling import CollocationPoints
from .variants import VariantSpec


FIELD_COMPONENTS = ("pde", "neumann", "dtn", "data")


@dataclass(frozen=True)
class PhysicsContext:
    geometry: GeometryConfig
    n_indices: jax.Array
    modal_scales: jax.Array
    y_quadrature_normalized: jax.Array
    quadrature_weights: jax.Array
    modal_basis_quadrature: jax.Array


@dataclass(frozen=True)
class CaseContext:
    case: Case
    b_base: jax.Array
    field_scale: float
    target_left: jax.Array
    target_right: jax.Array
    y_left_normalized: jax.Array
    y_right_normalized: jax.Array


@dataclass(frozen=True)
class PressureMetrics:
    l2_absolute: float
    l2_relative: float
    h1_absolute: float
    h1_relative: float


@dataclass(frozen=True)
class CelerityMetrics:
    anomaly_relative_l1: float | None
    global_relative_l1: float
    mean_absolute: float
    anomaly_mean_absolute: float | None
    background_mean_absolute: float | None
    rmse: float
    relative_l2: float
    max_absolute: float


def build_physics_context(
    geometry: GeometryConfig, cases: Sequence[Case]
) -> PhysicsContext:
    maximum_frequency = max(case.frequency for case in cases)
    n_modes = int(round(2.0 * geometry.height * maximum_frequency / geometry.c0)) + 5
    n_indices = jnp.arange(n_modes, dtype=jnp.float32)
    modal_scales = jnp.sqrt(2.0 / geometry.height) * jnp.ones(n_modes, dtype=jnp.float32)
    modal_scales = modal_scales.at[0].set(jnp.sqrt(1.0 / geometry.height))
    points, weights = np.polynomial.legendre.leggauss(3 * n_modes)
    y_physical = (points + 1.0) * geometry.height / 2.0
    physical_weights = weights * geometry.height / 2.0
    basis = np.asarray(modal_scales)[:, None] * np.cos(
        np.arange(n_modes)[:, None] * np.pi * y_physical[None, :] / geometry.height
    )
    return PhysicsContext(
        geometry,
        n_indices,
        modal_scales,
        jnp.asarray(points, dtype=jnp.float32),
        jnp.asarray(physical_weights, dtype=jnp.float32),
        jnp.asarray(basis, dtype=jnp.float32),
    )


def beta_modes(physics: PhysicsContext, frequency: float | jax.Array) -> jax.Array:
    k0 = 2.0 * jnp.pi * frequency / physics.geometry.c0
    return jnp.sqrt(k0**2 - (physics.n_indices * jnp.pi / physics.geometry.height) ** 2 + 0j)


def incident_wave_complex(
    physics: PhysicsContext,
    case: Case,
    x_normalized: jax.Array,
    y_normalized: jax.Array,
) -> jax.Array:
    beta = beta_modes(physics, case.frequency)[case.mode]
    x_physical = x_normalized * physics.geometry.half_length
    y_physical = (y_normalized + 1.0) * physics.geometry.height / 2.0
    mode_shape = physics.modal_scales[case.mode] * jnp.cos(
        case.mode * jnp.pi * y_physical / physics.geometry.height
    )
    return mode_shape * jnp.exp(-1j * case.incidence * beta * x_physical)


def complex_to_pair(value: jax.Array) -> jax.Array:
    return jnp.stack((jnp.real(value), jnp.imag(value)), axis=-1)


def pair_to_complex(value: jax.Array) -> jax.Array:
    return value[..., 0] + 1j * value[..., 1]


def build_case_context(
    physics: PhysicsContext,
    case: Case,
    b_base: jax.Array,
    boundary: BoundaryTrace,
    variant: VariantSpec,
) -> CaseContext:
    geometry = physics.geometry
    y_left = 2.0 * boundary.y_left / geometry.height - 1.0
    y_right = 2.0 * boundary.y_right / geometry.height - 1.0
    left = jnp.asarray(boundary.values_left)
    right = jnp.asarray(boundary.values_right)
    if variant.scattered:
        left = left - incident_wave_complex(
            physics, case, jnp.asarray(-1.0), jnp.asarray(y_left)
        )
        right = right - incident_wave_complex(
            physics, case, jnp.asarray(1.0), jnp.asarray(y_right)
        )
    scale = float(jnp.sqrt(jnp.max(jnp.concatenate((jnp.abs(left) ** 2, jnp.abs(right) ** 2)))))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Invalid boundary field scale for {case.id}")
    return CaseContext(
        case,
        b_base,
        scale,
        complex_to_pair(left / scale),
        complex_to_pair(right / scale),
        jnp.asarray(y_left, dtype=jnp.float32),
        jnp.asarray(y_right, dtype=jnp.float32),
    )


def pointwise_pde_residual(
    field_params: Mapping[str, Any],
    material_params: Mapping[str, Any],
    physics: PhysicsContext,
    context: CaseContext,
    variant: VariantSpec,
    x: jax.Array,
    y: jax.Array,
    *,
    homogeneous_material: bool = False,
) -> jax.Array:
    value, _, laplacian = field_value_gradient_laplacian(
        field_params, context.b_base, physics.geometry, variant, x, y
    )
    material = (
        jnp.asarray(physics.geometry.m0, dtype=jnp.float32)
        if homogeneous_material
        else material_value(material_params, physics.geometry, x, y)
    )
    omega_squared = (2.0 * jnp.pi * context.case.frequency) ** 2
    if variant.scattered:
        incident = complex_to_pair(
            incident_wave_complex(physics, context.case, x, y) / context.field_scale
        )
        coefficient = material * value + (material - physics.geometry.m0) * incident
    else:
        coefficient = material * value
    return laplacian + omega_squared * coefficient


def pressure_loss_components(
    field_params: Mapping[str, Any],
    material_params: Mapping[str, Any],
    physics: PhysicsContext,
    context: CaseContext,
    variant: VariantSpec,
    points: CollocationPoints,
    *,
    include_data: bool = True,
    homogeneous_material: bool = False,
) -> dict[str, jax.Array]:
    case = context.case
    k0 = 2.0 * jnp.pi * case.frequency / physics.geometry.c0
    kx = jnp.real(beta_modes(physics, case.frequency)[case.mode])
    ky = max(case.mode, 1) * jnp.pi / physics.geometry.height

    def value(x, y):
        return field_value(field_params, context.b_base, physics.geometry, variant, x, y)

    def terms(x, y):
        return field_value_gradient_laplacian(
            field_params, context.b_base, physics.geometry, variant, x, y
        )

    residual = jax.vmap(
        lambda x, y: pointwise_pde_residual(
            field_params,
            material_params,
            physics,
            context,
            variant,
            x,
            y,
            homogeneous_material=homogeneous_material,
        )
    )(points.x_pde, points.y_pde)
    pde = jnp.mean((residual / k0**2) ** 2)

    top = jax.vmap(lambda x: terms(x, 1.0)[1][1])(points.x_neumann)
    bottom = jax.vmap(lambda x: terms(x, -1.0)[1][1])(points.x_neumann)
    neumann = jnp.mean((top / ky) ** 2 + (bottom / ky) ** 2)

    def dtn_loss(x_boundary: float, sign: float) -> jax.Array:
        field_quad = jax.vmap(value, in_axes=(None, 0))(
            x_boundary, physics.y_quadrature_normalized
        )
        coefficients = physics.modal_basis_quadrature @ (
            physics.quadrature_weights * pair_to_complex(field_quad)
        )
        beta = beta_modes(physics, case.frequency)
        expected_modes = sign * 1j * beta * coefficients
        if not variant.scattered and int(x_boundary) == case.incidence:
            incident_coefficients = jnp.zeros(beta.shape, dtype=jnp.complex64)
            amplitude = jnp.exp(
                -1j
                * case.incidence
                * beta[case.mode]
                * x_boundary
                * physics.geometry.half_length
            ) / context.field_scale
            incident_coefficients = incident_coefficients.at[case.mode].set(amplitude)
            expected_modes = expected_modes - case.incidence * 2j * beta * incident_coefficients

        def expected_at(y_normalized):
            y_physical = (y_normalized + 1.0) * physics.geometry.height / 2.0
            basis = physics.modal_scales * jnp.cos(
                physics.n_indices * jnp.pi * y_physical / physics.geometry.height
            )
            return jnp.dot(basis, expected_modes)

        expected = jax.vmap(expected_at)(points.y_dtn)
        actual = jax.vmap(lambda y: terms(x_boundary, y)[1][0])(points.y_dtn)
        return jnp.mean(jnp.abs(pair_to_complex(actual) - expected) ** 2) / kx**2

    dtn_left = dtn_loss(-1.0, -1.0)
    dtn_right = dtn_loss(1.0, 1.0)
    dtn = dtn_left + dtn_right
    if include_data:
        prediction_left = jax.vmap(value, in_axes=(None, 0))(-1.0, context.y_left_normalized)
        prediction_right = jax.vmap(value, in_axes=(None, 0))(1.0, context.y_right_normalized)
        data = jnp.mean((prediction_left - context.target_left) ** 2) + jnp.mean(
            (prediction_right - context.target_right) ** 2
        )
    else:
        data = jnp.zeros((), dtype=jnp.float32)
    return {
        "pde": pde,
        "neumann": neumann,
        "dtn": dtn,
        "dtn_left": dtn_left,
        "dtn_right": dtn_right,
        "data": data,
    }


def pressure_objective(
    field_params: Mapping[str, Any],
    material_params: Mapping[str, Any],
    physics: PhysicsContext,
    context: CaseContext,
    variant: VariantSpec,
    points: CollocationPoints,
    weights: jax.Array | Sequence[float],
    *,
    include_data: bool = True,
    homogeneous_material: bool = False,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    components = pressure_loss_components(
        field_params,
        material_params,
        physics,
        context,
        variant,
        points,
        include_data=include_data,
        homogeneous_material=homogeneous_material,
    )
    weights = jnp.asarray(weights, dtype=jnp.float32)
    if weights.shape != (4,):
        raise ValueError("Pressure weights must have shape (4,)")
    objective = sum(weights[index] * components[name] for index, name in enumerate(FIELD_COMPONENTS))
    components = {
        **components,
        "objective": objective,
        "unweighted_total": sum(components[name] for name in FIELD_COMPONENTS),
    }
    return objective, components


def packed_pressure_objective(
    packed_field_params: Mapping[str, Any],
    material_params: Mapping[str, Any],
    physics: PhysicsContext,
    contexts: Sequence[CaseContext],
    b_bases: jax.Array,
    variant: VariantSpec,
    points: CollocationPoints,
    weights: jax.Array | Sequence[Sequence[float]],
    *,
    include_data: bool = True,
    homogeneous_material: bool = False,
    freeze_sigma: bool = False,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Evaluate one mean objective without exposing per-acquisition losses."""
    contexts = tuple(contexts)
    field_params = unpack_field_parameters(packed_field_params)
    weights = jnp.asarray(weights, dtype=jnp.float32)
    if weights.shape != (len(contexts), len(FIELD_COMPONENTS)):
        raise ValueError("Packed pressure weights have an invalid shape")
    if b_bases.shape[0] != len(contexts):
        raise ValueError("Packed Fourier bases have an invalid leading dimension")

    objectives = []
    components = []
    for params, context, b_base, case_weights in zip(
        field_params, contexts, b_bases, weights
    ):
        if freeze_sigma:
            params = {
                **params,
                "sigma": jax.lax.stop_gradient(params["sigma"]),
            }
        objective, values = pressure_objective(
            params,
            material_params,
            physics,
            replace(context, b_base=b_base),
            variant,
            points,
            case_weights,
            include_data=include_data,
            homogeneous_material=homogeneous_material,
        )
        objectives.append(objective)
        components.append(values)

    means = {
        name: jnp.mean(jnp.stack(tuple(values[name] for values in components)))
        for name in (
            "pde", "neumann", "dtn", "dtn_left", "dtn_right", "data",
            "unweighted_total",
        )
    }
    means["objective"] = jnp.mean(jnp.stack(tuple(objectives)))
    return means["objective"], means


def material_tv(
    material_params: Mapping[str, Any],
    geometry: GeometryConfig,
    points: CollocationPoints,
    epsilon_squared: float,
) -> jax.Array:
    def point_value(x, y):
        gradient = geometry.c0**2 * material_physical_gradient(
            material_params, geometry, x, y
        )
        return jnp.sqrt(jnp.sum(gradient**2) + epsilon_squared)

    return jnp.mean(jax.vmap(point_value)(points.x_pde, points.y_pde))


def material_case_pde(
    material_params: Mapping[str, Any],
    field_params: Mapping[str, Any],
    physics: PhysicsContext,
    context: CaseContext,
    variant: VariantSpec,
    points: CollocationPoints,
) -> jax.Array:
    residual = jax.vmap(
        lambda x, y: pointwise_pde_residual(
            field_params, material_params, physics, context, variant, x, y
        )
    )(points.x_pde, points.y_pde)
    k0 = 2.0 * jnp.pi * context.case.frequency / physics.geometry.c0
    return jnp.mean((residual / k0**2) ** 2)


def material_objective(
    material_params: Mapping[str, Any],
    field_params: Sequence[Mapping[str, Any]],
    physics: PhysicsContext,
    contexts: Sequence[CaseContext],
    variant: VariantSpec,
    points: CollocationPoints,
    case_weights: jax.Array | Sequence[float],
    *,
    tv_weight: float,
    tv_epsilon_squared: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    case_weights = jnp.asarray(case_weights, dtype=jnp.float32)
    if case_weights.shape != (len(contexts),):
        raise ValueError("Material case weights have an invalid shape")
    pdes = jnp.stack(
        tuple(
            material_case_pde(material_params, params, physics, context, variant, points)
            for params, context in zip(field_params, contexts)
        )
    )
    pde_objective = jnp.mean(case_weights * pdes)
    if float(tv_weight) != 0.0:
        tv = material_tv(
            material_params, physics.geometry, points, tv_epsilon_squared
        )
        weighted_tv = float(tv_weight) * tv
        objective = pde_objective + weighted_tv
    else:
        # Keep a stable auxiliary structure for JIT while completely removing
        # the TV value and its derivatives from the traced objective.
        tv = jnp.zeros((), dtype=pde_objective.dtype)
        weighted_tv = jnp.zeros((), dtype=pde_objective.dtype)
        objective = pde_objective
    return objective, {
        "case_pdes": pdes,
        "pde_objective": pde_objective,
        "tv": tv,
        "weighted_tv": weighted_tv,
        "objective": objective,
    }


def pressure_gradient_statistics(
    field_params,
    material_params,
    physics,
    context,
    variant,
    points,
    weights,
    *,
    freeze_sigma: bool,
    include_data: bool = True,
    homogeneous_material: bool = False,
) -> dict[str, jax.Array]:
    def component_gradient(component):
        def objective(params):
            if freeze_sigma:
                params = {**params, "sigma": jax.lax.stop_gradient(params["sigma"])}
            value, values = pressure_objective(
                params,
                material_params,
                physics,
                context,
                variant,
                points,
                weights,
                include_data=include_data,
                homogeneous_material=homogeneous_material,
            )
            return value if component == "objective" else values[component]
        return jax.grad(objective)(field_params)

    names = (*FIELD_COMPONENTS, "objective")
    gradients = {name: component_gradient(name) for name in names}
    return {f"{name}_gradient_l2_norm": tree_l2_norm(gradients[name]) for name in names}


def field_component_gradient_norms(
    field_params: Mapping[str, Any],
    material_params: Mapping[str, Any],
    physics: PhysicsContext,
    context: CaseContext,
    variant: VariantSpec,
    points: CollocationPoints,
    *,
    freeze_sigma: bool,
    include_data: bool = True,
    homogeneous_material: bool = False,
) -> jax.Array:
    """Return only the four raw field-component norms used by adweights."""

    def component_norm(component: str) -> jax.Array:
        def objective(candidate):
            if freeze_sigma:
                candidate = {
                    **candidate,
                    "sigma": jax.lax.stop_gradient(candidate["sigma"]),
                }
            return pressure_loss_components(
                candidate,
                material_params,
                physics,
                context,
                variant,
                points,
                include_data=include_data,
                homogeneous_material=homogeneous_material,
            )[component]

        return tree_l2_norm(jax.grad(objective)(field_params))

    return jnp.stack(tuple(component_norm(name) for name in FIELD_COMPONENTS))


def packed_field_component_gradient_norms(
    packed_field_params: Mapping[str, Any],
    material_params: Mapping[str, Any],
    physics: PhysicsContext,
    contexts: Sequence[CaseContext],
    b_bases: jax.Array,
    variant: VariantSpec,
    points: CollocationPoints,
    *,
    freeze_sigma: bool,
    include_data: bool = True,
    homogeneous_material: bool = False,
) -> jax.Array:
    """Return an ``(acquisition, component)`` matrix for a compiled evaluator."""
    contexts = tuple(contexts)
    field_params = unpack_field_parameters(packed_field_params)
    if len(field_params) != len(contexts):
        raise ValueError("Packed field parameters and contexts are not aligned")
    if b_bases.shape[0] != len(contexts):
        raise ValueError("Packed Fourier bases have an invalid leading dimension")
    return jnp.stack(
        tuple(
            field_component_gradient_norms(
                params,
                material_params,
                physics,
                replace(context, b_base=b_base),
                variant,
                points,
                freeze_sigma=freeze_sigma,
                include_data=include_data,
                homogeneous_material=homogeneous_material,
            )
            for params, context, b_base in zip(field_params, contexts, b_bases)
        )
    )


def material_snapshot_statistics(
    material_params,
    field_params: Sequence[Mapping[str, Any]],
    physics: PhysicsContext,
    contexts: Sequence[CaseContext],
    variant: VariantSpec,
    points: CollocationPoints,
    case_weights: Sequence[float] | jax.Array,
    *,
    tv_weight: float,
    tv_epsilon_squared: float,
) -> dict[str, Any]:
    gradients = []
    pde_values = []
    for params, context in zip(field_params, contexts):
        def objective(candidate):
            return material_case_pde(candidate, params, physics, context, variant, points)
        value, gradient = jax.value_and_grad(objective)(material_params)
        pde_values.append(value)
        gradients.append(gradient)

    norms = jnp.stack(tuple(tree_l2_norm(gradient) for gradient in gradients))
    dots = jnp.stack(
        tuple(
            jnp.stack(tuple(tree_dot(left, right) for right in gradients))
            for left in gradients
        )
    )
    denominator = norms[:, None] * norms[None, :]
    cosines = jnp.where(denominator > 0.0, jnp.clip(dots / denominator, -1.0, 1.0), jnp.nan)
    weights_array = jnp.asarray(case_weights, dtype=jnp.float32)
    weighted_pde_gradient = jax.tree_util.tree_map(
        lambda *leaves: sum(weight * leaf for weight, leaf in zip(weights_array, leaves)) / len(gradients),
        *gradients,
    )
    pde_values_array = jnp.stack(tuple(pde_values))
    aggregate_value = jnp.mean(weights_array * pde_values_array)
    if float(tv_weight) != 0.0:
        tv_value, tv_gradient = jax.value_and_grad(
            lambda candidate: material_tv(
                candidate, physics.geometry, points, tv_epsilon_squared
            )
        )(material_params)
        aggregate_value = aggregate_value + float(tv_weight) * tv_value
        aggregate_gradient = jax.tree_util.tree_map(
            lambda pde_leaf, tv_leaf: pde_leaf + float(tv_weight) * tv_leaf,
            weighted_pde_gradient,
            tv_gradient,
        )
        tv_gradient_norm: jax.Array | None = tree_l2_norm(tv_gradient)
    else:
        tv_value = None
        tv_gradient_norm = None
        aggregate_gradient = weighted_pde_gradient
    cancellation = tree_l2_norm(weighted_pde_gradient) / jnp.maximum(
        jnp.mean(weights_array * norms), jnp.finfo(jnp.float32).tiny
    )
    if len(gradients) > 1:
        upper = cosines[jnp.triu_indices(len(gradients), 1)]
        negative_fraction = jnp.mean(upper < 0.0)
    else:
        negative_fraction = jnp.asarray(jnp.nan, dtype=jnp.float32)
    return {
        "pde_values": pde_values_array,
        "pde_gradient_norms": norms,
        "cosines": cosines,
        "tv": tv_value,
        "tv_gradient_norm": tv_gradient_norm,
        "aggregate": aggregate_value,
        "aggregate_gradient_norm": tree_l2_norm(aggregate_gradient),
        "negative_pair_fraction": negative_fraction,
        "cancellation_ratio": cancellation,
    }


def material_case_gradient_norms(
    material_params,
    field_params: Sequence[Mapping[str, Any]],
    physics: PhysicsContext,
    contexts: Sequence[CaseContext],
    variant: VariantSpec,
    points: CollocationPoints,
) -> jax.Array:
    """Return only the PDE gradient norms required by material adweights.

    This deliberately does not form pairwise dot products. Full material-gradient
    diagnostics are reserved for configured snapshot events.
    """
    norms = []
    for params, context in zip(field_params, contexts):
        gradient = jax.grad(
            lambda candidate: material_case_pde(
                candidate, params, physics, context, variant, points
            )
        )(material_params)
        norms.append(tree_l2_norm(gradient))
    return jnp.stack(tuple(norms))


def packed_material_case_gradient_norms(
    material_params: Mapping[str, Any],
    packed_field_params: Mapping[str, Any],
    physics: PhysicsContext,
    contexts: Sequence[CaseContext],
    b_bases: jax.Array,
    variant: VariantSpec,
    points: CollocationPoints,
) -> jax.Array:
    """Return all material case-gradient norms from one compiled call."""
    contexts = tuple(contexts)
    field_params = unpack_field_parameters(packed_field_params)
    if len(field_params) != len(contexts):
        raise ValueError("Packed field parameters and contexts are not aligned")
    if b_bases.shape[0] != len(contexts):
        raise ValueError("Packed Fourier bases have an invalid leading dimension")
    dynamic_contexts = tuple(
        replace(context, b_base=b_base)
        for context, b_base in zip(contexts, b_bases)
    )
    return material_case_gradient_norms(
        material_params,
        field_params,
        physics,
        dynamic_contexts,
        variant,
        points,
    )


def physical_pressure_prediction(
    field_params,
    physics: PhysicsContext,
    context: CaseContext,
    variant: VariantSpec,
    x_physical: jax.Array,
    y_physical: jax.Array,
) -> jax.Array:
    x_normalized = x_physical / physics.geometry.half_length
    y_normalized = 2.0 * y_physical / physics.geometry.height - 1.0
    pairs = jax.vmap(
        lambda x, y: field_value(
            field_params, context.b_base, physics.geometry, variant, x, y
        )
    )(x_normalized, y_normalized)
    prediction = context.field_scale * pair_to_complex(pairs)
    if variant.scattered:
        prediction = prediction + jax.vmap(
            lambda x, y: incident_wave_complex(physics, context.case, x, y)
        )(x_normalized, y_normalized)
    return prediction


def pressure_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    mass: SymmetricCOOMatrix,
    stiffness: SymmetricCOOMatrix,
) -> PressureMetrics:
    reference = np.asarray(reference, dtype=np.complex128)
    prediction = np.asarray(prediction, dtype=np.complex128)
    error = prediction - reference
    l2_error = mass.quadratic_form(error)
    l2_reference = mass.quadratic_form(reference)
    h1_error = l2_error + stiffness.quadratic_form(error)
    h1_reference = l2_reference + stiffness.quadratic_form(reference)
    if l2_reference <= 0.0 or h1_reference <= 0.0:
        raise ValueError("Reference pressure has a zero FEM norm")
    return PressureMetrics(
        float(np.sqrt(l2_error)),
        float(np.sqrt(l2_error / l2_reference)),
        float(np.sqrt(h1_error)),
        float(np.sqrt(h1_error / h1_reference)),
    )


def celerity_metrics(
    truth: np.ndarray, prediction: np.ndarray, background: float
) -> CelerityMetrics:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - truth
    absolute = np.abs(error)
    anomaly = np.abs(truth - background)
    mask = anomaly > 1e-9 * max(1.0, background)
    background_mask = ~mask
    anomaly_reference = float(np.sum(anomaly))
    truth_l1 = float(np.sum(np.abs(truth)))
    truth_l2 = float(np.linalg.norm(truth.ravel()))
    return CelerityMetrics(
        float(np.sum(absolute) / anomaly_reference) if anomaly_reference > 0.0 else None,
        float(np.sum(absolute) / truth_l1),
        float(np.mean(absolute)),
        float(np.mean(absolute[mask])) if mask.any() else None,
        float(np.mean(absolute[background_mask])) if background_mask.any() else None,
        float(np.sqrt(np.mean(error**2))),
        float(np.linalg.norm(error.ravel()) / truth_l2),
        float(np.max(absolute)),
    )
