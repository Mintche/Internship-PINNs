"""Pressure and common bounded-slowness neural networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .config import Case, GeometryConfig, ModelConfig
from .variants import VariantSpec


Params = Mapping[str, Any]


@dataclass(frozen=True)
class FieldModel:
    params: dict[str, Any]
    b_base: jax.Array


def init_layers(key: jax.Array, dimensions: Sequence[int]) -> list[dict[str, jax.Array]]:
    dimensions = tuple(int(value) for value in dimensions)
    if len(dimensions) < 2 or min(dimensions) <= 0:
        raise ValueError("Layer dimensions must contain at least two positive values")
    keys = jax.random.split(key, len(dimensions) - 1)
    layers = []
    for layer_key, fan_in, fan_out in zip(keys, dimensions[:-1], dimensions[1:]):
        scale = jnp.sqrt(2.0 / (fan_in + fan_out))
        layers.append(
            {
                "W": jax.random.normal(layer_key, (fan_in, fan_out), dtype=jnp.float32) * scale,
                "b": jnp.zeros((fan_out,), dtype=jnp.float32),
            }
        )
    return layers


def field_sigma(case: Case, geometry: GeometryConfig) -> jax.Array:
    k0 = 2.0 * jnp.pi * case.frequency / geometry.c0
    transverse = case.mode * jnp.pi / geometry.height
    if not bool(k0 > transverse):
        raise ValueError(f"Incident case {case.id} is evanescent")
    kx = jnp.sqrt(k0**2 - transverse**2)
    ky = max(case.mode, 1) * jnp.pi / geometry.height
    return jnp.asarray((kx, ky), dtype=jnp.float32)


def initialize_field_model(
    key: jax.Array,
    models: ModelConfig,
    geometry: GeometryConfig,
    case: Case,
    variant: VariantSpec,
) -> FieldModel:
    basis_key, layers_key = jax.random.split(key)
    b_base = jax.random.normal(
        basis_key, (models.fourier_features, 2), dtype=jnp.float32
    )
    feature_width = 2 * models.fourier_features
    if variant.modified:
        if len(set(models.field_hidden_layers)) != 1:
            raise ValueError("fourier_modified requires constant field hidden widths")
        width = models.field_hidden_layers[0]
        keys = jax.random.split(layers_key, 4)
        encoder_u = init_layers(keys[0], (feature_width, width))[0]
        encoder_v = init_layers(keys[1], (feature_width, width))[0]
        gates = init_layers(keys[2], (feature_width, *models.field_hidden_layers))
        output = init_layers(keys[3], (width, 2))[0]
        output["W"] = output["W"] / 10.0
        layers = [encoder_u, encoder_v, *gates, output]
    else:
        layers = init_layers(
            layers_key, (feature_width, *models.field_hidden_layers, 2)
        )
        layers[-1]["W"] = layers[-1]["W"] / 10.0
    return FieldModel(
        params={"layers": layers, "sigma": field_sigma(case, geometry)},
        b_base=b_base,
    )


def initialize_material_model(
    key: jax.Array, models: ModelConfig, geometry: GeometryConfig
) -> dict[str, Any]:
    layers = init_layers(key, (2, *models.material_hidden_layers, 1))
    fraction = (geometry.m0 - geometry.m_min) / (geometry.m_max - geometry.m_min)
    if not 0.0 < fraction < 1.0:
        raise ValueError("The homogeneous slowness must lie strictly inside its bounds")
    bias = np.log(fraction / (1.0 - fraction))
    layers[-1]["b"] = jnp.full_like(layers[-1]["b"], bias)
    layers[-1]["W"] = layers[-1]["W"] / 10.0
    return {"layers": layers}


def pack_field_parameters(
    parameters: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Stack equal-architecture pressure networks along an acquisition axis."""
    parameters = tuple(parameters)
    if not parameters:
        raise ValueError("At least one pressure network is required")
    reference = jax.tree_util.tree_structure(parameters[0])
    if any(jax.tree_util.tree_structure(item) != reference for item in parameters[1:]):
        raise ValueError("All packed pressure networks must share one architecture")
    try:
        return jax.tree_util.tree_map(
            lambda *leaves: jnp.stack(leaves, axis=0), *parameters
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Pressure-network parameter shapes are incompatible") from error


def unpack_field_parameters(
    packed_parameters: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Return acquisition views from a packed pressure-parameter pytree."""
    leaves = jax.tree_util.tree_leaves(packed_parameters)
    if not leaves or leaves[0].ndim == 0:
        raise ValueError("Invalid packed pressure parameters")
    count = int(leaves[0].shape[0])
    if count <= 0 or any(leaf.ndim == 0 or leaf.shape[0] != count for leaf in leaves):
        raise ValueError("Packed pressure leaves must share a positive leading axis")
    return tuple(
        jax.tree_util.tree_map(lambda leaf, index=index: leaf[index], packed_parameters)
        for index in range(count)
    )


def _standard_layers(layers: Sequence[Params], values: jax.Array) -> jax.Array:
    for layer in layers[:-1]:
        values = jax.nn.tanh(values @ layer["W"] + layer["b"])
    return values @ layers[-1]["W"] + layers[-1]["b"]


def _modified_layers(layers: Sequence[Params], values: jax.Array) -> jax.Array:
    encoder_u = jax.nn.tanh(values @ layers[0]["W"] + layers[0]["b"])
    encoder_v = jax.nn.tanh(values @ layers[1]["W"] + layers[1]["b"])
    hidden = values
    for layer in layers[2:-1]:
        gate = jax.nn.tanh(hidden @ layer["W"] + layer["b"])
        hidden = encoder_u + gate * (encoder_v - encoder_u)
    return hidden @ layers[-1]["W"] + layers[-1]["b"]


def fourier_features(
    params: Params,
    b_base: jax.Array,
    geometry: GeometryConfig,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    wavevectors = b_base * params["sigma"]
    x_physical = x * geometry.half_length
    y_physical = (y + 1.0) * geometry.height / 2.0
    projection = wavevectors[:, 0] * x_physical + wavevectors[:, 1] * y_physical
    return jnp.concatenate((jnp.cos(projection), jnp.sin(projection)))


def field_value(
    params: Params,
    b_base: jax.Array,
    geometry: GeometryConfig,
    variant: VariantSpec,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    values = fourier_features(params, b_base, geometry, x, y)
    if variant.modified:
        return _modified_layers(params["layers"], values)
    return _standard_layers(params["layers"], values)


def _tanh_terms(value, gradient, laplacian, layer):
    preactivation = value @ layer["W"] + layer["b"]
    preactivation_gradient = gradient @ layer["W"]
    preactivation_laplacian = laplacian @ layer["W"]
    value = jax.nn.tanh(preactivation)
    first = 1.0 - value**2
    gradient = first[None, :] * preactivation_gradient
    laplacian = first * (
        preactivation_laplacian
        - 2.0 * value * jnp.sum(preactivation_gradient**2, axis=0)
    )
    return value, gradient, laplacian


def _linear_terms(value, gradient, laplacian, layer):
    return (
        value @ layer["W"] + layer["b"],
        gradient @ layer["W"],
        laplacian @ layer["W"],
    )


def field_value_gradient_laplacian(
    params: Params,
    b_base: jax.Array,
    geometry: GeometryConfig,
    variant: VariantSpec,
    x: jax.Array,
    y: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Propagate value, physical gradient, and physical Laplacian analytically."""
    wavevectors = b_base * params["sigma"]
    x_physical = x * geometry.half_length
    y_physical = (y + 1.0) * geometry.height / 2.0
    projection = wavevectors[:, 0] * x_physical + wavevectors[:, 1] * y_physical
    cosine, sine = jnp.cos(projection), jnp.sin(projection)
    value = jnp.concatenate((cosine, sine))
    projection_gradient = wavevectors.T
    gradient = jnp.concatenate(
        (-sine[None, :] * projection_gradient, cosine[None, :] * projection_gradient),
        axis=1,
    )
    frequency_squared = jnp.sum(projection_gradient**2, axis=0)
    laplacian = jnp.concatenate(
        (-cosine * frequency_squared, -sine * frequency_squared)
    )
    layers = params["layers"]
    if not variant.modified:
        for layer in layers[:-1]:
            value, gradient, laplacian = _tanh_terms(value, gradient, laplacian, layer)
        return _linear_terms(value, gradient, laplacian, layers[-1])

    encoder_u = _tanh_terms(value, gradient, laplacian, layers[0])
    encoder_v = _tanh_terms(value, gradient, laplacian, layers[1])
    hidden_value, hidden_gradient, hidden_laplacian = value, gradient, laplacian
    for layer in layers[2:-1]:
        gate_value, gate_gradient, gate_laplacian = _tanh_terms(
            hidden_value, hidden_gradient, hidden_laplacian, layer
        )
        delta_value = encoder_v[0] - encoder_u[0]
        delta_gradient = encoder_v[1] - encoder_u[1]
        delta_laplacian = encoder_v[2] - encoder_u[2]
        hidden_value = encoder_u[0] + gate_value * delta_value
        hidden_gradient = (
            encoder_u[1]
            + gate_gradient * delta_value[None, :]
            + gate_value[None, :] * delta_gradient
        )
        hidden_laplacian = (
            encoder_u[2]
            + gate_laplacian * delta_value
            + gate_value * delta_laplacian
            + 2.0 * jnp.sum(gate_gradient * delta_gradient, axis=0)
        )
    return _linear_terms(hidden_value, hidden_gradient, hidden_laplacian, layers[-1])


def material_value(
    params: Params,
    geometry: GeometryConfig,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    raw = _standard_layers(params["layers"], jnp.stack((x, y)))
    return (
        geometry.m_min
        + (geometry.m_max - geometry.m_min) * jax.nn.sigmoid(raw)
    ).squeeze()


def material_sound_speed(
    params: Params,
    geometry: GeometryConfig,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    return 1.0 / jnp.sqrt(material_value(params, geometry, x, y))


def material_physical_gradient(
    params: Params,
    geometry: GeometryConfig,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    def value_function(x_value, y_value):
        return material_value(params, geometry, x_value, y_value)

    dx = jax.grad(value_function, argnums=0)(x, y) / geometry.half_length
    dy = jax.grad(value_function, argnums=1)(x, y) * 2.0 / geometry.height
    return jnp.stack((dx, dy))


def parameter_count(params: Params) -> int:
    return int(sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params)))
