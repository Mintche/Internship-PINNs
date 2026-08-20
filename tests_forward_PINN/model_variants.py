from __future__ import annotations

from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

BASE_VARIANTS = (
    "classical_total",
    "fourier_total",
    "fourier_modified_total",
    "fourier_scattered",
    "fourier_modified_scattered",
)
ADWEIGHTS_VARIANTS = tuple(f"{variant}_adweights" for variant in BASE_VARIANTS)
RAD_VARIANTS = tuple(f"{variant}_rad" for variant in BASE_VARIANTS)
RAD_ADWEIGHTS_VARIANTS = tuple(
    f"{variant}_rad_adweights" for variant in BASE_VARIANTS
)
VARIANTS = (
    *BASE_VARIANTS,
    *ADWEIGHTS_VARIANTS,
    *RAD_VARIANTS,
    *RAD_ADWEIGHTS_VARIANTS,
)


def _tokens(variant: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in variant.split("_") if token)


def _validate_variant(variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}")


def uses_fourier_features(variant: str) -> bool:
    return "fourier" in _tokens(variant)


def uses_modified_mlp(variant: str) -> bool:
    return "modified" in _tokens(variant)


def uses_adweights(variant: str) -> bool:
    _validate_variant(variant)
    return "adweights" in _tokens(variant)


def base_variant(variant: str) -> str:
    """Return the architecture/physics variant without training modifiers."""
    _validate_variant(variant)
    for suffix in ("_rad_adweights", "_adweights", "_rad"):
        if variant.endswith(suffix):
            return variant.removesuffix(suffix)
    return variant


def is_scattered_variant(variant: str) -> bool:
    _validate_variant(variant)
    return "scattered" in _tokens(variant)


def is_rad_variant(variant: str) -> bool:
    _validate_variant(variant)
    return "rad" in _tokens(variant)


def variant_label(variant: str) -> str:
    _validate_variant(variant)
    pieces = []
    if "classical" in _tokens(variant):
        pieces.append("Classical")
    if uses_fourier_features(variant):
        pieces.append("Fourier")
    if uses_modified_mlp(variant):
        pieces.append("modified MLP")
    if is_scattered_variant(variant):
        pieces.append("scattered")
    elif "total" in _tokens(variant):
        pieces.append("total")
    if is_rad_variant(variant):
        pieces.append("RAD")
    if uses_adweights(variant):
        pieces.append("adaptive weights")
    return " ".join(pieces)


def variant_parameter_count(config: Any, variant: str) -> int:
    params, _ = initialize_model(jax.random.key(0), config, variant)
    return int(
        sum(
            np.prod(leaf.shape, dtype=np.int64)
            for leaf in jax.tree_util.tree_leaves(params)
        )
    )


def _init_layers(
    key: jax.Array, dimensions: Sequence[int]
) -> list[dict[str, jax.Array]]:
    keys = jax.random.split(key, len(dimensions) - 1)
    layers = []
    for layer_key, fan_in, fan_out in zip(keys, dimensions[:-1], dimensions[1:]):
        scale = jnp.sqrt(2.0 / (fan_in + fan_out))
        layers.append(
            {
                "W": jax.random.normal(layer_key, (fan_in, fan_out)) * scale,
                "b": jnp.zeros((fan_out,), dtype=jnp.float32),
            }
        )
    return layers


def _validate_modified_widths(config: Any) -> None:
    widths = tuple(int(width) for width in config.hidden_layers)
    if not widths or len(set(widths)) != 1:
        raise ValueError("modified MLP variants require constant hidden widths")


def _fourier_sigma_initialization(config: Any) -> jax.Array:
    k0 = 2.0 * jnp.pi * config.frequency / config.c0
    transverse = config.mode * jnp.pi / config.height
    if not bool(k0 > transverse):
        raise ValueError("The selected mode is evanescent")
    sigma_x = jnp.sqrt(k0**2 - transverse**2)
    sigma_y_index = config.mode + int(config.mode == 0)
    sigma_y = sigma_y_index * jnp.pi / config.height
    return jnp.asarray([sigma_x, sigma_y], dtype=jnp.float32)


def initialize_model(
    key: jax.Array,
    config: Any,
    variant: str,
) -> tuple[dict[str, Any], jax.Array]:
    """Initialize one field model and its fixed random Fourier basis."""
    _validate_variant(variant)
    basis_key, layers_key = jax.random.split(key)
    b_base = jax.random.normal(
        basis_key, (config.fourier_features, 2), dtype=jnp.float32
    )

    if not uses_fourier_features(variant):
        dimensions = (2, *config.hidden_layers, 2)
        return {"layers": _init_layers(layers_key, dimensions)}, b_base

    feature_width = 2 * config.fourier_features
    if uses_modified_mlp(variant):
        _validate_modified_widths(config)
        width = int(config.hidden_layers[0])
        keys = jax.random.split(layers_key, 4)
        encoders = [
            _init_layers(keys[0], (feature_width, width))[0],
            _init_layers(keys[1], (feature_width, width))[0],
        ]
        gate_dimensions = (feature_width, *config.hidden_layers)
        gate_layers = _init_layers(keys[2], gate_dimensions)
        output = _init_layers(keys[3], (width, 2))[0]
        output["W"] = output["W"] / 10.0
        return {
            "layers": [*encoders, *gate_layers, output],
            "sigma": _fourier_sigma_initialization(config),
        }, b_base

    dimensions = (feature_width, *config.hidden_layers, 2)
    layers = _init_layers(layers_key, dimensions)
    layers[-1]["W"] = layers[-1]["W"] / 10.0
    return {
        "layers": layers,
        "sigma": _fourier_sigma_initialization(config),
    }, b_base


def _forward_layers(
    layers: Sequence[Mapping[str, jax.Array]], values: jax.Array
) -> jax.Array:
    for layer in layers[:-1]:
        values = jax.nn.tanh(values @ layer["W"] + layer["b"])
    return values @ layers[-1]["W"] + layers[-1]["b"]


def _modified_forward_layers(
    layers: Sequence[Mapping[str, jax.Array]], values: jax.Array
) -> jax.Array:
    encoder_u = jax.nn.tanh(values @ layers[0]["W"] + layers[0]["b"])
    encoder_v = jax.nn.tanh(values @ layers[1]["W"] + layers[1]["b"])
    hidden = values
    for layer in layers[2:-1]:
        gate = jax.nn.tanh(hidden @ layer["W"] + layer["b"])
        hidden = (1.0 - gate) * encoder_u + gate * encoder_v
    output = layers[-1]
    return hidden @ output["W"] + output["b"]


def _fourier_features(
    params: Mapping[str, Any], context: Any, x: jax.Array, y: jax.Array
) -> jax.Array:
    x_physical = x * context.config.half_length
    y_physical = (y + 1.0) * context.config.height / 2.0
    wavevectors = context.b_base * params["sigma"]
    projection = wavevectors[:, 0] * x_physical + wavevectors[:, 1] * y_physical
    return jnp.concatenate((jnp.cos(projection), jnp.sin(projection)))


def model_value(
    params: Mapping[str, Any],
    context: Any,
    variant: str,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    _validate_variant(variant)
    if uses_fourier_features(variant):
        features = _fourier_features(params, context, x, y)
    else:
        features = jnp.stack((x, y))
    if uses_modified_mlp(variant):
        return _modified_forward_layers(params["layers"], features)
    return _forward_layers(params["layers"], features)


def _affine_tanh_terms(
    value: jax.Array,
    gradient: jax.Array,
    laplacian: jax.Array,
    layer: Mapping[str, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array]:
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


def _affine_linear_terms(
    value: jax.Array,
    gradient: jax.Array,
    laplacian: jax.Array,
    layer: Mapping[str, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return (
        value @ layer["W"] + layer["b"],
        gradient @ layer["W"],
        laplacian @ layer["W"],
    )


def _fourier_feature_terms(
    params: Mapping[str, Any], context: Any, x: jax.Array, y: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    wavevectors = context.b_base * params["sigma"]
    x_physical = x * context.config.half_length
    y_physical = (y + 1.0) * context.config.height / 2.0
    projection = wavevectors[:, 0] * x_physical + wavevectors[:, 1] * y_physical
    cosine = jnp.cos(projection)
    sine = jnp.sin(projection)
    value = jnp.concatenate((cosine, sine))
    projection_gradient = wavevectors.T
    gradient = jnp.concatenate(
        (
            -sine[None, :] * projection_gradient,
            cosine[None, :] * projection_gradient,
        ),
        axis=1,
    )
    frequency_squared = jnp.sum(projection_gradient**2, axis=0)
    laplacian = jnp.concatenate(
        (-cosine * frequency_squared, -sine * frequency_squared)
    )
    return value, gradient, laplacian


def fourier_value_gradient_laplacian(
    params: Mapping[str, Any],
    context: Any,
    variant: str,
    x: jax.Array,
    y: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Propagate value, physical gradient, and physical Laplacian analytically."""
    if not uses_fourier_features(variant):
        raise ValueError("Analytic Fourier derivatives require a Fourier variant")
    value, gradient, laplacian = _fourier_feature_terms(params, context, x, y)

    if not uses_modified_mlp(variant):
        for layer in params["layers"][:-1]:
            value, gradient, laplacian = _affine_tanh_terms(
                value, gradient, laplacian, layer
            )
        return _affine_linear_terms(value, gradient, laplacian, params["layers"][-1])

    encoder_u = _affine_tanh_terms(value, gradient, laplacian, params["layers"][0])
    encoder_v = _affine_tanh_terms(value, gradient, laplacian, params["layers"][1])
    hidden_value, hidden_gradient, hidden_laplacian = value, gradient, laplacian
    for layer in params["layers"][2:-1]:
        gate_value, gate_gradient, gate_laplacian = _affine_tanh_terms(
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
    return _affine_linear_terms(
        hidden_value, hidden_gradient, hidden_laplacian, params["layers"][-1]
    )


def classical_value_gradient_laplacian(
    params: Mapping[str, Any], context: Any, x: jax.Array, y: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    def value_function(x_value: jax.Array, y_value: jax.Array) -> jax.Array:
        return model_value(params, context, "classical_total", x_value, y_value)

    value = value_function(x, y)
    dx = jax.jacfwd(value_function, argnums=0)(x, y) / context.config.half_length
    dy = (
        jax.jacfwd(value_function, argnums=1)(x, y)
        * 2.0
        / context.config.height
    )
    dxx = (
        jax.jacfwd(jax.jacfwd(value_function, argnums=0), argnums=0)(x, y)
        / context.config.half_length**2
    )
    dyy = (
        jax.jacfwd(jax.jacfwd(value_function, argnums=1), argnums=1)(x, y)
        * (2.0 / context.config.height) ** 2
    )
    return value, jnp.stack((dx, dy)), dxx + dyy


def value_gradient_laplacian(
    params: Mapping[str, Any],
    context: Any,
    variant: str,
    x: jax.Array,
    y: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    _validate_variant(variant)
    if uses_fourier_features(variant):
        return fourier_value_gradient_laplacian(params, context, variant, x, y)
    return classical_value_gradient_laplacian(params, context, x, y)
