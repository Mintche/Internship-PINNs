import os
import functools
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", False)

import optax

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.data_loader import WaveguideBoundaryData, format_ratio_label
from tools.us_checkpoint import collect_us_norms, save_us_checkpoint

# ==============================================================================
# SECTION 1: CONFIGURATION & HYPERPARAMETERS
# ==============================================================================

# --- Domain Geometry ---
H = 0.6  # Height of the waveguide
L = 1.0  # Half-length of the waveguide

# --- Physics Parameters ---
c0 = 340.0
contrast_max = 0.4
cmin = c0 * (1 - contrast_max)
cmax = c0 * (1 + 0.01)
m0 = 1 / c0**2
ms_min = 1 / cmax**2 - m0
ms_max = 1 / cmin**2 - m0

# --- Data Configuration ---
# Data files follow the naming convention:
#   pinn_boundary_{left/right}_{defect_name}_ratio{c_defect/c0}.csv
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(REPOSITORY_ROOT, "FEM")
if jax.config.jax_compilation_cache_dir is None:
    jax.config.update(
        "jax_compilation_cache_dir",
        os.path.join(script_dir, "cache", "jax_compilation"),
    )
defect_name = "barhalf"
contrast_ratio = 0.8
data_ratio_label = format_ratio_label(contrast_ratio)

# Each package is trained as one packed selection of frequencies and modes.
# Example: [{600.0: [0, 1], 1200.0: [0, 1, 2]}]
training_packages = [
    {700.0: [0, 1, 2]},
    {700.0: [0, 1, 2], 1000.0: [0, 1, 2, 3]},
    #{600.0: [0, 2], 1000.0: [0, 3]},
    #{600.0: [0, 2], 1200.0: [0, 3]}
]

# Fractions of the Adam phase-2 budget where the slowness output is plotted.
m_snapshot = [0.3, 0.5, 0.8]

# --- Random Seed ---
random_seed = 0
key = jax.random.key(random_seed)

# --- Neural Network Architectures ---
m_fourier_features = 64
n_input = 2
n_layers_us = [2 * m_fourier_features, 128, 128, 64, 2]
n_layers_ms = [n_input, 128, 64, 1]

# --- Optimizer Learning Rates ---
lr_us = 1e-3
lr_ms = 3e-4
lr_sigma = 1e-2

# --- Training Hyperparameters ---
# Collocation sizes are ordered as (PDE interior, horizontal Neumann, vertical DtN).
N_adam = (4096, 256, 128)
N_lbfgs = (2048, 256, 128)
N_validation = (4096, 256, 128)
eval_interval = 200
switch_threshold = 0.0
switch_window = 10

max_steps_adam_warmup = 10001
max_steps_adam_inverse = 100001
max_steps_lbfgs_inverse = 201

SHOW_PLOTS = True

# Loss weights are ordered as (PDE, BC, data). Physics is active immediately
# and the data weight is ramped up, as in the full-field multimode inverse step.
weights_warmup = jnp.array([10.0, 1.0])
weights_inverse_final = jnp.array([1.0, 3.0, 10.0])

# ==============================================================================
# SECTION 2: DATA DISCOVERY
# ==============================================================================

def boundary_data_paths(data_dir, defect_name, contrast_ratio):
    ratio_label = format_ratio_label(contrast_ratio)
    suffix = f"{defect_name}_{ratio_label}.csv"
    left_path = os.path.join(data_dir, "pinn_data", f"pinn_boundary_left_{suffix}")
    right_path = os.path.join(data_dir, "pinn_data", f"pinn_boundary_right_{suffix}")
    missing = [path for path in (left_path, right_path) if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Missing combined FEM boundary file(s): " + ", ".join(missing)
        )
    return left_path, right_path


def flatten_package(package):
    """Return sorted (frequency, mode) pairs from one training package."""
    cases = []
    for frequency, modes in package.items():
        for mode_index in modes:
            cases.append((float(frequency), int(mode_index)))
    return sorted(cases, key=lambda item: (item[0], item[1]))


def flatten_packages(packages):
    cases = []
    for package in packages:
        cases.extend(flatten_package(package))
    return cases


def select_warmup_cases(package_index, cases, previously_trained_cases):
    """Warm up only new cases from packages after the first one."""
    if package_index == 0:
        return []
    return [
        (float(frequency), int(mode_index))
        for frequency, mode_index in cases
        if (float(frequency), int(mode_index)) not in previously_trained_cases
    ]


def frequency_label(frequency):
    return f"{float(frequency):g}".replace(".", "p")


def package_label(package_index, cases):
    by_frequency = {}
    for frequency, mode_index in cases:
        by_frequency.setdefault(float(frequency), []).append(int(mode_index))
    tokens = []
    for frequency in sorted(by_frequency):
        modes = "_".join(str(mode_index) for mode_index in sorted(by_frequency[frequency]))
        tokens.append(f"f{frequency_label(frequency)}_m{modes}")
    return f"pkg{package_index + 1:02d}_" + "__".join(tokens)


all_requested_cases = flatten_packages(training_packages)
if not all_requested_cases:
    raise ValueError("training_packages must contain at least one (frequency, mode) pair")

all_training_frequencies = sorted({frequency for frequency, _ in all_requested_cases})

# ==============================================================================
# SECTION 3: FOURIER FEATURE MAPPING, INCIDENT MODES & QUADRATURE SETUP
# ==============================================================================

key, subkey = jax.random.split(key)
B_base = jax.random.normal(subkey, (m_fourier_features, 2))

fmax = float(max(all_training_frequencies))
N_modes = int(np.round(2 * H * fmax / c0)) + 5

n_gauss_legendre = 3 * N_modes
y_gauss_legendre_np, w_gauss_legendre_np = np.polynomial.legendre.leggauss(
    n_gauss_legendre
)
y_gauss_legendre = jnp.asarray(y_gauss_legendre_np, dtype=jnp.float32)
y_quad = jnp.asarray((y_gauss_legendre_np + 1.0) * H / 2.0, dtype=jnp.float32)
w_quad = jnp.asarray(w_gauss_legendre_np * H / 2.0, dtype=jnp.float32)

n_modes = jnp.arange(N_modes, dtype=jnp.float32)
a_n = jnp.sqrt(2.0 / H) * jnp.ones(N_modes, dtype=jnp.float32)
a_n = a_n.at[0].set(jnp.sqrt(1.0 / H))
C_quad = a_n[:, None] * jnp.cos(n_modes[:, None] * jnp.pi * y_quad[None, :] / H)


def beta_for_frequency(frequency):
    k0 = 2 * jnp.pi * frequency / c0
    return jnp.sqrt(k0**2 - (n_modes * jnp.pi / H) ** 2 + 0j)


def incident_wave_complex(x, y, frequency, mode_index):
    """Evaluate the physical incident mode u0 at normalized coordinates."""
    mode_index = jnp.asarray(mode_index, dtype=jnp.int32)
    beta_n = beta_for_frequency(frequency)
    beta_mode = jnp.take(beta_n, mode_index)
    mode_float = mode_index.astype(jnp.float32)
    x_phys = x * L
    y_phys = (y + 1.0) * H / 2.0
    mode_shape = jnp.take(a_n, mode_index) * jnp.cos(mode_float * jnp.pi * y_phys / H)
    return mode_shape * jnp.exp(1j * beta_mode * x_phys)


def create_incident_wave_mode(x, y, frequency, mode_index, us_norm_val):
    """Evaluate the incident mode u0 scaled by the scattered-field norm."""
    value = incident_wave_complex(x, y, frequency, mode_index) / us_norm_val
    return jnp.stack([jnp.real(value), jnp.imag(value)])

# ==============================================================================
# SECTION 4: NEURAL NETWORK UTILITIES
# ==============================================================================

def init_layers(key, n_layers):
    layers = []
    keys = jax.random.split(key, len(n_layers) - 1)
    for index, layer_key in enumerate(keys):
        fan_in = n_layers[index]
        fan_out = n_layers[index + 1]
        scale = jnp.sqrt(2.0 / (fan_in + fan_out))
        W = jax.random.normal(layer_key, (fan_in, fan_out)) * scale
        b = jnp.zeros(fan_out)
        layers.append({"W": W, "b": b})
    return layers


def init_layers_us(key, n_layers, frequency, mode_index):
    kx = 2 * jnp.pi * frequency / c0
    ky = (mode_index + int(mode_index == 0)) * jnp.pi / H
    layers_us = init_layers(key, n_layers)
    layers_us[-1]["W"] = layers_us[-1]["W"] / 10.0
    return {"layers": layers_us, "sigma": jnp.array([kx, ky])}

# ==============================================================================
# SECTION 5: MODEL FORWARD PASSES
# ==============================================================================

def forward_func(layers, X):
    Z = X
    for layer in layers[:-1]:
        Z = jax.nn.tanh(Z @ layer["W"] + layer["b"])
    return Z @ layers[-1]["W"] + layers[-1]["b"]


def forward_ms(layers_ms, X):
    Z = X
    for layer in layers_ms[:-1]:
        Z = jax.nn.tanh(Z @ layer["W"] + layer["b"])
    raw = Z @ layers_ms[-1]["W"] + layers_ms[-1]["b"]
    return (((ms_max - ms_min) / 2.0) * jax.nn.tanh(raw) + (ms_max + ms_min) / 2.0).squeeze()


def compute_features(x, y, sigma):
    x_phys = x * L
    y_phys = (y + 1.0) * H / 2.0
    B_scaled = B_base * sigma
    projection = B_scaled[:, 0] * x_phys + B_scaled[:, 1] * y_phys
    return jnp.concatenate([jnp.cos(projection), jnp.sin(projection)])


def sound_speed(x, y, layers_ms):
    return 1.0 / jnp.sqrt(m0 + forward_ms(layers_ms, jnp.array([x, y])))

# ==============================================================================
# SECTION 6: COLLOCATION POINTS & SWITCH CRITERION
# ==============================================================================

def regular_collocation_points(sizes):
    """Return fixed PDE, Neumann and DtN cell-centred grids."""
    N_pde, N_neumann, N_dtn = sizes
    if min(N_pde, N_neumann, N_dtn) <= 0:
        raise ValueError("All collocation counts must be positive")

    def axis_points(count):
        return jnp.linspace(-1.0 + 1.0 / count, 1.0 - 1.0 / count, count)

    physical_aspect_ratio = (2.0 * L) / H
    factor_pairs = [
        (N_pde // ny, ny)
        for ny in range(1, int(np.sqrt(N_pde)) + 1)
        if N_pde % ny == 0
    ]
    nx, ny = min(
        factor_pairs,
        key=lambda shape: abs(np.log((shape[0] / shape[1]) / physical_aspect_ratio)),
    )
    x_axis = axis_points(nx)
    y_axis = axis_points(ny)
    x_grid, y_grid = jnp.meshgrid(x_axis, y_axis, indexing="xy")
    return x_grid.reshape(-1), y_grid.reshape(-1), axis_points(N_neumann), axis_points(N_dtn)


def sample_collocation_points(key, sizes):
    if key is None:
        return regular_collocation_points(sizes)

    N_pde, N_neumann, N_dtn = sizes
    key_x_pde, key_y_pde, key_neumann, key_dtn = jax.random.split(key, 4)
    x_pde = jax.random.uniform(key_x_pde, (N_pde,), minval=-1.0, maxval=1.0)
    y_pde = jax.random.uniform(key_y_pde, (N_pde,), minval=-1.0, maxval=1.0)
    x_neumann = jax.random.uniform(key_neumann, (N_neumann,), minval=-1.0, maxval=1.0)
    y_dtn = jax.random.uniform(key_dtn, (N_dtn,), minval=-1.0, maxval=1.0)
    return x_pde, y_pde, x_neumann, y_dtn


def check_switch_criterion(loss_history, window=10, threshold=1e-3):
    if len(loss_history) < window:
        return False
    old_loss = loss_history[-window]
    new_loss = loss_history[-1]
    relative_slope = abs(new_loss - old_loss) / max(abs(old_loss), 1e-12)
    return relative_slope < threshold and old_loss < 1e-2

# ==============================================================================
# SECTION 7: LOSS FUNCTION DEFINITIONS
# ==============================================================================

def us_apply(params_us, x, y):
    features = compute_features(x, y, params_us["sigma"])
    return forward_func(params_us["layers"], features)


def _fourier_feature_terms(params_us, x, y):
    """Return Fourier feature trigonometric terms and physical wavevectors."""
    wavevectors = B_base * params_us["sigma"]
    projection = (
        wavevectors[:, 0] * (x * L)
        + wavevectors[:, 1] * ((y + 1.0) * H / 2.0)
    )
    return jnp.cos(projection), jnp.sin(projection), wavevectors


def us_value_and_physical_derivative(params_us, x, y, axis):
    """Evaluate ``us`` and one physical-coordinate derivative analytically."""
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis!r}")

    cosine, sine, wavevectors = _fourier_feature_terms(params_us, x, y)
    value = jnp.concatenate([cosine, sine])
    feature_derivative = jnp.concatenate(
        [-sine * wavevectors[:, axis], cosine * wavevectors[:, axis]]
    )

    for layer in params_us["layers"][:-1]:
        preactivation = value @ layer["W"] + layer["b"]
        preactivation_derivative = feature_derivative @ layer["W"]
        value = jax.nn.tanh(preactivation)
        feature_derivative = (1.0 - value**2) * preactivation_derivative

    output_layer = params_us["layers"][-1]
    return (
        value @ output_layer["W"] + output_layer["b"],
        feature_derivative @ output_layer["W"],
    )


def us_value_and_physical_laplacian(params_us, x, y):
    """Evaluate ``us`` and its physical-coordinate Laplacian explicitly.

    A nonlinear layer's Laplacian needs its two-component spatial gradient but
    only one second-order accumulator.  This is narrower than propagating xx
    and yy independently when the PDE immediately adds them.
    """
    cosine, sine, wavevectors = _fourier_feature_terms(params_us, x, y)
    value = jnp.concatenate([cosine, sine])

    projection_gradient = wavevectors.T
    gradient = jnp.concatenate(
        [
            -sine[None, :] * projection_gradient,
            cosine[None, :] * projection_gradient,
        ],
        axis=1,
    )
    frequency_squared = jnp.sum(projection_gradient**2, axis=0)
    laplacian = jnp.concatenate(
        [-cosine * frequency_squared, -sine * frequency_squared]
    )

    for layer in params_us["layers"][:-1]:
        preactivation = value @ layer["W"] + layer["b"]
        preactivation_gradient = gradient @ layer["W"]
        preactivation_laplacian = laplacian @ layer["W"]
        value = jax.nn.tanh(preactivation)
        tanh_first = 1.0 - value**2
        gradient = tanh_first[None, :] * preactivation_gradient
        laplacian = tanh_first * (
            preactivation_laplacian
            - 2.0 * value * jnp.sum(preactivation_gradient**2, axis=0)
        )

    output_layer = params_us["layers"][-1]
    return (
        value @ output_layer["W"] + output_layer["b"],
        laplacian @ output_layer["W"],
    )


def scattered_data_loss_single(
    params_us, target_left, target_right, y_bnd_left, y_bnd_right,
):
    left_prediction = jax.vmap(us_apply, in_axes=(None, None, 0))(
        params_us, -1.0, y_bnd_left
    )
    right_prediction = jax.vmap(us_apply, in_axes=(None, None, 0))(
        params_us, 1.0, y_bnd_right
    )
    return (
        jnp.mean((left_prediction - target_left) ** 2)
        + jnp.mean((right_prediction - target_right) ** 2)
    )


def scattered_loss_single(
    params_us, layers_ms, x_pde, y_pde, x_neumann,
    y_dtn, frequency, mode_index, target_left, target_right,
    y_bnd_left, y_bnd_right, u_norm_val, weights,
):
    def us(x, y):
        return us_apply(params_us, x, y)

    def ms(x, y):
        return forward_ms(layers_ms, jnp.array([x, y]))

    def u0(x, y):
        return create_incident_wave_mode(x, y, frequency, mode_index, u_norm_val)

    def us_x(x, y):
        _, derivative = us_value_and_physical_derivative(
            params_us, x, y, axis=0
        )
        return derivative

    def us_y(x, y):
        _, derivative = us_value_and_physical_derivative(
            params_us, x, y, axis=1
        )
        return derivative

    def pde_residual(x, y):
        us_value, laplacian_us = us_value_and_physical_laplacian(
            params_us, x, y
        )
        ms_value = ms(x, y)
        omega = 2.0 * jnp.pi * frequency
        scattering = m0 * us_value + ms_value * (us_value + u0(x, y))
        return laplacian_us + omega**2 * scattering

    def compute_dtn_loss(x_bnd, y_eval, sign):
        us_quad = jax.vmap(us, in_axes=(None, 0))(x_bnd, y_gauss_legendre)
        us_quad_complex = us_quad[:, 0] + 1j * us_quad[:, 1]

        us_modes = C_quad @ (w_quad * us_quad_complex)
        beta_n = beta_for_frequency(frequency)
        dtn_modes = sign * 1j * beta_n * us_modes

        def dtn_eval(y_val):
            C_y = a_n * jnp.cos(n_modes * jnp.pi * (y_val + 1.0) / 2.0)
            return jnp.dot(C_y, dtn_modes)

        dtn_pred_complex = jax.vmap(dtn_eval)(y_eval)
        dtn_actual = jax.vmap(us_x, in_axes=(None, 0))(x_bnd, y_eval)
        dtn_actual_complex = dtn_actual[:, 0] + 1j * dtn_actual[:, 1]
        return jnp.mean(jnp.abs(dtn_actual_complex - dtn_pred_complex) ** 2)

    pde_values = jax.vmap(pde_residual, in_axes=(0, 0))(x_pde, y_pde)
    pde_loss = jnp.mean(pde_values**2)

    dtn_loss_left = compute_dtn_loss(-1.0, y_dtn, sign=-1)
    dtn_loss_right = compute_dtn_loss(1.0, y_dtn, sign=1)
    neumann_loss = jnp.mean(
        jax.vmap(us_y, in_axes=(0, None))(x_neumann, 1.0) ** 2
        + jax.vmap(us_y, in_axes=(0, None))(x_neumann, -1.0) ** 2
    )
    bc_loss = neumann_loss + dtn_loss_left + dtn_loss_right

    data_loss = scattered_data_loss_single(
        params_us, target_left, target_right, y_bnd_left, y_bnd_right,
    )

    total_loss = weights[0] * pde_loss + weights[1] * bc_loss + weights[2] * data_loss
    return total_loss, (pde_loss, bc_loss, data_loss)


def package_scattered_loss_fn(
    params_us_stacked, layers_ms, x_pde, y_pde, x_neumann,
    y_dtn, frequencies, mode_indices, targets_left, targets_right,
    y_bnds_left, y_bnds_right, u_norms, weights,
):
    """Evaluate the mean weighted scattered-field loss for one package."""
    def _loss_single(p_us, frequency, mode_index, target_left, target_right, y_left, y_right, u_norm_val):
        return scattered_loss_single(
            p_us, layers_ms, x_pde, y_pde, x_neumann, y_dtn,
            frequency, mode_index, target_left, target_right,
            y_left, y_right, u_norm_val, weights,
        )

    losses, (pdes, bcs, datas) = jax.vmap(
        _loss_single, in_axes=(0, 0, 0, 0, 0, 0, 0, 0)
    )(
        params_us_stacked, frequencies, mode_indices, targets_left,
        targets_right, y_bnds_left, y_bnds_right, u_norms,
    )
    return jnp.mean(losses), (jnp.mean(pdes), jnp.mean(bcs), jnp.mean(datas), datas)


@jax.jit
def evaluate_inverse_loss(
    params_us_stacked, layers_ms, x_pde, y_pde, x_neumann,
    y_dtn, frequencies, mode_indices, targets_left, targets_right,
    y_bnds_left, y_bnds_right, u_norms, weights,
):
    return package_scattered_loss_fn(
        params_us_stacked, layers_ms, x_pde, y_pde, x_neumann,
        y_dtn, frequencies, mode_indices, targets_left, targets_right,
        y_bnds_left, y_bnds_right, u_norms, weights,
    )

# ==============================================================================
# SECTION 8: OPTIMIZATION SCHEMES & TRAIN STEPS (JIT COMPILED)
# ==============================================================================

def make_train_step_warmup(adam_opt_us):
    """Update packed us networks against a frozen ms network."""

    @functools.partial(jax.jit, static_argnames=("N",))
    def train_step_warmup(
        params_us_stacked, layers_ms, opt_state_us, key,
        frequencies, mode_indices, targets_left, targets_right,
        y_bnds_left, y_bnds_right, u_norms, current_weights, N,
    ):
        x_pde, y_pde, x_neumann, y_dtn = sample_collocation_points(key, N)

        def loss_all_cases(p_us_stacked):
            return package_scattered_loss_fn(
                p_us_stacked, layers_ms, x_pde, y_pde, x_neumann, y_dtn,
                frequencies, mode_indices, targets_left, targets_right,
                y_bnds_left, y_bnds_right, u_norms, current_weights,
            )

        (loss, aux), grads_us = jax.value_and_grad(
            loss_all_cases, has_aux=True
        )(params_us_stacked)
        updates_us, opt_state_us = adam_opt_us.update(
            grads_us, opt_state_us, params_us_stacked
        )
        params_us_stacked = optax.apply_updates(params_us_stacked, updates_us)
        return params_us_stacked, opt_state_us, loss, aux

    return train_step_warmup


def make_train_step_inverse_package(adam_opt_us, adam_opt_ms, lbfgs_opt_packed):
    """Update packed us networks and the shared ms network jointly."""

    @functools.partial(jax.jit, static_argnames=("N", "use_lbfgs"))
    def train_step_inverse(
        params_us_stacked, layers_ms, opt_state_us, opt_state_ms,
        opt_state_lbfgs, key, frequencies, mode_indices,
        targets_left, targets_right, y_bnds_left, y_bnds_right,
        u_norms, current_weights, N, use_lbfgs=False,
    ):
        x_pde, y_pde, x_neumann, y_dtn = sample_collocation_points(key, N)

        def loss_all_cases(p_us_stacked, l_ms):
            return package_scattered_loss_fn(
                p_us_stacked, l_ms, x_pde, y_pde, x_neumann, y_dtn,
                frequencies, mode_indices, targets_left, targets_right,
                y_bnds_left, y_bnds_right, u_norms, current_weights,
            )

        (loss, aux), (grads_us, grads_ms) = jax.value_and_grad(
            loss_all_cases, argnums=(0, 1), has_aux=True
        )(params_us_stacked, layers_ms)

        if use_lbfgs:
            params_packed = {"us_list": params_us_stacked, "ms": layers_ms}
            grads_packed = {"us_list": grads_us, "ms": grads_ms}

            def value_fn_packed(packed):
                value, _ = loss_all_cases(packed["us_list"], packed["ms"])
                return value

            updates_packed, opt_state_lbfgs = lbfgs_opt_packed.update(
                grads_packed,
                opt_state_lbfgs,
                params_packed,
                value=loss,
                grad=grads_packed,
                value_fn=value_fn_packed,
            )
            params_packed = optax.apply_updates(params_packed, updates_packed)
            params_us_stacked = params_packed["us_list"]
            layers_ms = params_packed["ms"]
        else:
            updates_us, opt_state_us = adam_opt_us.update(
                grads_us, opt_state_us, params_us_stacked
            )
            params_us_stacked = optax.apply_updates(params_us_stacked, updates_us)

            updates_ms, opt_state_ms = adam_opt_ms.update(grads_ms, opt_state_ms, layers_ms)
            layers_ms = optax.apply_updates(layers_ms, updates_ms)

        return (
            params_us_stacked, layers_ms, opt_state_us, opt_state_ms,
            opt_state_lbfgs, loss, aux,
        )

    return train_step_inverse

# ==============================================================================
# SECTION 9: MAIN TRAINING LOOP
# ==============================================================================

def make_package_arrays(cases, mode_data):
    frequencies = jnp.array([frequency for frequency, _ in cases], dtype=jnp.float32)
    mode_indices = jnp.array([mode_index for _, mode_index in cases], dtype=jnp.int32)
    targets_left = jnp.stack(
        [mode_data[mode_index]["U_left_norm"][frequency] for frequency, mode_index in cases]
    )
    targets_right = jnp.stack(
        [mode_data[mode_index]["U_right_norm"][frequency] for frequency, mode_index in cases]
    )
    y_bnds_left = jnp.stack(
        [mode_data[mode_index]["Y_left"][frequency] for frequency, mode_index in cases]
    )
    y_bnds_right = jnp.stack(
        [mode_data[mode_index]["Y_right"][frequency] for frequency, mode_index in cases]
    )
    u_norms = jnp.array(
        [mode_data[mode_index]["U_norm"][frequency] for frequency, mode_index in cases],
        dtype=jnp.float32,
    )
    return frequencies, mode_indices, targets_left, targets_right, y_bnds_left, y_bnds_right, u_norms


def snapshot_steps(max_steps, ratios):
    if max_steps <= 0:
        return {}
    steps = {}
    for ratio in ratios:
        ratio = float(ratio)
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError("m_snapshot values must be between 0 and 1")
        step = int(round(ratio * max_steps))
        step = min(max(step, 0), max_steps - 1)
        steps[step] = ratio
    return steps


def save_celerity_plot(layers_ms, output_path, title, show=False):
    x_plot = jnp.linspace(-L, L, 120)
    y_plot = jnp.linspace(0, H, 60)
    c_grid = jax.vmap(
        jax.vmap(lambda x, y: sound_speed(x, y, layers_ms), in_axes=(0, None)),
        in_axes=(None, 0),
    )(x_plot / L, 2 * y_plot / H - 1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 3.5))
    plt.pcolormesh(x_plot, y_plot, jnp.asarray(c_grid), rasterized=True)
    plt.colorbar(label="Sound speed c(x, y)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def format_sigma_scales(params_us_stacked, cases):
    sigma_values = jnp.asarray(jax.device_get(params_us_stacked["sigma"]))
    return " | ".join(
        f"f{frequency_label(frequency)} M{mode_index} "
        f"sigma=({sigma[0]:.3g},{sigma[1]:.3g})"
        for (frequency, mode_index), sigma in zip(cases, sigma_values)
    )


def train(
    params_us, layers_ms, N_adam, N_lbfgs, max_steps_adam_warmup,
    max_steps_adam_inverse, max_steps_lbfgs_inverse, packages,
    mode_data, eval_interval, switch_threshold, switch_window, key,
):
    validation_points = regular_collocation_points(N_validation)
    cache_dir = Path(os.path.join(script_dir,"cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    best_validation_losses = {"warmup": [], "inverse": []}
    loss_inverse_by_package = []
    best_params_each_ms = []

    template_param_us = next(iter(params_us.values()))
    param_labels = jax.tree_util.tree_map(lambda _: "base", template_param_us)
    if "sigma" in param_labels:
        param_labels["sigma"] = jax.tree_util.tree_map(
            lambda _: "sigma", template_param_us["sigma"]
        )

    cosine_us_warmup = optax.schedules.cosine_decay_schedule(
        init_value=lr_us, decay_steps=max(max_steps_adam_warmup, 1), alpha=0.1
    )
    cosine_sigma_warmup = optax.schedules.cosine_decay_schedule(
        init_value=lr_sigma,
        decay_steps=max(int(0.5 * max_steps_adam_warmup), 1),
        alpha=0.001,
    )
    adam_us_warmup = optax.multi_transform(
        transforms={
            "base": optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.scale_by_adam(),
                optax.scale_by_schedule(cosine_us_warmup),
                optax.scale(-1.0),
            ),
            "sigma": optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.scale_by_adam(),
                optax.scale_by_schedule(cosine_sigma_warmup),
                optax.scale(-1.0),
            ),
        },
        param_labels=param_labels,
    )

    data_weight_schedule = optax.linear_schedule(
        init_value=0.1,
        end_value=float(weights_inverse_final[2]),
        transition_steps=max(round(0.2 * max_steps_adam_inverse), 1),
    )
    # Adam is nearly invariant to a uniform scaling of its input gradients.
    # Apply the same normalized data ramp to its *updates* so ms stays nearly
    # frozen until the scattered-field data term is fully introduced.
    ms_update_scale_schedule = optax.linear_schedule(
        init_value=0.1 / float(weights_inverse_final[2]),
        end_value=1.0,
        transition_steps=max(int(0.2 * max_steps_adam_inverse), 1),
    )

    cosine_us_inverse = optax.schedules.cosine_decay_schedule(
        init_value=lr_us, decay_steps=max(max_steps_adam_inverse, 1), alpha=0.1
    )
    cosine_sigma_inverse = optax.schedules.cosine_decay_schedule(
        init_value=lr_sigma, decay_steps=max(int(0.3 * max_steps_adam_inverse), 1), alpha=0.0001
    )
    cosine_ms_inverse = optax.schedules.cosine_decay_schedule(
        init_value=lr_ms, decay_steps=max(max_steps_adam_inverse, 1), alpha=0.1
    )
    adam_us_inverse = optax.multi_transform(
        transforms={
            "base": optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.scale_by_adam(),
                optax.scale_by_schedule(cosine_us_inverse),
                optax.scale(-1.0),
            ),
            "sigma": optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.scale_by_adam(),
                optax.scale_by_schedule(cosine_sigma_inverse),
                optax.scale(-1.0),
            ),
        },
        param_labels=param_labels,
    )
    adam_ms_inverse = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(cosine_ms_inverse),
        optax.scale_by_schedule(ms_update_scale_schedule),
        optax.scale(-1.0),
    )
    lbfgs_packed_inverse = optax.lbfgs()
    step_warmup = make_train_step_warmup(adam_us_warmup)
    step_inverse = make_train_step_inverse_package(
        adam_us_inverse, adam_ms_inverse, lbfgs_packed_inverse
    )

    previously_trained_cases = set()
    warmup_full_weights = jnp.concatenate(
        [weights_warmup, jnp.array([0.0], dtype=weights_warmup.dtype)]
    )

    for package_index, package in enumerate(packages):
        cases = flatten_package(package)
        if not cases:
            continue

        label = package_label(package_index, cases)
        print(f"\n{'=' * 70}")
        print(f"--- Package {package_index + 1}: {cases} ---")
        print(f"{'=' * 70}")

        warmup_cases = select_warmup_cases(
            package_index, cases, previously_trained_cases
        )
        if warmup_cases and max_steps_adam_warmup <= 0:
            print(
                f"Warmup cases {warmup_cases} selected, but "
                "max_steps_adam_warmup <= 0; skipping warmup."
            )
        elif warmup_cases:
            warmup_label = package_label(package_index, warmup_cases)
            warmup_param_us_list = [
                params_us[(frequency, mode_index)]
                for frequency, mode_index in warmup_cases
            ]
            warmup_params_us_stacked = jax.tree_util.tree_map(
                lambda *values: jnp.stack(values, axis=0), *warmup_param_us_list
            )
            warmup_arrays = make_package_arrays(warmup_cases, mode_data)
            warmup_loss_history = []
            best_warmup_loss = float("inf")
            best_warmup_params_us_stacked = jax.tree_util.tree_map(
                jnp.copy, warmup_params_us_stacked
            )
            best_warmup_summary = None

            print(
                f"\n--- Scattered warmup | Adam (max {max_steps_adam_warmup} steps) "
                f"| frozen ms from previous package | cases {warmup_cases} ---"
            )
            opt_state_us_warmup = adam_us_warmup.init(warmup_params_us_stacked)

            for step in range(max_steps_adam_warmup):
                key, subkey = jax.random.split(key)
                warmup_params_us_stacked, opt_state_us_warmup, _, _ = step_warmup(
                    warmup_params_us_stacked, layers_ms, opt_state_us_warmup,
                    subkey, *warmup_arrays, warmup_full_weights, N=N_adam,
                )

                if step % eval_interval == 0:
                    loss, validation_aux = evaluate_inverse_loss(
                        warmup_params_us_stacked,
                        layers_ms,
                        *validation_points,
                        *warmup_arrays,
                        warmup_full_weights,
                    )
                    pde_loss, bc_loss, _, _ = validation_aux
                    loss_value = float(loss)
                    sigma_summary = format_sigma_scales(
                        warmup_params_us_stacked, warmup_cases
                    )
                    print(
                        f"[Adam warmup] Step {step} | val pde {pde_loss:.2e} | "
                        f"val bc {bc_loss:.2e} | weighted total {loss:.2e} | "
                        f"{sigma_summary}"
                    )

                    warmup_loss_history.append(loss_value)
                    if loss_value < best_warmup_loss:
                        best_warmup_loss = loss_value
                        best_warmup_params_us_stacked = jax.tree_util.tree_map(
                            jnp.copy, warmup_params_us_stacked
                        )
                        best_warmup_summary = {
                            "package": package_index + 1,
                            "cases": [(float(f), int(m)) for f, m in warmup_cases],
                            "optimizer": "adam",
                            "step": int(step),
                            "pde": float(pde_loss),
                            "bc": float(bc_loss),
                            "weighted_total": loss_value,
                            "weights": [float(value) for value in weights_warmup],
                            "frozen_ms_from_previous_package": True,
                        }

                    if check_switch_criterion(
                        warmup_loss_history,
                        window=switch_window,
                        threshold=switch_threshold,
                    ):
                        print(
                            f"[Adam warmup] Convergence criterion met at step {step}. "
                            "Stopping warmup."
                        )
                        break

            if best_warmup_summary is None:
                raise RuntimeError(
                    f"No finite warmup validation loss for package {package_index + 1}"
                )

            warmup_params_us_stacked = best_warmup_params_us_stacked
            for index, case in enumerate(warmup_cases):
                params_us[case] = jax.tree_util.tree_map(
                    lambda value, idx=index: value[idx], warmup_params_us_stacked
                )
            best_validation_losses["warmup"].append(best_warmup_summary)

        param_us_list = [params_us[(frequency, mode_index)] for frequency, mode_index in cases]
        params_us_stacked = jax.tree_util.tree_map(
            lambda *values: jnp.stack(values, axis=0), *param_us_list
        )
        package_arrays = make_package_arrays(cases, mode_data)
        inverse_history = []

        # ============================================================
        # Scattered inverse problem
        # ============================================================
        best_loss = float("inf")
        best_params_us_stacked = jax.tree_util.tree_map(jnp.copy, params_us_stacked)
        best_params_ms = jax.tree_util.tree_map(jnp.copy, layers_ms)
        best_inverse_summary = None
        inverse_loss_history = []
        switched_to_lbfgs = False
        inverse_snapshot_steps = snapshot_steps(max_steps_adam_inverse, m_snapshot)

        if max_steps_adam_inverse > 0 or max_steps_lbfgs_inverse > 0:
            print(f"\n--- Scattered inverse | Adam (max {max_steps_adam_inverse} steps) ---")
            opt_state_us = adam_us_inverse.init(params_us_stacked)
            opt_state_ms = adam_ms_inverse.init(layers_ms)
            opt_state_lbfgs = None

            for step in range(max_steps_adam_inverse):
                current_weights = jnp.array(
                    [
                        weights_inverse_final[0],
                        weights_inverse_final[1],
                        data_weight_schedule(step),
                    ]
                )

                key, subkey = jax.random.split(key)
                (
                    params_us_stacked, layers_ms, opt_state_us, opt_state_ms,
                    opt_state_lbfgs, _, _,
                ) = step_inverse(
                    params_us_stacked, layers_ms, opt_state_us, opt_state_ms,
                    opt_state_lbfgs, subkey, *package_arrays, current_weights,
                    N=N_adam, use_lbfgs=False,
                )

                if step in inverse_snapshot_steps:
                    ratio = inverse_snapshot_steps[step]
                    save_celerity_plot(
                        layers_ms,
                        cache_dir / f"{label}_inverse_adam_snapshot_{defect_name}_{int(round(100 * ratio)):03d}pct.pdf",
                        f"Inverse Adam snapshot {100 * ratio:.0f}% - {label}",
                        show=False,
                    )

                if step % eval_interval == 0:
                    loss, validation_aux = evaluate_inverse_loss(
                        params_us_stacked,
                        layers_ms,
                        *validation_points,
                        *package_arrays,
                        weights_inverse_final,
                    )
                    pde_loss, bc_loss, data_loss, per_case_data = validation_aux
                    loss_value = float(loss)
                    sigma_summary = format_sigma_scales(params_us_stacked, cases)
                    inverse_history.append(
                        jnp.array([float(pde_loss), float(bc_loss), float(data_loss)])
                    )
                    print(
                        f"[Adam inverse] Step {step} | val pde {pde_loss:.2e} | "
                        f"val bc {bc_loss:.2e} | val data {data_loss:.2e} | "
                        f"weighted total {loss:.2e} | "
                        f"{sigma_summary}"
                    )

                    inverse_loss_history.append(loss_value)
                    if loss_value < best_loss:
                        best_loss = loss_value
                        best_params_us_stacked = jax.tree_util.tree_map(jnp.copy, params_us_stacked)
                        best_params_ms = jax.tree_util.tree_map(jnp.copy, layers_ms)
                        best_inverse_summary = {
                            "package": package_index + 1,
                            "cases": [(float(f), int(m)) for f, m in cases],
                            "optimizer": "adam",
                            "step": int(step),
                            "pde": float(pde_loss),
                            "bc": float(bc_loss),
                            "data": float(data_loss),
                            "per_case_data": [float(value) for value in jnp.asarray(per_case_data)],
                            "weighted_total": loss_value,
                            "weights": [float(value) for value in weights_inverse_final],
                        }

                    if check_switch_criterion(inverse_loss_history, window=switch_window, threshold=switch_threshold):
                        print(f"[Adam inverse] Convergence criterion met at step {step}. Switching to L-BFGS.")
                        switched_to_lbfgs = True
                        params_us_stacked = best_params_us_stacked
                        layers_ms = best_params_ms
                        break

            if max_steps_adam_inverse > 0:
                save_celerity_plot(
                    layers_ms,
                    cache_dir / f"{label}_inverse_adam_end.pdf",
                    f"End of inverse Adam - {label}",
                    show=False,
                )

            if switched_to_lbfgs or max_steps_lbfgs_inverse > 0:
                print(f"\n--- Scattered inverse | L-BFGS (max {max_steps_lbfgs_inverse} steps) ---")
                opt_state_lbfgs = lbfgs_packed_inverse.init(
                    {"us_list": params_us_stacked, "ms": layers_ms}
                )

                for step in range(max_steps_lbfgs_inverse):
                    (
                        params_us_stacked, layers_ms, opt_state_us, opt_state_ms,
                        opt_state_lbfgs, _, _,
                    ) = step_inverse(
                        params_us_stacked, layers_ms, opt_state_us, opt_state_ms,
                        opt_state_lbfgs, None, *package_arrays, weights_inverse_final,
                        N=N_lbfgs, use_lbfgs=True,
                    )

                    if step % eval_interval == 0:
                        loss, validation_aux = evaluate_inverse_loss(
                            params_us_stacked,
                            layers_ms,
                            *validation_points,
                            *package_arrays,
                            weights_inverse_final,
                        )
                        pde_loss, bc_loss, data_loss, per_case_data = validation_aux
                        loss_value = float(loss)
                        sigma_summary = format_sigma_scales(params_us_stacked, cases)
                        inverse_history.append(
                            jnp.array([float(pde_loss), float(bc_loss), float(data_loss)])
                        )
                        print(
                            f"[LBFGS inverse] Step {step} | val pde {pde_loss:.2e} | "
                            f"val bc {bc_loss:.2e} | val data {data_loss:.2e} | "
                            f"weighted total {loss:.2e} | {sigma_summary}"
                        )

                        if loss_value < best_loss:
                            best_loss = loss_value
                            best_params_us_stacked = jax.tree_util.tree_map(jnp.copy, params_us_stacked)
                            best_params_ms = jax.tree_util.tree_map(jnp.copy, layers_ms)
                            best_inverse_summary = {
                                "package": package_index + 1,
                                "cases": [(float(f), int(m)) for f, m in cases],
                                "optimizer": "lbfgs",
                                "step": int(step),
                                "pde": float(pde_loss),
                                "bc": float(bc_loss),
                                "data": float(data_loss),
                                "per_case_data": [float(value) for value in jnp.asarray(per_case_data)],
                                "weighted_total": loss_value,
                                "weights": [float(value) for value in weights_inverse_final],
                            }

                if max_steps_lbfgs_inverse > 0:
                    save_celerity_plot(
                        layers_ms,
                        cache_dir / f"{label}_inverse_lbfgs_end_{defect_name}.pdf",
                        f"End of inverse L-BFGS - {label}",
                        show=False,
                    )

        if best_inverse_summary is None:
            raise RuntimeError(f"No finite inverse validation loss for package {package_index + 1}")

        params_us_stacked = best_params_us_stacked
        layers_ms = best_params_ms
        for index, case in enumerate(cases):
            params_us[case] = jax.tree_util.tree_map(lambda value: value[index], params_us_stacked)

        best_params_each_ms.append((package_index, cases, jax.tree_util.tree_map(jnp.copy, layers_ms)))
        best_validation_losses["inverse"].append(best_inverse_summary)
        loss_inverse_by_package.append({"label": label, "cases": cases, "history": inverse_history})
        previously_trained_cases.update((float(f), int(m)) for f, m in cases)

    return (
        params_us,
        layers_ms,
        best_params_each_ms,
        key,
        loss_inverse_by_package,
        best_validation_losses,
    )

# ==============================================================================
# SECTION 10: MAIN ENTRY POINT
# ==============================================================================

def plot_inverse_losses(loss_inverse_by_package, output_dir):
    for package_history in loss_inverse_by_package:
        history = package_history["history"]
        if not history:
            continue
        losses = jnp.asarray(history)
        x_axis = eval_interval * jnp.arange(losses.shape[0])
        total_losses = (
            weights_inverse_final[0] * losses[:, 0]
            + weights_inverse_final[1] * losses[:, 1]
            + weights_inverse_final[2] * losses[:, 2]
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.semilogy(x_axis, losses[:, 0], label="PDE Loss")
        ax1.semilogy(x_axis, losses[:, 1], label="BC Loss")
        ax1.semilogy(x_axis, losses[:, 2], label="Data Loss")
        ax1.set_title("Inverse partial losses")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.grid(True)

        ax2.semilogy(x_axis, total_losses, label="Weighted Validation Loss", color="black")
        ax2.set_title("Inverse weighted validation loss")
        ax2.set_xlabel("Steps")
        ax2.legend(loc="upper right")
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"inverse_losses_{package_history['label']}.pdf")
        if SHOW_PLOTS:
            plt.show()
        plt.close()


def main():
    global key

    print("JAX Devices:", jax.devices())
    print("Default JAX dtype:", jnp.ones(1).dtype)

    left_path, right_path = boundary_data_paths(data_dir, defect_name, contrast_ratio)
    boundary_data = WaveguideBoundaryData(left_path, right_path)

    requested_cases = flatten_packages(training_packages)
    missing_pairs = [
        (mode_index, frequency)
        for frequency, mode_index in requested_cases
        if not boundary_data.has_pair(mode_index, frequency)
    ]
    if missing_pairs:
        raise ValueError(
            f"The combined FEM boundary files do not contain requested pairs: {missing_pairs}. "
            f"Available pairs: {boundary_data.available_pairs}"
        )

    mode_data = {}
    used_modes = sorted({mode_index for _, mode_index in requested_cases})
    for mode_index in used_modes:
        mode_frequencies = sorted(
            {frequency for frequency, candidate_mode in requested_cases if candidate_mode == mode_index}
        )
        Y_left = {}
        Y_right = {}
        U_norm = {}
        U_left_norm = {}
        U_right_norm = {}

        for frequency in mode_frequencies:
            pair = boundary_data.get_pair(mode_index, frequency)
            if not jnp.isclose(pair.x_left, -L, rtol=0.0, atol=1e-6) or not jnp.isclose(
                pair.x_right, L, rtol=0.0, atol=1e-6
            ):
                raise ValueError(
                    f"Boundary coordinates for mode {mode_index}, frequency {frequency} are "
                    f"x_left={pair.x_left}, x_right={pair.x_right}; expected {-L} and {L}"
                )

            y_left = 2 * jnp.asarray(pair.y_left, dtype=jnp.float32) / H - 1
            y_right = 2 * jnp.asarray(pair.y_right, dtype=jnp.float32) / H - 1
            Y_left[frequency] = y_left
            Y_right[frequency] = y_right

            total_left = (
                jnp.asarray(pair.u_re_left, dtype=jnp.float32)
                + 1j * jnp.asarray(pair.u_im_left, dtype=jnp.float32)
            )
            total_right = (
                jnp.asarray(pair.u_re_right, dtype=jnp.float32)
                + 1j * jnp.asarray(pair.u_im_right, dtype=jnp.float32)
            )
            scattered_left = total_left - incident_wave_complex(
                -1.0, y_left, frequency, mode_index
            )
            scattered_right = total_right - incident_wave_complex(
                1.0, y_right, frequency, mode_index
            )

            norm = jnp.sqrt(
                jnp.max(
                    jnp.concatenate(
                        (jnp.abs(scattered_left) ** 2, jnp.abs(scattered_right) ** 2)
                    )
                )
            )
            if not jnp.isfinite(float(norm)) or float(norm) <= 0.0:
                raise ValueError(
                    f"Invalid zero or non-finite scattered-field norm for mode {mode_index}, "
                    f"frequency {frequency}"
                )

            U_norm[frequency] = norm
            U_left_norm[frequency] = jnp.stack(
                [jnp.real(scattered_left) / norm, jnp.imag(scattered_left) / norm],
                axis=1,
            )
            U_right_norm[frequency] = jnp.stack(
                [jnp.real(scattered_right) / norm, jnp.imag(scattered_right) / norm],
                axis=1,
            )

        mode_data[mode_index] = {
            "Y_left": Y_left,
            "Y_right": Y_right,
            "U_left_norm": U_left_norm,
            "U_right_norm": U_right_norm,
            "U_norm": U_norm,
        }
        print(
            f"  Mode {mode_index}: {len(mode_frequencies)} frequencies loaded "
            f"({mode_frequencies[0]:.0f}-{mode_frequencies[-1]:.0f} Hz)"
        )

    for package_index, package in enumerate(training_packages):
        cases = flatten_package(package)
        left_counts = {
            mode_data[mode_index]["Y_left"][frequency].shape[0]
            for frequency, mode_index in cases
        }
        right_counts = {
            mode_data[mode_index]["Y_right"][frequency].shape[0]
            for frequency, mode_index in cases
        }
        if len(left_counts) != 1 or len(right_counts) != 1:
            raise ValueError(
                f"All cases in package {package_index + 1} must use the same number "
                "of boundary samples per side"
            )

    key, subkey_ms = jax.random.split(key)
    layers_ms = init_layers(subkey_ms, n_layers_ms)
    initial_ms_raw = jnp.arctanh(-(ms_max + ms_min) / (ms_max - ms_min))
    layers_ms[-1]["b"] = jnp.full_like(layers_ms[-1]["b"], initial_ms_raw)
    layers_ms[-1]["W"] = layers_ms[-1]["W"] / 10.0

    params_us = {}
    for frequency, mode_index in sorted(set(requested_cases)):
        key, subkey_us = jax.random.split(key)
        params_us[(frequency, mode_index)] = init_layers_us(
            subkey_us, n_layers_us, frequency, mode_index
        )

    fig_dir = Path(script_dir) / "fig"
    cache_dir = Path(script_dir) / "cache"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Plotting initial celerity profile...")
    save_celerity_plot(
        layers_ms,
        fig_dir / "initial_scattered_celerity.pdf",
        "Initial celerity field",
        show=SHOW_PLOTS,
    )

    key, subkey = jax.random.split(key)
    (
        params_us, layers_ms, best_params_each_ms,
        key, loss_inverse_by_package, best_validation_losses,
    ) = train(
        params_us, layers_ms, N_adam, N_lbfgs,
        max_steps_adam_warmup, max_steps_adam_inverse,
        max_steps_lbfgs_inverse, training_packages, mode_data, eval_interval,
        switch_threshold, switch_window, subkey,
    )

    all_used_modes = sorted({mode_index for _, mode_index in requested_cases})
    modes_str = "_".join(str(mode_index) for mode_index in all_used_modes)
    freqs_str = "_".join(frequency_label(frequency) for frequency in all_training_frequencies)

    checkpoint_dir = Path(script_dir) / "checkpoints"
    checkpoint_path = (
        checkpoint_dir
        / f"scattered_checkpoint_{defect_name}_{data_ratio_label}_modes{modes_str}_freqs{freqs_str}.npz"
    )
    save_us_checkpoint(
        checkpoint_path,
        params_us,
        B_base,
        collect_us_norms(params_us, mode_data),
        length=L,
        height=H,
        c0=c0,
        layers_ms=layers_ms,
        ms_min=ms_min,
        ms_max=ms_max,
        network_config={
            "us_layers": list(n_layers_us),
            "ms_layers": list(n_layers_ms),
            "fourier_features": int(m_fourier_features),
            "hidden_activation": "tanh",
            "us_feature_mapping": "random_fourier_cos_sin",
            "ms_output_parameterization": "bounded_scattered_slowness_tanh",
        },
        best_validation_losses=best_validation_losses,
        random_seed=random_seed,
        metadata={
            "defect_name": defect_name,
            "contrast_ratio": contrast_ratio,
            "ratio_label": data_ratio_label,
            "field_formulation": "u_total = u0 + us",
            "residual": "Delta us + omega^2*((m0+ms)*us + ms*u0)",
            "us_normalization": "boundary scattered field max norm after subtracting incident",
            "training_strategy": "new_cases_warmup_then_scattered_inverse",
            "warmup_scope": "packages after the first; cases absent from earlier packages",
            "warmup_weights": [float(value) for value in weights_warmup],
            "inverse_weight_schedule": "physics_fixed_data_ramp",
            "training_packages": [
                {str(frequency): list(modes) for frequency, modes in package.items()}
                for package in training_packages
            ],
        },
    )
    print(f"US/MS checkpoint saved to: {checkpoint_path}")

    plot_inverse_losses(loss_inverse_by_package, fig_dir)

    for package_index, cases, layers_ms_each in best_params_each_ms:
        label = package_label(package_index, cases)
        save_celerity_plot(
            layers_ms_each,
            fig_dir / f"c_map_scattered_{defect_name}_{label}.pdf",
            f"Inverted celerity - {label}",
            show=SHOW_PLOTS,
        )


if __name__ == "__main__":
    main()
