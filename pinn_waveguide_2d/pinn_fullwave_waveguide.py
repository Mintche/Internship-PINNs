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
from tools.training_run_config import load_training_run_config
from tools.uv_checkpoint import collect_u_norms, save_uv_checkpoint

# ==============================================================================
# SECTION 1: CONFIGURATION & HYPERPARAMETERS
# ==============================================================================

_DEFAULT_RUN_CONFIG = {
    "formulation": "total",
    "height": 0.6,
    "half_length": 1.0,
    "c0": 340.0,
    "contrast_max": 0.4,
    "celerity_upper_factor": 1.01,
    "data_dir": "FEM/pinn_data",
    "defect_name": "circlebottomright",
    "training_packages": [{"-1": {"1200": [0]}}],
    "random_seed": 0,
    "fourier_features": 64,
    "field_hidden_layers": [128, 128, 64],
    "material_hidden_layers": [128, 64],
    "lr_field": 1e-3,
    "lr_material": 3e-4,
    "lr_sigma": 1e-2,
    "gradient_clip_norm": 1.0,
    "schedule": {
        "forward_field_cosine_alpha": 0.1,
        "forward_sigma_cosine_alpha": 0.1,
        "inverse_field_cosine_alpha": 0.1,
        "inverse_sigma_decay_fraction": 0.2,
        "inverse_sigma_cosine_alpha": 0.001,
        "inverse_material_cosine_alpha": 0.1,
        "inverse_data_initial_weight": 0.1,
        "inverse_data_transition_fraction": 0.2,
    },
    "n_adam": [4096, 256, 64],
    "n_lbfgs": [2048, 256, 64],
    "n_validation": [4096, 256, 128],
    "eval_interval": 200,
    "switch_threshold": 0.0,
    "switch_window": 0,
    "max_steps_adam_forward": 50001,
    "max_steps_lbfgs_forward": 0,
    "max_steps_adam_inverse": 0,
    "max_steps_lbfgs_inverse": 0,
    "weights_forward": [1.0, 10.0, 0.0],
    "weights_inverse": [1.0, 1.0, 10.0],
    "show_plots": True,
    "output_root": "pinn_waveguide_2d",
}
RUN_CONFIG = load_training_run_config(
    _DEFAULT_RUN_CONFIG,
    formulation="total",
    argv=sys.argv[1:] if __name__ == "__main__" else (),
)
_config = RUN_CONFIG.values

# --- Domain Geometry ---
H = float(_config["height"])
L = float(_config["half_length"])

# --- Physics Parameters ---
c0 = float(_config["c0"])
contrast_max = float(_config["contrast_max"])
cmin = c0 * (1 - contrast_max)
cmax = c0 * float(_config["celerity_upper_factor"])
m0 = 1/c0**2
m_min = 1 / cmax**2
m_max = 1 / cmin**2

# --- Data Configuration ---
# Data files follow the naming convention:
#   pinn_boundary_{left/right}_{defect_name}_ratio{c_defect/c0}.csv
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = str(RUN_CONFIG.resolve_path(REPOSITORY_ROOT, "data_dir"))
defect_name = str(_config["defect_name"])
VALID_INCIDENCES = {-1, 1}


def parse_incidence(value):
    try:
        incidence = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("training_packages incidence keys must be -1 or 1") from error
    if incidence not in VALID_INCIDENCES:
        raise ValueError("training_packages incidence keys must be -1 or 1")
    return incidence


def parse_training_package(package):
    if not isinstance(package, dict) or not package:
        raise ValueError("Each training package must be a non-empty incidence mapping")
    parsed = {}
    for incidence_key, frequency_map in package.items():
        incidence = parse_incidence(incidence_key)
        if incidence in parsed:
            raise ValueError(f"Duplicate incidence {incidence} in one training package")
        if not isinstance(frequency_map, dict) or not frequency_map:
            raise ValueError(
                f"training_packages[{incidence}] must define at least one frequency"
            )
        parsed[incidence] = {}
        for frequency_key, modes in frequency_map.items():
            frequency = float(frequency_key)
            if not np.isfinite(frequency) or frequency <= 0.0:
                raise ValueError("Training frequencies must be finite and positive")
            mode_values = [int(mode) for mode in modes]
            if not mode_values or min(mode_values) < 0:
                raise ValueError("Training mode lists must be non-empty and non-negative")
            if len(set(mode_values)) != len(mode_values):
                raise ValueError(
                    f"Duplicate modes for frequency {frequency}, incidence {incidence}"
                )
            parsed[incidence][frequency] = mode_values
    return parsed


def flatten_package(package):
    """Return sorted (frequency, mode, incidence) cases from one package."""
    cases = []
    for incidence, frequency_map in package.items():
        for frequency, modes in frequency_map.items():
            for mode_index in modes:
                cases.append((float(frequency), int(mode_index), int(incidence)))
    return sorted(cases, key=lambda item: (item[0], item[1], item[2]))


def flatten_packages(packages):
    cases = []
    for package in packages:
        cases.extend(flatten_package(package))
    return cases


def frequency_label(frequency):
    return f"{float(frequency):g}".replace(".", "p")


def incidence_label(incidence):
    return "m1" if int(incidence) == -1 else "p1"


def package_label(package_index, cases):
    by_incidence_frequency = {}
    for frequency, mode_index, incidence in cases:
        by_incidence_frequency.setdefault(int(incidence), {}).setdefault(
            float(frequency), []
        ).append(int(mode_index))
    tokens = []
    for incidence in sorted(by_incidence_frequency):
        for frequency in sorted(by_incidence_frequency[incidence]):
            modes = "_".join(
                str(mode_index)
                for mode_index in sorted(by_incidence_frequency[incidence][frequency])
            )
            tokens.append(
                f"i{incidence_label(incidence)}_f{frequency_label(frequency)}_m{modes}"
            )
    return f"pkg{package_index + 1:02d}_" + "__".join(tokens)

training_packages = [
    parse_training_package(package)
    for package in _config["training_packages"]
]
all_requested_cases = flatten_packages(training_packages)
if not all_requested_cases:
    raise ValueError("training_packages must define at least one case")
all_training_frequencies = sorted({frequency for frequency, _, _ in all_requested_cases})

# --- Random Seed ---
random_seed = int(_config["random_seed"])
key = jax.random.key(random_seed)

# --- Neural Network Architectures ---
m_fourier_features = int(_config["fourier_features"])
n_input = 2
n_layers_uv = [
    2 * m_fourier_features,
    *(int(width) for width in _config["field_hidden_layers"]),
    2,
]
n_layers_m = [
    n_input,
    *(int(width) for width in _config["material_hidden_layers"]),
    1,
]

# --- Optimizer Learning Rates ---
lr_uv = float(_config["lr_field"])
lr_m = float(_config["lr_material"])
lr_sigma = float(_config["lr_sigma"])
gradient_clip_norm = float(_config["gradient_clip_norm"])
if gradient_clip_norm <= 0:
    raise ValueError("gradient_clip_norm must be positive")

schedule = _config["schedule"]
expected_schedule_keys = {
    "forward_field_cosine_alpha",
    "forward_sigma_cosine_alpha",
    "inverse_field_cosine_alpha",
    "inverse_sigma_decay_fraction",
    "inverse_sigma_cosine_alpha",
    "inverse_material_cosine_alpha",
    "inverse_data_initial_weight",
    "inverse_data_transition_fraction",
}
if not isinstance(schedule, dict) or set(schedule) != expected_schedule_keys:
    raise ValueError(
        "schedule must define exactly: " + ", ".join(sorted(expected_schedule_keys))
    )
schedule = {name: float(value) for name, value in schedule.items()}
for name in (
    "forward_field_cosine_alpha",
    "forward_sigma_cosine_alpha",
    "inverse_field_cosine_alpha",
    "inverse_sigma_cosine_alpha",
    "inverse_material_cosine_alpha",
):
    if not 0.0 <= schedule[name] <= 1.0:
        raise ValueError(f"schedule.{name} must lie between 0 and 1")
for name in ("inverse_sigma_decay_fraction", "inverse_data_transition_fraction"):
    if not 0.0 < schedule[name] <= 1.0:
        raise ValueError(f"schedule.{name} must lie in (0, 1]")
if schedule["inverse_data_initial_weight"] <= 0:
    raise ValueError("schedule.inverse_data_initial_weight must be positive")

# --- Training Hyperparameters ---
# Collocation sizes are ordered as (PDE interior, horizontal Neumann, vertical DtN).
# Keeping these tuples fixed avoids recompilation while Adam still resamples the points.
N_adam = tuple(int(value) for value in _config["n_adam"])
N_lbfgs = tuple(int(value) for value in _config["n_lbfgs"])
N_validation = tuple(int(value) for value in _config["n_validation"])
if any(len(values) != 3 or min(values) <= 0 for values in (N_adam, N_lbfgs, N_validation)):
    raise ValueError("n_adam, n_lbfgs and n_validation must contain three positive counts")
eval_interval = int(_config["eval_interval"])
switch_threshold = float(_config["switch_threshold"])
switch_window = int(_config["switch_window"])

max_steps_adam_phase1 = int(_config["max_steps_adam_forward"])
max_steps_lbfgs_phase1 = int(_config["max_steps_lbfgs_forward"])
max_steps_adam_phase2 = int(_config["max_steps_adam_inverse"])
max_steps_lbfgs_phase2 = int(_config["max_steps_lbfgs_inverse"])

SHOW_PLOTS = bool(_config["show_plots"])

weights_phase1 = jnp.asarray(_config["weights_forward"], dtype=jnp.float32)
weights_phase2_final = jnp.asarray(_config["weights_inverse"], dtype=jnp.float32)
if weights_phase1.shape != (3,) or weights_phase2_final.shape != (3,):
    raise ValueError("weights_forward and weights_inverse must each contain three values")

# ==============================================================================
# SECTION 2: DATA DISCOVERY
# ==============================================================================

def boundary_data_paths(data_dir, defect_name):
    suffix = f'{defect_name}.csv'
    left_path = os.path.join(data_dir, f'pinn_boundary_left_{suffix}')
    right_path = os.path.join(data_dir, f'pinn_boundary_right_{suffix}')
    missing = [path for path in (left_path, right_path) if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            'Missing combined FEM boundary file(s): ' + ', '.join(missing)
        )
    return left_path, right_path

# ==============================================================================
# SECTION 3: FOURIER FEATURE MAPPING & QUADRATURE SETUP
# ==============================================================================

# Initialize random projection matrix for Fourier features
key, subkey = jax.random.split(key)
B_base = jax.random.normal(subkey, (m_fourier_features, 2))

# Calculate dynamic boundary modes configuration using max training frequency
fmax = float(max(all_training_frequencies))
N_modes = int(np.round(2 * H * fmax / c0)) + 5

# --- Gauss-Legendre Quadrature Setup ---
n_gauss_legendre = 3*N_modes
y_gauss_legendre, w_gauss_legendre = np.polynomial.legendre.leggauss(n_gauss_legendre)
y_quad = (y_gauss_legendre + 1.0) * H / 2.0
w_quad = w_gauss_legendre * H / 2.0

n_modes = jnp.arange(N_modes, dtype=jnp.float32)
a_n = jnp.sqrt(2.0 / H) * jnp.ones(N_modes, dtype=jnp.float32)
a_n = a_n.at[0].set(jnp.sqrt(1.0 / H))
C_quad = a_n[:, None] * jnp.cos(n_modes[:, None] * jnp.pi * y_quad[None, :] / H)


def beta_for_frequency(frequency):
    k0 = 2 * jnp.pi * frequency / c0
    return jnp.sqrt(k0**2 - (n_modes * jnp.pi / H) ** 2 + 0j)

# ==============================================================================
# SECTION 4: NEURAL NETWORK UTILITIES
# ==============================================================================

# --- Xavier Initialization Utilities ---
def init_layers(key, n_layers):
    layers = []
    keys = jax.random.split(key, len(n_layers) - 1)
    for i, k in enumerate(keys):
        W = jax.random.normal(k, (n_layers[i], n_layers[i+1])) * jnp.sqrt(2.0 / (n_layers[i] + n_layers[i+1]))
        b = jnp.zeros(n_layers[i+1])
        layers.append({"W": W, "b": b})
    return layers

def init_layers_uv(key, n_layers, f_val, mode_val):
    kx = 2 * jnp.pi * f_val / c0
    ky = (mode_val + int(mode_val==0)) * jnp.pi / H
    layers_uv = init_layers(key, n_layers)
    return {'layers': layers_uv, 'sigma': jnp.array([kx, ky])}

# ==============================================================================
# SECTION 5: MODEL FORWARD PASSES
# ==============================================================================

def forward_func(layers, X):
    n = len(layers)
    Z = X
    for i in range(n-1):
        Z = jax.nn.tanh(Z @ layers[i]["W"] + layers[i]["b"])
    Z = Z @ layers[-1]["W"] + layers[-1]["b"]
    return Z
 
def forward_params(layers, X):
    n = len(layers)
    Z = X
    for i in range(n-1):
        Z = jax.nn.tanh(Z @ layers[i]["W"] + layers[i]["b"])
    Z = m_min + (m_max - m_min) * jax.nn.sigmoid(Z @ layers[-1]["W"] + layers[-1]["b"])
    return Z.squeeze()

def compute_features(x, y, sigma):
    """Compute Fourier features from normalized coordinates.
    Replaces the former `gamma` function — single source of truth."""
    x_phys = x * L 
    y_phys = (y + 1.0) * H / 2.0
    B_scaled = B_base * sigma
    proj = B_scaled[:, 0] * x_phys + B_scaled[:, 1] * y_phys
    return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)])

def c(x, y, layers_m):
    """Compute sound speed at (x, y) using the slowness network.
    layers_m must be passed explicitly to avoid stale closure issues."""
    return 1 / jnp.sqrt(forward_params(layers_m, jnp.array([x, y])))

# ==============================================================================
# SECTION 6: COLLOCATION POINTS & SWITCH CRITERION
# ==============================================================================

def sample_collocation_points(key, sizes):
    """Generate independent PDE and boundary points for one optimization step."""
    N_pde, N_neumann, N_dtn = sizes
    if key is None:
        return regular_collocation_points(sizes)

    key_x_pde, key_y_pde, key_neumann, key_dtn = jax.random.split(key, 4)
    x_pde = jax.random.uniform(
        key_x_pde, (N_pde,), minval=-1.0, maxval=1.0)
    y_pde = jax.random.uniform(
        key_y_pde, (N_pde,), minval=-1.0, maxval=1.0)
    x_neumann = jax.random.uniform(
        key_neumann, (N_neumann,), minval=-1.0, maxval=1.0)
    y_dtn = jax.random.uniform(
        key_dtn, (N_dtn,), minval=-1.0, maxval=1.0)
    return x_pde, y_pde, x_neumann, y_dtn


def regular_collocation_grid(N):
    """Return an approximately isotropic physical tensor grid with exactly N points."""
    if N <= 0:
        raise ValueError("The number of collocation points must be positive")

    # Pick an exact factorization whose axis ratio is closest to the physical
    # aspect ratio (2L)/H. For N=1000 this gives a 50 x 20 grid.
    physical_aspect_ratio = (2.0 * L) / H
    factor_pairs = [
        (N // ny, ny)
        for ny in range(1, int(np.sqrt(N)) + 1)
        if N % ny == 0
    ]
    nx, ny = min(
        factor_pairs,
        key=lambda shape: abs(np.log((shape[0] / shape[1]) / physical_aspect_ratio)),
    )

    # Cell-centred linspaces keep PDE collocation points off the boundary.
    x_axis = jnp.linspace(-1.0 + 1.0 / nx, 1.0 - 1.0 / nx, nx)
    y_axis = jnp.linspace(-1.0 + 1.0 / ny, 1.0 - 1.0 / ny, ny)
    x_grid, y_grid = jnp.meshgrid(x_axis, y_axis, indexing='xy')
    return x_grid.reshape(-1), y_grid.reshape(-1)


def regular_boundary_grid(N):
    """Return N cell-centred coordinates on a normalized one-dimensional edge."""
    if N <= 0:
        raise ValueError("The number of boundary points must be positive")
    return jnp.linspace(-1.0 + 1.0 / N, 1.0 - 1.0 / N, N)


def regular_collocation_points(sizes):
    """Return fixed, non-redundant PDE, Neumann and DtN collocation grids."""
    N_pde, N_neumann, N_dtn = sizes
    x_pde, y_pde = regular_collocation_grid(N_pde)
    x_neumann = regular_boundary_grid(N_neumann)
    y_dtn = regular_boundary_grid(N_dtn)
    return x_pde, y_pde, x_neumann, y_dtn


def check_switch_criterion(loss_history, window=10, threshold=1e-3):
    """Returns True if the relative slope of the loss is below threshold."""
    if len(loss_history) < window:
        return False
    old_loss = loss_history[-window]
    new_loss = loss_history[-1]
    relative_slope = abs(new_loss - old_loss) / max(abs(old_loss), 1e-12)
    return relative_slope < threshold and old_loss < 1e-2

# ==============================================================================
# SECTION 7: LOSS FUNCTION DEFINITIONS
# ==============================================================================

def uv_apply(params_uv, x, y):
    """Evaluate the normalized total-field network at normalized coordinates."""
    features = compute_features(x, y, params_uv["sigma"])
    return forward_func(params_uv["layers"], features)


def _fourier_feature_terms(params_uv, x, y):
    """Return Fourier feature trigonometric terms and physical wavevectors."""
    wavevectors = B_base * params_uv["sigma"]
    projection = (
        wavevectors[:, 0] * (x * L)
        + wavevectors[:, 1] * ((y + 1.0) * H / 2.0)
    )
    return jnp.cos(projection), jnp.sin(projection), wavevectors


def uv_value_and_physical_derivative(params_uv, x, y, axis):
    """Evaluate ``uv`` and one physical-coordinate derivative analytically."""
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis!r}")

    cosine, sine, wavevectors = _fourier_feature_terms(params_uv, x, y)
    value = jnp.concatenate([cosine, sine])
    feature_derivative = jnp.concatenate(
        [-sine * wavevectors[:, axis], cosine * wavevectors[:, axis]]
    )

    for layer in params_uv["layers"][:-1]:
        preactivation = value @ layer["W"] + layer["b"]
        preactivation_derivative = feature_derivative @ layer["W"]
        value = jax.nn.tanh(preactivation)
        feature_derivative = (1.0 - value**2) * preactivation_derivative

    output_layer = params_uv["layers"][-1]
    return (
        value @ output_layer["W"] + output_layer["b"],
        feature_derivative @ output_layer["W"],
    )


def uv_value_and_physical_laplacian(params_uv, x, y):
    """Evaluate ``uv`` and its physical-coordinate Laplacian analytically."""
    cosine, sine, wavevectors = _fourier_feature_terms(params_uv, x, y)
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

    for layer in params_uv["layers"][:-1]:
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

    output_layer = params_uv["layers"][-1]
    return (
        value @ output_layer["W"] + output_layer["b"],
        laplacian @ output_layer["W"],
    )


def loss_fn(params_uv, layers_m, x_pde, y_pde, x_neumann, y_dtn,
            f, mode_index, incidence, target_u_left, target_u_right, y_bnd_left,
            y_bnd_right, u_norm_val, weights,
            is_warmup=False, use_healthy_guide=False):
    beta_n = beta_for_frequency(f)
    
    def uv(x, y):
        return uv_apply(params_uv, x, y)
    
    def m(x, y):
        return forward_params(layers_m, jnp.array([x, y]))
    
    def k2(x, y):
        if use_healthy_guide:
            return (2 * jnp.pi * f / c0)**2
        else:
            return (2 * jnp.pi * f)**2 * m(x, y)
    
    def uv_x(x, y):
        _, derivative = uv_value_and_physical_derivative(
            params_uv, x, y, axis=0
        )
        return derivative
    
    def uv_y(x, y):
        _, derivative = uv_value_and_physical_derivative(
            params_uv, x, y, axis=1
        )
        return derivative

    def pde_residual(x, y):
        uv_value, laplacian_uv = uv_value_and_physical_laplacian(
            params_uv, x, y
        )
        return laplacian_uv + k2(x, y) * uv_value
    
    def compute_dtn_loss(x_bnd, y_eval, sign):
        uv_quad = jax.vmap(uv, in_axes=(None, 0))(x_bnd, y_gauss_legendre)
        U_quad_complex = uv_quad[:, 0] + 1j * uv_quad[:, 1]
        
        u_n = C_quad @ (w_quad * U_quad_complex)
        
        dtn_n = sign * 1j * beta_n * u_n
        A_inc = jnp.zeros(N_modes, dtype=jnp.complex64)
        x_phys_boundary = x_bnd * L
        amplitude_incidente = (
            jnp.exp(-1j * incidence * beta_n[mode_index] * x_phys_boundary)
            / u_norm_val
        )
        A_inc = A_inc.at[mode_index].set(amplitude_incidente)
        source_modes = -incidence * 2j * beta_n * A_inc
        incoming_boundary = jnp.asarray(int(x_bnd), dtype=jnp.int32) == incidence
        dtn_n = dtn_n + jnp.where(incoming_boundary, source_modes, 0.0)
        
        def DtN_eval(y_val):
            C_y = a_n * jnp.cos(n_modes * jnp.pi * (y_val + 1.0) / 2.0)
            return jnp.dot(C_y, dtn_n)
            
        dtn_pred_complex = jax.vmap(DtN_eval)(y_eval)
        
        dtn_actual = jax.vmap(uv_x, in_axes=(None, 0))(x_bnd, y_eval)
        dtn_actual_complex = dtn_actual[:, 0] + 1j * dtn_actual[:, 1]
        
        return jnp.mean(jnp.abs(dtn_actual_complex - dtn_pred_complex)**2)

    # PDE Residual Loss
    vmap_pde_residual = jax.vmap(pde_residual, in_axes=(0, 0))
    pde_loss = jnp.mean(vmap_pde_residual(x_pde, y_pde)**2)
    
    # Boundary Conditions Loss
    dtn_loss_left = compute_dtn_loss(-1.0, y_dtn, sign=-1)
    dtn_loss_right = compute_dtn_loss(1.0, y_dtn, sign=1)
    neum_loss = jnp.mean(
        jax.vmap(uv_y, in_axes=(0, None))(x_neumann, 1.0)**2
        + jax.vmap(uv_y, in_axes=(0, None))(x_neumann, -1.0)**2
    )
    bc_loss = neum_loss + dtn_loss_left + dtn_loss_right
    
    # Data Loss
    if is_warmup:
        data_loss = 0.0
    else:
        vmap_uv_y = jax.vmap(uv, (None, 0))
        data_loss = jnp.mean((vmap_uv_y(-1.0, y_bnd_left) - target_u_left)**2) 
        + jnp.mean((vmap_uv_y(1.0, y_bnd_right) - target_u_right)**2)

    total_loss = weights[0] * pde_loss + weights[1] * bc_loss + weights[2] * data_loss
    
    return total_loss, (pde_loss, bc_loss, data_loss)


def multimode_loss_fn(params_uv_stacked, layers_m, x_pde, y_pde,
                      x_neumann, y_dtn, frequencies, mode_indices, incidences,
                      targets_left, targets_right, y_bnds_left, y_bnds_right,
                      u_norms, weights):
    """Evaluate the mean weighted loss over all active incident modes."""
    def _loss_single(p_uv, frequency, mi, incidence, tl, tr, yl, yr, un):
        return loss_fn(
            p_uv, layers_m, x_pde, y_pde, x_neumann, y_dtn,
            frequency, mi, incidence, tl, tr, yl, yr, un, weights,
            is_warmup=False, use_healthy_guide=False,
        )

    vmapped = jax.vmap(_loss_single, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0))
    losses, (pdes, bcs, datas) = vmapped(
        params_uv_stacked, frequencies, mode_indices, incidences,
        targets_left, targets_right,
        y_bnds_left, y_bnds_right,
        u_norms,
    )
    return jnp.mean(losses), (jnp.mean(pdes), jnp.mean(bcs), jnp.mean(datas))


@functools.partial(jax.jit, static_argnames=('use_healthy_guide',))
def evaluate_forward_loss(params_uv, layers_m, x_pde, y_pde, x_neumann,
                          y_dtn, f_val, mode_index, incidence,
                          target_left, target_right, y_bnd_left, y_bnd_right,
                          u_norm, weights,
                          use_healthy_guide):
    """Evaluate phase 1 parameters without applying an optimizer update."""
    return loss_fn(
        params_uv, layers_m, x_pde, y_pde, x_neumann, y_dtn,
        f_val, mode_index, incidence,
        target_left, target_right, y_bnd_left, y_bnd_right,
        u_norm, weights,
        is_warmup=True, use_healthy_guide=use_healthy_guide,
    )


@jax.jit
def evaluate_inverse_loss(params_uv_stacked, layers_m, x_pde, y_pde,
                          x_neumann, y_dtn, frequencies, mode_indices, incidences,
                          targets_left, targets_right, y_bnds_left, y_bnds_right,
                          u_norms, weights):
    """Evaluate phase 2 parameters without applying an optimizer update."""
    return multimode_loss_fn(
        params_uv_stacked, layers_m, x_pde, y_pde, x_neumann, y_dtn,
        frequencies, mode_indices, incidences,
        targets_left, targets_right, y_bnds_left, y_bnds_right,
        u_norms, weights,
    )

# ==============================================================================
# SECTION 8: OPTIMIZATION SCHEMES & TRAIN STEPS (JIT COMPILED)
# ==============================================================================

def make_train_step_forward(adam_opt, lbfgs_opt):
    """Train step for Phase 1: only update one u-network, m is frozen."""

    @functools.partial(jax.jit, static_argnames=('N', 'use_lbfgs', 'use_healthy_guide'))
    def train_step_forward(params_uv, layers_m, opt_state_uv, key, f_val, mode_index,
                           incidence, target_left, target_right, y_bnd_left,
                           y_bnd_right, u_norm, current_weights,
                           N, use_lbfgs=False, use_healthy_guide=True):
        x_pde, y_pde, x_neumann, y_dtn = sample_collocation_points(key, N)

        def loss_uv(p_uv):
            return loss_fn(
                p_uv, layers_m,
                x_pde, y_pde, x_neumann, y_dtn,
                f_val, mode_index, incidence,
                target_left, target_right,
                y_bnd_left, y_bnd_right, u_norm, current_weights,
                is_warmup=True, use_healthy_guide=use_healthy_guide,
            )

        (loss, aux), grads_uv = jax.value_and_grad(loss_uv, has_aux=True)(params_uv)

        if use_lbfgs:
            def value_fn(p_uv):
                v, _ = loss_uv(p_uv)
                return v
            updates_uv, opt_state_uv = lbfgs_opt.update(
                grads_uv, opt_state_uv, params_uv,
                value=loss, grad=grads_uv, value_fn=value_fn)
        else:
            updates_uv, opt_state_uv = adam_opt.update(grads_uv, opt_state_uv, params_uv)

        params_uv = optax.apply_updates(params_uv, updates_uv)
        return params_uv, opt_state_uv, loss, aux

    return train_step_forward


def make_train_step_inverse_multimode(adam_opt_uv, adam_opt_m, lbfgs_opt_packed):
    """Train step for Phase 2: update all u-networks + shared m jointly."""

    @functools.partial(jax.jit, static_argnames=('N', 'use_lbfgs'))
    def train_step_inverse(params_uv_stacked, layers_m, opt_state_uv, opt_state_m,
                           opt_state_lbfgs, key, frequencies, mode_indices, incidences,
                           targets_left, targets_right,
                           y_bnds_left, y_bnds_right, u_norms,
                           current_weights, N, use_lbfgs=False):

        x_pde, y_pde, x_neumann, y_dtn = sample_collocation_points(key, N)
        
        def loss_all_modes(p_uv_stacked, l_m):
            return multimode_loss_fn(
                p_uv_stacked, l_m,
                x_pde, y_pde, x_neumann, y_dtn,
                frequencies, mode_indices, incidences,
                targets_left, targets_right, y_bnds_left, y_bnds_right,
                u_norms, current_weights,
            )

        (loss, aux), (grads_uv, grads_m) = jax.value_and_grad(
            loss_all_modes, argnums=(0, 1), has_aux=True)(params_uv_stacked, layers_m)

        if use_lbfgs:
            params_packed = {'uv_list': params_uv_stacked, 'm': layers_m}
            grads_packed = {'uv_list': grads_uv, 'm': grads_m}
            
            def value_fn_packed(p_packed):
                v, _ = loss_all_modes(p_packed['uv_list'], p_packed['m'])
                return v

            updates_packed, opt_state_lbfgs = lbfgs_opt_packed.update(
                grads_packed, opt_state_lbfgs, params_packed,
                value=loss, grad=grads_packed, value_fn=value_fn_packed)
            
            params_packed = optax.apply_updates(params_packed, updates_packed)
            params_uv_stacked = params_packed['uv_list']
            layers_m = params_packed['m']
        else:
            updates_uv, opt_state_uv = adam_opt_uv.update(grads_uv, opt_state_uv, params_uv_stacked)
            params_uv_stacked = optax.apply_updates(params_uv_stacked, updates_uv)

            updates_m, opt_state_m = adam_opt_m.update(grads_m, opt_state_m, layers_m)
            layers_m = optax.apply_updates(layers_m, updates_m)

        return params_uv_stacked, layers_m, opt_state_uv, opt_state_m, opt_state_lbfgs, loss, aux

    return train_step_inverse

# ==============================================================================
# SECTION 9: MAIN TRAINING LOOP
# ==============================================================================

def make_package_arrays(cases, mode_data):
    frequencies = jnp.asarray([frequency for frequency, _, _ in cases], dtype=jnp.float32)
    mode_indices = jnp.asarray([mode_index for _, mode_index, _ in cases], dtype=jnp.int32)
    incidences = jnp.asarray([incidence for _, _, incidence in cases], dtype=jnp.int32)
    targets_left = jnp.stack([mode_data[case]['U_left_norm'] for case in cases])
    targets_right = jnp.stack([mode_data[case]['U_right_norm'] for case in cases])
    y_bnds_left = jnp.stack([mode_data[case]['Y_left'] for case in cases])
    y_bnds_right = jnp.stack([mode_data[case]['Y_right'] for case in cases])
    u_norms = jnp.asarray([mode_data[case]['U_norm'] for case in cases], dtype=jnp.float32)
    return (
        frequencies,
        mode_indices,
        incidences,
        targets_left,
        targets_right,
        y_bnds_left,
        y_bnds_right,
        u_norms,
    )


def format_case(case):
    frequency, mode_index, incidence = case
    return f"f{frequency_label(frequency)} M{mode_index} I{incidence}"


def train(params_uv, layers_m, N_adam, N_lbfgs,
          max_steps_adam_phase1, max_steps_lbfgs_phase1,
          max_steps_adam_phase2, max_steps_lbfgs_phase2,
          packages, mode_data, eval_interval,
          switch_threshold, switch_window, key):
    """Train fullwave UV/M networks over incidence-aware training packages."""

    all_used_cases = sorted(set(flatten_packages(packages)))
    loss_forward = {case: [] for case in all_used_cases}
    loss_inverse = []
    best_validation_losses = {'forward': [], 'inverse': []}

    validation_points = regular_collocation_points(N_validation)
    best_params_uv = dict(params_uv)
    best_params_each_m = []

    template_param_uv = next(iter(params_uv.values()))
    param_labels_p1 = jax.tree_util.tree_map(lambda _: 'base', template_param_uv)
    if "sigma" in param_labels_p1:
        param_labels_p1["sigma"] = jax.tree_util.tree_map(lambda _: 'sigma', template_param_uv["sigma"])

    cosine_p1 = optax.schedules.cosine_decay_schedule(
        init_value=lr_uv,
        decay_steps=max(max_steps_adam_phase1, 1),
        alpha=schedule["forward_field_cosine_alpha"],
    )
    cosine_sigma_p1 = optax.schedules.cosine_decay_schedule(
        init_value=lr_sigma,
        decay_steps=max(max_steps_adam_phase1, 1),
        alpha=schedule["forward_sigma_cosine_alpha"],
    )

    adam_base_p1 = optax.chain(
        optax.clip_by_global_norm(gradient_clip_norm),
        optax.scale_by_adam(),
        optax.scale_by_schedule(cosine_p1),
        optax.scale(-1.0))
    adam_sigma_p1 = optax.chain(
        optax.clip_by_global_norm(gradient_clip_norm),
        optax.scale_by_adam(),
        optax.scale_by_schedule(cosine_sigma_p1),
        optax.scale(-1.0))

    adam_uv_p1 = optax.multi_transform(
        transforms={'base': adam_base_p1, 'sigma': adam_sigma_p1},
        param_labels=param_labels_p1)
    lbfgs_uv_p1 = optax.lbfgs()
    step_forward = make_train_step_forward(adam_uv_p1, lbfgs_uv_p1)

    cosine_uv_p2 = optax.schedules.cosine_decay_schedule(
        init_value=lr_uv,
        decay_steps=max(max_steps_adam_phase2, 1),
        alpha=schedule["inverse_field_cosine_alpha"],
    )
    cosine_sigma_p2 = optax.schedules.cosine_decay_schedule(
        init_value=lr_sigma,
        decay_steps=max(
            int(schedule["inverse_sigma_decay_fraction"] * max_steps_adam_phase2),
            1,
        ),
        alpha=schedule["inverse_sigma_cosine_alpha"],
    )
    cosine_m_p2 = optax.schedules.cosine_decay_schedule(
        init_value=lr_m,
        decay_steps=max(max_steps_adam_phase2, 1),
        alpha=schedule["inverse_material_cosine_alpha"],
    )

    adam_base_p2 = optax.chain(
        optax.clip_by_global_norm(gradient_clip_norm),
        optax.scale_by_adam(),
        optax.scale_by_schedule(cosine_uv_p2),
        optax.scale(-1.0))
    adam_sigma_p2 = optax.chain(
        optax.clip_by_global_norm(gradient_clip_norm),
        optax.scale_by_adam(),
        optax.scale_by_schedule(cosine_sigma_p2),
        optax.scale(-1.0))

    param_labels_p2 = jax.tree_util.tree_map(lambda _: 'base', template_param_uv)
    if "sigma" in param_labels_p2:
        param_labels_p2["sigma"] = jax.tree_util.tree_map(lambda _: 'sigma', template_param_uv["sigma"])

    adam_uv_p2 = optax.multi_transform(
        transforms={'base': adam_base_p2, 'sigma': adam_sigma_p2},
        param_labels=param_labels_p2)
    adam_m_p2 = optax.chain(
        optax.clip_by_global_norm(gradient_clip_norm),
        optax.scale_by_adam(),
        optax.scale_by_schedule(cosine_m_p2),
        optax.scale(-1.0))
    lbfgs_packed_p2 = optax.lbfgs()
    step_inverse = make_train_step_inverse_multimode(
        adam_uv_p2, adam_m_p2, lbfgs_packed_p2)

    for package_index, package in enumerate(packages):
        cases = flatten_package(package)
        if not cases:
            continue
        label = package_label(package_index, cases)
        use_healthy_guide_flag = package_index == 0

        print(f"\n{'='*60}")
        print(f"--- Package {package_index + 1}: {cases} ---")
        print(f"{'='*60}")

        for case in cases:
            freq, mode_idx, incidence = case
            param_uv = params_uv[case]
            best_params_uv[case] = param_uv
            best_loss = float('inf')
            best_forward_summary = None
            md = mode_data[case]
            loss_history_p1 = []
            switched_to_lbfgs = False

            print(
                f"\n--- Phase 1: {format_case(case)} | "
                f"Adam (max {max_steps_adam_phase1} steps) ---"
            )
            opt_state_uv = adam_uv_p1.init(param_uv)

            for step in range(max_steps_adam_phase1):
                key, subkey = jax.random.split(key)
                param_uv, opt_state_uv, _, _ = step_forward(
                    param_uv, layers_m, opt_state_uv, subkey, freq, mode_idx,
                    incidence, md['U_left_norm'], md['U_right_norm'],
                    md['Y_left'], md['Y_right'], md['U_norm'], weights_phase1,
                    N=N_adam, use_lbfgs=False,
                    use_healthy_guide=use_healthy_guide_flag)

                if step % eval_interval == 0:
                    loss, validation_aux = evaluate_forward_loss(
                        param_uv, layers_m, *validation_points,
                        freq, mode_idx, incidence,
                        md['U_left_norm'], md['U_right_norm'],
                        md['Y_left'], md['Y_right'], md['U_norm'],
                        weights_phase1, use_healthy_guide_flag,
                    )
                    pde_loss, bc_loss, _ = validation_aux
                    loss_value = float(loss)
                    loss_forward[case].append((pde_loss, bc_loss))
                    print(
                        f"[Adam {format_case(case)}] Step {step} | "
                        f"val pde {pde_loss:.2e} | val bc {bc_loss:.2e} | "
                        f"weighted val total {loss:.2e} | scales {param_uv['sigma']}"
                    )
                    loss_history_p1.append(loss_value)

                    if loss_value < best_loss:
                        best_loss = loss_value
                        best_params_uv[case] = param_uv
                        best_forward_summary = {
                            'frequency': float(freq),
                            'mode': int(mode_idx),
                            'incidence': int(incidence),
                            'optimizer': 'adam',
                            'step': int(step),
                            'pde': float(pde_loss),
                            'bc': float(bc_loss),
                            'data': 0.0,
                            'weighted_total': loss_value,
                            'weights': [float(value) for value in weights_phase1],
                        }

                    if check_switch_criterion(
                        loss_history_p1,
                        window=switch_window,
                        threshold=switch_threshold,
                    ):
                        print(
                            f"[Adam {format_case(case)}] Convergence criterion met "
                            f"at step {step}. Switching to L-BFGS."
                        )
                        switched_to_lbfgs = True
                        param_uv = best_params_uv[case]
                        break

            if switched_to_lbfgs or max_steps_lbfgs_phase1 > 0:
                print(
                    f"\n--- Phase 1: {format_case(case)} | "
                    f"L-BFGS (max {max_steps_lbfgs_phase1} steps) ---"
                )
                opt_state_uv = lbfgs_uv_p1.init(param_uv)

                for step in range(max_steps_lbfgs_phase1):
                    param_uv, opt_state_uv, _, _ = step_forward(
                        param_uv, layers_m, opt_state_uv, None, freq, mode_idx,
                        incidence, md['U_left_norm'], md['U_right_norm'],
                        md['Y_left'], md['Y_right'], md['U_norm'], weights_phase1,
                        N=N_lbfgs, use_lbfgs=True,
                        use_healthy_guide=use_healthy_guide_flag)

                    if step % eval_interval == 0:
                        loss, validation_aux = evaluate_forward_loss(
                            param_uv, layers_m, *validation_points,
                            freq, mode_idx, incidence,
                            md['U_left_norm'], md['U_right_norm'],
                            md['Y_left'], md['Y_right'], md['U_norm'],
                            weights_phase1, use_healthy_guide_flag,
                        )
                        pde_loss, bc_loss, _ = validation_aux
                        loss_value = float(loss)
                        loss_forward[case].append((pde_loss, bc_loss))
                        print(
                            f"[LBFGS {format_case(case)}] Step {step} | "
                            f"val pde {pde_loss:.2e} | val bc {bc_loss:.2e} | "
                            f"weighted val total {loss:.2e} | scales {param_uv['sigma']}"
                        )
                        if loss_value < best_loss:
                            best_loss = loss_value
                            best_params_uv[case] = param_uv
                            best_forward_summary = {
                                'frequency': float(freq),
                                'mode': int(mode_idx),
                                'incidence': int(incidence),
                                'optimizer': 'lbfgs',
                                'step': int(step),
                                'pde': float(pde_loss),
                                'bc': float(bc_loss),
                                'data': 0.0,
                                'weighted_total': loss_value,
                                'weights': [float(value) for value in weights_phase1],
                            }

            if best_forward_summary is None:
                raise RuntimeError(f'No finite validation loss for forward {case}')
            params_uv[case] = best_params_uv[case]
            best_validation_losses['forward'].append(best_forward_summary)

        best_loss = float('inf')
        best_params_m = layers_m
        best_inverse_summary = None

        if max_steps_adam_phase2 > 0 or max_steps_lbfgs_phase2 > 0:
            print(
                f"\n--- Phase 2: Package inverse | Adam "
                f"(max {max_steps_adam_phase2} steps) ---"
            )
            param_uv_list = [params_uv[case] for case in cases]
            params_uv_stacked = jax.tree_util.tree_map(
                lambda *args: jnp.stack(args, axis=0), *param_uv_list
            )
            best_params_uv_stacked = jax.tree_util.tree_map(jnp.copy, params_uv_stacked)
            opt_state_uv = adam_uv_p2.init(params_uv_stacked)
            opt_state_m = adam_m_p2.init(layers_m)
            opt_state_lbfgs = None
            package_arrays = make_package_arrays(cases, mode_data)

            data_weight_schedule = optax.linear_schedule(
                init_value=schedule["inverse_data_initial_weight"],
                end_value=float(weights_phase2_final[2]),
                transition_steps=max(
                    int(
                        schedule["inverse_data_transition_fraction"]
                        * max_steps_adam_phase2
                    ),
                    1,
                ),
            )

            loss_history_p2 = []
            switched_to_lbfgs_p2 = False

            for step in range(max_steps_adam_phase2):
                weights_phase2 = weights_phase2_final.at[2].set(
                    data_weight_schedule(step)
                )

                key, subkey = jax.random.split(key)
                (
                    params_uv_stacked, layers_m, opt_state_uv, opt_state_m,
                    opt_state_lbfgs, _, _,
                ) = step_inverse(
                    params_uv_stacked, layers_m, opt_state_uv, opt_state_m,
                    opt_state_lbfgs, subkey, *package_arrays, weights_phase2,
                    N=N_adam, use_lbfgs=False)

                if step % eval_interval == 0:
                    loss, validation_aux = evaluate_inverse_loss(
                        params_uv_stacked, layers_m, *validation_points,
                        *package_arrays, weights_phase2_final,
                    )
                    pde_loss, bc_loss, data_loss = validation_aux
                    loss_value = float(loss)
                    loss_inverse.append((pde_loss, bc_loss, data_loss))
                    sigmas = []
                    for case_item, sigma_val in zip(cases, params_uv_stacked['sigma']):
                        sigmas.append(f"{format_case(case_item)} sigma={sigma_val}")
                    sigmas_str = " | ".join(sigmas)
                    print(
                        f"[Adam inverse] Step {step} | val pde {pde_loss:.2e} | "
                        f"val bc {bc_loss:.2e} | val data {data_loss:.2e} | "
                        f"weighted val total {loss:.2e} | "
                        f"train data weight {weights_phase2[2]:.2e} | {sigmas_str}"
                    )
                    loss_history_p2.append(loss_value)

                    if loss_value < best_loss:
                        best_loss = loss_value
                        best_params_uv_stacked = jax.tree_util.tree_map(
                            jnp.copy, params_uv_stacked
                        )
                        best_params_m = jax.tree_util.tree_map(jnp.copy, layers_m)
                        best_inverse_summary = {
                            'package': package_index + 1,
                            'label': label,
                            'cases': [
                                (float(f), int(m), int(i))
                                for f, m, i in cases
                            ],
                            'optimizer': 'adam',
                            'step': int(step),
                            'pde': float(pde_loss),
                            'bc': float(bc_loss),
                            'data': float(data_loss),
                            'weighted_total': loss_value,
                            'weights': [float(value) for value in weights_phase2_final],
                        }

                    if check_switch_criterion(
                        loss_history_p2,
                        window=switch_window,
                        threshold=switch_threshold,
                    ):
                        print(
                            f"[Adam inverse] Convergence criterion met at step {step}. "
                            "Switching to L-BFGS."
                        )
                        switched_to_lbfgs_p2 = True
                        params_uv_stacked = best_params_uv_stacked
                        layers_m = best_params_m
                        break

            if switched_to_lbfgs_p2 or max_steps_lbfgs_phase2 > 0:
                print(
                    f"\n--- Phase 2: Package inverse | L-BFGS "
                    f"(max {max_steps_lbfgs_phase2} steps) ---"
                )
                opt_state_lbfgs = lbfgs_packed_p2.init(
                    {'uv_list': params_uv_stacked, 'm': layers_m}
                )

                for step in range(max_steps_lbfgs_phase2):
                    (
                        params_uv_stacked, layers_m, opt_state_uv, opt_state_m,
                        opt_state_lbfgs, _, _,
                    ) = step_inverse(
                        params_uv_stacked, layers_m, opt_state_uv, opt_state_m,
                        opt_state_lbfgs, None, *package_arrays, weights_phase2_final,
                        N=N_lbfgs, use_lbfgs=True)

                    if step % eval_interval == 0:
                        loss, validation_aux = evaluate_inverse_loss(
                            params_uv_stacked, layers_m, *validation_points,
                            *package_arrays, weights_phase2_final,
                        )
                        pde_loss, bc_loss, data_loss = validation_aux
                        loss_value = float(loss)
                        loss_inverse.append((pde_loss, bc_loss, data_loss))
                        print(
                            f"[LBFGS inverse] Step {step} | val pde {pde_loss:.2e} | "
                            f"val bc {bc_loss:.2e} | val data {data_loss:.2e} | "
                            f"weighted val total {loss:.2e}"
                        )
                        if loss_value < best_loss:
                            best_loss = loss_value
                            best_params_uv_stacked = jax.tree_util.tree_map(
                                jnp.copy, params_uv_stacked
                            )
                            best_params_m = jax.tree_util.tree_map(jnp.copy, layers_m)
                            best_inverse_summary = {
                                'package': package_index + 1,
                                'label': label,
                                'cases': [
                                    (float(f), int(m), int(i))
                                    for f, m, i in cases
                                ],
                                'optimizer': 'lbfgs',
                                'step': int(step),
                                'pde': float(pde_loss),
                                'bc': float(bc_loss),
                                'data': float(data_loss),
                                'weighted_total': loss_value,
                                'weights': [float(value) for value in weights_phase2_final],
                            }

            if best_inverse_summary is None:
                raise RuntimeError(
                    f'No finite validation loss for inverse package {package_index + 1}'
                )
            for index, case in enumerate(cases):
                params_uv[case] = jax.tree_util.tree_map(
                    lambda value, idx=index: value[idx], best_params_uv_stacked
                )
            layers_m = best_params_m
            best_params_each_m.append(
                (package_index, cases, jax.tree_util.tree_map(jnp.copy, layers_m))
            )
            best_validation_losses['inverse'].append(best_inverse_summary)

    return (
        params_uv,
        layers_m,
        best_params_each_m,
        key,
        loss_forward,
        loss_inverse,
        best_validation_losses,
    )

# ==============================================================================
# SECTION 10: MAIN ENTRY POINT
# ==============================================================================

def main():
    global key

    output_root = RUN_CONFIG.prepare_output_root(REPOSITORY_ROOT)
    fig_dir = output_root / "fig"
    checkpoint_dir = output_root / "checkpoints"
    fig_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("JAX Devices:", jax.devices())
    print("Default JAX dtype:", jnp.ones(1).dtype)
    print("Run config ID:", RUN_CONFIG.identifier)
    print("Run output root:", output_root)

    # ==================================================================
    # Data Loading & Preprocessing
    # ==================================================================
    left_path, right_path = boundary_data_paths(
        data_dir, defect_name
    )
    boundary_data = WaveguideBoundaryData(left_path, right_path)

    requested_cases = sorted(set(flatten_packages(training_packages)))
    missing_cases = [
        case for case in requested_cases
        if not boundary_data.has_pair(case[1], case[0], case[2])
    ]
    if missing_cases:
        raise ValueError(
            f'The combined FEM boundary files do not contain requested cases: {missing_cases}. '
            f'Available triplets: {boundary_data.available_triplets}'
        )

    mode_data = {}
    for frequency, mode_idx, incidence in requested_cases:
        pair = boundary_data.get_pair(mode_idx, frequency, incidence)
        if not np.isclose(pair.x_left, -L, rtol=0.0, atol=1e-6) or not np.isclose(
            pair.x_right, L, rtol=0.0, atol=1e-6
        ):
            raise ValueError(
                f'Boundary coordinates for mode {mode_idx}, frequency {frequency}, '
                f'incidence {incidence} are x_left={pair.x_left}, x_right={pair.x_right}; '
                f'expected {-L} and {L}'
            )

        y_left = 2 * jnp.asarray(pair.y_left, dtype=jnp.float32) / H - 1
        y_right = 2 * jnp.asarray(pair.y_right, dtype=jnp.float32) / H - 1
        re_l = jnp.asarray(pair.u_re_left, dtype=jnp.float32)
        im_l = jnp.asarray(pair.u_im_left, dtype=jnp.float32)
        re_r = jnp.asarray(pair.u_re_right, dtype=jnp.float32)
        im_r = jnp.asarray(pair.u_im_right, dtype=jnp.float32)

        norm = jnp.sqrt(jnp.max(jnp.concatenate((
            re_l**2 + im_l**2,
            re_r**2 + im_r**2
        ))))
        if not np.isfinite(float(norm)) or float(norm) <= 0.0:
            raise ValueError(
                f'Invalid zero or non-finite field norm for mode {mode_idx}, '
                f'frequency {frequency}, incidence {incidence}'
            )

        mode_data[(float(frequency), int(mode_idx), int(incidence))] = {
            'Y_left': y_left,
            'Y_right': y_right,
            'U_left_norm': jnp.stack([re_l / norm, im_l / norm], axis=1),
            'U_right_norm': jnp.stack([re_r / norm, im_r / norm], axis=1),
            'U_norm': norm,
        }

    used_modes = sorted({mode_idx for _, mode_idx, _ in requested_cases})
    for mode_idx in used_modes:
        mode_cases = [case for case in requested_cases if case[1] == mode_idx]
        print(
            f"  Mode {mode_idx}: {len(mode_cases)} frequency/incidence cases loaded"
        )

    for package_index, package in enumerate(training_packages):
        cases = flatten_package(package)
        left_counts = {mode_data[case]['Y_left'].shape[0] for case in cases}
        right_counts = {mode_data[case]['Y_right'].shape[0] for case in cases}
        if len(left_counts) != 1 or len(right_counts) != 1:
            raise ValueError(
                f'All cases in package {package_index + 1} must use the same number '
                'of samples per side'
            )

    # ==================================================================
    # Network Initialization
    # ==================================================================

    # Slowness parameter network (layers_m) — shared across all modes and frequencies
    key, subkey_m = jax.random.split(key)
    layers_m = init_layers(subkey_m, n_layers_m)
    initial_output_bias = -jnp.log((m_max-m_min)/(m0-m_min) - 1)
    layers_m[-1]["b"] = jnp.full_like(
        layers_m[-1]["b"], initial_output_bias
    )
    layers_m[-1]["W"] /= 10.0

    # Wavefield networks: one per (frequency, mode, incidence)
    params_uv = {}
    for freq, mode_idx, incidence in requested_cases:
        key, subkey_f = jax.random.split(key)
        params_uv[(freq, mode_idx, incidence)] = init_layers_uv(
            subkey_f, n_layers_uv, freq, mode_idx
        )

    # ==================================================================
    # Initial Sound Speed Plot
    # ==================================================================
    print("Plotting initial sound speed profile...")
    x_plot = np.linspace(-L, L, 100)
    y_plot = np.linspace(0, H, 50)
    c_grid = jax.vmap(
        jax.vmap(lambda x, y: c(x, y, layers_m), in_axes=(0, None)),
        in_axes=(None, 0)
    )(x_plot / L, 2 * y_plot / H - 1)

    plt.figure(figsize=(7, 3.5))
    plt.pcolormesh(x_plot, y_plot, c_grid, rasterized=True)
    plt.colorbar(label="Sound speed c(x, y)")
    plt.title("Initial Sound Speed Field")
    plt.tight_layout()
    #plt.savefig(fig_dir / 'initial_sound_speed.pdf')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    # ==================================================================
    # Training
    # ==================================================================
    key, subkey = jax.random.split(key)
    (
        params_uv,
        layers_m,
        best_params_each_m,
        key,
        loss_forward,
        loss_inverse,
        best_validation_losses,
    ) = train(
        params_uv, layers_m, N_adam, N_lbfgs,
        max_steps_adam_phase1, max_steps_lbfgs_phase1,
        max_steps_adam_phase2, max_steps_lbfgs_phase2,
        training_packages, mode_data,
        eval_interval, switch_threshold, switch_window, subkey
    )

    # ==================================================================
    # Post-processing Plots
    # ==================================================================

    # Precompute common strings for filenames (computed once)
    all_used_modes = sorted({mode_idx for _, mode_idx, _ in requested_cases})
    all_used_incidences = sorted({incidence for _, _, incidence in requested_cases})
    modes_str = '_'.join(str(mi) for mi in all_used_modes)
    freqs_str = '_'.join(frequency_label(frequency) for frequency in all_training_frequencies)
    incidences_str = '_'.join(incidence_label(incidence) for incidence in all_used_incidences)

    # Save every UV network together with the Fourier basis and physical scaling.
    checkpoint_path = os.path.join(
        checkpoint_dir,
        (
            f'checkpoint_fullwave_{defect_name}_inc{incidences_str}'
            f'_modes{modes_str}_freqs{freqs_str}.npz'
        )
    )
    save_uv_checkpoint(
        checkpoint_path,
        params_uv,
        B_base,
        collect_u_norms(params_uv, mode_data),
        length=L,
        height=H,
        c0=c0,
        layers_m=layers_m,
        c_min=cmin,
        c_max=cmax,
        network_config={
            'uv_layers': list(n_layers_uv),
            'm_layers': list(n_layers_m),
            'fourier_features': int(m_fourier_features),
            'hidden_activation': 'tanh',
            'uv_feature_mapping': 'random_fourier_cos_sin',
            'm_output_parameterization': 'bounded_slowness_sigmoid',
        },
        best_validation_losses=best_validation_losses,
        random_seed=random_seed,
        metadata={
            'defect_name': defect_name,
            'training_packages': [
                {
                    str(incidence): {
                        str(frequency): list(modes)
                        for frequency, modes in frequency_map.items()
                    }
                    for incidence, frequency_map in package.items()
                }
                for package in training_packages
            ],
            'run_config_id': RUN_CONFIG.identifier,
            'run_config_source': (
                str(RUN_CONFIG.source) if RUN_CONFIG.source is not None else None
            ),
            'run_config': RUN_CONFIG.values,
        },
    )
    print(f'UV/M checkpoint saved to: {checkpoint_path}')

    # --- Plot Phase 1 (Forward) Training Losses per mode ---
    for case in sorted(loss_forward.keys()):
        if len(loss_forward[case]) == 0:
            continue
        loss_fwd_arr = np.array(loss_forward[case])
        pde_losses = loss_fwd_arr[:, 0]
        bc_losses = loss_fwd_arr[:, 1]
        total_losses = weights_phase1[0] * pde_losses + weights_phase1[1] * bc_losses

        x_axis = eval_interval * np.arange(len(loss_fwd_arr))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.semilogy(x_axis, pde_losses, label="PDE Loss")
        ax1.semilogy(x_axis, bc_losses, label="BC Loss")
        case_title = format_case(case)
        ax1.set_title(f"Phase 1 {case_title}: Partial Loss")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.grid(True)

        ax2.semilogy(x_axis, total_losses, label="Weighted Validation Loss", color='black')
        ax2.set_title(f"Phase 1 {case_title}: Weighted Validation Loss")
        ax2.set_xlabel("Steps")
        ax2.legend(loc="upper right")
        ax2.grid(True)
        plt.tight_layout()
        if SHOW_PLOTS:
            plt.show()
        plt.close()

    # --- Plot Phase 2 (Inverse) Training Losses ---
    if len(loss_inverse) > 0:
        loss_inverse_arr = np.array(loss_inverse)
        pde_losses_inv = loss_inverse_arr[:, 0]
        bc_losses_inv = loss_inverse_arr[:, 1]
        data_losses_inv = loss_inverse_arr[:, 2]
        total_losses_inv = (
            weights_phase2_final[0] * pde_losses_inv
            + weights_phase2_final[1] * bc_losses_inv
            + weights_phase2_final[2] * data_losses_inv
        )

        x_axis_inv = eval_interval * np.arange(len(loss_inverse_arr))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.semilogy(x_axis_inv, pde_losses_inv, label="PDE Loss")
        ax1.semilogy(x_axis_inv, bc_losses_inv, label="BC Loss")
        ax1.semilogy(x_axis_inv, data_losses_inv, label="Data Loss")
        ax1.set_title("Phase 2 (Multi-mode): Partial Loss")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.grid(True)

        ax2.semilogy(x_axis_inv, total_losses_inv, label="Weighted Validation Loss", color='black')
        ax2.set_title("Phase 2 (Multi-mode): Weighted Validation Loss")
        ax2.set_xlabel("Steps")
        ax2.legend(loc="upper right")
        ax2.grid(True)
        plt.tight_layout()
        plt.savefig(fig_dir / f'losses_fullwave_{defect_name}_modes{modes_str}_freqs{freqs_str}.pdf')
        if SHOW_PLOTS:
            plt.show()
        plt.close()

    # --- Plot final reconstructed sound speed field c(x, y) ---
    for package_index, cases, layers_m_each in best_params_each_m:
        def c_final(x, y, _lm=layers_m_each):
            return c(x, y, _lm)

        x_c = np.linspace(-L, L, 100)
        y_c = np.linspace(0, H, 50)
        c_grid_final = jax.vmap(jax.vmap(c_final, in_axes=(0, None)), in_axes=(None, 0))(x_c / L, 2 * y_c / H - 1)

        plt.figure(figsize=(7, 3.5))
        plt.pcolormesh(x_c, y_c, c_grid_final, rasterized=True)
        plt.colorbar(label="Sound speed c(x, y)")
        label = package_label(package_index, cases)
        plt.title(f"Reconstructed c(x,y) - {label}")
        plt.tight_layout()
        plt.savefig(
            fig_dir
            / f'c_map_fullwave_{defect_name}_{label}.pdf'
        )
        if SHOW_PLOTS:
            plt.show()
        plt.close()


if __name__ == "__main__":
    main()
