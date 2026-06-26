import os
import functools
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

# Enforce float32 globally to prevent silent float64 promotion
jax.config.update("jax_enable_x64", False)

import optax
from data_loader import WaveguideBoundaryData

# Print JAX devices to verify setup
print("JAX Devices:", jax.devices())
print("Default JAX dtype:", jnp.ones(1).dtype)

# ==============================================================================
# SECTION 1: CONFIGURATION & HYPERPARAMETERS
# ==============================================================================

# --- Domain Geometry ---
H = 0.6  # Height of the waveguide
L = 1.0  # Half-length of the waveguide

# --- Physics Parameters ---
c0 = 340.0         # Reference speed of sound
cmax = 1.5 * c0    # Maximum speed of sound
cmin = 0.5 * c0    # Minimum speed of sound
contrast = 1.0

# Calculated inverse-squared speed boundaries
msq_min = 1.0 / (cmax**2)
msq_max = 1.0 / (cmin**2)

# --- Data Configuration ---
# Data files follow the naming convention:
#   pinn_boundary_{left/right}_{defect_name}_contrast{X}percent_mode{N}.csv
script_dir = os.path.dirname(os.path.abspath(__file__))
defect_name = 'barrehalf'
contrast_label = '20percent'

# --- Mode Selection ---
# max_mode_index: highest mode index to attempt loading (None = auto-detect all)
# enforce_even_modes: if True, use an even number of excitation modes per frequency
#   to balance symmetric (n=0,2,4..) and antisymmetric (n=1,3,5..) modes about y=H/2,
#   avoiding a symmetry prior in the reconstructed celerity map.
max_mode_index = None       # None = load all available, or set e.g. 1 to cap at mode 1
enforce_even_modes = True

# Frequencies to run training on (curriculum learning: low to high)
training_frequencies = np.array([600.0, 900.0])

# Auto-discover available mode data files
def discover_mode_files(script_dir, defect_name, contrast_label, max_mode_index=None):
    """Find all mode data files that exist on disk, up to max_mode_index."""
    data_paths = {}
    mode_idx = 0
    while True:
        if max_mode_index is not None and mode_idx > max_mode_index:
            break
        left_path = os.path.join(script_dir, 'data',
            f'pinn_boundary_left_{defect_name}_contrast{contrast_label}_mode{mode_idx}.csv')
        right_path = os.path.join(script_dir, 'data',
            f'pinn_boundary_right_{defect_name}_contrast{contrast_label}_mode{mode_idx}.csv')
        if os.path.isfile(left_path) and os.path.isfile(right_path):
            data_paths[mode_idx] = (left_path, right_path)
            mode_idx += 1
        else:
            break
    return data_paths

data_paths = discover_mode_files(script_dir, defect_name, contrast_label, max_mode_index)
available_modes = sorted(data_paths.keys())
print(f"Discovered mode data files for modes: {available_modes}")

# --- Random Seed ---
random_seed = 0
key = jax.random.key(random_seed)

# --- Neural Network Architectures ---
m = 64  # Dimensionality of Fourier feature mapping
n_input = 2
n_layers_uv = [2 * m, 128, 128, 64, 2]
n_layers_m = [n_input, 128, 64, 1]

# --- Gauss-Legendre Quadrature ---
n_gauss_legendre = 30

# --- Optimizer Learning Rates ---
lr_uv = 3e-4
lr_m = 3e-5
lr_sigma = 1e-2

# --- Training Hyperparameters ---
N_adam = 4000            # Number of collocation points for Adam
N_lbfgs = 1000           # Number of collocation points for L-BFGS
eval_interval = 200      # Interval for evaluating losses and printing progress
switch_threshold = 0.0   # Threshold for L-BFGS switch criterion
switch_window = 10       # Window size for switch criterion

max_steps_adam_phase1 = 40001
max_steps_lbfgs_phase1 = 601
max_steps_adam_phase2 = 100001
max_steps_lbfgs_phase2 = 3001

SHOW_PLOTS = True

weights_phase1_ = jnp.array([1.0, 10.0, 0.0])

# ==============================================================================
# SECTION 2: DATA LOADING & PREPROCESSING
# ==============================================================================

# Load boundary data for each discovered mode.
# mode_data[mode_idx] stores Y coords, normalized fields, and available frequencies.
mode_data = {}

for mode_idx in available_modes:
    left_path, right_path = data_paths[mode_idx]
    loader_left = WaveguideBoundaryData(left_path)
    loader_right = WaveguideBoundaryData(right_path)

    X_left, Y_left_raw, U_re_left, U_im_left, mode_freq = loader_left.get_training_data()
    X_right, Y_right_raw, U_re_right, U_im_right, _ = loader_right.get_training_data()

    # Process Y coordinates (normalize to [-1, 1])
    Y_left = jnp.array(Y_left_raw[mode_freq[0]], dtype=jnp.float32)
    Y_right = jnp.array(Y_right_raw[mode_freq[0]], dtype=jnp.float32)

    Y_norm_val = jnp.max(Y_left)
    Y_left = 2 * Y_left / Y_norm_val - 1
    Y_right = 2 * Y_right / Y_norm_val - 1

    # Normalize complex wave fields per frequency
    U_norm = {}
    U_left_norm = {}
    U_right_norm = {}

    for f in mode_freq:
        re_l = jnp.array(U_re_left[f], dtype=jnp.float32)
        im_l = jnp.array(U_im_left[f], dtype=jnp.float32)
        re_r = jnp.array(U_re_right[f], dtype=jnp.float32)
        im_r = jnp.array(U_im_right[f], dtype=jnp.float32)

        norm = jnp.sqrt(jnp.max(jnp.concatenate((
            re_l**2 + im_l**2,
            re_r**2 + im_r**2
        ))))
        
        U_norm[f] = norm
        re_l /= norm
        im_l /= norm
        re_r /= norm
        im_r /= norm
        
        U_left_norm[f] = jnp.stack([re_l, im_l], axis=1)
        U_right_norm[f] = jnp.stack([re_r, im_r], axis=1)

    mode_data[mode_idx] = {
        'Y_left': Y_left,
        'Y_right': Y_right,
        'U_left_norm': U_left_norm,
        'U_right_norm': U_right_norm,
        'U_norm': U_norm,
        'dataset_freq': set(float(f) for f in mode_freq),  # set for O(1) lookup
    }
    print(f"  Mode {mode_idx}: {len(mode_freq)} frequencies loaded ({float(mode_freq[0]):.0f}–{float(mode_freq[-1]):.0f} Hz)")

# Use mode 0's frequencies as the reference dataset (superset of all)
dataset_freq = mode_data[0]['dataset_freq']

# ==============================================================================
# SECTION 2b: DYNAMIC MODE SELECTION PER FREQUENCY
# ==============================================================================

def compute_active_modes(freq, c0, H, available_modes, mode_data, enforce_even=True):
    active = []
    for n in available_modes:
        f_cutoff = n * c0 / (2 * H)
        if freq <= f_cutoff:
            continue  # evanescent in healthy guide
        # Check data availability
        if freq not in mode_data[n]['dataset_freq']:
            continue
        active.append(n)
    
    if len(active) == 0:
        # Fallback: mode 0 should always be available
        raise ValueError(f"No propagative modes with data at {freq} Hz. "
                         f"Check your training_frequencies.")
    
    # Enforce even count (drop highest mode if needed)
    if enforce_even and len(active) > 1 and len(active) % 2 != 0:
        active = active[:-1]  # drop the highest mode
    
    return active

# Pre-compute active modes for each training frequency
active_modes_per_freq = {}
for freq in training_frequencies:
    active = compute_active_modes(float(freq), c0, H, available_modes,
                                   mode_data, enforce_even=enforce_even_modes)
    active_modes_per_freq[float(freq)] = active
    print(f"  f={freq:.0f} Hz → active modes: {active}")

active_modes_per_freq[600.0] = [0, 1, 2] #to customize the training

active_modes_per_freq[900.0] = [0, 1,  2, 3]

# Collect all (freq, mode) pairs that will be used
all_fm_pairs = set()
for freq, modes in active_modes_per_freq.items():
    for mi in modes:
        all_fm_pairs.add((freq, mi))

# ==============================================================================
# SECTION 3: FOURIER FEATURE MAPPING & SETUP
# ==============================================================================

# Initialize random projection matrix for Fourier features
key, subkey = jax.random.split(key)
B_base = jax.random.normal(subkey, (m, 2))

def gamma(x, y, sigma):
    x_phys = x * L 
    y_phys = (y + 1.0) * H / 2.0
    v_phys = jnp.array([x_phys, y_phys])
    B = B_base * sigma
    return jnp.concatenate([jnp.cos(B @ v_phys), jnp.sin(B @ v_phys)])

# --- Gauss-Legendre Quadrature Setup ---
y_gauss_legendre, w_gauss_legendre = np.polynomial.legendre.leggauss(n_gauss_legendre)
y_quad = (y_gauss_legendre + 1.0) * H / 2.0
w_quad = w_gauss_legendre * H / 2.0

# Calculate dynamic boundary modes configuration using max training frequency
fmax = float(max(training_frequencies))
N_modes = int(np.round(2 * H * fmax / c0)) + 5

n_modes = jnp.arange(N_modes, dtype=jnp.float32)
a_n = jnp.sqrt(2.0 / H) * jnp.ones(N_modes, dtype=jnp.float32)
a_n = a_n.at[0].set(jnp.sqrt(1.0 / H))
C_quad = a_n[:, None] * jnp.cos(n_modes[:, None] * jnp.pi * y_quad[None, :] / H)

def precompute_C_y(y_data, n_modes_arr, a_n_arr):
    return a_n_arr[None, :] * jnp.cos(n_modes_arr[None, :] * jnp.pi * (y_data[:, None] + 1.0) / 2.0)

# Add precomputed C_y_left and C_y_right to mode_data
for mi in mode_data:
    mode_data[mi]['C_y_left'] = precompute_C_y(mode_data[mi]['Y_left'], n_modes, a_n)
    mode_data[mi]['C_y_right'] = precompute_C_y(mode_data[mi]['Y_right'], n_modes, a_n)

# ==============================================================================
# SECTION 4: NEURAL NETWORK UTILITIES & INITIALIZATION
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
    kx = 2 * jnp.pi * f_val / (contrast * c0)
    ky = (mode_val + int(mode_val==0)) * jnp.pi / H
    layers_uv = init_layers(key, n_layers)
    return {'layers': layers_uv, 'sigma': jnp.array([kx, ky])}

# --- Network Initialization ---
# Slowness parameter network (layers_m) — shared across all modes and frequencies
key, subkey_m = jax.random.split(key)
layers_m = init_layers(subkey_m, n_layers_m)
layers_m[-1]["b"] = -jnp.log(27.0 / 5.0)
layers_m[-1]["W"] /= 10.0

# Wavefield networks: one per (frequency, mode)
# params_uv[(freq, mode_idx)] = {'layers': ..., 'sigma': ...}
params_uv = {}
for (f, mode_idx) in all_fm_pairs:
    key, subkey_f = jax.random.split(key)
    params_uv[(f, mode_idx)] = init_layers_uv(subkey_f, n_layers_uv, f, mode_idx)

# ==============================================================================
# SECTION 5: MODEL FORWARD PASSES & INITIAL PLOT
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
    Z = msq_min + (msq_max - msq_min) * jax.nn.sigmoid(Z @ layers[-1]["W"] + layers[-1]["b"])
    return Z.squeeze()

def c(x, y):
    return 1 / jnp.sqrt(forward_params(layers_m, jnp.array([x, y])))

# Plot initial sound speed field
print("Plotting initial sound speed profile...")
x_plot = np.linspace(-1, 1, 100)
y_plot = np.linspace(0, 0.6, 50)
c_grid = jax.vmap(jax.vmap(c, in_axes=(0, None)), in_axes=(None, 0))(x_plot, 2 * y_plot / H - 1)

os.makedirs(os.path.join(script_dir, 'fig'), exist_ok=True)
plt.figure(figsize=(7, 3.5))
plt.pcolormesh(x_plot, y_plot, c_grid)
plt.colorbar(label="Sound speed c(x, y)")
plt.title("Initial Sound Speed Field")
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'fig', f'initial_sound_speed.png'), dpi=150)
if SHOW_PLOTS:
    plt.show()
plt.close()

# ==============================================================================
# SECTION 6: LOSS FUNCTION DEFINITIONS
# ==============================================================================

def loss_fn(params_uv, layers_m, x, y, f, mode_index, target_u_left, target_u_right,
            y_bnd_left, y_bnd_right, u_norm_val, weights, k0, beta_n, C_y_left, C_y_right,
            is_warmup=False, use_healthy_guide=False):
    B_scaled = B_base * params_uv['sigma']
    
    def uv(x, y):
        x_phys = x * L 
        y_phys = (y + 1.0) * H / 2.0
        proj = B_scaled[:, 0] * x_phys + B_scaled[:, 1] * y_phys
        features = jnp.concatenate([jnp.cos(proj), jnp.sin(proj)])
        return forward_func(params_uv['layers'], features)
    
    def m(x, y):
        return forward_params(layers_m, jnp.array([x, y]))
    
    def k2(x, y):
        if use_healthy_guide:
            return (2 * jnp.pi * f / c0)**2
        else:
            return (2 * jnp.pi * f)**2 * m(x, y)
    
    def uv_x(x, y):
        return jax.jacfwd(uv, argnums=0)(x, y) / L
    
    def uv_y(x, y):
        return jax.jacfwd(uv, argnums=1)(x, y) * (2 / H)

    def pde_residual(x, y):
        uv_xx = jax.jacfwd(jax.jacfwd(uv, argnums=0), argnums=0)(x, y) / L**2
        uv_yy = jax.jacfwd(jax.jacfwd(uv, argnums=1), argnums=1)(x, y) * (2 / H)**2
        return (uv_xx + uv_yy) + k2(x,y)*uv(x, y)
    
    def compute_dtn_loss(x_bnd, y_data, C_y_data, sign, A_inc=None):
        uv_quad = jax.vmap(uv, in_axes=(None, 0))(x_bnd, y_gauss_legendre)
        U_quad_complex = uv_quad[:, 0] + 1j * uv_quad[:, 1]
        
        u_n = C_quad @ (w_quad * U_quad_complex)
        
        dtn_n = sign * 1j * beta_n * u_n
        if A_inc is not None:
            dtn_n = dtn_n + 2j * beta_n * A_inc
        
        dtn_pred_complex = C_y_data @ dtn_n
        
        dtn_actual = jax.vmap(uv_x, in_axes=(None, 0))(x_bnd, y_data)
        dtn_actual_complex = dtn_actual[:, 0] + 1j * dtn_actual[:, 1]
        
        return jnp.mean(jnp.abs(dtn_pred_complex - dtn_actual_complex)**2)

    # Incident field: mode `mode_index` propagating from the left
    A_inc_left = jnp.zeros(N_modes, dtype=jnp.complex64)
    amplitude_incidente = jnp.exp(-1j * beta_n[mode_index] * L) / u_norm_val
    A_inc_left = A_inc_left.at[mode_index].set(amplitude_incidente)
    
    # PDE Residual Loss
    vmap_pde_residual = jax.vmap(pde_residual, in_axes=(0, 0))
    pde_loss = jnp.mean(vmap_pde_residual(x, y)**2)
    
    # Boundary Conditions Loss
    dtn_loss_left = compute_dtn_loss(-1.0, y_bnd_left, C_y_left, sign=-1, A_inc=A_inc_left)
    dtn_loss_right = compute_dtn_loss(1.0, y_bnd_right, C_y_right, sign=1, A_inc=None)
    neum_loss = jnp.mean(jax.vmap(uv_y, in_axes=(0, None))(x, 1.0)**2 + jax.vmap(uv_y, in_axes=(0, None))(x, -1.0)**2)
    bc_loss = neum_loss + dtn_loss_left + dtn_loss_right
    
    # Data Loss
    if is_warmup:
        data_loss = 0.0
    else:
        vmap_uv_y = jax.vmap(uv, (None, 0))
        data_loss = jnp.mean((vmap_uv_y(-1.0, y_bnd_left) - target_u_left)**2) + jnp.mean((vmap_uv_y(1.0, y_bnd_right) - target_u_right)**2)

    total_loss = weights[0] * pde_loss + weights[1] * bc_loss + weights[2] * data_loss
    
    return total_loss, (pde_loss, bc_loss, data_loss)

# ==============================================================================
# SECTION 7: OPTIMIZATION SCHEMES & TRAIN STEPS (JIT COMPILED)
# ==============================================================================

def make_train_step_forward(adam_opt, lbfgs_opt):
    """Train step for Phase 1: only update one u-network, m is frozen."""

    @functools.partial(jax.jit, static_argnames=('N', 'use_lbfgs', 'use_healthy_guide'))
    def train_step_forward(params_uv, layers_m, opt_state_uv, key, f_val, mode_index,
                           target_left, target_right, y_bnd_left, y_bnd_right, u_norm,
                           current_weights, k0, beta_n, C_y_left, C_y_right,
                           N, use_lbfgs=False, use_healthy_guide=True):
        if use_lbfgs:
            subkey1, subkey2 = jax.random.split(key, 2)
            x = jax.random.uniform(subkey1, (N,), minval=-1.0, maxval=1.0)
            y = jax.random.uniform(subkey2, (N,), minval=-1.0, maxval=1.0)
            new_key = key
        else:
            new_key, subkey1, subkey2 = jax.random.split(key, 3)
            x = jax.random.uniform(subkey1, (N,), minval=-1.0, maxval=1.0)
            y = jax.random.uniform(subkey2, (N,), minval=-1.0, maxval=1.0)

        def loss_uv(p_uv):
            return loss_fn(p_uv, layers_m, x, y, f_val, mode_index,
                           target_left, target_right,
                           y_bnd_left, y_bnd_right, u_norm, current_weights,
                           k0, beta_n, C_y_left, C_y_right,
                           is_warmup=True, use_healthy_guide=use_healthy_guide)

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
        return params_uv, opt_state_uv, loss, aux, new_key

    return train_step_forward


def make_train_step_inverse_multimode(adam_opt_uv, adam_opt_m, lbfgs_opt_packed):

    @functools.partial(jax.jit, static_argnames=('N', 'use_lbfgs'))
    def train_step_inverse(params_uv_stacked, layers_m, opt_state_uv, opt_state_m,
                           opt_state_lbfgs, key, f_val, mode_indices,
                           targets_left, targets_right,
                           y_bnds_left, y_bnds_right, u_norms,
                           current_weights, k0, beta_n, C_y_left_stacked, C_y_right_stacked,
                           N, use_lbfgs=False):

        if use_lbfgs:
            subkey1, subkey2 = jax.random.split(key, 2)
            x = jax.random.uniform(subkey1, (N,), minval=-1.0, maxval=1.0)
            y = jax.random.uniform(subkey2, (N,), minval=-1.0, maxval=1.0)
            new_key = key
        else:
            new_key, subkey1, subkey2 = jax.random.split(key, 3)
            x = jax.random.uniform(subkey1, (N,), minval=-1.0, maxval=1.0)
            y = jax.random.uniform(subkey2, (N,), minval=-1.0, maxval=1.0)
        
        def loss_all_modes(p_uv_stacked, l_m):
            def _loss_single(p_uv, mi, tl, tr, yl, yr, un, Cyl, Cyr):
                return loss_fn(p_uv, l_m, x, y, f_val, mi,
                               tl, tr, yl, yr, un, current_weights, k0, beta_n, Cyl, Cyr,
                               is_warmup=False, use_healthy_guide=False)

            vmapped = jax.vmap(_loss_single, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0))
            losses, (pdes, bcs, datas) = vmapped(
                p_uv_stacked, mode_indices,
                targets_left, targets_right,
                y_bnds_left, y_bnds_right,
                u_norms, C_y_left_stacked, C_y_right_stacked)
            return jnp.mean(losses), (jnp.mean(pdes), jnp.mean(bcs), jnp.mean(datas))

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

        return params_uv_stacked, layers_m, opt_state_uv, opt_state_m, opt_state_lbfgs, loss, aux, new_key

    return train_step_inverse


def check_switch_criterion(loss_history, window=10, threshold=1e-3):
    """Returns True if the relative slope of the loss is below threshold."""
    if len(loss_history) < window:
        return False
    old_loss = loss_history[-window]
    new_loss = loss_history[-1]
    relative_slope = abs(new_loss - old_loss) / max(abs(old_loss), 1e-12)
    return relative_slope < threshold and jnp.mean(old_loss) < 1e-2

# ==============================================================================
# SECTION 8: MAIN TRAINING LOOP
# ==============================================================================

def train(params_uv, layers_m, N_adam, N_lbfgs,
          max_steps_adam_phase1, max_steps_lbfgs_phase1,
          max_steps_adam_phase2, max_steps_lbfgs_phase2,
          freqs, active_modes_per_freq, mode_data, eval_interval,
          switch_threshold, switch_window, key):
    """
    Main training loop with curriculum learning over frequencies.
    
    For each frequency:
      - Dynamically select which modes to use (from active_modes_per_freq)
      - Phase 1: Train each mode's u-network independently (forward problem)
      - Phase 2: Train all active mode u-networks + shared m jointly (inverse)
    
    The m-network carries over between frequencies (curriculum learning).
    """

    # Collect all modes that appear in any frequency
    all_used_modes = set()
    for modes in active_modes_per_freq.values():
        all_used_modes.update(modes)
    
    loss_forward = {mi: [] for mi in all_used_modes}
    loss_inverse = []

    weights_phase1 = weights_phase1_

    best_params_uv = dict(params_uv)  # shallow copy of the dict

    for i, freq in enumerate(freqs):
        freq = float(freq)
        freq_modes = active_modes_per_freq[freq]
        n_modes_used = len(freq_modes)

        print(f"\n{'='*60}")
        print(f"--- f = {freq} Hz | active modes = {freq_modes} ({n_modes_used} modes) ---")
        print(f"{'='*60}")

        k0 = jnp.float32(2 * jnp.pi * freq / c0)
        beta_n = jnp.sqrt(k0**2 - (n_modes * jnp.pi / H)**2 + 0j)

        use_healthy_guide_flag = (i == 0)

        # ============================================================
        # PHASE 1: Forward (train each mode's u-network independently)
        # ============================================================

        for mode_idx in freq_modes:
            fm_key = (freq, mode_idx)
            param_uv = params_uv[fm_key]
            best_params_uv[fm_key] = param_uv
            best_loss = float('inf')

            # Rescale sigma from previous frequency if available
            if i > 0:
                prev_freq = float(freqs[i-1])
                prev_modes = active_modes_per_freq[prev_freq]
                if mode_idx in prev_modes:
                    prev_key = (prev_freq, mode_idx)
                    scale = jnp.abs(best_params_uv[prev_key]["sigma"][1] / best_params_uv[prev_key]["sigma"][0])
                    sigma_x = param_uv["sigma"][0]
                    param_uv["sigma"] = param_uv["sigma"].at[1].set(scale * sigma_x)

            print(f"\n--- Phase 1: Mode {mode_idx} | Adam (max {max_steps_adam_phase1} steps) ---")

            # Build optimizer with separate lr for sigma
            cosine_p1 = optax.schedules.cosine_decay_schedule(
                init_value=lr_uv, decay_steps=max_steps_adam_phase1, alpha=0.1)
            cosine_sigma_p1 = optax.schedules.cosine_decay_schedule(
                init_value=lr_sigma, decay_steps=max_steps_adam_phase1, alpha=0.1)

            adam_base_p1 = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.scale_by_adam(),
                optax.scale_by_schedule(cosine_p1),
                optax.scale(-1.0))
            adam_sigma_p1 = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.scale_by_adam(),
                optax.scale_by_schedule(cosine_sigma_p1),
                optax.scale(-1.0))

            param_labels_p1 = jax.tree_util.tree_map(lambda _: 'base', param_uv)
            if "sigma" in param_labels_p1:
                param_labels_p1["sigma"] = jax.tree_util.tree_map(lambda _: 'sigma', param_uv["sigma"])

            adam_uv_p1 = optax.multi_transform(
                transforms={'base': adam_base_p1, 'sigma': adam_sigma_p1},
                param_labels=param_labels_p1)
            lbfgs_uv_p1 = optax.lbfgs()

            step_forward = make_train_step_forward(adam_uv_p1, lbfgs_uv_p1)
            opt_state_uv = adam_uv_p1.init(param_uv)

            md = mode_data[mode_idx]
            loss_history_p1 = []
            switched_to_lbfgs = False

            for step in range(max_steps_adam_phase1):
                param_uv, opt_state_uv, loss, aux, key = step_forward(
                    param_uv, layers_m, opt_state_uv, key, freq, mode_idx,
                    md['U_left_norm'][freq], md['U_right_norm'][freq],
                    md['Y_left'], md['Y_right'], md['U_norm'][freq], weights_phase1,
                    k0, beta_n, md['C_y_left'], md['C_y_right'],
                    N=N_adam, use_lbfgs=False, use_healthy_guide=use_healthy_guide_flag)

                if step % eval_interval == 0:
                    pde_loss, bc_loss, _ = aux
                    loss_forward[mode_idx].append((pde_loss, bc_loss))
                    print(f"[Adam M{mode_idx}] Step {step} | pde {pde_loss:.2e} | bc {bc_loss:.2e} | total {loss:.2e} | scales {param_uv['sigma']}")
                    
                    loss_history_p1.append(float(loss))
                    
                    if loss < best_loss:
                        best_loss = loss
                        best_params_uv[fm_key] = param_uv

                    if check_switch_criterion(loss_history_p1, window=switch_window, threshold=switch_threshold):
                        print(f"[Adam M{mode_idx}] Convergence criterion met at step {step}. Switching to L-BFGS.")
                        switched_to_lbfgs = True
                        param_uv = best_params_uv[fm_key]
                        break

            # --- Phase 1: L-BFGS ---
            if switched_to_lbfgs or max_steps_lbfgs_phase1 > 0:
                print(f"\n--- Phase 1: Mode {mode_idx} | L-BFGS (max {max_steps_lbfgs_phase1} steps) ---")
                opt_state_uv = lbfgs_uv_p1.init(param_uv)
                key, lbfgs_key = jax.random.split(key)

                for step in range(max_steps_lbfgs_phase1):
                    param_uv, opt_state_uv, loss, aux, _ = step_forward(
                        param_uv, layers_m, opt_state_uv, lbfgs_key, freq, mode_idx,
                        md['U_left_norm'][freq], md['U_right_norm'][freq],
                        md['Y_left'], md['Y_right'], md['U_norm'][freq], weights_phase1,
                        k0, beta_n, md['C_y_left'], md['C_y_right'],
                        N=N_lbfgs, use_lbfgs=True, use_healthy_guide=use_healthy_guide_flag)

                    if step % eval_interval == 0:
                        pde_loss, bc_loss, _ = aux
                        loss_forward[mode_idx].append((pde_loss, bc_loss))
                        print(f"[LBFGS M{mode_idx}] Step {step} | pde {pde_loss:.2e} | bc {bc_loss:.2e} | total {loss:.2e} | scales {param_uv['sigma']}")
                        if loss < best_loss:
                            best_loss = loss
                            best_params_uv[fm_key] = param_uv

            # Store the best u-params for this (freq, mode)
            params_uv[fm_key] = best_params_uv[fm_key]

        # ============================================================
        # PHASE 2: Inverse (all modes' u-networks + shared m jointly)
        # ============================================================

        best_loss = float('inf')
        best_params_m = layers_m
        
        if max_steps_adam_phase2 > 0 or max_steps_lbfgs_phase2 > 0:

            print(f"\n--- Phase 2: Multi-mode inverse | Adam (max {max_steps_adam_phase2} steps) ---")

            # Build u-params as a list and stack for vmap
            param_uv_list = [params_uv[(freq, mi)] for mi in freq_modes]
            params_uv_stacked = jax.tree_util.tree_map(lambda *args: jnp.stack(args, axis=0), *param_uv_list)

            # Build optimizer for u-networks
            cosine_uv_p2 = optax.schedules.cosine_decay_schedule(
                init_value=lr_uv, decay_steps=100000, alpha=0.1)
            cosine_sigma_p2 = optax.schedules.cosine_decay_schedule(
                init_value=lr_sigma, decay_steps=60000, alpha=0.001)
            cosine_m_p2 = optax.schedules.cosine_decay_schedule(
                init_value=lr_m, decay_steps=100000, alpha=0.1)

            adam_base_p2 = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.scale_by_adam(),
                optax.scale_by_schedule(cosine_uv_p2),
                optax.scale(-1.0))
            adam_sigma_p2 = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.scale_by_adam(),
                optax.scale_by_schedule(cosine_sigma_p2),
                optax.scale(-1.0))

            # Use the first mode's param structure for label template
            param_labels_p2 = jax.tree_util.tree_map(lambda _: 'base', param_uv_list[0])
            if "sigma" in param_labels_p2:
                param_labels_p2["sigma"] = jax.tree_util.tree_map(lambda _: 'sigma', param_uv_list[0]["sigma"])

            adam_uv_p2 = optax.multi_transform(
                transforms={'base': adam_base_p2, 'sigma': adam_sigma_p2},
                param_labels=param_labels_p2)

            adam_m_p2 = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.scale_by_adam(),
                optax.scale_by_schedule(cosine_m_p2),
                optax.scale(-1.0))
            
            lbfgs_packed_p2 = optax.lbfgs()

            step_inverse = make_train_step_inverse_multimode(
                adam_uv_p2, adam_m_p2, lbfgs_packed_p2)

            # Initialize optimizer states
            opt_state_uv = adam_uv_p2.init(params_uv_stacked)
            opt_state_m = adam_m_p2.init(layers_m)
            opt_state_lbfgs = None

            # Prepare data arrays as stacked tensors (batch dimension for modes)
            mode_indices_arr = jnp.array([mi for mi in freq_modes], dtype=jnp.int32)
            targets_left = jnp.stack([mode_data[mi]['U_left_norm'][freq] for mi in freq_modes])
            targets_right = jnp.stack([mode_data[mi]['U_right_norm'][freq] for mi in freq_modes])
            y_bnds_left = jnp.stack([mode_data[mi]['Y_left'] for mi in freq_modes])
            y_bnds_right = jnp.stack([mode_data[mi]['Y_right'] for mi in freq_modes])
            u_norms = jnp.array([mode_data[mi]['U_norm'][freq] for mi in freq_modes])
            C_y_left_stacked = jnp.stack([mode_data[mi]['C_y_left'] for mi in freq_modes])
            C_y_right_stacked = jnp.stack([mode_data[mi]['C_y_right'] for mi in freq_modes])

            data_weight_schedule = optax.linear_schedule(
                init_value=0.1, end_value=10.0,
                transition_steps=max(10000, 1))

            loss_history_p2 = []
            switched_to_lbfgs_p2 = False

            for step in range(max_steps_adam_phase2):
                weights_phase2 = jnp.array([1.0, 1.0, data_weight_schedule(step)])

                params_uv_stacked, layers_m, opt_state_uv, opt_state_m, opt_state_lbfgs, loss, aux, key = step_inverse(
                    params_uv_stacked, layers_m, opt_state_uv, opt_state_m, opt_state_lbfgs,
                    key, freq, mode_indices_arr,
                    targets_left, targets_right, y_bnds_left, y_bnds_right, u_norms,
                    weights_phase2, k0, beta_n, C_y_left_stacked, C_y_right_stacked,
                    N=N_adam, use_lbfgs=False)

                if step % eval_interval == 0:
                    pde_loss, bc_loss, data_loss = aux
                    loss_inverse.append((pde_loss, bc_loss, data_loss))
                    # Quick unstack of sigma for printing
                    sigmas = []
                    for idx, mi in enumerate(freq_modes):
                        sigma_val = params_uv_stacked['sigma'][idx]
                        sigmas.append(f"M{mi} σ={sigma_val}")
                    sigmas_str = " | ".join(sigmas)
                    print(f"[Adam] Step {step} | pde {pde_loss:.2e} | bc {bc_loss:.2e} | data {data_loss:.2e} | total {loss:.2e} | {sigmas_str}")
                    
                    loss_history_p2.append(float(loss))
                    
                    if loss < best_loss:
                        best_loss = loss
                        best_params_uv_stacked = jax.tree_util.tree_map(jnp.copy, params_uv_stacked)
                        best_params_m = layers_m

                    if check_switch_criterion(loss_history_p2, window=switch_window, threshold=switch_threshold):
                        print(f"[Adam] Convergence criterion met at step {step}. Switching to L-BFGS.")
                        switched_to_lbfgs_p2 = True
                        params_uv_stacked = best_params_uv_stacked
                        layers_m = best_params_m
                        break

            # --- Phase 2: L-BFGS ---
            if switched_to_lbfgs_p2 or max_steps_lbfgs_phase2 > 0:
                print(f"\n--- Phase 2: Multi-mode inverse | L-BFGS (max {max_steps_lbfgs_phase2} steps) ---")
                opt_state_lbfgs = lbfgs_packed_p2.init({'uv_list': params_uv_stacked, 'm': layers_m})
                weights_phase2_lbfgs = jnp.array([1.0, 1.0, 1.0])

                key, lbfgs_key = jax.random.split(key)

                for step in range(max_steps_lbfgs_phase2):
                    params_uv_stacked, layers_m, opt_state_uv, opt_state_m, opt_state_lbfgs, loss, aux, _ = step_inverse(
                        params_uv_stacked, layers_m, opt_state_uv, opt_state_m, opt_state_lbfgs,
                        lbfgs_key, freq, mode_indices_arr,
                        targets_left, targets_right, y_bnds_left, y_bnds_right, u_norms,
                        weights_phase2_lbfgs, k0, beta_n, C_y_left_stacked, C_y_right_stacked,
                        N=N_lbfgs, use_lbfgs=True)

                    if step % eval_interval == 0:
                        pde_loss, bc_loss, data_loss = aux
                        loss_inverse.append((pde_loss, bc_loss, data_loss))
                        print(f"[LBFGS] Step {step} | pde {pde_loss:.2e} | bc {bc_loss:.2e} | data {data_loss:.2e} | total {loss:.2e}")
                        if loss < best_loss:
                            best_loss = loss
                            best_params_uv_stacked = jax.tree_util.tree_map(jnp.copy, params_uv_stacked)
                            best_params_m = layers_m

            # Unstack and store the best parameters back into params_uv
            for idx, mi in enumerate(freq_modes):
                param_uv_i = jax.tree_util.tree_map(lambda x: x[idx], best_params_uv_stacked)
                params_uv[(freq, mi)] = param_uv_i
            layers_m = best_params_m

    return best_params_uv, layers_m, key, loss_forward, loss_inverse, active_modes_per_freq

# ==============================================================================
# SECTION 9: MODEL TRAINING & POST-PROCESSING
# ==============================================================================

# --- Run training ---
key, subkey = jax.random.split(key)
params_uv, layers_m, key, loss_forward, loss_inverse, active_modes_per_freq = train(
    params_uv, layers_m, N_adam, N_lbfgs,
    max_steps_adam_phase1, max_steps_lbfgs_phase1,
    max_steps_adam_phase2, max_steps_lbfgs_phase2,
    training_frequencies, active_modes_per_freq, mode_data,
    eval_interval, switch_threshold, switch_window, key
)

# Create figures directory if it doesn't exist
os.makedirs(os.path.join(script_dir, 'fig'), exist_ok=True)

# --- Plot wavefield predictions (Re(u) PINN) for each (freq, mode) ---
for f in training_frequencies:
    f = float(f)
    freq_modes = active_modes_per_freq[f]
    for mode_idx in freq_modes:
        fm_key = (f, mode_idx)
        p_uv = params_uv[fm_key]
        u_norm_val = mode_data[mode_idx]['U_norm'][f]

        def u_real(x, y, _p=p_uv, _u_norm=u_norm_val):
            features = gamma(x, y, _p['sigma'])
            return forward_func(_p['layers'], features)[0] * _u_norm

        x_val = np.linspace(-1, 1, 500)
        y_val = np.linspace(0, 0.6, 150)

        u_grid = jax.vmap(jax.vmap(u_real, in_axes=(0, None)), in_axes=(None, 0))(x_val, 2 * y_val / H - 1)

        fig, ax = plt.subplots(figsize=(8, 4))
        pcm = ax.pcolormesh(x_val, y_val, u_grid, cmap='RdBu_r', rasterized=True)
        ax.set_title(f'Re(u) PINN at {f} Hz — Mode {mode_idx}')
        fig.colorbar(pcm, ax=ax, shrink=0.8, location='left')
        plt.tight_layout()
        if SHOW_PLOTS:
            plt.show()
        plt.close()

# --- Plot Phase 1 (Forward) Training Losses per mode ---
for mode_idx in sorted(loss_forward.keys()):
    if len(loss_forward[mode_idx]) == 0:
        continue
    loss_fwd_arr = np.array(loss_forward[mode_idx])
    pde_losses = loss_fwd_arr[:, 0]
    bc_losses = loss_fwd_arr[:, 1]
    total_losses = pde_losses + bc_losses

    x_axis = eval_interval * np.arange(len(loss_fwd_arr))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.semilogy(x_axis, pde_losses, label="PDE Loss")
    ax1.semilogy(x_axis, bc_losses, label="BC Loss")
    ax1.set_title(f"Phase 1 Mode {mode_idx}: Partial Loss")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    ax2.semilogy(x_axis, total_losses, label="Total Loss", color='black')
    ax2.set_title(f"Phase 1 Mode {mode_idx}: Total Loss")
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
    total_losses_inv = pde_losses_inv + bc_losses_inv + data_losses_inv

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

    ax2.semilogy(x_axis_inv, total_losses_inv, label="Total Loss", color='black')
    ax2.set_title("Phase 2 (Multi-mode): Total Loss")
    ax2.set_xlabel("Steps")
    ax2.legend(loc="upper right")
    ax2.grid(True)
    plt.tight_layout()
    all_used = sorted(set(mi for modes in active_modes_per_freq.values() for mi in modes))
    modes_str = '_'.join([str(mi) for mi in all_used])
    freqs_str = '_'.join([str(int(f)) for f in training_frequencies])
    plt.savefig(os.path.join(script_dir, 'fig', f'2D_WG_losses_modes{modes_str}_freqs{freqs_str}.pdf'))
    if SHOW_PLOTS:
        plt.show()
    plt.close()

# --- Plot final reconstructed sound speed field c(x, y) ---
def c_final(x, y):
    return 1 / jnp.sqrt(forward_params(layers_m, jnp.array([x, y])))

x_c = np.linspace(-1, 1, 100)
y_c = np.linspace(0, 0.6, 50)
c_grid_final = jax.vmap(jax.vmap(c_final, in_axes=(0, None)), in_axes=(None, 0))(x_c, 2 * y_c / H - 1)

plt.figure(figsize=(7, 3.5))
plt.pcolormesh(x_c, y_c, c_grid_final, rasterized=True)
plt.colorbar(label="Sound speed c(x, y)")
all_used_modes_final = sorted(set(mi for modes in active_modes_per_freq.values() for mi in modes))
modes_str = ', '.join([str(mi) for mi in all_used_modes_final])
freqs_str = ', '.join([str(int(f)) for f in training_frequencies])
modes_per_freq_str = ' | '.join([f"{int(f)}Hz:M{active_modes_per_freq[float(f)]}" for f in training_frequencies])
plt.title(f"Reconstructed c(x,y) — {modes_per_freq_str}")
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'fig',
    f'2D_WG_c_map_modes{"_".join([str(mi) for mi in all_used_modes_final])}_freqs{"_".join([str(int(f)) for f in training_frequencies])}.pdf'))
if SHOW_PLOTS:
    plt.show()
plt.close()