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
from tools.uv_checkpoint import collect_u_norms, save_uv_checkpoint

# ==============================================================================
# SECTION 1: CONFIGURATION & HYPERPARAMETERS
# ==============================================================================

# --- Domain Geometry ---
H = 0.6  # Height of the waveguide
L = 1.0  # Half-length of the waveguide

# --- Physics Parameters ---
c0 = 340.0
contrast_max = 0.4
cmin = c0 * (1-contrast_max)
cmax = c0 * (1+0.01)
m0 = 1/c0**2
m_min = 1 / cmax**2
m_max = 1 / cmin**2

# --- Data Configuration ---
# Data files follow the naming convention:
#   pinn_boundary_{left/right}_{defect_name}_ratio{c_defect/c0}.csv
script_dir = os.path.dirname(os.path.abspath(__file__))
defect_name = 'circlebottomright'
contrast_ratio = 0.8
data_ratio_label = format_ratio_label(contrast_ratio)

# Frequencies to run training on (curriculum learning: low to high)
training_frequencies = np.array([1200.0])

# Active modes per frequency: edit these to select which modes to use
active_modes_per_freq = {
    1200.0: [0, 1, 2, 3, 4]
}

# --- Random Seed ---
random_seed = 0
key = jax.random.key(random_seed)

# --- Neural Network Architectures ---
m_fourier_features = 64  # Dimensionality of Fourier feature mapping
n_input = 2
n_layers_uv = [2 * m_fourier_features, 128, 128, 64, 2]
n_layers_m = [n_input, 128, 64, 1]

# --- Gauss-Legendre Quadrature ---
n_gauss_legendre = 40

# --- Optimizer Learning Rates ---
lr_uv = 1e-3
lr_m = 3e-4
lr_sigma = 1e-2

# --- Training Hyperparameters ---
# Collocation sizes are ordered as (PDE interior, horizontal Neumann, vertical DtN).
# Keeping these tuples fixed avoids recompilation while Adam still resamples the points.
N_adam = (4096, 256, 64)
N_lbfgs = (2048, 256, 64)
N_validation = (4096, 256, 128)
eval_interval = 200      # Interval for evaluating losses and printing progress
switch_threshold = 0.0   # Threshold for L-BFGS switch criterion
switch_window = 10       # Window size for switch criterion

max_steps_adam_phase1 = 40001
max_steps_lbfgs_phase1 = 3001
max_steps_adam_phase2 = 100001
max_steps_lbfgs_phase2 = 3001

SHOW_PLOTS = True

weights_phase1 = jnp.array([1.0, 10.0, 0.0])
weights_phase2_final = jnp.array([1.0, 1.0, 10.0])

# ==============================================================================
# SECTION 2: DATA DISCOVERY
# ==============================================================================

def boundary_data_paths(script_dir, defect_name, contrast_ratio):
    ratio_label = format_ratio_label(contrast_ratio)
    suffix = f'{defect_name}_{ratio_label}.csv'
    left_path = os.path.join(script_dir, 'data', f'pinn_boundary_left_{suffix}')
    right_path = os.path.join(script_dir, 'data', f'pinn_boundary_right_{suffix}')
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

def loss_fn(params_uv, layers_m, x_pde, y_pde, x_neumann, y_dtn,
            f, mode_index, target_u_left, target_u_right, y_bnd_left,
            y_bnd_right, u_norm_val, weights, beta_n,
            is_warmup=False, use_healthy_guide=False):
    
    def uv(x, y):
        features = compute_features(x, y, params_uv['sigma'])
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
    
    def compute_dtn_loss(x_bnd, y_eval, sign, A_inc=None):
        uv_quad = jax.vmap(uv, in_axes=(None, 0))(x_bnd, y_gauss_legendre)
        U_quad_complex = uv_quad[:, 0] + 1j * uv_quad[:, 1]
        
        u_n = C_quad @ (w_quad * U_quad_complex)
        
        dtn_n = sign * 1j * beta_n * u_n
        if A_inc is not None:
            dtn_n = dtn_n + 2j * beta_n * A_inc
        
        def DtN_eval(y_val):
            C_y = a_n * jnp.cos(n_modes * jnp.pi * (y_val + 1.0) / 2.0)
            return jnp.dot(C_y, dtn_n)
            
        dtn_pred_complex = jax.vmap(DtN_eval)(y_eval)
        
        dtn_actual = jax.vmap(uv_x, in_axes=(None, 0))(x_bnd, y_eval)
        dtn_actual_complex = dtn_actual[:, 0] + 1j * dtn_actual[:, 1]
        
        return jnp.mean(jnp.abs(dtn_actual_complex - dtn_pred_complex)**2)

    # Incident field: mode `mode_index` propagating from the left to the right
    A_inc_left = jnp.zeros(N_modes, dtype=jnp.complex64)
    amplitude_incidente = jnp.exp(-1j * beta_n[mode_index] * L) / u_norm_val
    A_inc_left = A_inc_left.at[mode_index].set(amplitude_incidente)
    
    # PDE Residual Loss
    vmap_pde_residual = jax.vmap(pde_residual, in_axes=(0, 0))
    pde_loss = jnp.mean(vmap_pde_residual(x_pde, y_pde)**2)
    
    # Boundary Conditions Loss
    dtn_loss_left = compute_dtn_loss(
        -1.0, y_dtn, sign=-1, A_inc=A_inc_left)
    dtn_loss_right = compute_dtn_loss(
        1.0, y_dtn, sign=1, A_inc=None)
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
        data_loss = jnp.mean((vmap_uv_y(-1.0, y_bnd_left) - target_u_left)**2) + jnp.mean((vmap_uv_y(1.0, y_bnd_right) - target_u_right)**2)

    total_loss = weights[0] * pde_loss + weights[1] * bc_loss + weights[2] * data_loss
    
    return total_loss, (pde_loss, bc_loss, data_loss)


def multimode_loss_fn(params_uv_stacked, layers_m, x_pde, y_pde,
                      x_neumann, y_dtn, f_val, mode_indices, targets_left,
                      targets_right, y_bnds_left, y_bnds_right, u_norms,
                      weights, beta_n):
    """Evaluate the mean weighted loss over all active incident modes."""
    def _loss_single(p_uv, mi, tl, tr, yl, yr, un):
        return loss_fn(
            p_uv, layers_m, x_pde, y_pde, x_neumann, y_dtn, f_val, mi,
            tl, tr, yl, yr, un, weights, beta_n,
            is_warmup=False, use_healthy_guide=False,
        )

    vmapped = jax.vmap(_loss_single, in_axes=(0, 0, 0, 0, 0, 0, 0))
    losses, (pdes, bcs, datas) = vmapped(
        params_uv_stacked, mode_indices,
        targets_left, targets_right,
        y_bnds_left, y_bnds_right,
        u_norms,
    )
    return jnp.mean(losses), (jnp.mean(pdes), jnp.mean(bcs), jnp.mean(datas))


@functools.partial(jax.jit, static_argnames=('use_healthy_guide',))
def evaluate_forward_loss(params_uv, layers_m, x_pde, y_pde, x_neumann,
                          y_dtn, f_val, mode_index, target_left, target_right,
                          y_bnd_left, y_bnd_right, u_norm, weights, beta_n,
                          use_healthy_guide):
    """Evaluate phase 1 parameters without applying an optimizer update."""
    return loss_fn(
        params_uv, layers_m, x_pde, y_pde, x_neumann, y_dtn,
        f_val, mode_index,
        target_left, target_right, y_bnd_left, y_bnd_right,
        u_norm, weights, beta_n,
        is_warmup=True, use_healthy_guide=use_healthy_guide,
    )


@jax.jit
def evaluate_inverse_loss(params_uv_stacked, layers_m, x_pde, y_pde,
                          x_neumann, y_dtn, f_val, mode_indices, targets_left,
                          targets_right, y_bnds_left, y_bnds_right, u_norms,
                          weights, beta_n):
    """Evaluate phase 2 parameters without applying an optimizer update."""
    return multimode_loss_fn(
        params_uv_stacked, layers_m, x_pde, y_pde, x_neumann, y_dtn,
        f_val, mode_indices,
        targets_left, targets_right, y_bnds_left, y_bnds_right,
        u_norms, weights, beta_n,
    )

# ==============================================================================
# SECTION 8: OPTIMIZATION SCHEMES & TRAIN STEPS (JIT COMPILED)
# ==============================================================================

def make_train_step_forward(adam_opt, lbfgs_opt):
    """Train step for Phase 1: only update one u-network, m is frozen."""

    @functools.partial(jax.jit, static_argnames=('N', 'use_lbfgs', 'use_healthy_guide'))
    def train_step_forward(params_uv, layers_m, opt_state_uv, key, f_val, mode_index,
                           target_left, target_right, y_bnd_left, y_bnd_right, u_norm,
                           current_weights, beta_n,
                           N, use_lbfgs=False, use_healthy_guide=True):
        x_pde, y_pde, x_neumann, y_dtn = sample_collocation_points(key, N)

        def loss_uv(p_uv):
            return loss_fn(
                p_uv, layers_m,
                x_pde, y_pde, x_neumann, y_dtn,
                f_val, mode_index,
                target_left, target_right,
                y_bnd_left, y_bnd_right, u_norm, current_weights,
                beta_n,
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
                           opt_state_lbfgs, key, f_val, mode_indices,
                           targets_left, targets_right,
                           y_bnds_left, y_bnds_right, u_norms,
                           current_weights, beta_n,
                           N, use_lbfgs=False):

        x_pde, y_pde, x_neumann, y_dtn = sample_collocation_points(key, N)
        
        def loss_all_modes(p_uv_stacked, l_m):
            return multimode_loss_fn(
                p_uv_stacked, l_m,
                x_pde, y_pde, x_neumann, y_dtn,
                f_val, mode_indices,
                targets_left, targets_right, y_bnds_left, y_bnds_right,
                u_norms, current_weights, beta_n,
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
    best_validation_losses = {'forward': [], 'inverse': []}

    # This deterministic grid is used only for convergence checks and best-model
    # selection. It is denser than, and therefore distinct from, the L-BFGS grid.
    validation_points = regular_collocation_points(N_validation)

    best_params_uv = dict(params_uv)  # shallow copy of the dict
    best_params_each_m = []

    # ------------------------------------------------------------------
    # Create optimizers and JIT-compiled step functions ONCE to avoid
    # recompilation at each mode/frequency.
    # Only the optimizer STATE is re-initialized inside the loop.
    # ------------------------------------------------------------------

    # Phase 1 optimizer (u-network only, m frozen)
    template_param_uv = next(iter(params_uv.values()))
    param_labels_p1 = jax.tree_util.tree_map(lambda _: 'base', template_param_uv)
    if "sigma" in param_labels_p1:
        param_labels_p1["sigma"] = jax.tree_util.tree_map(lambda _: 'sigma', template_param_uv["sigma"])

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

    adam_uv_p1 = optax.multi_transform(
        transforms={'base': adam_base_p1, 'sigma': adam_sigma_p1},
        param_labels=param_labels_p1)
    lbfgs_uv_p1 = optax.lbfgs()

    step_forward = make_train_step_forward(adam_uv_p1, lbfgs_uv_p1)

    # Phase 2 optimizers (u-networks + m jointly)
    cosine_uv_p2 = optax.schedules.cosine_decay_schedule(
        init_value=lr_uv, decay_steps=max_steps_adam_phase2, alpha=0.1)
    cosine_sigma_p2 = optax.schedules.cosine_decay_schedule(
        init_value=lr_sigma, decay_steps=0.2*max_steps_adam_phase2, alpha=0.001)
    cosine_m_p2 = optax.schedules.cosine_decay_schedule(
        init_value=lr_m, decay_steps=max_steps_adam_phase2, alpha=0.1)

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

    param_labels_p2 = jax.tree_util.tree_map(lambda _: 'base', template_param_uv)
    if "sigma" in param_labels_p2:
        param_labels_p2["sigma"] = jax.tree_util.tree_map(lambda _: 'sigma', template_param_uv["sigma"])

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

    # ------------------------------------------------------------------
    # Curriculum training loop over frequencies
    # ------------------------------------------------------------------

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
            best_forward_summary = None

            print(f"\n--- Phase 1: Mode {mode_idx} | Adam (max {max_steps_adam_phase1} steps) ---")

            # Re-initialize optimizer state (resets cosine schedule counter)
            opt_state_uv = adam_uv_p1.init(param_uv)

            md = mode_data[mode_idx]
            loss_history_p1 = []
            switched_to_lbfgs = False

            for step in range(max_steps_adam_phase1):
                key, subkey = jax.random.split(key)
                param_uv, opt_state_uv, _, _ = step_forward(
                    param_uv, layers_m, opt_state_uv, subkey, freq, mode_idx,
                    md['U_left_norm'][freq], md['U_right_norm'][freq],
                    md['Y_left'][freq], md['Y_right'][freq], md['U_norm'][freq], weights_phase1,
                    beta_n,
                    N=N_adam, use_lbfgs=False, use_healthy_guide=use_healthy_guide_flag)

                if step % eval_interval == 0:
                    loss, validation_aux = evaluate_forward_loss(
                        param_uv, layers_m, *validation_points,
                        freq, mode_idx,
                        md['U_left_norm'][freq], md['U_right_norm'][freq],
                        md['Y_left'][freq], md['Y_right'][freq], md['U_norm'][freq],
                        weights_phase1, beta_n, use_healthy_guide_flag,
                    )
                    pde_loss, bc_loss, _ = validation_aux
                    loss_value = float(loss)
                    loss_forward[mode_idx].append((pde_loss, bc_loss))
                    print(f"[Adam M{mode_idx}] Step {step} | val pde {pde_loss:.2e} | val bc {bc_loss:.2e} | weighted val total {loss:.2e} | scales {param_uv['sigma']}")
                    
                    loss_history_p1.append(loss_value)
                    
                    if loss_value < best_loss:
                        best_loss = loss_value
                        best_params_uv[fm_key] = param_uv
                        best_forward_summary = {
                            'frequency': freq,
                            'mode': int(mode_idx),
                            'optimizer': 'adam',
                            'step': int(step),
                            'pde': float(pde_loss),
                            'bc': float(bc_loss),
                            'data': 0.0,
                            'weighted_total': loss_value,
                            'weights': [float(value) for value in weights_phase1],
                        }

                    if check_switch_criterion(loss_history_p1, window=switch_window, threshold=switch_threshold):
                        print(f"[Adam M{mode_idx}] Convergence criterion met at step {step}. Switching to L-BFGS.")
                        switched_to_lbfgs = True
                        param_uv = best_params_uv[fm_key]
                        break

            # --- Phase 1: L-BFGS ---
            if switched_to_lbfgs or max_steps_lbfgs_phase1 > 0:
                print(f"\n--- Phase 1: Mode {mode_idx} | L-BFGS (max {max_steps_lbfgs_phase1} steps) ---")
                opt_state_uv = lbfgs_uv_p1.init(param_uv)

                for step in range(max_steps_lbfgs_phase1):
                    param_uv, opt_state_uv, _, _ = step_forward(
                        param_uv, layers_m, opt_state_uv, None, freq, mode_idx,
                        md['U_left_norm'][freq], md['U_right_norm'][freq],
                        md['Y_left'][freq], md['Y_right'][freq], md['U_norm'][freq], weights_phase1,
                        beta_n,
                        N=N_lbfgs, use_lbfgs=True, use_healthy_guide=use_healthy_guide_flag)

                    if step % eval_interval == 0:
                        loss, validation_aux = evaluate_forward_loss(
                            param_uv, layers_m, *validation_points,
                            freq, mode_idx,
                            md['U_left_norm'][freq], md['U_right_norm'][freq],
                            md['Y_left'][freq], md['Y_right'][freq], md['U_norm'][freq],
                            weights_phase1, beta_n, use_healthy_guide_flag,
                        )
                        pde_loss, bc_loss, _ = validation_aux
                        loss_value = float(loss)
                        loss_forward[mode_idx].append((pde_loss, bc_loss))
                        print(f"[LBFGS M{mode_idx}] Step {step} | val pde {pde_loss:.2e} | val bc {bc_loss:.2e} | weighted val total {loss:.2e} | scales {param_uv['sigma']}")
                        if loss_value < best_loss:
                            best_loss = loss_value
                            best_params_uv[fm_key] = param_uv
                            best_forward_summary = {
                                'frequency': freq,
                                'mode': int(mode_idx),
                                'optimizer': 'lbfgs',
                                'step': int(step),
                                'pde': float(pde_loss),
                                'bc': float(bc_loss),
                                'data': 0.0,
                                'weighted_total': loss_value,
                                'weights': [float(value) for value in weights_phase1],
                            }

            # Store the best u-params for this (freq, mode)
            if best_forward_summary is None:
                raise RuntimeError(
                    f'No finite validation loss for forward f={freq}, mode={mode_idx}'
                )
            params_uv[fm_key] = best_params_uv[fm_key]
            best_validation_losses['forward'].append(best_forward_summary)

        # ============================================================
        # PHASE 2: Inverse (all modes' u-networks + shared m jointly)
        # ============================================================

        best_loss = float('inf')
        best_params_m = layers_m
        best_inverse_summary = None
        
        if max_steps_adam_phase2 > 0 or max_steps_lbfgs_phase2 > 0:

            print(f"\n--- Phase 2: Multi-mode inverse | Adam (max {max_steps_adam_phase2} steps) ---")

            # Build u-params as a list and stack for vmap
            param_uv_list = [params_uv[(freq, mi)] for mi in freq_modes]
            params_uv_stacked = jax.tree_util.tree_map(lambda *args: jnp.stack(args, axis=0), *param_uv_list)

            # Initialize best to current values (avoids NameError if first loss is NaN)
            best_params_uv_stacked = jax.tree_util.tree_map(jnp.copy, params_uv_stacked)

            # Re-initialize optimizer states (resets schedule counters)
            opt_state_uv = adam_uv_p2.init(params_uv_stacked)
            opt_state_m = adam_m_p2.init(layers_m)
            opt_state_lbfgs = None

            # Prepare data arrays as stacked tensors (batch dimension for modes)
            mode_indices_arr = jnp.array([mi for mi in freq_modes], dtype=jnp.int32)
            targets_left = jnp.stack([mode_data[mi]['U_left_norm'][freq] for mi in freq_modes])
            targets_right = jnp.stack([mode_data[mi]['U_right_norm'][freq] for mi in freq_modes])
            y_bnds_left = jnp.stack([mode_data[mi]['Y_left'][freq] for mi in freq_modes])
            y_bnds_right = jnp.stack([mode_data[mi]['Y_right'][freq] for mi in freq_modes])
            u_norms = jnp.array([mode_data[mi]['U_norm'][freq] for mi in freq_modes])

            data_weight_schedule = optax.linear_schedule(
                init_value=0.1, end_value=10.0,
                transition_steps=max(0.2*max_steps_adam_phase2, 1))

            loss_history_p2 = []
            switched_to_lbfgs_p2 = False

            for step in range(max_steps_adam_phase2):
                weights_phase2 = jnp.array([1.0, 1.0, data_weight_schedule(step)])

                key, subkey = jax.random.split(key)
                params_uv_stacked, layers_m, opt_state_uv, opt_state_m, opt_state_lbfgs, _, _ = step_inverse(
                    params_uv_stacked, layers_m, opt_state_uv, opt_state_m, opt_state_lbfgs,
                    subkey, freq, mode_indices_arr,
                    targets_left, targets_right, y_bnds_left, y_bnds_right, u_norms,
                    weights_phase2, beta_n,
                    N=N_adam, use_lbfgs=False)

                if step % eval_interval == 0:
                    loss, validation_aux = evaluate_inverse_loss(
                        params_uv_stacked, layers_m, *validation_points,
                        freq, mode_indices_arr,
                        targets_left, targets_right, y_bnds_left, y_bnds_right,
                        u_norms, weights_phase2_final, beta_n,
                    )
                    pde_loss, bc_loss, data_loss = validation_aux
                    loss_value = float(loss)
                    loss_inverse.append((pde_loss, bc_loss, data_loss))
                    # Quick unstack of sigma for printing
                    sigmas = []
                    for idx, mi in enumerate(freq_modes):
                        sigma_val = params_uv_stacked['sigma'][idx]
                        sigmas.append(f"M{mi} σ={sigma_val}")
                    sigmas_str = " | ".join(sigmas)
                    print(f"[Adam] Step {step} | val pde {pde_loss:.2e} | val bc {bc_loss:.2e} | val data {data_loss:.2e} | weighted val total {loss:.2e} | train data weight {weights_phase2[2]:.2e} | {sigmas_str}")
                    
                    loss_history_p2.append(loss_value)
                    
                    if loss_value < best_loss:
                        best_loss = loss_value
                        best_params_uv_stacked = jax.tree_util.tree_map(jnp.copy, params_uv_stacked)
                        best_params_m = layers_m
                        best_inverse_summary = {
                            'frequency': freq,
                            'modes': [int(mode) for mode in freq_modes],
                            'optimizer': 'adam',
                            'step': int(step),
                            'pde': float(pde_loss),
                            'bc': float(bc_loss),
                            'data': float(data_loss),
                            'weighted_total': loss_value,
                            'weights': [float(value) for value in weights_phase2_final],
                        }

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

                for step in range(max_steps_lbfgs_phase2):
                    params_uv_stacked, layers_m, opt_state_uv, opt_state_m, opt_state_lbfgs, _, _ = step_inverse(
                        params_uv_stacked, layers_m, opt_state_uv, opt_state_m, opt_state_lbfgs,
                        None, freq, mode_indices_arr,
                        targets_left, targets_right, y_bnds_left, y_bnds_right, u_norms,
                        weights_phase2_final, beta_n,
                        N=N_lbfgs, use_lbfgs=True)

                    if step % eval_interval == 0:
                        loss, validation_aux = evaluate_inverse_loss(
                            params_uv_stacked, layers_m, *validation_points,
                            freq, mode_indices_arr,
                            targets_left, targets_right, y_bnds_left, y_bnds_right,
                            u_norms, weights_phase2_final, beta_n,
                        )
                        pde_loss, bc_loss, data_loss = validation_aux
                        loss_value = float(loss)
                        loss_inverse.append((pde_loss, bc_loss, data_loss))
                        print(f"[LBFGS] Step {step} | val pde {pde_loss:.2e} | val bc {bc_loss:.2e} | val data {data_loss:.2e} | weighted val total {loss:.2e}")
                        if loss_value < best_loss:
                            best_loss = loss_value
                            best_params_uv_stacked = jax.tree_util.tree_map(jnp.copy, params_uv_stacked)
                            best_params_m = layers_m
                            best_inverse_summary = {
                                'frequency': freq,
                                'modes': [int(mode) for mode in freq_modes],
                                'optimizer': 'lbfgs',
                                'step': int(step),
                                'pde': float(pde_loss),
                                'bc': float(bc_loss),
                                'data': float(data_loss),
                                'weighted_total': loss_value,
                                'weights': [float(value) for value in weights_phase2_final],
                            }

            # Unstack and store the best parameters back into params_uv
            if best_inverse_summary is None:
                raise RuntimeError(
                    f'No finite validation loss for inverse f={freq}'
                )
            for idx, mi in enumerate(freq_modes):
                param_uv_i = jax.tree_util.tree_map(lambda x: x[idx], best_params_uv_stacked)
                params_uv[(freq, mi)] = param_uv_i
            layers_m = best_params_m
            best_params_each_m.append(jax.tree_util.tree_map(jnp.copy, layers_m))
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

    # Print JAX devices to verify setup
    print("JAX Devices:", jax.devices())
    print("Default JAX dtype:", jnp.ones(1).dtype)

    # ==================================================================
    # Data Loading & Preprocessing
    # ==================================================================
    left_path, right_path = boundary_data_paths(
        script_dir, defect_name, contrast_ratio
    )
    boundary_data = WaveguideBoundaryData(left_path, right_path)

    requested_frequencies = [float(frequency) for frequency in training_frequencies]
    missing_frequency_configs = [
        frequency for frequency in requested_frequencies
        if frequency not in active_modes_per_freq
    ]
    if missing_frequency_configs:
        raise ValueError(
            f'Missing active_modes_per_freq entries for {missing_frequency_configs}'
        )

    requested_pairs = [
        (mode_idx, frequency)
        for frequency in requested_frequencies
        for mode_idx in active_modes_per_freq[frequency]
    ]
    missing_pairs = [
        pair for pair in requested_pairs
        if not boundary_data.has_pair(pair[0], pair[1])
    ]
    if missing_pairs:
        raise ValueError(
            f'The combined FEM boundary files do not contain requested pairs: {missing_pairs}. '
            f'Available pairs: {boundary_data.available_pairs}'
        )

    mode_data = {}
    used_modes = sorted({mode_idx for mode_idx, _ in requested_pairs})

    for mode_idx in used_modes:
        mode_freqs = sorted({
            frequency for candidate_mode, frequency in requested_pairs
            if candidate_mode == mode_idx
        })
        Y_left = {}
        Y_right = {}
        U_norm = {}
        U_left_norm = {}
        U_right_norm = {}

        for f in mode_freqs:
            pair = boundary_data.get_pair(mode_idx, f)
            if not np.isclose(pair.x_left, -L, rtol=0.0, atol=1e-6) or not np.isclose(
                pair.x_right, L, rtol=0.0, atol=1e-6
            ):
                raise ValueError(
                    f'Boundary coordinates for mode {mode_idx}, frequency {f} are '
                    f'x_left={pair.x_left}, x_right={pair.x_right}; expected {-L} and {L}'
                )

            Y_left[f] = 2 * jnp.asarray(pair.y_left, dtype=jnp.float32) / H - 1
            Y_right[f] = 2 * jnp.asarray(pair.y_right, dtype=jnp.float32) / H - 1
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
                    f'Invalid zero or non-finite field norm for mode {mode_idx}, frequency {f}'
                )
            
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
        }
        print(
            f"  Mode {mode_idx}: {len(mode_freqs)} frequencies loaded "
            f"({mode_freqs[0]:.0f}–{mode_freqs[-1]:.0f} Hz)"
        )

    for frequency in requested_frequencies:
        modes = active_modes_per_freq[frequency]
        left_counts = {mode_data[mode]['Y_left'][frequency].shape[0] for mode in modes}
        right_counts = {mode_data[mode]['Y_right'][frequency].shape[0] for mode in modes}
        if len(left_counts) != 1 or len(right_counts) != 1:
            raise ValueError(
                f'All active modes at {frequency} Hz must use the same number of samples per side'
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

    # Wavefield networks: one per (frequency, mode)
    params_uv = {}
    for freq, modes in active_modes_per_freq.items():
        for mode_idx in modes:
            key, subkey_f = jax.random.split(key)
            params_uv[(freq, mode_idx)] = init_layers_uv(subkey_f, n_layers_uv, freq, mode_idx)

    # ==================================================================
    # Initial Sound Speed Plot
    # ==================================================================
    print("Plotting initial sound speed profile...")
    x_plot = np.linspace(-1, 1, 100)
    y_plot = np.linspace(0, 0.6, 50)
    c_grid = jax.vmap(
        jax.vmap(lambda x, y: c(x, y, layers_m), in_axes=(0, None)),
        in_axes=(None, 0)
    )(x_plot, 2 * y_plot / H - 1)

    os.makedirs(os.path.join(script_dir, 'fig'), exist_ok=True)
    plt.figure(figsize=(7, 3.5))
    plt.pcolormesh(x_plot, y_plot, c_grid, rasterized=True)
    plt.colorbar(label="Sound speed c(x, y)")
    plt.title("Initial Sound Speed Field")
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'fig', 'initial_sound_speed.pdf'))
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
        training_frequencies, active_modes_per_freq, mode_data,
        eval_interval, switch_threshold, switch_window, subkey
    )

    # ==================================================================
    # Post-processing Plots
    # ==================================================================

    # Precompute common strings for filenames (computed once)
    all_used_modes = sorted(set(mi for modes in active_modes_per_freq.values() for mi in modes))
    modes_str = '_'.join(str(mi) for mi in all_used_modes)
    freqs_str = '_'.join(str(int(f)) for f in training_frequencies)

    # Save every UV network together with the Fourier basis and physical scaling.
    checkpoint_dir = os.path.join(script_dir, 'checkpoints')
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f'checkpoint_{defect_name}_{data_ratio_label}_modes{modes_str}_freqs{freqs_str}.npz'
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
            'contrast_ratio': contrast_ratio,
            'ratio_label': data_ratio_label,
            'active_modes_per_freq': {
                str(frequency): list(modes)
                for frequency, modes in active_modes_per_freq.items()
            },
        },
    )
    print(f'UV/M checkpoint saved to: {checkpoint_path}')

    # --- Plot Phase 1 (Forward) Training Losses per mode ---
    for mode_idx in sorted(loss_forward.keys()):
        if len(loss_forward[mode_idx]) == 0:
            continue
        loss_fwd_arr = np.array(loss_forward[mode_idx])
        pde_losses = loss_fwd_arr[:, 0]
        bc_losses = loss_fwd_arr[:, 1]
        total_losses = weights_phase1[0] * pde_losses + weights_phase1[1] * bc_losses

        x_axis = eval_interval * np.arange(len(loss_fwd_arr))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.semilogy(x_axis, pde_losses, label="PDE Loss")
        ax1.semilogy(x_axis, bc_losses, label="BC Loss")
        ax1.set_title(f"Phase 1 Mode {mode_idx}: Partial Loss")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.grid(True)

        ax2.semilogy(x_axis, total_losses, label="Weighted Validation Loss", color='black')
        ax2.set_title(f"Phase 1 Mode {mode_idx}: Weighted Validation Loss")
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
        plt.savefig(os.path.join(script_dir, 'fig', f'2D_WG_losses_{defect_name}_modes{modes_str}_freqs{freqs_str}.pdf'))
        if SHOW_PLOTS:
            plt.show()
        plt.close()

    # --- Plot final reconstructed sound speed field c(x, y) ---
    for i, layers_m_each in enumerate(best_params_each_m):
        def c_final(x, y, _lm=layers_m_each):
            return c(x, y, _lm)

        x_c = np.linspace(-1, 1, 100)
        y_c = np.linspace(0, 0.6, 50)
        c_grid_final = jax.vmap(jax.vmap(c_final, in_axes=(0, None)), in_axes=(None, 0))(x_c, 2 * y_c / H - 1)

        plt.figure(figsize=(7, 3.5))
        plt.pcolormesh(x_c, y_c, c_grid_final, rasterized=True)
        plt.colorbar(label="Sound speed c(x, y)")
        modes_per_freq_str = ' | '.join([f"{int(f)}Hz:M{active_modes_per_freq[float(f)]}" for f in training_frequencies[:i+1]])
        plt.title(f"Reconstructed c(x,y) — {modes_per_freq_str}")
        plt.tight_layout()
        plt.savefig(os.path.join(script_dir, 'fig',
            f'c_map_{defect_name}_freq{int(training_frequencies[i])}_m{modes_str}_f{freqs_str}.pdf'))
        if SHOW_PLOTS:
            plt.show()
        plt.close()


if __name__ == "__main__":
    main()
