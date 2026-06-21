import os
import functools
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
from data_loader import WaveguideBoundaryData

# Print JAX devices to verify setup
print("JAX Devices:", jax.devices())

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

# --- Data Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_left_path = os.path.join(script_dir, 'data', 'pinn_boundary_left_barrehalf_contrast20percent_mode0.csv')
data_right_path = os.path.join(script_dir, 'data', 'pinn_boundary_right_barrehalf_contrast20percent_mode0.csv')

# --- Random Seed ---
random_seed = 0
key = jax.random.key(random_seed)

# --- Neural Network Architectures ---
m = 64  # Dimensionality of Fourier feature mapping
n_input = 2
n_layers_uv = [2 * m, 64, 64, 64, 64, 2]
n_layers_m = [n_input, 64, 64, 1]

# --- Gauss-Legendre Quadrature ---
n_gauss_legendre = 20

# --- Optimizer Learning Rates ---
lr_uv = 1e-3
lr_m = 3e-4
lr_sigma = 1e-2

# --- Training Hyperparameters ---
N_adam = 4000            # Number of collocation points for Adam
N_lbfgs = 1000           # Number of collocation points for L-BFGS
eval_interval = 200      # Interval for evaluating losses and printing progress
switch_threshold = 0.0   # Threshold for L-BFGS switch criterion
switch_window = 10       # Window size for switch criterion

max_steps_adam_phase1 = 20001
max_steps_lbfgs_phase1 = 201
max_steps_adam_phase2 = 100001
max_steps_lbfgs_phase2 = 201

# Frequencies to run training on
training_frequencies = np.array([300.0,500.0,700.0])

# ==============================================================================
# SECTION 2: DATA LOADING & PREPROCESSING
# ==============================================================================

# Load boundary data
data_loader_left = WaveguideBoundaryData(data_left_path)
data_loader_right = WaveguideBoundaryData(data_right_path)

X_left, Y_left, U_re_left, U_im_left, dataset_freq = data_loader_left.get_training_data()
X_right, Y_right, U_re_right, U_im_right, _ = data_loader_right.get_training_data()

# Process Y coordinates
Y_left = Y_left[dataset_freq[0]]
Y_right = Y_right[dataset_freq[0]]

Y_norm = jnp.max(Y_left)
Y_left = 2 * Y_left / Y_norm - 1
Y_right = 2 * Y_right / Y_norm - 1

# Normalize complex wave fields
U_norm = {}
U_left_norm = {}
U_right_norm = {}

for f in dataset_freq:
    U_norm[f] = jnp.sqrt(jnp.max(jnp.concatenate((
        U_re_left[f]**2 + U_im_left[f]**2,
        U_re_right[f]**2 + U_im_right[f]**2
    ))))
    U_re_left[f] /= U_norm[f]
    U_im_left[f] /= U_norm[f]
    U_re_right[f] /= U_norm[f]
    U_im_right[f] /= U_norm[f]
    U_left_norm[f] = jnp.stack([U_re_left[f], U_im_left[f]], axis=1)
    U_right_norm[f] = jnp.stack([U_re_right[f], U_im_right[f]], axis=1)

# Calculate dynamic boundary modes configuration
fmax = jnp.max(dataset_freq)
N_modes = int(jnp.round(2 * H * fmax / c0)) + 5

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

n_modes = jnp.arange(N_modes)
a_n = jnp.sqrt(2 / H) * jnp.ones(N_modes)
a_n = a_n.at[0].set(jnp.sqrt(1 / H))
C_quad = a_n[:, None] * jnp.cos(n_modes[:, None] * jnp.pi * y_quad[None, :] / H)

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
    ky = mode_val * jnp.pi / H
    layers_uv = init_layers(key, n_layers)
    return {'layers': layers_uv, 'sigma': jnp.array([kx, ky])}

# --- Network Initialization ---
# Sound speed parameter network (layers_m)
key, subkey_m = jax.random.split(key)
layers_m = init_layers(subkey_m, n_layers_m)
layers_m[-1]["b"] = -jnp.log(27.0 / 5.0)
layers_m[-1]["W"] /= 10.0

# Wavefield networks (params_uv) per frequency
params_uv = {}
for f in dataset_freq:
    key, subkey_f = jax.random.split(key)
    params_uv[float(f)] = init_layers_uv(subkey_f, n_layers_uv, f, 0)

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

plt.figure(figsize=(7, 3.5))
plt.pcolormesh(x_plot, y_plot, c_grid)
plt.colorbar(label="Sound speed c(x, y)")
plt.title("Initial Sound Speed Field")
plt.tight_layout()
plt.show()

# ==============================================================================
# SECTION 6: LOSS FUNCTION DEFINITIONS
# ==============================================================================

def loss_fn(params_uv, layers_m, x, y, f, target_u_left, target_u_right, y_bnd_left, y_bnd_right, u_norm_val, weights, is_warmup=False, use_healthy_guide=False):

    def uv(x, y):
        features = gamma(x, y, params_uv['sigma'])
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
    
    k0 = 2 * jnp.pi * f / c0
    beta_n = jnp.sqrt(k0**2 - (n_modes * jnp.pi / H)**2 + 0j)
    
    def compute_dtn_loss(x_bnd, y_data, sign, A_inc=None):
        uv_quad = jax.vmap(uv, in_axes=(None, 0))(x_bnd, y_gauss_legendre)
        U_quad_complex = uv_quad[:, 0] + 1j * uv_quad[:, 1]
        
        u_n = C_quad @ (w_quad * U_quad_complex)
        
        dtn_n = sign * 1j * beta_n * u_n
        if A_inc is not None:
            dtn_n = dtn_n + 2j * beta_n * A_inc
        
        def DtN_eval(y_eval):
            C_y = a_n * jnp.cos(n_modes * jnp.pi * (y_eval + 1) / 2)
            return C_y @ dtn_n
            
        dtn_pred_complex = jax.vmap(DtN_eval)(y_data)
        
        dtn_actual = jax.vmap(uv_x, in_axes=(None, 0))(x_bnd, y_data)
        dtn_actual_complex = dtn_actual[:, 0] + 1j * dtn_actual[:, 1]
        
        return jnp.mean(jnp.abs(dtn_pred_complex - dtn_actual_complex)**2)

    A_inc_left = jnp.zeros(N_modes, dtype=jnp.complex64)
    amplitude_incidente = jnp.exp(-1j * beta_n[0] * L) / u_norm_val
    A_inc_left = A_inc_left.at[0].set(amplitude_incidente)
    
    # PDE Residual Loss
    vmap_pde_residual = jax.vmap(pde_residual, in_axes=(0, 0))
    pde_loss = jnp.mean(vmap_pde_residual(x, y)**2)
    
    # Boundary Conditions Loss
    dtn_loss_left = compute_dtn_loss(-1.0, y, sign=-1, A_inc=A_inc_left)
    dtn_loss_right = compute_dtn_loss(1.0, y, sign=1, A_inc=None)
    neum_loss = jnp.mean(jax.vmap(uv_y, in_axes=(0, None))(x, 1.0)**2 + jax.vmap(uv_y, in_axes=(0, None))(x, -1.0)**2)
    bc_loss = neum_loss + dtn_loss_left + dtn_loss_right
    
    # Data
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

    @functools.partial(jax.jit, static_argnames=('N', 'use_lbfgs', 'use_healthy_guide'))
    def train_step_forward(params_uv, layers_m, opt_state_uv, key, f_val, target_left, target_right, y_bnd_left, y_bnd_right, u_norm, current_weights, N, use_lbfgs=False, use_healthy_guide=True):
        if use_lbfgs:
            # Fixed uniform generation
            subkey1, subkey2 = jax.random.split(key, 2)
            x = jax.random.uniform(subkey1, (N,), minval=-1, maxval=1)
            y = jax.random.uniform(subkey2, (N,), minval=-1, maxval=1)
            new_key = key
        else:
            # Dynamic uniform generation for Adam
            new_key, subkey1, subkey2 = jax.random.split(key, 3)
            x = jax.random.uniform(subkey1, (N,), minval=-1, maxval=1)
            y = jax.random.uniform(subkey2, (N,), minval=-1, maxval=1)

        def loss_uv(p_uv):
            return loss_fn(p_uv, layers_m, x, y, f_val, target_left, target_right,
                           y_bnd_left, y_bnd_right, u_norm, current_weights, is_warmup=True, use_healthy_guide=use_healthy_guide)

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


def make_train_step_inverse(adam_opt_uv, adam_opt_m, lbfgs_opt_packed):

    @functools.partial(jax.jit, static_argnames=('N', 'use_lbfgs',))
    def train_step_inverse(params_uv, layers_m, opt_state_uv, opt_state_m, opt_state_lbfgs, key, f_val, target_left, target_right, y_bnd_left, y_bnd_right, u_norm, current_weights, N, use_lbfgs=False):
        if use_lbfgs:
            # Fixed uniform generation
            subkey1, subkey2 = jax.random.split(key, 2)
            x = jax.random.uniform(subkey1, (N,), minval=-1, maxval=1)
            y = jax.random.uniform(subkey2, (N,), minval=-1, maxval=1)
            new_key = key
        else:
            # Dynamic uniform generation for Adam
            new_key, subkey1, subkey2 = jax.random.split(key, 3)
            x = jax.random.uniform(subkey1, (N,), minval=-1, maxval=1)
            y = jax.random.uniform(subkey2, (N,), minval=-1, maxval=1)
        
        def loss_full(p_uv, l_m):
            return loss_fn(p_uv, l_m, x, y, f_val, target_left, target_right,
                           y_bnd_left, y_bnd_right, u_norm, current_weights, is_warmup=False, use_healthy_guide=False)
        
        (loss, aux), (grads_uv, grads_m) = jax.value_and_grad(loss_full, argnums=(0, 1), has_aux=True)(params_uv, layers_m)

        if use_lbfgs:
            params_packed = {'uv': params_uv, 'm': layers_m}
            grads_packed = {'uv': grads_uv, 'm': grads_m}
            
            def value_fn_packed(p_packed):
                v, _ = loss_full(p_packed['uv'], p_packed['m'])
                return v

            updates_packed, opt_state_lbfgs = lbfgs_opt_packed.update(
                grads_packed, opt_state_lbfgs, params_packed,
                value=loss, grad=grads_packed, value_fn=value_fn_packed)
            
            params_packed = optax.apply_updates(params_packed, updates_packed)
            params_uv = params_packed['uv']
            layers_m = params_packed['m']
        else:
            updates_uv, opt_state_uv = adam_opt_uv.update(grads_uv, opt_state_uv, params_uv)
            params_uv = optax.apply_updates(params_uv, updates_uv)

            updates_m, opt_state_m = adam_opt_m.update(grads_m, opt_state_m, layers_m)
            layers_m = optax.apply_updates(layers_m, updates_m)

        return params_uv, layers_m, opt_state_uv, opt_state_m, opt_state_lbfgs, loss, aux, new_key

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
          freqs, eval_interval, switch_threshold, switch_window, key):

    loss_forward = []
    loss_inverse = []

    weights_phase1 = jnp.array([1.0, 10.0, 0.0])

    best_params_uv = params_uv

    for i, freq in enumerate(freqs):
        print(f"\n{'='*60}")
        print(f"--- f = {freq} Hz ---")
        print(f"{'='*60}")

        param_uv = params_uv[freq]
        best_params_uv[freq] = param_uv
        best_params_m = layers_m
        best_loss = float('inf')

        use_healthy_guide_flag = (i == 0)

        if i > 0:
            scale = jnp.abs(best_params_uv[freqs[i-1]]["sigma"][1]/best_params_uv[freqs[i-1]]["sigma"][0])
            sigma_x = param_uv["sigma"][0]
            param_uv["sigma"] = param_uv["sigma"].at[1].set(scale*sigma_x)

        # ============================================================
        # PHASE 1: Forward (only param_uv)
        # ============================================================

        print(f"\n--- Phase 1: Adam (max {max_steps_adam_phase1} steps) ---")

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
            param_labels=param_labels_p1
        )
        lbfgs_uv_p1 = optax.lbfgs()

        step_forward = make_train_step_forward(adam_uv_p1, lbfgs_uv_p1)
        opt_state_uv = adam_uv_p1.init(param_uv)

        loss_history_p1 = []
        switched_to_lbfgs = False

        for step in range(max_steps_adam_phase1):
            param_uv, opt_state_uv, loss, aux, key = step_forward(
                param_uv, layers_m, opt_state_uv, key, freq,
                U_left_norm[freq], U_right_norm[freq],
                Y_left, Y_right, U_norm[freq], weights_phase1,
                N=N_adam, use_lbfgs=False, use_healthy_guide=use_healthy_guide_flag)

            if step % eval_interval == 0:
                pde_loss, bc_loss, _ = aux
                loss_forward.append((pde_loss, bc_loss))
                print(f"[Adam] Step {step} | pde {pde_loss:.2e} | bc {bc_loss:.2e} | total {loss:.2e} | scales {param_uv['sigma']}")
                
                loss_history_p1.append(float(loss))
                
                if loss < best_loss:
                    best_loss = loss
                    best_params_uv[freq] = param_uv

                if check_switch_criterion(loss_history_p1, window=switch_window, threshold=switch_threshold):
                    print(f"[Adam] Convergence criterion met at step {step}. Switching to L-BFGS.")
                    switched_to_lbfgs = True
                    param_uv = best_params_uv[freq]
                    break

        # --- Phase 1: L-BFGS ---
        if switched_to_lbfgs or max_steps_lbfgs_phase1 > 0:
            print(f"\n--- Phase 1: L-BFGS (max {max_steps_lbfgs_phase1} steps) ---")
            opt_state_uv = lbfgs_uv_p1.init(param_uv)
            
            # Split key once to get fixed key for L-BFGS.
            # Passing this lbfgs_key at each step keeps the generated coordinates constant.
            key, lbfgs_key = jax.random.split(key)

            for step in range(max_steps_lbfgs_phase1):
                param_uv, opt_state_uv, loss, aux, _ = step_forward(
                    param_uv, layers_m, opt_state_uv, lbfgs_key, freq,
                    U_left_norm[freq], U_right_norm[freq],
                    Y_left, Y_right, U_norm[freq], weights_phase1,
                    N=N_lbfgs, use_lbfgs=True, use_healthy_guide=use_healthy_guide_flag)

                if step % eval_interval == 0:
                    pde_loss, bc_loss, _ = aux
                    loss_forward.append((pde_loss, bc_loss))
                    print(f"[LBFGS] Step {step} | pde {pde_loss:.2e} | bc {bc_loss:.2e} | total {loss:.2e} | scales {param_uv['sigma']}")
                    if loss < best_loss:
                        best_loss = loss
                        best_params_uv[freq] = param_uv

        # ============================================================
        # PHASE 2: Inverse (param_uv + layers_m)
        # ============================================================

        best_loss = float('inf')
        
        if max_steps_adam_phase2 > 0 or max_steps_lbfgs_phase2 > 0:

            print(f"\n--- Phase 2: Adam (max {max_steps_adam_phase2} steps) ---")

            cosine_uv_p2 = optax.schedules.cosine_decay_schedule(
                init_value=lr_uv, decay_steps=max_steps_adam_phase2, alpha=0.1)
            cosine_sigma_p2 = optax.schedules.cosine_decay_schedule(
                init_value=lr_sigma, decay_steps=max_steps_adam_phase2//3, alpha=0.1)
            
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

            param_labels_p2 = jax.tree_util.tree_map(lambda _: 'base', param_uv)
            if "sigma" in param_labels_p2:
                param_labels_p2["sigma"] = jax.tree_util.tree_map(lambda _: 'sigma', param_uv["sigma"])

            adam_uv_p2 = optax.multi_transform(
                transforms={'base': adam_base_p2, 'sigma': adam_sigma_p2},
                param_labels=param_labels_p2
            )

            adam_m_p2 = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.scale_by_adam(),
                optax.scale_by_schedule(cosine_m_p2),
                optax.scale(-1.0))
            
            lbfgs_packed_p2 = optax.lbfgs()

            step_inverse = make_train_step_inverse(adam_uv_p2, adam_m_p2, lbfgs_packed_p2)

            opt_state_uv = adam_uv_p2.init(param_uv)
            opt_state_m = adam_m_p2.init(layers_m)
            opt_state_lbfgs = None

            data_weight_schedule = optax.linear_schedule(
                init_value=0.1, end_value=10.0,
                transition_steps=max(max_steps_adam_phase2 // 3, 1))

            loss_history_p2 = []
            switched_to_lbfgs_p2 = False

            for step in range(max_steps_adam_phase2):
                weights_phase2 = jnp.array([1.0, 1.0, data_weight_schedule(step), 0.0])

                param_uv, layers_m, opt_state_uv, opt_state_m, opt_state_lbfgs, loss, aux, key = step_inverse(
                    param_uv, layers_m, opt_state_uv, opt_state_m, opt_state_lbfgs, key, freq,
                    U_left_norm[freq], U_right_norm[freq],
                    Y_left, Y_right, U_norm[freq], weights_phase2,
                    N=N_adam, use_lbfgs=False)

                if step % eval_interval == 0:
                    pde_loss, bc_loss, data_loss = aux
                    loss_inverse.append((pde_loss, bc_loss, data_loss))
                    print(f"[Adam] Step {step} | pde {pde_loss:.2e} | bc {bc_loss:.2e} | data {data_loss:.2e} | total {loss:.2e} | scales {param_uv['sigma']}")
                    
                    loss_history_p2.append(float(loss))
                    
                    if loss < best_loss:
                        best_loss = loss
                        best_params_uv[freq] = param_uv
                        best_params_m = layers_m

                    if check_switch_criterion(loss_history_p2, window=switch_window, threshold=switch_threshold):
                        print(f"[Adam] Convergence criterion met at step {step}. Switching to L-BFGS.")
                        switched_to_lbfgs_p2 = True
                        param_uv = best_params_uv[freq]
                        layers_m = best_params_m
                        break

            # --- Phase 2: L-BFGS ---
            if switched_to_lbfgs_p2 or max_steps_lbfgs_phase2 > 0:
                print(f"\n--- Phase 2: L-BFGS (max {max_steps_lbfgs_phase2} steps) ---")
                opt_state_lbfgs = lbfgs_packed_p2.init({'uv': param_uv, 'm': layers_m})
                weights_phase2_lbfgs = jnp.array([1.0, 1.0, 1.0])

                key, lbfgs_key = jax.random.split(key)

                for step in range(max_steps_lbfgs_phase2):
                    param_uv, layers_m, opt_state_uv, opt_state_m, opt_state_lbfgs, loss, aux, _ = step_inverse(
                        param_uv, layers_m, opt_state_uv, opt_state_m, opt_state_lbfgs, lbfgs_key, freq,
                        U_left_norm[freq], U_right_norm[freq],
                        Y_left, Y_right, U_norm[freq], weights_phase2_lbfgs,
                        N=N_lbfgs, use_lbfgs=True)

                    if step % eval_interval == 0:
                        pde_loss, bc_loss, data_loss = aux
                        loss_inverse.append((pde_loss, bc_loss, data_loss))
                        print(f"[LBFGS] Step {step} | pde {pde_loss:.2e} | bc {bc_loss:.2e} | data {data_loss:.2e} | total {loss:.2e} | scales {param_uv['sigma']}")
                        if loss < best_loss:
                            best_loss = loss
                            best_params_uv[freq] = param_uv
                            best_params_m = layers_m

    return best_params_uv, best_params_m, key, loss_forward, loss_inverse

# ==============================================================================
# SECTION 9: MODEL TRAINING & POST-PROCESSING
# ==============================================================================

# --- Run training ---
key, subkey = jax.random.split(key)
params_uv, layers_m, key, loss_forward, loss_inverse = train(
    params_uv, layers_m, N_adam, N_lbfgs,
    max_steps_adam_phase1, max_steps_lbfgs_phase1,
    max_steps_adam_phase2, max_steps_lbfgs_phase2,
    training_frequencies, eval_interval, switch_threshold, switch_window, key
)

# Create figures directory if it doesn't exist
os.makedirs(os.path.join(script_dir, 'fig'), exist_ok=True)

# --- Plot wavefield predictions (Re(u) PINN) ---
for f in training_frequencies:
    def u_real(x, y):
        features = gamma(x, y, params_uv[f]['sigma'])
        return forward_func(params_uv[f]['layers'], features)[0] * U_norm[f]

    x_val = np.linspace(-1, 1, 500)
    y_val = np.linspace(0, 0.6, 150)

    u_grid = jax.vmap(jax.vmap(u_real, in_axes=(0, None)), in_axes=(None, 0))(x_val, 2 * y_val / H - 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    pcm = ax.pcolormesh(x_val, y_val, u_grid, cmap='RdBu_r', rasterized=True)
    ax.set_title(f'Re(u) PINN at {f} Hz')
    fig.colorbar(pcm, ax=ax, shrink=0.8, location='left')
    plt.tight_layout()
    # plt.savefig(os.path.join(script_dir, 'fig', '2D_WG_PINN_forward.pdf'), dpi=300, bbox_inches='tight')
    plt.show()

# --- Plot Phase 1 (Forward) Training Losses ---
loss_forward_arr = np.array(loss_forward)
pde_losses = loss_forward_arr[:, 0]
bc_losses = loss_forward_arr[:, 1]
total_losses = pde_losses + bc_losses

x_axis = eval_interval * np.arange(len(loss_forward))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.semilogy(x_axis, pde_losses, label="PDE Loss")
ax1.semilogy(x_axis, bc_losses, label="BC Loss")
ax1.set_title("Phase 1: Partial Loss")
ax1.set_xlabel("Steps")
ax1.set_ylabel("Loss")
ax1.legend(loc="upper right")
ax1.grid(True)

ax2.semilogy(x_axis, total_losses, label="Total Loss", color='black')
ax2.set_title("Phase 1: Total Loss")
ax2.set_xlabel("Steps")
ax2.legend(loc="upper right")
ax2.grid(True)
plt.tight_layout()
plt.show()

# --- Plot Phase 2 (Inverse) Training Losses ---
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
ax1.set_title("Phase 2: Partial Loss")
ax1.set_xlabel("Steps")
ax1.set_ylabel("Loss")
ax1.legend(loc="upper right")
ax1.grid(True)

ax2.semilogy(x_axis_inv, total_losses_inv, label="Total Loss", color='black')
ax2.set_title("Phase 2: Total Loss")
ax2.set_xlabel("Steps")
ax2.legend(loc="upper right")
ax2.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'fig', '2D_Wave_guide_losses.pdf'))
plt.show()

# --- Plot final reconstructed sound speed field c(x, y) ---
def c_final(x, y):
    return 1 / jnp.sqrt(forward_params(layers_m, jnp.array([x, y])))

x_c = np.linspace(-1, 1, 100)
y_c = np.linspace(0, 0.6, 50)
c_grid_final = jax.vmap(jax.vmap(c_final, in_axes=(0, None)), in_axes=(None, 0))(x_c, 2 * y_c / H - 1)

plt.figure(figsize=(7, 3.5))
plt.pcolormesh(x_c, y_c, c_grid_final, rasterized=True)
plt.colorbar(label="Sound speed c(x, y)")
plt.title("Reconstructed Sound Speed Field")
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'fig', '2D_Wave_guide_c_map_third.pdf'))
plt.show()