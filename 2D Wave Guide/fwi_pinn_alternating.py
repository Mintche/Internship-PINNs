import jax
import jax.numpy as jnp
import functools
import numpy as np
import optax
import matplotlib.pyplot as plt
from data_load import WaveguideBoundaryData
import os

print(jax.devices())

# Domain Omega : [-1,1]x[0,0.6]
# conditions de neuman en y=0 et y=0.6
H = 0.6
L = 1

data_loader_left = WaveguideBoundaryData('data/pinn_boundary_left_circle_contrast20percent.csv')
data_loader_right = WaveguideBoundaryData('data/pinn_boundary_right_circle_contrast20percent.csv')

X_left, Y_left, U_re_left, U_im_left, freq = data_loader_left.get_training_data()
X_right, Y_right, U_re_right, U_im_right, _ = data_loader_right.get_training_data()

f_test = 600.0
Y_left = Y_left[f_test]
Y_right = Y_right[f_test]

# x is already in [-1,1]
# y and u [-1,1]

Y_norm = jnp.max(Y_left)
Y_left = 2*Y_left/Y_norm - 1
Y_right = 2*Y_right/Y_norm - 1

U_norm = {}
U_left_norm = {}
U_right_norm = {}

for f in freq:
    U_norm[f] = np.sqrt(jnp.max(np.concatenate((U_re_left[f]**2 + U_im_left[f]**2, U_re_right[f]**2 + U_im_right[f]**2))))
    U_re_left[f] /= U_norm[f]
    U_im_left[f] /= U_norm[f]
    U_re_right[f] /= U_norm[f]
    U_im_right[f] /= U_norm[f]
    U_left_norm[f] = jnp.stack([U_re_left[f], U_im_left[f]], axis=1)
    U_right_norm[f] = jnp.stack([U_re_right[f], U_im_right[f]], axis=1)

key = jax.random.key(0)

# Fouriers Features init
fmax = f_test
c0 = 340
cmax = 1.5 * c0
cmin = 0.5 * c0
contrast = 1
N_modes = int(jnp.round(2*H*fmax/c0)) + 5
kmax = 2*jnp.pi*fmax/(contrast*340)
m = 64

msq_min = 1.0 / (cmax**2)
msq_max = 1.0 / (cmin**2)

key, subkey = jax.random.split(key)
B_base = jax.random.normal(subkey, (m, 2))

def gamma(x, y, sigma):
    x_phys = x * L 
    y_phys = (y + 1.0) * H / 2.0
    v_phys = jnp.array([x_phys, y_phys])
    B = B_base * sigma
    return jnp.concatenate([jnp.cos(B@v_phys), jnp.sin(B@v_phys)])

n_gauss_legendre = 20
y_gauss_legendre, w_gauss_legendre = np.polynomial.legendre.leggauss(n_gauss_legendre)

n_input = 2
n_layers_uv = [2*m, 64, 64, 64, 64, 2]
n_layers_m = [n_input, 128, 128, 128, 1] # Augmentation de la taille du réseau m

def init_layers(key, n_layers):
    layers = []
    for i in range(len(n_layers)-1):
        key, subkey = jax.random.split(key)
        W = jax.random.normal(subkey, (n_layers[i], n_layers[i+1])) * jnp.sqrt(2/(n_layers[i] + n_layers[i+1]))
        b = jnp.zeros(n_layers[i+1])
        layers.append({"W": W, "b": b})
    return layers, key

key, subkey_uv, subkey_m = jax.random.split(key, 3)
layers_uv, key = init_layers(subkey_uv, n_layers_uv)
layers_m, key = init_layers(subkey_m, n_layers_m)

layers_m[-1]["b"] = -jnp.log(27/5)

params_uv = {
    'layers': layers_uv,
    'sigma': jnp.array([kmax, 0.5*kmax])
}

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
    Z = msq_min + (msq_max-msq_min) * jax.nn.sigmoid(Z @ layers[-1]["W"] + layers[-1]["b"])
    return Z.squeeze()

y_quad = (y_gauss_legendre + 1.0) * H / 2.0
w_quad = w_gauss_legendre * H / 2.0
n_modes = jnp.arange(N_modes)
a_n = jnp.sqrt(2/H) * jnp.ones(N_modes)
a_n = a_n.at[0].set(jnp.sqrt(1/H))
C_quad = a_n[:, None] * jnp.cos(n_modes[:, None] * jnp.pi * y_quad[None, :] / H)

# Fonction de coût pour la phase U (met à jour params_uv)
def loss_fn_u(params_uv, layers_m, x, y, f, target_u_left, target_u_right, y_bnd_left, y_bnd_right, u_norm_val, weights):
    def uv(x,y):
        features = gamma(x, y, params_uv['sigma'])
        return forward_func(params_uv['layers'], features)
    
    def m(x,y):
        return forward_params(layers_m, jnp.array([x, y]))
    
    def k2(x,y):
        return (2*jnp.pi*f)**2*m(x,y)
    
    def uv_x(x, y):
        return jax.jacfwd(uv, argnums=0)(x, y)/L
    
    def uv_y(x, y):
        return jax.jacfwd(uv, argnums=1)(x, y)*(2/H)
    
    def pde_residual(x, y):
        uv_xx = jax.jacfwd(jax.jacfwd(uv, argnums=0), argnums=0)(x, y) / L**2
        uv_yy = jax.jacfwd(jax.jacfwd(uv, argnums=1), argnums=1)(x, y)*(2/H)**2
        return uv_xx + uv_yy + k2(x,y)*uv(x,y)
    
    k0 = 2*jnp.pi*f/c0
    beta_n = jnp.sqrt(k0**2 - (n_modes*jnp.pi/H)**2 +0j)
    
    def compute_dtn_loss(x_bnd, y_data, sign, A_inc=None):
        uv_quad = jax.vmap(uv, in_axes=(None, 0))(x_bnd, y_gauss_legendre)
        U_quad_complex = uv_quad[:, 0] + 1j * uv_quad[:, 1]
        
        u_n = C_quad @ (w_quad * U_quad_complex)
        
        dtn_n = sign * 1j * beta_n * u_n
        if A_inc is not None:
            dtn_n = dtn_n + 2j * beta_n * A_inc
        
        def DtN_eval(y_eval):
            C_y = a_n * jnp.cos(n_modes * jnp.pi * (y_eval+1)/2)
            return C_y @ dtn_n
            
        dtn_pred_complex = jax.vmap(DtN_eval)(y_data)
        
        dtn_actual = jax.vmap(uv_x, in_axes=(None, 0))(x_bnd, y_data)
        dtn_actual_complex = dtn_actual[:, 0] + 1j * dtn_actual[:, 1]
        
        return jnp.mean(jnp.abs(dtn_pred_complex - dtn_actual_complex)**2)

    A_inc_left = jnp.zeros(N_modes, dtype=jnp.complex64)
    amplitude_incidente = jnp.exp(-1j * beta_n[0] * L) / u_norm_val
    A_inc_left = A_inc_left.at[0].set(amplitude_incidente)
    
    # PDE residual
    vmap_pde_residual = jax.vmap(pde_residual, in_axes=(0, 0))
    pde_loss = jnp.mean(vmap_pde_residual(x, y)**2)
    
    # Boundary conditions
    dtn_loss_left = compute_dtn_loss(-1.0, y, sign=-1, A_inc=A_inc_left)
    dtn_loss_right = compute_dtn_loss(1.0, y, sign=1, A_inc=None)
    neum_loss = jnp.mean(jax.vmap(uv_y, in_axes=(0, None))(x, 1.0)**2 + jax.vmap(uv_y, in_axes=(0, None))(x, -1.0)**2)
    bc_loss = neum_loss + dtn_loss_left + dtn_loss_right

    # Data loss (Phase U: on fitte u aux donnees)
    vmap_uv = jax.vmap(uv, (None,0))
    data_loss = jnp.mean((vmap_uv(-1.0,y_bnd_left) - target_u_left)**2) + jnp.mean((vmap_uv(1.0,y_bnd_right) - target_u_right)**2)

    total_loss = weights[0]*pde_loss + weights[1]*bc_loss + weights[2]*data_loss
    
    return total_loss, (pde_loss, bc_loss, data_loss, 0.0)

# Fonction de coût pour la phase M (met à jour layers_m)
def loss_fn_m(params_uv, layers_m, x, y, f, target_u_left, target_u_right, y_bnd_left, y_bnd_right, u_norm_val, weights):
    def uv(x,y):
        features = gamma(x, y, params_uv['sigma'])
        return forward_func(params_uv['layers'], features)
    
    def m(x,y):
        return forward_params(layers_m, jnp.array([x, y]))

    def k2(x,y):
        return (2*jnp.pi*f)**2*m(x,y)
    
    def pde_residual(x, y):
        # Pour m, on a besoin du residu PDE
        uv_xx = jax.jacfwd(jax.jacfwd(uv, argnums=0), argnums=0)(x, y) / L**2
        uv_yy = jax.jacfwd(jax.jacfwd(uv, argnums=1), argnums=1)(x, y)*(2/H)**2
        return uv_xx + uv_yy + k2(x,y)*uv(x,y)
    
    def m_x(x,y):
        return jax.grad(m, argnums=0)(x, y)/L
    
    def m_y(x,y):
        return jax.grad(m, argnums=1)(x, y)*(2/H)

    # PDE residual (Phase M: on ajuste m pour que l'EDP soit satisfaite)
    vmap_pde_residual = jax.vmap(pde_residual, in_axes=(0, 0))
    pde_loss = jnp.mean(vmap_pde_residual(x, y)**2)

    # Régularisation TV
    vmap_m_x = jax.vmap(m_x, (0,0))
    vmap_m_y = jax.vmap(m_y, (0,0))
    TV_loss = jnp.mean(jnp.sqrt(c0**2*vmap_m_x(x,y)**2 + c0**2*vmap_m_y(x,y)**2 + 10**-8))
    
    total_loss = weights[0]*pde_loss + weights[3]*TV_loss
    
    return total_loss, (pde_loss, 0.0, 0.0, TV_loss)



lr_uv = 1e-3
lr_m = 1e-3 # Learning rate pour le réseau de slowness

def make_train_step_u(adam_opt):
    @functools.partial(jax.jit, static_argnames=('N',))
    def train_step_u(params_uv, layers_m, opt_state_uv, N, f_val, target_left, target_right, y_bnd_left, y_bnd_right, u_norm, current_weights, key):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        x = jax.random.uniform(subkey1, (N,), minval=-1, maxval=1)
        y = jax.random.uniform(subkey2, (N,), minval=-1, maxval=1)

        def loss_u(p_uv):
            return loss_fn_u(p_uv, layers_m, x, y, f_val, target_left, target_right,
                y_bnd_left, y_bnd_right, u_norm, current_weights)

        (loss, aux), grads_uv = jax.value_and_grad(loss_u, has_aux=True)(params_uv)

        updates_uv, opt_state_uv = adam_opt.update(grads_uv, opt_state_uv, params_uv)
        params_uv = optax.apply_updates(params_uv, updates_uv)
        return params_uv, opt_state_uv, loss, aux, key
    return train_step_u

def make_train_step_m(adam_opt_m):
    @functools.partial(jax.jit, static_argnames=('N',))
    def train_step_m(params_uv, layers_m, opt_state_m, N, f_val, target_left, target_right, y_bnd_left, y_bnd_right, u_norm, current_weights, key):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        x = jax.random.uniform(subkey1, (N,), minval=-1, maxval=1)
        y = jax.random.uniform(subkey2, (N,), minval=-1, maxval=1)

        def loss_m(l_m):
            # params_uv est figé grâce à jax.lax.stop_gradient, les gradients ne passeront pas au réseau de champ
            return loss_fn_m(jax.lax.stop_gradient(params_uv), l_m, x, y, f_val, target_left, target_right,
                y_bnd_left, y_bnd_right, u_norm, current_weights)

        (loss, aux), grads_m = jax.value_and_grad(loss_m, has_aux=True)(layers_m)

        updates_m, opt_state_m = adam_opt_m.update(grads_m, opt_state_m, layers_m)
        layers_m = optax.apply_updates(layers_m, updates_m)

        return layers_m, opt_state_m, loss, aux, key
    return train_step_m

# --- Boucle Principale d'Entraînement ---
def train(params_uv, layers_m, N, num_cycles, steps_u, steps_m, freq, eval_interval, key):
    loss_u_hist = []
    loss_m_hist = []

    # Initialisation des optimiseurs
    adam_uv = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_uv)
    )
    adam_m = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_m)
    )

    step_u = make_train_step_u(adam_uv)
    step_m = make_train_step_m(adam_m)

    opt_state_uv = adam_uv.init(params_uv)
    opt_state_m = adam_m.init(layers_m)

    # Poids pour les loss : [pde, bc, data, tv]
    weights_u = jnp.array([1.0, 1.0, 1.0, 0.0])
    weights_m = jnp.array([1.0, 0.0, 0.0, 1.0])

    print(f"\n{'='*60}")
    print(f"--- Entraînement Alterné FWI (Corrigé) : f = {freq} Hz ---")
    print(f"{'='*60}")

    for cycle in range(num_cycles):
        print(f"\n--- Cycle {cycle+1}/{num_cycles} ---")
        
        # --- Phase U ---
        print(f"  Phase U ({steps_u} steps) - Mise à jour de u (PDE + BC + Data)")
        for step in range(steps_u):
            params_uv, opt_state_uv, loss, aux, key = step_u(
                params_uv, layers_m, opt_state_uv, N, freq,
                U_left_norm[freq], U_right_norm[freq],
                Y_left, Y_right, U_norm[freq], weights_u, key)
            
            if step % eval_interval == 0 or step == steps_u - 1:
                pde_loss, bc_loss, data_loss, _ = aux
                loss_u_hist.append((float(pde_loss), float(bc_loss), float(data_loss)))
                print(f"  [Phase U] Step {step} | PDE: {pde_loss:.2e} | BC: {bc_loss:.2e} | Data: {data_loss:.2e} | Total: {loss:.2e}")

        # --- Phase M ---
        print(f"  Phase M ({steps_m} steps) - Mise à jour de m (PDE + TV)")
        for step in range(steps_m):
            layers_m, opt_state_m, loss, aux, key = step_m(
                params_uv, layers_m, opt_state_m, N, freq,
                U_left_norm[freq], U_right_norm[freq],
                Y_left, Y_right, U_norm[freq], weights_m, key)
            
            if step % eval_interval == 0 or step == steps_m - 1:
                pde_loss, _, _, TV_loss = aux
                loss_m_hist.append((float(pde_loss), float(TV_loss)))
                print(f"  [Phase M] Step {step} | PDE: {pde_loss:.2e} | TV: {TV_loss:.2e} | Total: {loss:.2e}")

    return params_uv, layers_m, key, loss_u_hist, loss_m_hist

N = 1000
eval_interval = 200

# Paramètres du schéma alterné
num_cycles = 10
steps_u = 1000
steps_m = 500

key, subkey = jax.random.split(key)

# Entraînement initial Phase U (optionnel pour bien commencer)
print("\n--- Préchauffage de u (Phase initiale) ---")
adam_uv_pre = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr_uv))
step_u_pre = make_train_step_u(adam_uv_pre)
opt_state_uv_pre = adam_uv_pre.init(params_uv)
weights_u_pre = jnp.array([1.0, 1.0, 0.0, 0.0]) # PDE, BC, Data, TV

for step in range(2000):
    params_uv, opt_state_uv_pre, loss, aux, key = step_u_pre(
        params_uv, layers_m, opt_state_uv_pre, N, f_test,
        U_left_norm[f_test], U_right_norm[f_test],
        Y_left, Y_right, U_norm[f_test], weights_u_pre, key)
    if step % 500 == 0:
        pde_loss, bc_loss, data_loss, _ = aux
        print(f"  [Pre-U] Step {step} | PDE: {pde_loss:.2e} | BC: {bc_loss:.2e} | Data: {data_loss:.2e} | Total: {loss:.2e}")

# Entraînement Alterné
params_uv, layers_m, key, loss_u_hist, loss_m_hist = train(
    params_uv, layers_m, N, num_cycles, steps_u, steps_m,
    f_test, eval_interval, key)

# --- Visualisation des Résultats ---
def plot_results(params_uv, layers_m):
    # Celerity
    def c(x,y):
        return 1/jnp.sqrt(forward_params(layers_m, jnp.array([x, y])))

    x_val = np.linspace(-1,1,100)
    y_val = np.linspace(0,0.6,50)

    c_grid = jax.vmap(jax.vmap(c, in_axes=(0, None)), in_axes=(None, 0))(x_val, 2*y_val/H-1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    pcm_c = plt.pcolormesh(x_val, y_val, c_grid)
    plt.colorbar(pcm_c, label='Celerity c(x,y)')
    plt.title('Reconstructed Celerity')

    # Wavefield
    def u_real(x,y):
        features = gamma(x, y, params_uv['sigma'])
        return forward_func(params_uv['layers'], features)[0] * U_norm[f_test]

    x_u = np.linspace(-1, 1, 500)
    y_u = np.linspace(0, 0.6, 150)

    u_grid = jax.vmap(jax.vmap(u_real, in_axes=(0, None)), in_axes=(None, 0))(x_u, 2*y_u/H - 1)

    plt.subplot(1, 2, 2)
    pcm_u = plt.pcolormesh(x_u, y_u, u_grid, cmap='RdBu_r', rasterized=True)
    plt.colorbar(pcm_u, label='Re(u)')
    plt.title('Re(u) PINN')

    plt.tight_layout()
    os.makedirs('fig', exist_ok=True)
    plt.savefig('fig/fwi_pinn_alternating_results.png', dpi=300, bbox_inches='tight')
    print("Results saved to fig/fwi_pinn_alternating_results.png")

plot_results(params_uv, layers_m)
