# Controlled forward PINN benchmark

This directory validates the forward PINN field architectures on known
coefficient maps.  It does not train a material network, and the FEM field is
used only for reference metrics, never by the physics loss or checkpoint
selection.

The heterogeneous report benchmark is the one-circle waveguide at 1200 Hz,
incident mode 0:

- background speed: `c0 = 340 m/s`;
- circle `(0.2, 0.2)`, radius `0.1`, speed ratio `0.8`;
- fine FEM reference: 89372 P2 degrees of freedom.

The homogeneous controls are
`forward_homogeneous_600_m0_uniform.json`,
`forward_homogeneous_600_m0_manual.json`, and
`forward_homogeneous_1200_m0_manual.json`.  Their exact total field is the
incident waveguide mode

```text
u(x,y) = a_m cos(m pi y/H) exp(i beta_m x),
```

and its exact scattered field is zero.  This control loads no FEM field or FEM
matrix.  L2/H1 errors are evaluated on an independent structured P1
triangulation: each rectangular cell is divided into two triangles and the
corresponding mass and stiffness matrices are assembled.

## Variants

The active campaign variants are the names exposed by `VARIANTS` in
`model_variants.py`.  The available formulations are:

- `classical_total`: coordinate MLP, tanh activations, automatic spatial
  derivatives, constant Adam learning rate;
- `fourier_total`: Fourier features, standard tanh MLP, analytic spatial
  derivatives, total-field equation;
- `fourier_modified_total`: Fourier features plus the modified MLP from Wang,
  Teng and Perdikaris, total-field equation;
- `fourier_scattered`: Fourier features, standard tanh MLP, analytic spatial
  derivatives, scattered-field equation;
- `fourier_modified_scattered`: Fourier features plus modified MLP,
  scattered-field equation.

Each formulation also has an adaptive-loss-weight counterpart with the
`_adweights` suffix, for example `classical_total_adweights` and
`fourier_modified_scattered_adweights`.  The suffix changes only the loss
weighting: architecture, physical formulation, optimizer, and parameter count
remain those of the base variant.  These variants are accepted by `run` and by
`campaign --variants`, but the campaign default remains the five static
variants listed above.

RAD is available independently with the `_rad` suffix, and can be combined
with adaptive weights using `_rad_adweights`.  For example,
`fourier_total_rad` uses residual-based adaptive sampling with static loss
weights, while `fourier_total_rad_adweights` enables both mechanisms.  RAD
also leaves the architecture and physical formulation of its base variant
unchanged.

Variant behavior is token based: `fourier` activates the trainable Fourier
scales, `modified` activates the gated modified MLP, and `scattered` switches
from total-field to scattered-field physics; `rad` and `adweights` activate
their respective training strategies.  The modified MLP uses two tanh encoders
`U` and `V`, tanh gates `Z`, the blend `(1 - Z) U + Z V`, and a linear
two-component complex output.

The RAD suffix activates the configured mixture of fresh uniform points and
points bootstrapped from the residual-weighted candidate distribution.  Its
candidate count, retained RAD point count, residual parameters, batch size,
and refresh interval are controlled by the existing `rad_*` JSON fields.

Static loss weights are configured as `[PDE, Neumann, DtN]`.  The CSV histories
still include `bc_loss = neumann_loss + dtn_loss`, which is used for the PDE/BC
gradient cosine.

The physical Helmholtz residual is divided by `k0**2` before its mean-square
loss is computed.  The physical Neumann and DtN residuals are first order and
are normalized by `k0`; in the implementation this is applied equivalently by
dividing their squared losses by `k0**2`.  Each Neumann abscissa is evaluated
on both rigid walls and each DtN ordinate on both ports.

## Optimization

All active variants are trained with Adam only.  There is no L-BFGS phase, no
gradient clipping, and no learning-rate decay for field weights.

Fourier variants train `sigma` for

```text
N_sigma = max(1, int(fourier_sigma_decay_fraction * adam_steps))
```

updates.  Its learning rate follows a cosine decay from
`fourier_sigma_learning_rate` to
`fourier_sigma_cosine_alpha * fourier_sigma_learning_rate`.  After that point,
the training step applies `stop_gradient` to `sigma` and forces the Optax sigma
update to zero, while field weights continue with the same Adam state.

### Adaptive loss weights

An `_adweights` run requires this JSON object:

```json
"adweights": {
  "epsilon": 1e-8,
  "alpha": 0.1,
  "initial_lambdas": [0.01, 0.001, 1.0],
  "update_interval_adam": 500,
  "custom_weights": [1.0, 1.0, 1.0]
}
```

All triplets use the `[PDE, Neumann, DtN]` order.  At Adam step `1`, the three
raw component losses are differentiated on the exact step batch and recorded,
while the configured initial weights are retained.  Starting at step
`1 + update_interval_adam`, the moving averages and weights are updated at the
configured interval.  For each component update,

```text
lambda_tilde_i = L2 norm(gradient(loss_i))
lambda_i = (1 - alpha) lambda_i + alpha lambda_tilde_i
raw_weight_i = 1 / (epsilon + lambda_i)
effective_weight_i = custom_weight_i * raw_weight_i
```

Only currently trainable parameters contribute to the norm, so Fourier
`sigma` is excluded after it freezes.  The weights are not normalized or
clipped and are treated as constants during the corresponding Adam update.
The adaptive objective is reported in the loss history, while
`checkpoint_best_monitor.npz` is selected using the comparable raw sum
`PDE + Neumann + DtN`.

## Install-time verification

The tests use deliberately tiny networks and batches.  They do not load the
fine FEM files or launch a scientific training run:

```bash
MPLCONFIGDIR=/tmp/matplotlib JAX_PLATFORMS=cpu \
  .venv/bin/python -m unittest tests.test_forward_ablation -v
```

## One run

The Adam budget is required explicitly.  This example is illustrative; choose
the final report budget before starting:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
  .venv/bin/python tests_forward_PINN/forward_ablation.py run \
  --config tests_forward_PINN/forward_circlebottomright_1200_m0.json \
  --variant fourier_modified_scattered \
  --seed 0 \
  --adam-steps 30000
```

The generated run directory is exclusive.  Repeating the same configuration
refuses to overwrite it.

For the homogeneous analytic control, replace the configuration:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
  .venv/bin/python tests_forward_PINN/forward_ablation.py run \
  --config tests_forward_PINN/forward_homogeneous_1200_m0_manual.json \
  --variant fourier_modified_total \
  --seed 0 \
  --adam-steps 30000
```

For scattered variants, reported L2/H1 errors still compare the reconstructed
physical total field `u0 + us` to the reference total field.

## Campaign

The campaign launches the active variants sequentially in fresh Python
subprocesses.  By default it launches only the five static variants, and its
default seed list is deliberately the single seed `0`:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
  .venv/bin/python tests_forward_PINN/forward_ablation.py campaign \
  --config tests_forward_PINN/forward_circlebottomright_1200_m0.json \
  --adam-steps 30000
```

No campaign is launched automatically by tests or imports.

Each run contains:

- `manifest.json` and `effective_config.json`;
- `loss_history.csv` at the fixed-monitor cadence;
- `gradient_history.csv` at the configured optimizer-batch cadence;
- `adweights_history.csv` for adaptive runs, with the initial state and one
  row per component at every weight update;
- `fem_metrics.csv` at the reference cadence;
- `rad_resampling_history.csv` for RAD runs with at least one probability
  refresh;
- `checkpoint_final.npz` and `checkpoint_best_monitor.npz`;
- `summary.json`, including `optimizer_seconds`, `rad_resampling_seconds`,
  `adweights_seconds`, `training_seconds`, adaptive final weights,
  `sigma_train_steps`, and final L2/H1 metrics.

Synchronized Adam updates contribute to `optimizer_seconds`.  RAD candidate
generation, residual evaluation, and probability/CDF construction contribute
to `rad_resampling_seconds`.  Required adaptive component-gradient
norms contribute to `adweights_seconds`; their sum with the other two timings
is `training_seconds`.
Compilation, reference inference, matrix norms, checkpoint writes, and plots
are excluded.

## Aggregate and plot completed runs

After the campaign, give its generated `campaign_cfg...` directory to:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python tests_forward_PINN/plot_campaign.py \
  --campaign-root tests_forward_PINN/results/campaign_cfgXXXXXXXXXXXX \
  --output-dir tests_forward_PINN/results/report_figures
```

The output directory must not already exist.  It receives PDF and PNG figures
for raw losses, gradient statistics, reference errors versus iterations,
reference errors versus measured training time, training times, and adaptive
raw/effective weights when present, together with `runs.csv`, `aggregate.csv`,
and concatenated histories including `adweights_history_all.csv`.  Thin lines
show individual seeds; the main iteration-based lines and bands show the
arithmetic mean plus or minus one sample standard deviation.

All metrics compare the physical total field.  Relative norms use the mass and
stiffness matrices of either the fine FEM mesh or the independent analytic
triangulation:

```text
relative L2 = sqrt(e* M e) / sqrt(u_ref* M u_ref)
relative H1 = sqrt(e* (M + K) e) / sqrt(u_ref* (M + K) u_ref)
```

The finest solution is a numerical reference, not an exact solution.  Its
difference from the preceding 44209-DOF mesh is approximately 0.217% in L2 and
2.60% in H1.
