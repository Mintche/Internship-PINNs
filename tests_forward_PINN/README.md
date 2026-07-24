# Controlled forward PINN benchmark

This directory validates the two field architectures used by the inverse PINNs
on a known coefficient map.  It does **not** train a material network and the FEM
field is never used by the physics loss or by checkpoint selection.

The benchmark case is the two-circle waveguide at 1200 Hz, incident mode 0:

- background speed: `c0 = 340 m/s`;
- circle `(-0.3, 0.4)`, radius `0.1`, speed ratio `0.8`;
- circle `(0.3, 0.2)`, radius `0.1`, speed ratio `1.1`;
- fine FEM reference: 173365 P2 degrees of freedom.

An additional homogeneous control is provided in
`forward_homogeneous_1200_m0.json`.  Its exact total field is the incident
waveguide mode

```text
u(x,y) = a_m cos(m pi y/H) exp(i beta_m x),
```

and its exact scattered field is zero.  This control loads no FEM field or FEM
matrix.  L2/H1 errors are evaluated on an independent structured P1
triangulation: each rectangular cell is divided into two triangles and the
corresponding mass and stiffness matrices are assembled.  The default
`[256, 80]` resolution means 40,960 triangles.

The five variants are:

- `classical_total`: coordinate MLP, automatic derivatives, constant Adam rate;
- `adapted_total`: Fourier architecture and analytic derivatives, total field;
- `adapted_scattered`: exactly the same adapted architecture and optimizer,
  but trained on the scattered-field equation;
- `adapted_total_RAD`: `adapted_total` with residual-based adaptive
  distribution (RAD) sampling;
- `adapted_scattered_RAD`: `adapted_scattered` with the same RAD sampling.

The RAD implementation follows Wu et al., [*A comprehensive study of
non-adaptive and residual-based adaptive sampling for physics-informed neural
networks*](https://arxiv.org/abs/2207.10289), and its [official reference
implementation](https://github.com/lu-group/pinn-sampling).  A dense uniform
candidate pool is assigned the discrete probabilities

```text
p_i ∝ epsilon_i^k / mean(epsilon^k) + c,
```

where `epsilon_i` is the Euclidean magnitude of the complex Helmholtz residual.
The supplied configurations use `k = 1`, `c = 0.1`, 100,000 candidates, and
refresh their probabilities every 500 Adam iterations.

`collocation_adam[0]` remains the total PDE batch size for every variant.  With
the supplied `rad_points = 512`, each RAD Adam update therefore concatenates
3,584 fresh uniform points with a fresh 512-point bootstrap **with replacement**
from the cached candidate distribution.  Before the first adaptive refresh,
the cached distribution is uniform.  The paired non-RAD and RAD variants keep
the same boundary clouds and the RAD uniform points are the prefix of the
non-RAD PDE cloud generated from the same key.  L-BFGS similarly keeps its
configured total PDE count and replaces 512 regular points by one fixed RAD
bootstrap because its line search requires a deterministic objective.
FEM/reference values are never used for sampling or checkpoint selection.

The classical model has the same hidden widths but fewer trainable parameters
because its two coordinate inputs replace 128 Fourier inputs.  Parameter counts
are written to every run summary.

## Install-time verification

The tests use deliberately tiny networks and batches.  They do not load the fine
FEM files or launch a scientific training run:

```bash
MPLCONFIGDIR=/tmp/matplotlib JAX_PLATFORMS=cpu \
  .venv/bin/python -m unittest tests_forward_PINN.test_forward_ablation -v
```

## One run

The optimizer budgets are required explicitly.  This example is illustrative;
choose the final report budget before starting:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
  .venv/bin/python tests_forward_PINN/forward_ablation.py run \
  --config tests_forward_PINN/forward_2circles_1200_m0.json \
  --variant adapted_total \
  --seed 0 \
  --adam-steps 30000 \
  --lbfgs-steps 500
```

The generated run directory is exclusive.  Repeating the same configuration
refuses to overwrite it.

For the homogeneous analytic control, replace the configuration only:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
  .venv/bin/python tests_forward_PINN/forward_ablation.py run \
  --config tests_forward_PINN/forward_homogeneous_1200_m0.json \
  --variant adapted_total \
  --seed 0 \
  --adam-steps 30000 \
  --lbfgs-steps 500
```

For `adapted_scattered`, the analytic target is `u_s = 0`; reported L2/H1
errors nevertheless compare the reconstructed physical total field `u0 + us`
to the exact mode `u0`, as for the other variants.

The same statement applies to `adapted_scattered_RAD`.

## One-seed RAD comparison

The campaign launches the five variants sequentially in fresh Python
subprocesses.  Its default is deliberately the single seed `0` for this first
RAD screening:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
  .venv/bin/python tests_forward_PINN/forward_ablation.py campaign \
  --config tests_forward_PINN/forward_2circles_1200_m0.json \
  --adam-steps 30000 \
  --lbfgs-steps 500
```

No campaign is launched automatically by tests or imports.

After changing only the RAD protocol, rerun just the two affected variants with:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
  .venv/bin/python tests_forward_PINN/forward_ablation.py campaign \
  --config tests_forward_PINN/forward_circlebottomright_1200_m0.json \
  --variants adapted_total_RAD,adapted_scattered_RAD \
  --adam-steps 50000 \
  --lbfgs-steps 0
```

Each run contains:

- `manifest.json` and `effective_config.json`;
- `loss_history.csv` at the inexpensive fixed-monitor cadence;
- `gradient_history.csv` at the configured optimizer-batch cadence;
- `fem_metrics.csv` at the less frequent reference cadence (the historical
  filename is retained for plotting compatibility);
- `rad_resampling_history.csv` for RAD runs with at least one probability
  refresh;
- `checkpoint_final.npz` and `checkpoint_best_monitor.npz`;
- `summary.json`, including synchronized Adam and L-BFGS times.

By default, fixed losses are evaluated every 200 Adam steps and every 100
L-BFGS steps.  Raw gradient statistics are evaluated every 500 Adam steps and
every 250 L-BFGS steps.  Reference L2/H1 errors are evaluated every 1000 Adam
steps and every 500 L-BFGS steps, plus the initial state, optimizer transition,
and final state.  These intervals are editable in the JSON configuration.

Synchronized optimizer updates, including the inexpensive per-step bootstrap,
contribute to `optimizer_seconds`.  Candidate generation, residual evaluation,
and probability/CDF construction contribute to `rad_resampling_seconds`; their
sum is `training_seconds`.  Compilation, reference inference, matrix norms,
checkpoint writes, and plots are excluded.

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
reference errors versus measured optimizer-plus-RAD time, and training times,
together with `runs.csv`, `aggregate.csv`, and concatenated histories.  Thin
lines show individual seeds;
the main iteration-based lines and bands show the arithmetic mean plus or minus
one sample standard deviation.  L2/H1 points retain their native, coarser
cadence.

All metrics compare the physical total field.  For the scattered model this
means evaluating `u0 + us`.  Relative norms use the mass and stiffness matrices
of either the fine FEM mesh or the independent analytic triangulation:

```text
relative L2 = sqrt(e* M e) / sqrt(u_ref* M u_ref)
relative H1 = sqrt(e* (M + K) e) / sqrt(u_ref* (M + K) u_ref)
```

The finest solution is a numerical reference, not an exact solution.  Its
difference from the preceding 44209-DOF mesh is approximately 0.217% in L2 and
2.60% in H1.
