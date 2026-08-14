# Modular inverse PINN

`inverse_PINN` is the current all-at-once inverse workflow. The parameters of
all active pressure networks are packed into one pytree and updated from their
mean loss with a single optimizer state. It jointly learns those complex
pressure fields and a bounded slowness/material network. Training and plotting
are separate: the training module never imports Matplotlib.

## Commands

Generate the FEM files referenced by the JSON configuration first; see
[`../FEM/README.md`](../FEM/README.md). Then run one variant and seed:

```bash
python -m inverse_PINN.cli run \
  --config inverse_PINN/configs/circlebottomright_smoke.json \
  --variant fourier_total \
  --seed 0
```

Run a complete variant/seed campaign:

```bash
python -m inverse_PINN.cli campaign \
  --config inverse_PINN/configs/circlebottomright_smoke.json \
  --variants fourier_total,fourier_scattered \
  --seeds 0,1
```

Both commands reject an existing output directory. The campaign creates
`<output_root>/<config-stem>_campaign/runs/`; each run has one directory per
training package. The checked-in configuration is deliberately a tiny smoke
test and should not be used as a scientific budget without modification.

## Variants

Variant names are strict and token-ordered:

- architecture: `fourier` or `fourier_modified`;
- formulation: `total` or `scattered`;
- optional suffixes, in this order: `_field_adweights`,
  `_material_adweights`, `_tv`.

For example,
`fourier_modified_scattered_field_adweights_material_adweights_tv` enables all
three optional mechanisms. The two architectures, two formulations, and eight
modifier combinations produce 32 accepted names. Fourier scales are trained
for the configured fraction of Adam updates and then frozen.

The four field loss components are PDE, Neumann, combined DtN, and boundary-data
terms. Monitoring records only their package-wide means and the common pressure
objective; it does not recompute or save one pressure loss per acquisition. The
material history is global as well. Per-acquisition work is reserved for the
explicit gradient diagnostics, snapshots and final FEM metrics. The material
objective can include per-case PDE terms and total variation. L-BFGS budgets
are supported separately for the field and material phases.

## Configuration

`config.py` validates the JSON strictly: unknown or missing keys, invalid
geometry/material regions, evanescent incident cases, and missing data files
are rejected before training. Paths are resolved relative to the repository
root unless they are absolute.

The `training_packages` list defines the curriculum. Each package declares its
active cases plus warm-up and inverse budgets. The configuration also controls
collocation counts, Sobol sampling, logging intervals, snapshot fractions,
learning-rate schedules, loss weights, adaptive weights, and TV regularization.
`optimization.data_transition_steps` fixes the number of Adam updates used to
ramp the data-loss factor from `data_initial_factor` to one, independently of
the total inverse budget.
The field and material Adam rates stay constant through
`optimization.cosine_decay_start`, follow a cosine decay until
`optimization.consine_decay_stop`, then remain at
`optimization.cosine_decay_alpha` times their initial values. This schedule is
restarted for every warm-up or inverse Adam phase; the sigma schedule remains
independent.

Console progress reports show the current global pressure objective and the
package-wide mean PDE, boundary (`Neumann + DtN`), and data components. The
three displayed components are unweighted; `global` uses the current configured
or adaptive weights.

The default smoke configuration expects these generated files in
`FEM/pinn_data/`:

```text
pinn_boundary_left_circlebottomrightforward_ratio0p8.csv
pinn_boundary_right_circlebottomrightforward_ratio0p8.csv
fem_field_circlebottomrightforward_ratio0p8.csv
Mass_matrix_circlebottomrightforward.csv
Stiff_matrix_circlebottomrightforward.csv
mesh_nodes_circlebottomrightforward.csv
mesh_triangles_circlebottomrightforward.csv
```

## Run artifacts and plotting

Every package writes the following main artifacts:

- `pressure_common_loss_history.csv` et `material_loss_history.csv`;
- `pressure_gradient_history.csv`, `material_snapshot_diagnostics.csv`, and
  `material_gradient_cosines.csv`;
- `adweights_history.csv`, `sigma_history.csv`, and `timing.csv`;
- `pressure_weights_best.npz` and `slowness_weights_best.npz`;
- `celerity_snapshots.npz`, `pressure_metrics.csv`,
  `celerity_metrics.json`, and `summary.json`.

Plot completed runs offline:

```bash
MPLCONFIGDIR=/tmp/matplotlib python -m inverse_PINN.plot_results \
  --campaign-root inverse_PINN/results/circlebottomright_smoke_campaign \
  --cosines
```

The command creates PDF field, pressure-misfit, common-pressure-loss,
reconstructed-celerity, celerity-snapshot, and optional gradient-cosine figures
inside each package's `figures/` directory. The `--cosines` option reads the
snapshot diagnostics already saved during training; it does not recompute
training gradients.

JAX's persistent compilation cache is enabled automatically in
`inverse_PINN/cache/` (ignored by Git). Consequently, later processes can reuse
compatible executables. A change in shapes, architecture, variant, platform,
JAX/XLA version, or relevant static configuration still requires compilation.
`JAX_COMPILATION_CACHE_DIR` can be set before launch to use another directory.
