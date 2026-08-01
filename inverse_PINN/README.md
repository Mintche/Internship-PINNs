# Modular inverse PINN

`inverse_PINN` is the current all-at-once inverse workflow. It jointly learns
the complex pressure fields and a bounded slowness/material network for a set
of frequency, mode, and incidence cases. Training and plotting are separate:
the training module never imports Matplotlib.

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
terms as recorded in the CSV histories; the DtN contribution is also split into
left and right columns. The material objective can include per-case PDE terms
and total variation. L-BFGS budgets are supported separately for the field and
material phases.

## Configuration

`config.py` validates the JSON strictly: unknown or missing keys, invalid
geometry/material regions, evanescent incident cases, and missing data files
are rejected before training. Paths are resolved relative to the repository
root unless they are absolute.

The `training_packages` list defines the curriculum. Each package declares its
active cases plus warm-up and inverse budgets. The configuration also controls
collocation counts, Sobol sampling, logging intervals, snapshot fractions,
learning-rate schedules, loss weights, adaptive weights, and TV regularization.

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

- `pressure_loss_history.csv`, `material_loss_history.csv`;
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

The command creates PDF field, pressure-misfit, reconstructed-celerity,
celerity-snapshot, and optional gradient-cosine figures inside each package's
`figures/` directory. The `--cosines` option reads the snapshot diagnostics
already saved during training; it does not recompute training gradients.
