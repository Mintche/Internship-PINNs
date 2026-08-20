# Modular inverse PINN

`inverse_PINN` jointly learns one complex pressure field per acquisition and a
bounded slowness/material map. Training and plotting are separate; training
does not import Matplotlib.

Run every command from the repository root.

## Prepare data

Each JSON configuration names its boundary traces, FEM reference, matrices,
and reordered P2 mesh. Generate those files first with the tool documented in
[`../FEM/README.md`](../FEM/README.md).

For example, `circlebottomright_1200_A.json` expects data generated with
`--defectname circlebottomright`, frequency `1200`, modes `0,1,2,3`, incidence
`-1`, and `--tag-contrasts 2:0.8`.

## Run

One variant and seed:

```bash
.venv/bin/python -m inverse_PINN.cli run \
  --config inverse_PINN/configs/circlebottomright_1200_A.json \
  --variant fourier_total \
  --seed 0
```

A Cartesian product of variants and seeds:

```bash
.venv/bin/python -m inverse_PINN.cli campaign \
  --config inverse_PINN/configs/circlebottomright_1200_A.json \
  --variants fourier_total,fourier_scattered \
  --seeds 0,1
```

Both commands reject an existing output directory. A single run is written to
`<output_root>/<config-stem>/<variant>_seed<seed>/`; a campaign is written to
`<output_root>/<config-stem>_campaign/runs/`.

The checked-in configurations use full optimization and sampling budgets.
Copy one and reduce its `training_packages`, sampling sizes, and logging grid
before using it as a smoke test.

## Variants

Canonical names contain, in order:

1. architecture: `fourier` or `fourier_modified`;
2. formulation: `total` or `scattered`;
3. optional `_field_adweights`, `_material_adweights`, and `_tv` suffixes in
   that order.

For example,
`fourier_modified_scattered_field_adweights_material_adweights_tv` enables all
three modifiers. The combinations produce 32 accepted variants.

## Configuration notes

Configuration parsing is strict: unknown or missing keys, missing files,
invalid material regions, and evanescent incident cases are rejected before
training. Relative paths are resolved from the repository root.

`training_packages` defines the acquisition curriculum and the warm-up/inverse
Adam and L-BFGS budgets. Field loss weights use the order
`[PDE, Neumann, DtN, boundary data]`. `data_transition_steps` ramps the data
term during inverse Adam training. The learning-rate schedule uses the JSON
key `consine_decay_stop` (spelling kept for compatibility).

Adaptive-weight gradient evaluators are compiled before the comparable
training timer starts. Their update intervals directly affect cost; the
checked-in configurations use `update_interval_adam: 500`.

Fourier scales are trained for `sigma_decay_fraction` of each Adam phase and
then frozen. The persistent JAX cache defaults to `inverse_PINN/cache/`; set
`JAX_COMPILATION_CACHE_DIR` before launch to use another directory.

## Results and figures

Each run contains a manifest and summary, then one `packages/pkg.../`
directory per curriculum stage. A package stores loss, gradient, adaptive
weight, sigma and timing histories; best pressure/material checkpoints;
celerity snapshots; and final pressure/celerity metrics.

Generate figures after training:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m inverse_PINN.plot_results \
  --campaign-root inverse_PINN/results/circlebottomright_1200_A_campaign \
  --cosines
```

The command searches below the supplied root and creates a `figures/`
directory in each package. `--cosines` plots diagnostics already recorded
during training; it does not recompute gradients.

## Tests

```bash
MPLCONFIGDIR=/tmp/matplotlib JAX_PLATFORMS=cpu \
  .venv/bin/python -m pytest -q tests/test_inverse_pinn_*.py
```
