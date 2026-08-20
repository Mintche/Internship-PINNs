# Controlled forward PINN benchmark

This workflow compares pressure-field architectures while keeping the
material map fixed. The FEM or analytic field is used only for metrics, never
in the physics loss or checkpoint selection.

Run every command from the repository root.

## Reference configurations

The homogeneous configurations need no generated data:

- `forward_homogeneous_600_m0_uniform.json`;
- `forward_homogeneous_600_m0_manual.json`;
- `forward_homogeneous_1200_m0_manual.json`.

The heterogeneous configuration uses a circle at `(0.2, 0.2)`, radius `0.1`,
speed ratio `0.8`, at 1200 Hz. Generate its FEM reference with:

```bash
make -C FEM
FEM/generate_pinn_data.x \
  --mesh FEM/data/test_us_defaut_circlebottomright.msh \
  --defectname circlebottomrightforward \
  --freqs 1200 \
  --modes 0 \
  --outputdir FEM/pinn_data \
  --c0 340 \
  --tag-contrasts 2:0.8 \
  --incidence -1
```

## Variants

Five base variants are available:

- `classical_total`;
- `fourier_total` and `fourier_modified_total`;
- `fourier_scattered` and `fourier_modified_scattered`.

Append `_adweights` for adaptive loss weights, `_rad` for residual-based
adaptive sampling, or `_rad_adweights` for both. This gives 20 accepted names;
`--help` prints the complete list.

All variants use Adam only. Fourier scales follow their configured cosine
schedule and are then frozen. Loss triplets use the order
`[PDE, Neumann, DtN]`; the corresponding JSON sections configure static or
adaptive weights and RAD sampling.

## One run

`--adam-steps` is always explicit. This small homogeneous run is suitable for
a quick workflow check:

```bash
JAX_PLATFORMS=cpu .venv/bin/python tests_forward_PINN/forward_ablation.py run \
  --config tests_forward_PINN/forward_homogeneous_600_m0_manual.json \
  --variant fourier_modified_total \
  --seed 0 \
  --adam-steps 100
```

For the heterogeneous benchmark, replace the configuration with
`tests_forward_PINN/forward_circlebottomright_1200_m0.json` and choose the
scientific training budget explicitly.

## Campaign and figures

Campaigns run variants sequentially in isolated subprocesses. If omitted,
`--variants` selects the five base variants and `--seeds` defaults to `0`.

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
  .venv/bin/python tests_forward_PINN/forward_ablation.py campaign \
  --config tests_forward_PINN/forward_circlebottomright_1200_m0.json \
  --variants classical_total,fourier_total,fourier_scattered \
  --seeds 0,1 \
  --adam-steps 30000
```

The command prints the new `campaign_cfg...` directory. Aggregate it with:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python tests_forward_PINN/plot_campaign.py \
  --campaign-root tests_forward_PINN/results/campaign_cfgXXXXXXXXXXXX \
  --output-dir tests_forward_PINN/results/report_figures
```

Run and figure directories are exclusive and are never overwritten.

## Results and tests

Each run records its manifest and effective configuration, loss/gradient/FEM
histories, final and best-monitor checkpoints, timing, and a JSON summary. RAD
and adaptive-weight variants add their respective histories. Campaign plots
also write per-run and aggregated CSV tables.

Reported relative L2 and H1 errors compare the physical total field (including
`u0 + us` for scattered variants) with the FEM field or exact homogeneous
mode, using the corresponding mass and stiffness matrices.

```bash
MPLCONFIGDIR=/tmp/matplotlib JAX_PLATFORMS=cpu \
  .venv/bin/python -m unittest tests.test_forward_ablation -v
```
