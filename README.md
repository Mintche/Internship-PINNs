# Internship PINNs — 2D acoustic waveguide

Research code for a rectangular acoustic waveguide:

- `FEM/`: quadratic (`P2`) finite-element solver and reference-data generator;
- `inverse_PINN/`: all-at-once reconstruction of pressure fields and material;
- `tests_forward_PINN/`: fixed-material benchmark for field architectures;
- `tests/`: unit and integration tests.

Run every command below from the repository root.

## Installation

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The requirements install the standard JAX build. For GPU execution, install
the JAX/JAXlib build matching the CUDA version of the machine. To force CPU:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -c "import jax; print(jax.devices())"
```

## Generate reference data

Build the FEM generator:

```bash
make -C FEM
```

This example creates the files used by
`inverse_PINN/configs/circlebottomright_1200_A.json`:

```bash
FEM/generate_pinn_data.x \
  --mesh FEM/data/test_us_defaut_circlebottomright.msh \
  --defectname circlebottomright \
  --freqs 1200 \
  --modes 0,1,2,3 \
  --outputdir FEM/pinn_data \
  --c0 340 \
  --tag-contrasts 2:0.8 \
  --incidence -1 \
  --numberofdatapoints 31
```

Generated data are written to `FEM/pinn_data/` and ignored by Git. See
[`FEM/README.md`](FEM/README.md) for the mesh tags, material options, source
directions, and output schemas.

## Run the inverse PINN

The checked-in inverse configurations use full training budgets. Copy and
reduce one before using it as a quick smoke test.

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m inverse_PINN.cli run \
  --config inverse_PINN/configs/circlebottomright_1200_A.json \
  --variant fourier_total \
  --seed 0
```

Run several variants and seeds as an isolated campaign:

```bash
.venv/bin/python -m inverse_PINN.cli campaign \
  --config inverse_PINN/configs/circlebottomright_1200_A.json \
  --variants fourier_total,fourier_scattered \
  --seeds 0,1
```

Results are created under `inverse_PINN/results/`; existing run directories
are never overwritten. Generate figures after a campaign with:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m inverse_PINN.plot_results \
  --campaign-root inverse_PINN/results/circlebottomright_1200_A_campaign \
  --cosines
```

Variants, configuration fields, and artifacts are summarized in
[`inverse_PINN/README.md`](inverse_PINN/README.md).

## Run the forward benchmark

The homogeneous controls need no FEM files, so they are the simplest way to
check the workflow:

```bash
JAX_PLATFORMS=cpu .venv/bin/python tests_forward_PINN/forward_ablation.py run \
  --config tests_forward_PINN/forward_homogeneous_600_m0_manual.json \
  --variant fourier_modified_total \
  --seed 0 \
  --adam-steps 100
```

For campaigns, heterogeneous reference data, variants, and plotting, see
[`tests_forward_PINN/README.md`](tests_forward_PINN/README.md).

## Tests

```bash
MPLCONFIGDIR=/tmp/matplotlib JAX_PLATFORMS=cpu \
  .venv/bin/python -m pytest -q
```

Tests use small fixtures and smoke runs; they do not launch a full scientific
campaign.
