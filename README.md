# Internship PINNs — 2D Acoustic Waveguide

This repository contains a quadratic (`P2`) finite-element solver and two JAX
PINN workflows for a rectangular acoustic waveguide:

- `inverse_PINN/` is the current modular all-at-once inverse solver. It learns
  the pressure fields and the slowness/material map together.
- `tests_forward_PINN/` is a controlled forward benchmark. The material is
  fixed, so it isolates field-representation and optimization errors.
- `FEM/` generates the reference fields, boundary traces, mesh coordinates,
  and geometric matrices consumed by the Python code.

The repository is research code. The small inverse configuration is a smoke
test, not a production training budget.

## Repository layout

```text
.
├── FEM/
│   ├── data/                         # Gmsh v2 meshes
│   ├── include/                      # FEM and matrix headers
│   └── src/generate_pinn_data.cpp    # FEM/PINN data generator
├── inverse_PINN/                     # Current inverse-PINN package
│   ├── configs/circlebottomright_smoke.json
│   ├── cli.py                        # run/campaign entry points
│   └── plot_results.py               # offline figures
├── tests_forward_PINN/               # Fixed-material forward benchmark
│   ├── forward_ablation.py
│   └── plot_campaign.py
├── tools/                            # Data loaders and diagnostics
├── tests/                            # Unit and integration tests
└── requirements.txt
```

The older `pinn_waveguide_2d/` training scripts and their comparison tools are
no longer part of the active workflow. Do not use paths from historical notes
unless the corresponding file exists in the current checkout.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For GPU execution, install the JAX/JAXlib build matching the CUDA version on
the machine. The commands below force CPU execution where reproducibility is
more important than speed:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -c "import jax; print(jax.devices())"
```

## Generate FEM data

The generator is described in detail in [`FEM/README.md`](FEM/README.md). Build
it once from the repository root:

```bash
make -C FEM
```

The following command creates the files expected by the checked-in inverse
smoke configuration:

```bash
FEM/generate_pinn_data.x \
  --mesh FEM/data/test_us_defaut_circlebottomright_ref.msh \
  --defectname circlebottomrightforward \
  --freqs 1200 \
  --modes 0 \
  --outputdir FEM/pinn_data \
  --c0 340 \
  --contrast 0.8 \
  --numberofdatapoints 31
```

The output directory is created if necessary. Generated data are intentionally
ignored by Git. A generated dataset contains the two boundary CSVs, the FEM
field, the lower-triangular mass and stiffness matrices, and the RCM-reordered
P2 node/triangle CSVs.

For multiple material tags, replace `--contrast` with for example
`--tag-contrasts 2:0.8,3:0.9`. For a continuous reconstructed map, use
`--nodal-sound-speed PATH`; the CSV must contain every reordered P2 degree of
freedom with the header `node_id,x,y,c`. Use `--incidence -1,1` to export both
source directions; the default is incidence `-1` (left-to-right).

## Run the modular inverse PINN

The CLI has two commands. A run trains one variant and seed; a campaign runs
the Cartesian product of explicitly provided variants and seeds:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m inverse_PINN.cli run \
  --config inverse_PINN/configs/circlebottomright_smoke.json \
  --variant fourier_total \
  --seed 0

JAX_PLATFORMS=cpu .venv/bin/python -m inverse_PINN.cli campaign \
  --config inverse_PINN/configs/circlebottomright_smoke.json \
  --variants fourier_total,fourier_scattered \
  --seeds 0,1
```

Runs and campaigns refuse to overwrite an existing output directory. Each run
records its effective configuration and writes one package per curriculum
stage. The accepted inverse variants are the two architectures
`fourier`/`fourier_modified`, the two formulations `total`/`scattered`, and any
ordered combination of `_field_adweights`, `_material_adweights`, and `_tv`.
This gives 32 canonical variant names.

Each package contains pressure and material loss histories, gradient and sigma
histories, adaptive-weight history, material snapshots, NPZ checkpoints,
pressure/material metrics, timing information, and a JSON summary. Training
does not import Matplotlib. Generate figures afterwards:

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python -m inverse_PINN.plot_results \
  --campaign-root inverse_PINN/results/circlebottomright_smoke_campaign \
  --cosines
```

The smoke configuration uses one tiny package and a few optimizer steps. Copy
it before a scientific run and update the geometry, data paths, curriculum,
sampling, loss weights, and optimizer budgets together.

## Run the controlled forward benchmark

This workflow keeps the material fixed and compares five base formulations:
classical total field, Fourier total field, modified Fourier total field,
Fourier scattered field, and modified Fourier scattered field. Adaptive loss
weights (`_adweights`), residual-based adaptive sampling (`_rad`), and their
combination are also supported.

Run one configuration:

```bash
JAX_PLATFORMS=cpu .venv/bin/python tests_forward_PINN/forward_ablation.py run \
  --config tests_forward_PINN/forward_homogeneous_600_m0_manual.json \
  --variant fourier_modified_total \
  --seed 0 \
  --adam-steps 100
```

Run an isolated campaign and then aggregate it:

```bash
JAX_PLATFORMS=cpu .venv/bin/python tests_forward_PINN/forward_ablation.py campaign \
  --config tests_forward_PINN/forward_circlebottomright_1200_m0.json \
  --variants classical_total,fourier_total,fourier_scattered \
  --seeds 0,1 \
  --adam-steps 30000

MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python \
  tests_forward_PINN/plot_campaign.py \
  --campaign-root tests_forward_PINN/results/campaign_cfgXXXXXXXXXXXX \
  --output-dir tests_forward_PINN/results/report_figures
```

See [`tests_forward_PINN/README.md`](tests_forward_PINN/README.md) for the
normalizations, reference metrics, adaptive weights, RAD sampling protocol,
and complete artifact list.

## Diagnostics and data conventions

`tools/data_loader.py` validates the CSV schema, including the `incidence`
column. `tools/experiment_manifest.py` provides stable configuration hashes
and overwrite guards. `tools/ground_truth.py` is the explicit registry used by
the legacy checkpoint diagnostics; an unknown geometry is rejected rather than
silently evaluated against the wrong mask. `tools/compare_boundary_fields.py`
compares scattered traces from two FEM exports. Gradient diagnostics for current
inverse runs are recorded by `inverse_PINN` during configured material snapshots and
plotted by `inverse_PINN.plot_results`.

All matrices are stored as lower-triangular COO data with zero-based indices.
The field and mesh files use the same RCM ordering. Boundary and field CSVs
contain one row per `(incidence, frequency, mode, point)` case.

## Tests

Use the project environment and a writable Matplotlib cache directory:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m pytest -q
```

The tests use small synthetic fixtures and smoke runs. They do not replace a
full scientific campaign. FEM integration tests build `FEM/generate_pinn_data.x`
locally and create their temporary outputs under the system temporary folder.
