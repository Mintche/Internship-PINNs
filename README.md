# Internship PINNs — 2D Acoustic Waveguide

This repository brings together a P2 FEM solver and a JAX PINN for inverting
the wave speed in a rectangular acoustic waveguide.

## Project Structure

```text
Internship-PINNs/
├── FEM/                          # P2 solver, generation, and pinn_data/
├── pinn_waveguide_2d/            # PINN training and checkpoints
│   ├── checkpoints/
│   ├── pinn_waveguide_multi_modes.py
│   └── pinn_scattered_waveguide.py
├── tools/
│   ├── data_loader.py            # Load boundaries, fields, and matrices
│   ├── uv_checkpoint.py          # Save and evaluate UV/M networks
│   ├── us_checkpoint.py          # Save and evaluate scattered US/MS networks
│   ├── compare_pinn_fem.py       # Nodal FEM–PINN comparison
│   ├── compare_pinn_pinn.py      # Nodal PINN–PINN comparison
│   ├── compare_sound_speed.py    # Sound-speed reconstruction and misfit
│   ├── summarize_sound_speed_checkpoints.py
│   ├── summarize_total_field_diagnostics.py
│   ├── resolve_checkpoint_with_fem.py # Re-solve a reconstructed material with FEM
│   ├── analyze_scattered_gradient_conflicts.py
│   ├── compare_boundary_fields.py
│   ├── compare_scattered_pinn_fem.py
│   ├── compare_scattered_pinn_pinn.py
│   └── compare_scattered_sound_speed.py
├── notebooks/
└── requirements.txt
```

## Python Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a CUDA environment, install the `jaxlib` variant that matches the machine's
CUDA version.

## FEM Generation

Compilation, physical tags, the CLI, and CSV formats are described in
[`FEM/README.md`](FEM/README.md).

For multi-material defects, the FEM generator accepts one contrast ratio per
physical surface tag:

```bash
cd FEM
./generate_pinn_data.x \
  --mesh data/test_us_2triangles_diffcontrast.msh \
  --defectname triangles_diffcontrast \
  --freqs 600,1200 \
  --modes 0,1,2,3 \
  --outputdir pinn_data \
  --c0 340 \
  --tag-contrasts 2:0.8,3:0.9
```

The `Mass_matrix_*.csv` and `Stiff_matrix_*.csv` files contain the lower
triangles of the matrices in COO format. The `fem_field_*.csv` file contains
the complex field at all P2 degrees of freedom, in the same RCM ordering as
the matrices.

## Multi-Mode Training

The historical no-argument command still uses the defaults at the beginning of
`pinn_waveguide_2d/pinn_waveguide_multi_modes.py`:

```bash
python3 pinn_waveguide_2d/pinn_waveguide_multi_modes.py
```

For a report run, use the strict JSON configuration instead. Inspect the
effective configuration and its SHA-256 identifier without training:

```bash
JAX_PLATFORMS=cpu .venv/bin/python \
  pinn_waveguide_2d/pinn_waveguide_multi_modes.py \
  --config experiments/configs/inverse_halfbar_600_m012_total.json \
  --print-config
```

Then start one seed in a fresh, exclusive output directory:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 .venv/bin/python \
  pinn_waveguide_2d/pinn_waveguide_multi_modes.py \
  --config experiments/configs/inverse_halfbar_600_m012_total.json \
  --seed 0 \
  --output-root results/inverse_runs/total_halfbar_f600_m012_seed0 \
  --no-show-plots
```

The command refuses an existing output root. It archives the effective config in
`run_config.json`; figures and checkpoints go into subdirectories of that run.
Use a distinct root for every seed. The supplied JSON is a reproducible template
for the current total-field protocol, not by itself a causal total/scattered
ablation. It includes the optimizer schedules, data-weight ramp and gradient
clipping in addition to architecture, collocation, loss weights and budgets.

The no-argument run saves networks in `pinn_waveguide_2d/checkpoints/`; a
config-driven run uses `<output-root>/checkpoints/`. Version 2 of the portable
NPZ checkpoint contains the UV and sound-speed-network weights,
Fourier basis, `sigma` parameters, `U_norm`, speed bounds, inference
architectures, best validation losses, random seed, Git state, dependency
versions, and physical metadata. Pickle is not used. Version-1 UV checkpoints
remain readable by the field-comparison tools.

The manifest can be inspected with:

```bash
python3 tools/uv_checkpoint.py inspect \
  --checkpoint pinn_waveguide_2d/checkpoints/checkpoint_barhalf_ratio0p8_modes0_1_2_3_4_freqs600_1200.npz
```

## Scattered Training

The no-argument command preserves the historical defaults:

```bash
python3 pinn_waveguide_2d/pinn_scattered_waveguide.py
```

For a managed run, inspect and then execute the scattered template:

```bash
JAX_PLATFORMS=cpu .venv/bin/python \
  pinn_waveguide_2d/pinn_scattered_waveguide.py \
  --config experiments/configs/inverse_halfbar_600_m012_scattered.json \
  --print-config

XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 .venv/bin/python \
  pinn_waveguide_2d/pinn_scattered_waveguide.py \
  --config experiments/configs/inverse_halfbar_600_m012_scattered.json \
  --seed 0 \
  --output-root results/inverse_runs/scattered_halfbar_f600_m012_seed0 \
  --no-show-plots
```

This trains scattered fields `us` and the shared scattered-slowness map `ms`.
For every package after the first, the script first warms only the new
`(frequency, mode)` cases with Adam and the frozen best `ms` from the previous
package, then runs the joint inverse phase. The checkpoint is written as
`scattered_checkpoint_*.npz`. Boundary data are converted from total field to
scattered field by subtracting the incident mode before normalization.

Inspect or export a scattered checkpoint with:

```bash
python3 tools/us_checkpoint.py inspect \
  --checkpoint pinn_waveguide_2d/checkpoints/scattered_checkpoint_barhalf_ratio0p8_modes0_1_2_3_freqs600_1200.npz

python3 tools/us_checkpoint.py export \
  --checkpoint pinn_waveguide_2d/checkpoints/scattered_checkpoint_barhalf_ratio0p8_modes0_1_2_3_freqs600_1200.npz \
  --grid-map FEM/pinn_data/pinn_grid_barhalf.csv \
  --output pinn_waveguide_2d/predictions/scattered_total.csv \
  --field total
```

Compare scattered checkpoints with FEM or with another scattered checkpoint:

```bash
python3 tools/compare_scattered_pinn_fem.py \
  --checkpoint pinn_waveguide_2d/checkpoints/scattered_checkpoint_barhalf_ratio0p8_modes0_1_2_3_freqs600_1200.npz \
  --mass-matrix FEM/pinn_data/Mass_matrix_barhalf.csv \
  --stiffness-matrix FEM/pinn_data/Stiff_matrix_barhalf.csv \
  --fem-field FEM/pinn_data/fem_field_barhalf_ratio0p8.csv

python3 tools/compare_scattered_pinn_pinn.py \
  --checkpoint1 pinn_waveguide_2d/checkpoints/first_scattered.npz \
  --checkpoint2 pinn_waveguide_2d/checkpoints/second_scattered.npz \
  --mass-matrix FEM/pinn_data/Mass_matrix_barhalf.csv \
  --stiffness-matrix FEM/pinn_data/Stiff_matrix_barhalf.csv \
  --fem-field FEM/pinn_data/fem_field_barhalf_ratio0p8.csv
```

For scattered sound-speed diagnostics, the checkpoint `defect_name` selects a
geometry from `tools/ground_truth.py`. Unknown geometries are rejected until an
explicit definition is added to that registry:

```bash
python3 tools/compare_scattered_sound_speed.py \
  --checkpoint pinn_waveguide_2d/checkpoints/scattered_checkpoint_barhalf_ratio0p8_modes0_1_2_3_freqs600_1200.npz \
  --nx 201 --ny 121
```

Compare scattered FEM boundary outputs from two generated datasets:

```bash
python3 tools/compare_boundary_fields.py \
  --left-a FEM/pinn_data/pinn_boundary_left_barhalf_ratio0p8.csv \
  --right-a FEM/pinn_data/pinn_boundary_right_barhalf_ratio0p8.csv \
  --left-b FEM/pinn_data/pinn_boundary_left_triangles_diffcontrast.csv \
  --right-b FEM/pinn_data/pinn_boundary_right_triangles_diffcontrast.csv \
  --freqs 600,1200 \
  --modes 0,1,2,3 \
  --label-a barhalf \
  --label-b triangles_diffcontrast \
  --output-dir results/boundary_comparison
```

The tool subtracts the incident mode, prints relative/RMS/max differences for
the scattered boundary traces, and reports both a reference-relative and a
symmetric relative L2. With `--output-dir`, it writes a metrics CSV and one PDF
per case; without it, the Matplotlib figures are shown interactively.

## FEM–PINN Comparison

```bash
python3 tools/compare_pinn_fem.py \
  --checkpoint pinn_waveguide_2d/checkpoints/checkpoint_barhalf_ratio0p8_modes0_1_2_3_4_freqs600_1200.npz \
  --mass-matrix FEM/pinn_data/Mass_matrix_barhalf.csv \
  --stiffness-matrix FEM/pinn_data/Stiff_matrix_barhalf.csv \
  --fem-field FEM/pinn_data/fem_field_barhalf_ratio0p8.csv
```

The `--frequency` and `--mode` options can be used to filter the cases. If
`--stiffness-matrix` is omitted, only the L2 misfit is computed.

For the complex error `e = U_PINN - U_FEM`, the tool prints the absolute and
relative norms:

```text
||e||L2 = sqrt(Re(e* M e))
||e||H1 = sqrt(Re(e* (M + K) e))
```

Each relative value is divided by the corresponding FEM norm. For every
`(frequency, mode)` pair, an interactive figure displays `Re(U FEM)`,
`Re(U PINN)`, and the signed difference. The maps are rasterized, use physical
coordinates, and are not saved automatically; they can be saved from the
Matplotlib interface.

### Re-solving a reconstructed material map

`compare_pinn_fem.py` compares the auxiliary field learned by the PINN with the
reference FEM field. To test the stronger all-at-once consistency question, use
`resolve_checkpoint_with_fem.py`: it evaluates the checkpoint's reconstructed
sound speed at every P2 degree of freedom, asks the FEM solver to solve that
material independently, and compares the three fields/traces `PINN`, `FEM of the
reconstructed material`, and `FEM truth`.

Build the generator and run one total-field checkpoint at 600 Hz, modes 0--2:

```bash
make -C FEM

MPLCONFIGDIR=/tmp/matplotlib JAX_PLATFORMS=cpu .venv/bin/python \
  tools/resolve_checkpoint_with_fem.py \
  --checkpoint pinn_waveguide_2d/checkpoints/uv_barhalf_ratio0p8_modes0_1_2_3_4_freqs600_1200_1per.npz \
  --formulation total \
  --mesh FEM/data/test_us_barhalf_centree.msh \
  --fem-field-truth FEM/pinn_data/fem_field_barhalf_ratio0p8.csv \
  --mass-matrix FEM/pinn_data/Mass_matrix_barhalf.csv \
  --stiffness-matrix FEM/pinn_data/Stiff_matrix_barhalf.csv \
  --boundary-left-truth FEM/pinn_data/pinn_boundary_left_barhalf_ratio0p8.csv \
  --boundary-right-truth FEM/pinn_data/pinn_boundary_right_barhalf_ratio0p8.csv \
  --frequency 600 --mode 0 --mode 1 --mode 2 \
  --output-dir results/fem_checkpoint_resolutions/total_barhalf_f600
```

The selected cases must form a complete frequency-by-mode product because that
is the FEM generator's CLI convention. The output directory is exclusive and is
never overwritten. It contains the nodal material CSV, the independent FEM
outputs, `metrics.csv`, and a manifest with hashes of every input. This command is
a diagnostic, not a training run; its results should be interpreted only after
checking `metrics.csv` and the manifest.

## PINN–PINN Comparison

```bash
python3 tools/compare_pinn_pinn.py \
  --checkpoint1 pinn_waveguide_2d/checkpoints/first.npz \
  --checkpoint2 pinn_waveguide_2d/checkpoints/second.npz \
  --mass-matrix FEM/pinn_data/Mass_matrix_barhalf.csv \
  --stiffness-matrix FEM/pinn_data/Stiff_matrix_barhalf.csv \
  --fem-field FEM/pinn_data/fem_field_barhalf_ratio0p8.csv
```

The FEM field supplies the common physical P2 node coordinates. As in the
FEM–PINN tool, frequency and mode filters are optional.

To build one CSV containing both interior FEM errors and boundary-data errors
for several archived total-field checkpoints, use:

```bash
python3 tools/summarize_total_field_diagnostics.py \
  --checkpoint constrained=pinn_waveguide_2d/checkpoints/uv_barhalf_ratio0p8_modes0_1_2_freqs600_1per.npz \
  --checkpoint free=pinn_waveguide_2d/checkpoints/uv_barhalf_ratio0p8_modes0_1_2_freqs600_free.npz \
  --mass-matrix FEM/pinn_data/Mass_matrix_barhalf.csv \
  --stiffness-matrix FEM/pinn_data/Stiff_matrix_barhalf.csv \
  --fem-field FEM/pinn_data/fem_field_barhalf_ratio0p8.csv \
  --boundary-left FEM/pinn_data/pinn_boundary_left_barhalf_ratio0p8.csv \
  --boundary-right FEM/pinn_data/pinn_boundary_right_barhalf_ratio0p8.csv \
  --frequency 600 \
  --output results/halfbar_total_field_diagnostics.csv
```

The boundary errors use the full complex traces on both ports. Both total-field
and scattered-field relative errors are reported; the latter is computed after
subtracting the analytic incident mode.

## Sound-Speed Comparison

The ground-truth map is selected from `tools/ground_truth.py` using the
checkpoint `defect_name`, `c0`, and `contrast_ratio`. Unknown or multi-material
geometries must be added explicitly before metrics can be computed; the tool
does not silently fall back to another defect.

```bash
python3 tools/compare_sound_speed.py \
  --checkpoint pinn_waveguide_2d/checkpoints/checkpoint_barhalf_ratio0p8_modes0_1_2_3_4_freqs600_1200.npz \
  --nx 201 --ny 121
```

The primary score is the anomaly-relative L1, normalized by
`sum(|c_GT - c0|)`. The tool also reports MAE separately in the anomalous and
background regions, the homogeneous `c=c0` baseline, and improvement over that
baseline. Global L1, RMSE, relative L2, and maximum absolute error are retained
as secondary diagnostics. The three rasterized figures show the reconstructed
speed, the ground-truth/reconstruction comparison with a shared scale, and the
absolute error `|c_PINN - c_GT|`. A version-2 checkpoint containing `layers_m`
is required.

Generate the machine-readable inventory and the report-ready diagnostic plot
for every archived total/scattered checkpoint with:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python \
  tools/summarize_sound_speed_checkpoints.py
```

The command writes `results/checkpoint_sound_speed_metrics.csv` and
`results/checkpoint_monitor_vs_reconstruction.pdf`. Format-v1 UV checkpoints
without a stored sound-speed network remain listed with an explicit reason but
are not scored.

For a fixed scattered checkpoint, measure the per-acquisition PDE gradients of
the coefficient network and their pairwise cosines with:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python tools/analyze_scattered_gradient_conflicts.py \
  --checkpoint pinn_waveguide_2d/checkpoints/scattered_checkpoint_barhalf_ratio0p8_modes0_1_2_3_freqs500_1000.npz \
  --n-pde 4096
```

This is a checkpoint-local diagnostic. It does not replace gradient snapshots
recorded during training when making a causal optimization claim.

## Tests

The checkpoint compatibility and sound-speed plotting tests do not require
JAX and can be run with:

```bash
python3 -m unittest discover -s tests -v
```

The Python tooling expects the packages in `requirements.txt`: `jax`, `jaxlib`,
`optax`, `numpy`, `pandas`, and `matplotlib`. If a test import fails with
`ModuleNotFoundError` for one of those packages, install the requirements in the
active environment first.
