# Internship PINNs — 2D Acoustic Waveguide

This repository brings together a P2 FEM solver and a JAX PINN for inverting
the wave speed in a rectangular acoustic waveguide.

## Project Structure

```text
Internship-PINNs/
├── FEM/                          # P2 solver and data generation
├── pinn_waveguide_2d/            # PINN training and checkpoints
│   ├── checkpoints/
│   ├── data/
│   └── pinn_waveguide_multi_mode.py
├── tools/
│   ├── data_loader.py            # Load boundaries, fields, and matrices
│   ├── uv_checkpoint.py          # Save and evaluate UV/M networks
│   ├── compare_pinn_fem.py       # Nodal FEM–PINN comparison
│   ├── compare_pinn_pinn.py      # Nodal PINN–PINN comparison
│   └── compare_sound_speed.py    # Sound-speed reconstruction and misfit
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

The `Mass_matrix_*.csv` and `Stiff_matrix_*.csv` files contain the lower
triangles of the matrices in COO format. The `fem_field_*.csv` file contains
the complex field at all P2 degrees of freedom, in the same RCM ordering as
the matrices.

## Multi-Mode Training

Configure the defect, ratio, frequencies, and modes at the beginning of
`pinn_waveguide_2d/pinn_waveguide_multi_mode.py`, then run:

```bash
python3 pinn_waveguide_2d/pinn_waveguide_multi_mode.py
```

The script saves the networks in `pinn_waveguide_2d/checkpoints/`. Version 2 of
the portable NPZ checkpoint contains the UV and sound-speed-network weights,
Fourier basis, `sigma` parameters, `U_norm`, speed bounds, inference
architectures, best validation losses, random seed, Git state, dependency
versions, and physical metadata. Pickle is not used. Version-1 UV checkpoints
remain readable by the field-comparison tools.

The manifest can be inspected with:

```bash
python3 tools/uv_checkpoint.py inspect \
  --checkpoint pinn_waveguide_2d/checkpoints/uv_barhalfup_ratio0p8_modes0_1_2_freqs600.npz
```

## FEM–PINN Comparison

```bash
python3 tools/compare_pinn_fem.py \
  --checkpoint pinn_waveguide_2d/checkpoints/uv_barhalfup_ratio0p8_modes0_1_2_freqs600.npz \
  --mass-matrix FEM/pinn_data/Mass_matrix_barhalfup.csv \
  --stiffness-matrix FEM/pinn_data/Stiff_matrix_barhalfup.csv \
  --fem-field FEM/pinn_data/fem_field_barhalfup_ratio0p8.csv
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

## Sound-Speed Comparison

The ground-truth map is intentionally user-defined. Before running the tool,
edit `GROUND_TRUTH_NAME` and `ground_truth_sound_speed(x, y, checkpoint)` at
the beginning of `tools/compare_sound_speed.py`. The function returns the full
sound-speed map at physical coordinates, so it can assign independent speeds
to any number of defects. The default example uses `c0` and `contrast_ratio`
from the checkpoint for `barhalf`.

```bash
python3 tools/compare_sound_speed.py \
  --checkpoint pinn_waveguide_2d/checkpoints/uv_barhalf_ratio0p8_modes0_1_2_freqs600.npz \
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

## Tests

The checkpoint compatibility and sound-speed plotting tests do not require
JAX and can be run with:

```bash
python3 -m unittest discover -s tests -v
```
