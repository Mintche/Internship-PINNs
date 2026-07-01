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
│   ├── uv_checkpoint.py          # Save and evaluate UV networks
│   └── compare_pinn_fem.py       # Nodal comparison and L2/H1 misfits
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

The script saves the UV networks in `pinn_waveguide_2d/checkpoints/`. The NPZ
checkpoint contains the weights, Fourier basis, `sigma` parameters, `U_norm`,
and the physical geometry. Its contents can be inspected with:

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
