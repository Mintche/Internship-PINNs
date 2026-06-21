# Internship PINNs - Acoustic Waveguide & FWI

This repository contains implementations of **Physics-Informed Neural Networks (PINNs)** applied to acoustic waveguide analysis and **Full Waveform Inversion (FWI)** for reconstructing material properties (celerity/slowness fields).

---

## Directory Structure

The repository is organized as follows:

```
Internship-PINNs/
├── waveguide_2d/                  # 2D Waveguide PINN codes
│   ├── data/                      # Boundary csv data files
│   │   └── pinn_boundary_*.csv
│   ├── fig/                       # Saved plots and figures
│   ├── data_loader.py             # Data loading utility for CSV files
│   ├── pinn_waveguide.py          # Field reconstruction PINN (standard formulation)
│   └── pinn_alternating.py        # Alternating minimization/training FWI PINN
├── notebooks/                     # Exploratory Jupyter notebooks
│   ├── Heat_eq_1D_test_PINN_DG.ipynb
│   ├── Heat_eq_1D_test_PINN_base.ipynb
│   ├── WaveGuide_PINN_scd.ipynb
│   ├── WaveGuide_PINN_simple.ipynb
│   └── WaveGuide_PINN_trird.ipynb
│   └── fig/                       # Figures for notebooks
├── requirements.txt               # Project dependencies
├── README.md                      # This documentation
└── .gitignore                     # Git ignored patterns (.venv, pycache, etc.)
```

---

## Core Features

- **Dynamic Path Resolution**: Scripts dynamically locate CSV datasets and save directories relative to their file locations, meaning you can execute them from any working directory without `FileNotFoundError` or relative path failures.
- **Alternating Optimization (`pinn_alternating.py`)**: A multi-cycle scheme that alternately fits the pressure wavefield $u(x,y)$ to the boundaries and satisfies the Helmholz PDE, then optimizes the slowness field $m(x,y)$ with Total Variation (TV) regularization.
- **Spectral Bias Control**: Fourier Features projection (`gamma`) is incorporated to resolve high-frequency fields more efficiently.

---

## Installation & Setup

1. **Virtual Environment**:
   It is recommended to use a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Dependencies**:
   Install the required python libraries using:
   ```bash
   pip install -r requirements.txt
   ```
   > [!NOTE]
   > The default installation will set up JAX on CPU. If you wish to run with GPU/CUDA support, please refer to the [JAX installation guidelines](https://github.com/google/jax#installation) to install the correct CUDA-enabled `jaxlib` version.

---

## Usage

You can run the python scripts directly from the workspace root (or any directory):

### 1. Standard 2D Waveguide PINN
```bash
python waveguide_2d/pinn_waveguide.py
```

### 2. Alternating FWI PINN
```bash
python waveguide_2d/pinn_alternating.py
```
Output figures will be saved automatically in the `waveguide_2d/fig/` subdirectory.