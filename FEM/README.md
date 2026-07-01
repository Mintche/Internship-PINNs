# FEM Data Generator for the PINN

This directory contains a P2 FEM solver for a rectangular acoustic waveguide
and the `generate_pinn_data.x` executable. The generator solves the Helmholtz
problem for multiple frequencies and incident modes, then exports the complex
traces used by the PINN along with the P2 geometric matrices.

## Expected Mesh

The mesh must use the Gmsh v2 ASCII format and the following physical groups:

- background surface: tag `1`;
- defect: tag `2`;
- left port at `x=-L`: tag `11`;
- right port at `x=+L`: tag `12`;
- top and bottom boundaries: tags `13` and `14`.

The domain must be horizontally centered, and both ports must span the full
height of the waveguide.

## Building

With Make:

```bash
make
```

With CMake:

```bash
cmake -S . -B cmake-build
cmake --build cmake-build
```

Make creates `./generate_pinn_data.x`; CMake creates
`cmake-build/generate_pinn_data.x`.

## Usage

```bash
./generate_pinn_data.x \
  --mesh data/test_us_barhalfup_centree.msh \
  --defectname barhalfup \
  --freqs 600,800 \
  --modes 0,1,2 \
  --outputdir ../pinn_waveguide_2d/data \
  --c0 340 \
  --contrast 0.8 \
  --numberofdatapoints 31
```

Required arguments:

- `--mesh`: path to the mesh;
- `--defectname`: identifier used in file names;
- `--freqs`: comma-separated positive frequencies;
- `--modes`: comma-separated mode indices;
- `--outputdir`: output directory, created if necessary;
- `--c0`: wave speed in the background medium;
- `--contrast`: `c_defect/c0` ratio.

`--numberofdatapoints N` requests exactly `N` uniformly spaced points per port,
including the endpoints. Values are interpolated using the P2 shape functions.
If this option is omitted, all P2 degrees of freedom on the ports are exported.

The ratio is encoded without a decimal point: `0.8` becomes `ratio0p8`.

## Generated Files

For `defectname=barhalfup` and `contrast=0.8`:

- `pinn_boundary_left_barhalfup_ratio0p8.csv`;
- `pinn_boundary_right_barhalfup_ratio0p8.csv`;
- `Stiff_matrix_barhalfup.csv`;
- `Mass_matrix_barhalfup.csv`;
- `fem_field_barhalfup_ratio0p8.csv`.

The boundary files contain all frequencies and modes:

```text
f,k0,mode,x,y,Re_U,Im_U
```

The field file contains all P2 degrees of freedom in the RCM ordering shared
with the matrices:

```text
f,k0,mode,node_id,x,y,Re_U,Im_U
```

The stiffness and mass matrices are purely geometric and independent of the
frequency and wave-speed ratio. They are stored in COO format using only their
lower triangles:

```text
row,col,value
```

To reconstruct a full matrix, copy each off-diagonal entry to its symmetric
position. Indices are zero-based and correspond directly to the field file's
`node_id` values.
