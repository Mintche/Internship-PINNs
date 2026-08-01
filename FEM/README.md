# P2 FEM data generator

This directory contains the P2 finite-element solver for a rectangular acoustic
waveguide and the `generate_pinn_data.x` executable. The generator solves the
Helmholtz problem for one or more frequencies, incident modes, and source
directions, then exports the complex fields and boundary traces used by the
Python PINN workflows.

## Mesh contract

Input meshes must use Gmsh v2 ASCII format and these physical groups:

- background surface: tag `1`;
- defect/material surfaces: tag `2` or additional positive tags;
- left port at `x=-L`: tag `11`;
- right port at `x=+L`: tag `12`;
- top and bottom boundaries: tags `13` and `14`.

The domain must be horizontally centered, and both ports must cover the full
height of the waveguide. The solver reorders P2 degrees of freedom with RCM
before exporting them.

## Build

From this directory, either build with Make:

```bash
make
```

or with CMake:

```bash
cmake -S . -B cmake-build
cmake --build cmake-build
```

Make creates `./generate_pinn_data.x`; CMake creates
`cmake-build/generate_pinn_data.x`.

## Usage

From the repository root, a single-contrast dataset can be generated with:

```bash
make -C FEM
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

For distinct physical surface tags, provide one ratio per tag:

```bash
FEM/generate_pinn_data.x \
  --mesh FEM/data/test_us_2triangles_diffcontrast.msh \
  --defectname triangles_diffcontrast \
  --freqs 600,1200 \
  --modes 0,1,2 \
  --outputdir FEM/pinn_data \
  --c0 340 \
  --tag-contrasts 2:0.8,3:0.9
```

For an independently reconstructed continuous material map, provide one
physical sound speed for every reordered P2 degree of freedom:

```bash
FEM/generate_pinn_data.x \
  --mesh FEM/data/test_us_barhalf_centree.msh \
  --defectname reconstructed_map \
  --freqs 600 \
  --modes 0,1,2 \
  --outputdir /tmp/reconstructed_forward \
  --c0 340 \
  --nodal-sound-speed /tmp/reconstructed_sound_speed.csv \
  --incidence -1,1 \
  --numberofdatapoints 31
```

Required arguments are `--mesh`, `--defectname`, `--freqs`, `--modes`,
`--outputdir`, and `--c0`, plus exactly one material specification:

- `--contrast RATIO`: legacy `c_defect/c0` ratio for surface tag `2`;
- `--tag-contrasts T:R,T:R`: one `c_tag/c0` ratio per physical surface tag;
- `--nodal-sound-speed PATH`: CSV material map described below.

Surface tag `1` is the healthy background and uses `c0`. Any surface tag not
listed in `--tag-contrasts` also uses `c0`. `--incidence` accepts `-1`, `1`, or
`-1,1` and defaults to `-1`. `--numberofdatapoints N` requests `N` uniformly
spaced points per port, including endpoints; if omitted, all P2 port degrees of
freedom are exported.

The nodal CSV must start with `node_id,x,y,c` and contain every reordered P2
degree of freedom exactly once. Node IDs and coordinates are checked to prevent
a silent permutation error. The nodal sound speed is interpolated with the P2
shape functions inside the material term.

## Generated files

For `defectname=circlebottomrightforward` and `--contrast 0.8`, the output is:

```text
pinn_boundary_left_circlebottomrightforward_ratio0p8.csv
pinn_boundary_right_circlebottomrightforward_ratio0p8.csv
fem_field_circlebottomrightforward_ratio0p8.csv
Stiff_matrix_circlebottomrightforward.csv
Mass_matrix_circlebottomrightforward.csv
mesh_nodes_circlebottomrightforward.csv
mesh_triangles_circlebottomrightforward.csv
```

With `--tag-contrasts`, the material-dependent files use the defect name
without a ratio suffix. The boundary CSV schema is:

```text
incidence,f,k0,mode,x,y,Re_U,Im_U
```

The field CSV schema is:

```text
incidence,f,k0,mode,node_id,x,y,Re_U,Im_U
```

The matrices are purely geometric and independent of frequency and material
ratio. They are lower-triangular COO files with the schema
`row,col,value`; indices are zero-based and match `node_id` in the field file.
The node and triangle exports use the same RCM ordering and are required by the
modular inverse-PINN data loader.
