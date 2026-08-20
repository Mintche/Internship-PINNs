# P2 FEM data generator

`generate_pinn_data.x` solves the Helmholtz problem in a rectangular acoustic
waveguide and exports the fields, boundary traces, mesh, mass matrix, and
stiffness matrix used by the PINN workflows.

Run the examples below from the repository root.

## Build

```bash
make -C FEM
```

The executable is `FEM/generate_pinn_data.x`. A CMake build is also available:

```bash
cmake -S FEM -B FEM/cmake-build
cmake --build FEM/cmake-build
```

## Mesh contract

Input meshes must be Gmsh v2 ASCII files with these physical tags:

- surface `1`: healthy background;
- surfaces `2`, `3`, ...: material regions;
- curves `11` and `12`: complete left and right ports;
- curves `13` and `14`: top and bottom rigid walls.

The domain must be horizontally centered. Exported P2 degrees of freedom use
reverse Cuthill–McKee (RCM) ordering.

## Generate a dataset

This command generates the data expected by the
`circlebottomright_1200_A.json` inverse configuration:

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

Required options are `--mesh`, `--defectname`, `--freqs`, `--modes`,
`--outputdir`, and `--c0`, plus exactly one material definition:

- `--contrast 0.8`: legacy `c_defect/c0` ratio for surface `2`;
- `--tag-contrasts 2:0.8,3:1.2`: one `c_tag/c0` ratio per material tag;
- `--nodal-sound-speed PATH`: continuous sound-speed map with the header
  `node_id,x,y,c` and one row per reordered P2 degree of freedom.

Unlisted material tags use `c0`. `--incidence` accepts `-1`, `1`, or `-1,1`
and defaults to `-1` (source from the left). `--numberofdatapoints N` exports
`N` uniformly spaced samples per port; without it, all port P2 nodes are used.

Use `FEM/generate_pinn_data.x --help` for the complete CLI reference.

## Outputs

With `--defectname circlebottomright` and `--tag-contrasts`, the generated
files are:

```text
pinn_boundary_left_circlebottomright.csv
pinn_boundary_right_circlebottomright.csv
fem_field_circlebottomright.csv
Stiff_matrix_circlebottomright.csv
Mass_matrix_circlebottomright.csv
mesh_nodes_circlebottomright.csv
mesh_triangles_circlebottomright.csv
```

The legacy `--contrast` option adds a `_ratio...` suffix to the boundary and
field filenames. Matrices and mesh files always use only `defectname`.

Boundary rows follow
`incidence,f,k0,mode,x,y,Re_U,Im_U`; field rows add `node_id` before `x`.
Mass and stiffness matrices are lower-triangular COO files with
`row,col,value`; their zero-based indices match the exported mesh and field.
