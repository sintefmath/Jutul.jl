# Meshes and mesh utilities

## Mesh types

Jutul has two main internal mesh types: Cartesian meshes and unstructured meshes. The unstructured format is a general polyhedral mesh format, and a Cartesian mesh can easily be converted to an unstructured mesh.

```@docs
JutulMesh
CartesianMesh
UnstructuredMesh
```

## Mesh generation


### Radial meshes

```@docs
Jutul.RadialMeshes.spiral_mesh
```

```@example spiral
using Jutul, CairoMakie
import Jutul.RadialMeshes: spiral_mesh
n_angular_sections = 10
nrotations = 4
spacing = [0.0, 0.5, 1.0]
rmesh = spiral_mesh(n_angular_sections, nrotations, spacing = spacing)
num_cells = number_of_cells(rmesh)

fig, ax, plt = plot_cell_data(rmesh, 1:number_of_cells(rmesh))
fig
```

```@docs
Jutul.RadialMeshes.spiral_mesh_tags
```

```@example spiral
import Jutul.RadialMeshes: spiral_mesh_tags
tags = spiral_mesh_tags(rmesh, spacing)
fig = Figure(size = (400, 1800))
for (figno, pp) in enumerate(pairs(tags))
    k, val = pp
    ax = Axis(fig[figno, 1], title = "Spiral tag $k")
    plot_cell_data!(ax, rmesh, val)
end
fig
```

## Mesh API functions

```@docs
number_of_cells
number_of_faces
number_of_boundary_faces
```

### Misc

### Geometry

```@docs
TwoPointFiniteVolumeGeometry
Jutul.tpfv_geometry
```
