# Meshes and mesh utilities

## Mesh types

Jutul has two main internal mesh types: Cartesian meshes and unstructured meshes. The unstructured format is a general polyhedral mesh format, and a Cartesian mesh can easily be converted to an unstructured mesh.

```@docs
JutulMesh
CartesianMesh
UnstructuredMesh
```

## Plotting functions

Plotting requires that a Makie backend is loaded (typically GLMakie or CairoMakie). The documentation uses `CairoMakie` to work on machines without OpenGL enabled, but if you want fast and interactive plots, `GLMakie` should be preferred.

### Non-mutating

```@docs
plot_mesh
plot_cell_data
plot_mesh_edges
plot_interactive
```

### Mutating

```@docs
plot_mesh!
plot_cell_data!
plot_mesh_edges!
```

## Example: Cartesian meshes

For example, we can make a small 2D mesh with given physical dimensions and convert it:

```@example cart_mesh
using Jutul
nx = 10
ny = 5
g2d_cart = CartesianMesh((nx, ny), (100.0, 50.0))
```

```@example cart_mesh
g2d = UnstructuredMesh(g2d_cart)
```

We can then plot it, colorizing each cell by its enumeration:

```@example cart_mesh
using CairoMakie
fig, ax, plt = plot_cell_data(g2d, 1:number_of_cells(g2d))
plot_mesh_edges!(ax, g2d)
fig
```

We can make a 3D mesh in the same manner:

```@example cart_mesh
nz = 3
g3d = UnstructuredMesh(CartesianMesh((nx, ny, nz), (100.0, 50.0, 30.0)))
```

And plot it the same way:

```@example cart_mesh
using CairoMakie
nc = number_of_cells(g3d)
fig, ax, plt = plot_cell_data(g3d, 1:nc)
plot_mesh_edges!(ax, g3d)
fig
```

We can also plot only a subset of cells:

```@example cart_mesh
using CairoMakie
fig, ax, plt = plot_cell_data(g3d, 1:nc, cells = 1:2:nc)
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

## Mesh generation

### Radial mesh

```@example radial
using Jutul, CairoMakie
import Jutul.RadialMeshes: radial_mesh
import Jutul: plot_mesh_edges
nangle = 10
radii = [0.2, 0.5, 1.0]
m = radial_mesh(nangle, radii; centerpoint = true)
plot_mesh_edges(m)
```

Radial meshes can still be indexed as Cartesian meshes:

```@example radial
IJ = map(i -> cell_ijk(m, i), 1:number_of_cells(m))
```

```@example radial
fig, ax, plt = plot_cell_data(m, map(first, IJ))
fig
```

```@example radial
fig, ax, plt = plot_cell_data(m, map(ijk -> ijk[2], IJ))
fig
```

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
