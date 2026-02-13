# Meshes and mesh utilities

## Mesh types

Jutul has two main internal mesh types: Cartesian meshes and unstructured meshes. The unstructured format is a general polyhedral mesh format, and a Cartesian mesh can easily be converted to an unstructured mesh. Coarsened meshes can be created by a fine scale mesh and a partition vector.

```@docs
JutulMesh
CartesianMesh
UnstructuredMesh
CoarseMesh
MRSTWrapMesh
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

If we want to drill down a bit further, we can make a plot:

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

### Queries

```@docs
number_of_cells
number_of_faces
number_of_boundary_faces
```

### Manipulation

```@docs
Jutul.extrude_mesh
Jutul.extract_submesh
```

## Example: Mesh manipulation

We can quickly build new meshes by applying transformations to an already existing mesh. Let us create a Cartesian mesh and extract the cells that lie within a circle:

```@example extrude_submesh
using Jutul, CairoMakie
g = UnstructuredMesh(CartesianMesh((10, 10), (1.0, 1.0)))
geo = tpfv_geometry(g)
keep = Int[]
for c in 1:number_of_cells(g)
    x, y = geo.cell_centroids[:, c]
    if (x - 0.5)^2 + (y - 0.5)^2 < 0.25
        push!(keep, c)
    end
end
subg = extract_submesh(g, keep)
fig, ax, plt = plot_mesh(subg)
plot_mesh_edges!(ax, g, color = :red)
fig
```

We can turn this into a 3D mesh by extruding it, and then tweak the nodes:

```@example extrude_submesh
g3d = Jutul.extrude_mesh(subg, 20)
for node in eachindex(subg.node_points)
    g3d.node_points[node] += 0.01*rand(3)
end
fig, ax, plt = plot_mesh(g3d)
fig
```

### Geometry

```@docs
TwoPointFiniteVolumeGeometry
Jutul.tpfv_geometry
Jutul.find_enclosing_cells
Jutul.cells_inside_bounding_box
```

## Example: Cell intersection

```@example
using CairoMakie, Jutul
# 3D mesh
G = CartesianMesh((4, 4, 5), (100.0, 100.0, 100.0))
trajectory = [
    50.0 25.0 1;
    55 35.0 25;
    65.0 40.0 50.0;
    70.0 70.0 90.0
]

cells = Jutul.find_enclosing_cells(G, trajectory)

# Optional plotting, requires Makie:
fig, ax, plt = Jutul.plot_mesh_edges(G)
plot_mesh!(ax, G, cells = cells, alpha = 0.5, transparency = true)
lines!(ax, trajectory, linewidth = 10)
fig
```

2D version:

```@example
using CairoMakie, Jutul
# 2D mesh
G = CartesianMesh((50, 50), (1.0, 2.0))
trajectory = [
    0.1 0.1;
    0.2 0.4;
    0.3 1.2
]
fig, ax, plt = Jutul.plot_mesh_edges(G)
cells = Jutul.find_enclosing_cells(G, trajectory)
# Plotting, needs Makie
fig, ax, plt = Jutul.plot_mesh_edges(G)
plot_mesh!(ax, G, cells = cells, alpha = 0.5, transparency = true)
lines!(ax, trajectory[:, 1], trajectory[:, 2], linewidth = 3)
fig
```

## Mesh generation

### Gmsh support

```@docs
Jutul.mesh_from_gmsh
```

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

We can also plot the faces by using the nodes together with standard Makie plotting calls. We regenerate the mesh and make it contain a single cell in the middle before plotting it:

```@example radial
m = radial_mesh(nangle, radii; centerpoint = false)
ncells = number_of_cells(m)
fig, ax, plt = plot_cell_data(m, 1:ncells, alpha = 0.25)
scatter!(ax, m.node_points)
for face in 1:number_of_faces(m)
    n1, n2 = m.faces.faces_to_nodes[face]
    pt1 = m.node_points[n1]
    pt2 = m.node_points[n2]
    lines!(ax, [pt1, pt2], color = :red)
end

for bface in 1:number_of_boundary_faces(m)
    n1, n2 = m.boundary_faces.faces_to_nodes[bface]
    pt1 = m.node_points[n1]
    pt2 = m.node_points[n2]
    lines!(ax, [pt1, pt2], color = :blue)
end
fig
```

We can also zoom in on a single cell and plot the oriented normals:

```@example radial
using LinearAlgebra
geo = tpfv_geometry(m)
cellno = 1
fig, ax, plt = plot_mesh(m, cells = cellno)
for face in m.faces.cells_to_faces[cellno]
    n1, n2 = m.faces.faces_to_nodes[face]
    @info "Interior edge $n1 to $n2"
    pt1 = m.node_points[n1]
    pt2 = m.node_points[n2]
    lines!(ax, [pt1, pt2], color = :red)
    midpt = (pt1 + pt2) / 2
    if m.faces.neighbors[face][1] == cellno
        sgn = 1
    else
        sgn = -1
    end
    lines!(ax, [midpt, midpt + sgn*norm(pt2 - pt1, 2)*geo.normals[:, face]], color = :orange)
end
for bface in m.boundary_faces.cells_to_faces[cellno]
    n1, n2 = m.boundary_faces.faces_to_nodes[bface]
    @info "Exterior edge $n1 to $n2"
    pt1 = m.node_points[n1]
    pt2 = m.node_points[n2]
    lines!(ax, [pt1, pt2], color = :blue)
    midpt = (pt1 + pt2) / 2
    lines!(ax, [midpt, midpt + norm(pt2 - pt1, 2)*geo.boundary_normals[:, bface]], color = :green)
end
scatter!(ax, m.node_points)
fig
```

### Spiral meshes

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

### PEBI/Voronoi meshes

PEBI (Perpendicular Bisector) meshes, also known as Voronoi diagrams, can be generated from a set of points. Each point becomes a cell center in the resulting mesh.

```@docs
Jutul.VoronoiMeshes.PEBIMesh2D
```

#### Basic PEBI mesh example

```@example pebi
using Jutul, CairoMakie
import Jutul.VoronoiMeshes: PEBIMesh2D

# Create a simple PEBI mesh from 4 points
points = [0.0 1.0 0.0 1.0; 
          0.0 0.0 1.0 1.0]

mesh = PEBIMesh2D(points)

# Plot the mesh
fig, ax, plt = plot_mesh(mesh)
plot_mesh_edges!(ax, mesh)
scatter!(ax, mesh.node_points, color = :red, markersize = 10)
fig
```

#### PEBI mesh with random points

```@example pebi
using Random
Random.seed!(42)

# Generate random points
npts = 20
points = rand(2, npts)

# Create mesh
mesh = PEBIMesh2D(points)

# Plot with cell numbers
fig, ax, plt = plot_cell_data(mesh, 1:number_of_cells(mesh))
plot_mesh_edges!(ax, mesh)
fig
```

#### PEBI mesh with custom bounding box

```@example pebi
# Create points in a specific region
points = rand(2, 15) .* [100.0; 50.0]  # Scale to 100x50 domain

# Specify custom bounding box
bbox = ((0.0, 100.0), (0.0, 50.0))

mesh = PEBIMesh2D(points, bbox=bbox)

fig, ax, plt = plot_mesh(mesh)
plot_mesh_edges!(ax, mesh)
fig
```

#### PEBI mesh with constraints

Linear constraints can be added to ensure certain edges are respected in the mesh:

```@example pebi
# Create points
points = rand(2, 25)

# Add a vertical constraint line at x=0.5
constraint = ([0.5, 0.0], [0.5, 1.0])

mesh = PEBIMesh2D(points, constraints=[constraint])

fig, ax, plt = plot_mesh(mesh)
plot_mesh_edges!(ax, mesh, color = :blue)
# Highlight the constraint
lines!(ax, [0.5, 0.5], [0.0, 1.0], color = :red, linewidth = 3)
fig
```

```@docs
Jutul.VoronoiMeshes.PEBIMesh3D
```

