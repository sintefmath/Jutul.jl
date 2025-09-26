# Plotting and Visualization

Jutul.jl provides comprehensive plotting capabilities for mesh visualization, performance analysis, and model structure visualization. All plotting functionality requires a Makie backend to be loaded.

## Requirements

Plotting requires that a Makie backend is loaded (typically GLMakie or CairoMakie). The documentation uses `CairoMakie` to work on machines without OpenGL enabled, but if you want fast and interactive plots, `GLMakie` should be preferred.

```julia
using GLMakie  # For interactive plots
# or
using CairoMakie  # For static plots/headless systems
```

## Mesh Visualization

### Basic mesh plotting

```@docs
plot_mesh
plot_mesh!
plot_cell_data
plot_cell_data!
plot_mesh_edges
plot_mesh_edges!
```

### Interactive plotting

```@docs
plot_interactive
plot_multimodel_interactive
```

## Performance Analysis

These functions help analyze simulation performance and solver behavior.

```@docs
plot_solve_breakdown
plot_cumulative_solve
plot_cumulative_solve!
plot_linear_convergence
plot_linear_convergence!
```

## Model Structure Visualization

These functions require the GraphMakie.jl package to be loaded in addition to a Makie backend.

```julia
using GraphMakie
```

```@docs
plot_variable_graph
plot_model_graph
```

## Utilities

```@docs
check_plotting_availability
```

## Examples

### Basic mesh plotting

```julia
using Jutul, CairoMakie

# Create a simple mesh
nx, ny = 10, 5
mesh = CartesianMesh((nx, ny), (100.0, 50.0))

# Plot the mesh structure
fig, ax, p = plot_mesh(mesh)

# Plot cell data with values
cell_values = 1:number_of_cells(mesh)
fig, ax, p = plot_cell_data(mesh, cell_values)

# Add mesh edges
plot_mesh_edges!(ax, mesh)
fig
```

### Performance analysis

```julia
using Jutul, CairoMakie

# After running simulations and collecting reports
reports = [report1, report2, report3]
names = ["Configuration A", "Configuration B", "Configuration C"]

# Plot solver breakdown
fig, data = plot_solve_breakdown(reports, names)

# Plot cumulative solve time
fig, ax, alldata, t = plot_cumulative_solve(reports, names = names)

# Plot linear convergence for a single report
fig = plot_linear_convergence(reports[1])
```

### Interactive visualization

```julia
using Jutul, GLMakie

# Create mesh and simulation states
mesh = CartesianMesh((10, 10, 5), (100.0, 100.0, 50.0))
states = [state1, state2, state3, ...]  # From simulation

# Launch interactive plot
plot_interactive(mesh, states)
```

### Model structure visualization

```julia
using Jutul, GraphMakie, GLMakie

# Plot variable dependencies
fig = plot_variable_graph(model)

# For multi-models, plot the full structure
fig = plot_model_graph(multimodel)
```