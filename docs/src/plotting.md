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

For comprehensive mesh plotting documentation including `plot_mesh`, `plot_mesh!`, `plot_cell_data`, `plot_cell_data!`, `plot_mesh_edges`, `plot_mesh_edges!`, and `plot_interactive`, see the [Mesh](mesh.md) section.

### Interactive multi-model plotting

`plot_multimodel_interactive(model, states, model_keys = keys(model.models); plot_type = :mesh, shift = Dict(), kwarg...)`

Launch an interactive plot for multi-model simulations with multiple coupled domains.

**Arguments:**
- `model`: `MultiModel` instance containing multiple coupled simulation models
- `states`: Vector of simulation states or single state
- `model_keys = keys(model.models)`: Which models to include in the plot

**Keyword Arguments:**
- `plot_type = :mesh`: Type of plot (`:mesh`, `:meshscatter`, `:lines`)
- `shift = Dict()`: Dictionary of spatial shifts to apply to each model for visualization
- Additional keyword arguments are passed to `plot_interactive`

This function creates an interactive visualization for multi-physics simulations where multiple models are coupled together.

## Performance Analysis

These functions help analyze simulation performance and solver behavior.

### Solver timing breakdown

`plot_solve_breakdown(allreports, names; per_it = false, include_local_solves = nothing, t_scale = ("s", 1.0))`

Plot a breakdown of solver performance timing for multiple simulation reports.

**Arguments:**
- `allreports`: Vector of simulation reports to analyze
- `names`: Vector of names for each report (used for labeling)

**Keyword Arguments:**
- `per_it = false`: If `true`, show time per iteration instead of total time
- `include_local_solves = nothing`: Whether to include local solve timing (auto-detected if `nothing`)
- `t_scale = ("s", 1.0)`: Time scale unit and conversion factor for display

Returns `(fig, D)`: Figure object and processed timing data.

### Cumulative solve time

`plot_cumulative_solve(allreports, dt = nothing, names = nothing; kwarg...)`

Plot cumulative solver performance over time or steps for multiple simulation reports.

`plot_cumulative_solve!(f, allreports, dt = nothing, names = nothing; kwarg...)`

Mutating version that plots into an existing figure layout.

### Linear solver convergence

`plot_linear_convergence(report; kwarg...)`

Plot the convergence history of linear solver iterations from a simulation report.

`plot_linear_convergence!(ax, report)`

Mutating version that plots into an existing axis.

## Model Structure Visualization

These functions require the GraphMakie.jl package to be loaded in addition to a Makie backend.

```julia
using GraphMakie
```

### Variable dependency graph

`plot_variable_graph(model)`

Plot a graph visualization of variable dependencies in a simulation model.

**Arguments:**
- `model`: A Jutul simulation model

Returns a Figure object containing the variable dependency graph showing relationships between primary variables, secondary variables, and parameters.

### Model structure graph

`plot_model_graph(model; kwarg...)`

Plot a graph visualization of model structure and equation relationships. For `MultiModel` instances, this shows the full structure including cross-terms and model coupling.

## Utilities

`check_plotting_availability(; throw = true, interactive = false)`

Check if plotting through at least one Makie backend is available in the Julia session. Returns `true` if available, `false` otherwise (or throws an error if `throw=true`).

## Examples

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

### Model structure visualization

```julia
using Jutul, GraphMakie, GLMakie

# Plot variable dependencies
fig = plot_variable_graph(model)

# For multi-models, plot the full structure
fig = plot_model_graph(multimodel)
```

For mesh plotting examples, see the [Mesh](mesh.md) section.