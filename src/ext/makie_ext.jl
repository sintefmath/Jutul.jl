export plot_interactive, plot_multimodel_interactive
export plot_mesh, plot_mesh!
export plot_cell_data!, plot_cell_data
export plot_solve_breakdown
export plot_cumulative_solve, plot_cumulative_solve!


"""
    plot_interactive(mesh, vector_of_dicts; kwarg...)

Launch an interactive plot of a mesh with the given `vector_of_dicts` (or just a
dict). Each dict can have cell data either as vectors (one value per cell) or
matrices (one column per cell).
"""
function plot_interactive(arg...; kwarg...)
    check_plotting_availability(interactive = true)
    plot_interactive_impl(arg...; kwarg...)
end

function plot_interactive_impl

end

"""
    plot_multimodel_interactive(model, states, model_keys = keys(model.models); plot_type = :mesh, shift = Dict(), kwarg...)

Launch an interactive plot for multi-model simulations with multiple coupled domains.

# Arguments
- `model`: `MultiModel` instance containing multiple coupled simulation models
- `states`: Vector of simulation states or single state
- `model_keys = keys(model.models)`: Which models to include in the plot

# Keyword Arguments
- `plot_type = :mesh`: Type of plot (`:mesh`, `:meshscatter`, `:lines`)
- `shift = Dict()`: Dictionary of spatial shifts to apply to each model for visualization
- Additional keyword arguments are passed to `plot_interactive`

This function creates an interactive visualization for multi-physics simulations
where multiple models are coupled together. Each model domain can be spatially
shifted for better visualization, and the resulting plot allows interactive
exploration of the combined multi-model states over time.

The function automatically handles data mapping between different model domains
and creates a unified visualization interface.
"""
function plot_multimodel_interactive(arg...; kwarg...)
    check_plotting_availability(interactive = true)
    plot_multimodel_interactive_impl(arg...; kwarg...)
end

function plot_multimodel_interactive_impl

end

"""
    plot_mesh(mesh)
    plot_mesh(mesh;
        cells = nothing,
        faces = nothing,
        boundaryfaces = nothing,
        outer = false,
        color = :lightblue,
    )

Plot a `mesh` with uniform colors. Optionally, indices `cells`, `faces` or
`boundaryfaces` can be passed to limit the plotting to a specific selection of
entities.
"""
function plot_mesh(arg...; kwarg...)
    check_plotting_availability()
    plot_mesh_impl(arg...; kwarg...)
end

function plot_mesh_impl

end


"""
    plot_mesh!(ax, mesh)

Mutating version of `plot_mesh` that plots into an existing Makie `Axis`
instance.
"""
function plot_mesh!(arg...; kwarg...)
    check_plotting_availability()
    plot_mesh_impl!(arg...; kwarg...)
end

function plot_mesh_impl!

end

export plot_mesh_edges, plot_mesh_edges!
"""
    plot_mesh_edges(mesh; kwarg...)

Plot the edges of all cells on the exterior of a mesh.
"""
function plot_mesh_edges(arg...; kwarg...)
    check_plotting_availability()
    plot_mesh_edges_impl(arg...; kwarg...)
end

function plot_mesh_edges_impl

end

"""
    plot_mesh_edges!(ax, mesh; kwarg...)

Plot the edges of all cells on the exterior of a mesh into existing Makie
`Axis` `ax`.
"""
function plot_mesh_edges!(arg...; kwarg...)
    check_plotting_availability()
    plot_mesh_edges_impl!(arg...; kwarg...)
end

function plot_mesh_edges_impl!

end

"""
    plot_cell_data(mesh::JutulMesh, data::Vector; kwarg...)
    plot_cell_data(mesh, data;
        cells = nothing,
        faces = nothing,
        boundaryfaces = nothing
    )

Plot cell-wise values (as a vector) on the mesh. Optionally, indices `cells`,
`faces` or `boundaryfaces` can be passed to limit the plotting to a specific
selection of entities.
"""
function plot_cell_data(arg...; kwarg...)
    check_plotting_availability()
    plot_cell_data_impl(arg...; kwarg...)
end

function plot_cell_data_impl

end

"""
    plot_cell_data!(ax, mesh, data; kwarg...)

Mutating version of `plot_cell_data` that plots into an existing Makie `Axis`
"""
function plot_cell_data!(arg...; kwarg...)
    check_plotting_availability()
    plot_cell_data_impl!(arg...; kwarg...)
end

function plot_cell_data_impl!

end

function plotting_check_interactive

end

"""
    check_plotting_availability(; throw = true, interactive = false)

Check if plotting through at least one `Makie` backend is available in the Julia
session (after package has been loaded by for example `using GLMakie`). The
argument `throw` can be used to control if this function acts as a programmatic
check (`throw=false`) there the return value indicates availability, or if an
error message is to be printed telling the user how to get plotting working
(`throw=true`)

An additional check for specifically `interactive` plots can also be added.
"""
function check_plotting_availability(; throw = true, interactive = false)
    ok = true
    try
        ok = check_plotting_availability_impl()
    catch e
        if throw
            if e isa MethodError
                error("Plotting is not available. You need to have a Makie backend available. For 3D plots, GLMakie is recommended. To fix: using Pkg; Pkg.add(\"GLMakie\") and then call using GLMakie to enable plotting.")
            else
                rethrow(e)
            end
        else
            ok = false
        end
    end
    if interactive
        plotting_check_interactive()
    end
    return ok
end

function check_plotting_availability_impl

end

"""
    plot_solve_breakdown(allreports, names; kwarg...)

Plot a breakdown of solver performance timing for multiple simulation reports.
This function is implemented when a Makie backend is loaded.

See the Makie extension documentation for detailed arguments and usage.
"""
function plot_solve_breakdown

end

"""
    plot_cumulative_solve(allreports, args...; kwarg...)

Plot cumulative solver performance over time or steps for multiple simulation reports.
This function is implemented when a Makie backend is loaded.

See the Makie extension documentation for detailed arguments and usage.
"""
function plot_cumulative_solve

end

"""
    plot_cumulative_solve!(f, allreports, args...; kwarg...)

Mutating version of `plot_cumulative_solve` that plots into an existing figure layout.
This function is implemented when a Makie backend is loaded.

See the Makie extension documentation for detailed arguments and usage.
"""
function plot_cumulative_solve!

end

"""
    plot_linear_convergence(report; kwarg...)

Plot the convergence history of linear solver iterations from a simulation report.
This function is implemented when a Makie backend is loaded.

See the Makie extension documentation for detailed arguments and usage.
"""
function plot_linear_convergence

end

"""
    plot_linear_convergence!(ax, report)

Mutating version of `plot_linear_convergence` that plots into an existing axis.
This function is implemented when a Makie backend is loaded.

See the Makie extension documentation for detailed arguments and usage.
"""
function plot_linear_convergence!

end