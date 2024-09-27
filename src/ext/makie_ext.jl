export plot_interactive, plot_multimodel_interactive
export plot_mesh, plot_mesh!
export plot_cell_data!, plot_cell_data
export plot_solve_breakdown
export plot_cumulative_solve, plot_cumulative_solve!


function plot_interactive(arg...; kwarg...)
    check_plotting_availability(interactive = true)
    plot_interactive_impl(arg...; kwarg...)
end

function plot_interactive_impl

end

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

function plot_cell_data(arg...; kwarg...)
    check_plotting_availability()
    plot_cell_data_impl(arg...; kwarg...)
end

function plot_cell_data_impl

end

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

function plot_solve_breakdown

end

function plot_cumulative_solve

end

function plot_cumulative_solve!

end

function plot_linear_convergence

end

function plot_linear_convergence!

end