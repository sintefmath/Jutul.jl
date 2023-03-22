function plot_interactive(arg...; kwarg...)
    check_plotting_availability()
    plot_interactive_impl(arg...; kwarg...)
end

function plot_interactive_impl

end

function plot_multimodel_interactive(arg...; kwarg...)
    check_plotting_availability()
    plot_multimodel_interactive_impl(arg...; kwarg...)
end

function plot_multimodel_interactive_impl

end


function plot_mesh(arg...; kwarg...)
    check_plotting_availability()
    plot_mesh_impl(arg...; kwarg...)
end

function plot_mesh_impl

end

function plot_mesh!(arg...; kwarg...)
    check_plotting_availability()
    plot_mesh_impl!(arg...; kwarg...)
end

function plot_mesh_impl!

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

function check_plotting_availability()
    ok = true
    try
        ok = check_plotting_availability_impl()
    catch e
        if e isa MethodError
            error("Plotting is not available. You need to have a Makie backend available. For 3D plots, GLMakie is recommended. To fix: using Pkg; Pkg.add(\"GLMakie\")")
        else
            rethrow(e)
        end
    end
    return ok
end

function check_plotting_availability_impl

end
