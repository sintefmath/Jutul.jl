module JutulMakieExt

using Jutul, Makie, Statistics
function Jutul.allocate_plot_buffer(values)
    backend = Makie.current_backend()
    screen_type = isnothing(backend) ? Nothing : backend.Screen
    return Jutul.allocate_plot_buffer(values, screen_type)
end

Jutul.allocate_plot_buffer(values, ::Type) = Observable(values)

function Jutul.update_plot_buffer!(buffer::Observable, values)
    copyto!(buffer[], values)
    notify(buffer)
    return buffer
end

function Jutul.check_plotting_availability_impl()
    return true
end

include("mesh_plots.jl")
include("interactive_3d.jl")
include("performance.jl")
include("variable_graphs.jl")
include("variables.jl")
include("explorer_3d.jl")
end
