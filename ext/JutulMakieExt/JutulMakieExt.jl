module JutulMakieExt

using Jutul, Makie, Statistics
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
