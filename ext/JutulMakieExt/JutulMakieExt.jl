module JutulMakieExt

using Jutul, Makie
    function Jutul.check_plotting_availability_impl()
        return true
    end

    include("mesh_plots.jl")

end
