module CutCellMeshes
    using Jutul
    using StaticArrays
    using LinearAlgebra

    include("types.jl")
    include("geometry.jl")
    include("cutting.jl")

    export cut_mesh
end
