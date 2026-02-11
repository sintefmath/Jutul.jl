module CutCellMeshes
    using Jutul
    using StaticArrays
    using LinearAlgebra

    include("types.jl")
    include("geometry.jl")
    include("cutting.jl")
    include("gluing.jl")

    export cut_mesh, glue_mesh, mesh_fault_slip
end
