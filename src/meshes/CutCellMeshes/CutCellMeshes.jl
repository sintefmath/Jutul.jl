module CutCellMeshes
    using Jutul
    using StaticArrays
    using LinearAlgebra

    include("types.jl")
    include("geometry.jl")
    include("cutting.jl")
    include("merge_faces.jl")
    include("gluing.jl")
    include("layered.jl")
    include("embedding.jl")

    export cut_mesh, glue_mesh, cut_and_displace_mesh
    export layered_mesh, depth_grid_to_surface
    export embed_mesh
end
