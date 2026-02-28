module MeshUtils
    using Jutul, StaticArrays, LinearAlgebra
    import Jutul: cellmap_to_posmap, compute_centroid_and_measure

    include("refinement.jl")
    include("radial_refinement.jl")
    include("merge_cells.jl")

    export refine_mesh, refine_mesh_radial, merge_cells
end
