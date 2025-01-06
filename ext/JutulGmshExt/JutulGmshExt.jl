module JutulGmshExt
    using Jutul
    import Jutul: UnstructuredMesh, IndirectionMap, check_equal_perm
    using Gmsh: Gmsh, gmsh

    using StaticArrays
    using OrderedCollections

    const QUAD_T = SVector{4, Int}
    const TRI_T = SVector{3, Int}

    include("utils.jl")
    include("interface.jl")
end
