module EmbeddedMeshes

    using Jutul, LinearAlgebra, StaticArrays
    include("types.jl")
    include("embedded.jl")
    include("geometry.jl")
    include("finite-volume.jl")
    include("plotting.jl")

end
