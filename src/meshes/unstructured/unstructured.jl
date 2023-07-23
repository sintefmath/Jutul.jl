export UnstructuredMesh
include("types.jl")
include("utils.jl")
include("geometry.jl")
include("plotting.jl")

dim(t::UnstructuredMesh{D}) where D = D::Int

function get_neighborship(G::UnstructuredMesh)
    nf = number_of_faces(G)
    N = zeros(Int, 2, nf)
    for (i, lr) in enumerate(G.faces.neighbors)
        N[1, i] = lr[1]
        N[2, i] = lr[2]
    end
    return N
end
