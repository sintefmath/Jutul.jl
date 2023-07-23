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

function face_normal(G::UnstructuredMesh, f, e = Faces())
    get_nodes(::Faces) = G.faces
    get_nodes(::BoundaryFaces) = G.boundary_faces
    nodes = get_nodes(e).faces_to_nodes[f]
    pts = G.node_points
    a = pts[nodes[1]]
    b = pts[nodes[2]]
    c = pts[nodes[3]]

    normal = cross(c - b, a - b)
    normal /= norm(normal, 2)
    return normal
end
