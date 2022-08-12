# using Meshes, MeshViz

export MRSTWrapMesh, CartesianMesh, TwoPointFiniteVolumeGeometry, dim
export triangulate_outer_surface, tpfv_geometry, discretized_domain_tpfv_flow

abstract type JutulGeometry end

"""
    tpfv_geometry(g)

Generate two-point finite-volume geometry for a given grid, if supported.

See also [`TwoPointFiniteVolumeGeometry`](@ref).
"""
function tpfv_geometry end

"""
    TwoPointFiniteVolumeGeometry(neighbors, areas, volumes, normals, cell_centers, face_centers)

Store two-point geometry information for a given list of `neighbors` specified as a `2` by `n` matrix
where `n` is the number of faces such that face `i` connectes cells `N[1, i]` and `N[2, i]`.

The two-point finite-volume geometry contains the minimal set of geometry information
required to compute standard finite-volume discretizations.
"""
struct TwoPointFiniteVolumeGeometry <: JutulGeometry
    neighbors
    areas
    volumes
    normals
    cell_centroids
    face_centroids
    function TwoPointFiniteVolumeGeometry(neighbors, A, V, N, C_c, C_f)
        nf = size(neighbors, 2)
        dim, nc = size(C_c)

        # Sanity check
        @assert dim == 2 || dim == 3
        # Check cell centroids
        @assert size(C_c) == (dim, nc)
        # Check face centroids
        @assert size(C_f) == (dim, nf)
        # Check normals
        @assert size(N) == (dim, nf)
        # Check areas
        @assert length(A) == nf
        @assert length(V) == nc
        return new(neighbors, vec(A), vec(V), N, C_c, C_f)
    end
end

dim(g::TwoPointFiniteVolumeGeometry) = size(g.cell_centroids, 1)

"""
    dim(g)::Integer

Get the dimension of a mesh.
"""
dim(t::AbstractJutulMesh) = 2

"""
    number_of_cells(g)::Integer

Get the number of cells in a mesh.
"""
number_of_cells(t::AbstractJutulMesh) = 1

"""
    number_of_faces(g)::Integer

Get the number of faces in a mesh.
"""
number_of_faces(G) = size(get_neighborship(G), 2)

include("mrst.jl")
include("cart.jl")

