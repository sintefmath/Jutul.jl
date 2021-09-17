# using Meshes, MeshViz

export MRSTWrapMesh, CartesianMesh, TwoPointFiniteVolumeGeometry
export triangulate_outer_surface, tpfv_geometry, discretized_domain_tpfv_flow

abstract type TervGeometry end

struct TwoPointFiniteVolumeGeometry <: TervGeometry
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

abstract type AbstractTervMesh end
dim(t::AbstractTervMesh) = 2
number_of_cells(t::AbstractTervMesh) = 1
number_of_faces(t::AbstractTervMesh) = 0

include("mrst.jl")
include("cart.jl")

