# using Meshes, MeshViz

export MRSTWrapMesh, CartesianMesh, TwoPointFiniteVolumeGeometry
export triangulate_outer_surface, tpfv_geometry

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
cell_to_surface(m::AbstractTervMesh, celldata) = celldata

include("mrst.jl")
include("cart.jl")

