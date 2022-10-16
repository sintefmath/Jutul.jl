# using Meshes, MeshViz

export MRSTWrapMesh, CartesianMesh, TwoPointFiniteVolumeGeometry, dim
export triangulate_mesh, tpfv_geometry, discretized_domain_tpfv_flow

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

export plot_primitives
function plot_primitives(mesh, plot_type; kwarg...)
    # By default, no plotting is supported
    return nothing
end

function meshscatter_primitives(g; line = false, kwarg...)
    tp = tpfv_geometry(g)
    pts = collect(tp.cell_centroids')
    for i in axes(pts, 1)
        pts[i, 3] *= -1
    end
    mapper = (Cells = identity, )
    @assert size(pts, 2) == 3 "Only supported for 3D meshes"
    vol = tp.volumes
    sizes = meshscatter_primitives_inner(pts, vol)
    return (points = pts, mapper = mapper, sizes = sizes, line = line)
end

function meshscatter_primitives_inner(pts, vol)
    dim = size(pts, 2)
    urng = maximum(pts, dims = 1) - minimum(pts, dims = 1)
    @. urng[urng == 0] = 1.0
    sizes = similar(pts)
    for i in axes(sizes, 1)
        v = vol[i]
        # Assume that the grid scaling holds for each cell and solve for the diameter 
        # in each direction
        gamma = (v/prod(urng))^(1.0/dim)
        for d in 1:dim
            sizes[i, d] = gamma*urng[d]/2
        end
    end
    return sizes
end

include("mrst.jl")
include("cart.jl")

