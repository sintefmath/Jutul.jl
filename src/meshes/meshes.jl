# using Meshes, MeshViz

export MRSTWrapMesh, CartesianMesh, TwoPointFiniteVolumeGeometry, dim
export triangulate_mesh, tpfv_geometry

abstract type FiniteVolumeMesh <: JutulMesh end

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
    "Neighbor list 2 x nf for interior faces"
    neighbors
    "Area of interior faces"
    areas
    "Volumes of cells"
    volumes
    "Unit-normalized normals for each face"
    normals
    "Cell centroids"
    cell_centroids
    "Face centroids for interior faces"
    face_centroids
    "Half face map to cells"
    half_face_cells
    "Half face map to faces"
    half_face_faces
    "Boundary face areas"
    boundary_areas
    "Centroid of boundary faces"
    boundary_centroids
    "Boundary face normals"
    boundary_normals
    "Boundary cell neighbors"
    boundary_neighbors
    function TwoPointFiniteVolumeGeometry(neighbors, A, V, N, C_c, C_f, HF_c, HF_f, ∂A, ∂F_c, ∂neighbors, ∂N)
        nf = size(neighbors, 2)
        dim, nc = size(C_c)

        # Sanity check
        @assert dim == 2 || dim == 3 || dim == 1
        # Check cell centroids
        @assert size(C_c) == (dim, nc)
        # Check face centroids
        @assert size(C_f) == (dim, nf)
        # Check normals
        @assert size(N) == (dim, nf)
        # Check areas
        @assert length(A) == nf
        @assert length(V) == nc
        if ismissing(HF_c)
            @assert ismissing(HF_f)
        else
            # TODO: Add asserts here.
        end
        @assert ismissing(∂A) == ismissing(∂neighbors) == ismissing(∂N) == ismissing(∂F_c)
        if !ismissing(∂A)
            # TODO: Add asserts here.
        end
        return new(neighbors, vec(A), vec(V), N, C_c, C_f, HF_c, HF_f, ∂A, ∂F_c, ∂neighbors, ∂N)
    end
end

function TwoPointFiniteVolumeGeometry(
        neighbors, areas, volumes, normals, cell_centroids, face_centroids;
        half_face_cells = missing,
        half_face_faces = missing,
        boundary_areas = missing,
        boundary_normals = missing,
        boundary_centroids = missing,
        boundary_neighbors = missing
    )
    if ismissing(half_face_cells)
        @assert ismissing(half_face_faces)
        nc = length(volumes)
        half_face_faces, facepos = get_facepos(neighbors, nc)
        half_face_cells = similar(half_face_faces)
        for i in 1:nc
            for j in facepos[i]:(facepos[i+1]-1)
                half_face_cells[j] = i
            end
        end
    end
    # Call full constructor
    TwoPointFiniteVolumeGeometry(
        neighbors,
        areas,
        volumes,
        normals,
        cell_centroids,
        face_centroids,
        half_face_cells,
        half_face_faces,
        boundary_areas,
        boundary_centroids,
        boundary_normals,
        boundary_neighbors
    )
end

function count_entities(g::JutulMesh, ::Cells)
    return number_of_cells(g)
end

function count_entities(g::JutulMesh, ::Faces)
    return number_of_faces(g)
end

function count_entities(g::JutulMesh, ::BoundaryFaces)
    return number_of_boundary_faces(g)
end

function mesh_z_is_depth(g::JutulMesh)
    return false
end

dim(g::TwoPointFiniteVolumeGeometry) = size(g.cell_centroids, 1)

"""
    dim(g)::Integer

Get the dimension of a mesh.
"""
dim(t::JutulMesh) = 2

"""
    number_of_cells(g)::Integer

Get the number of cells in a mesh.
"""
number_of_cells(t::JutulMesh) = 1

"""
    number_of_faces(g)::Integer

Get the number of faces in a mesh.
"""
number_of_faces(G) = size(get_neighborship(G), 2)

"""
    number_of_boundary_faces(g)

Get the number of boundary/exterior faces in a mesh.
"""
function number_of_boundary_faces

end

export plot_primitives
function plot_primitives(mesh, plot_type; kwarg...)
    # By default, no plotting is supported
    return nothing
end

function meshscatter_primitives(g; line = false, kwarg...)
    tp = tpfv_geometry(g)
    pts = collect(tp.cell_centroids')
    if size(pts, 2) == 3
        for i in axes(pts, 1)
            pts[i, 3] *= -1
        end
    end
    mapper = (Cells = identity, )
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
include("unstructured/unstructured.jl")
include("coarse.jl")
include("trajectories.jl")
include("extruded.jl")

function declare_entities(G::JutulMesh)
    nf = number_of_faces(G)
    nc = number_of_cells(G)
    nbnd = number_of_boundary_faces(G)
    return [
            (entity = Cells(), count = nc),
            (entity = Faces(), count = nf),
            (entity = BoundaryFaces(), count = nbnd),
            (entity = HalfFaces(), count = 2*nf)
        ]
end

function add_default_domain_data!(Ω::DataDomain, m::FiniteVolumeMesh; geometry = missing)
    if ismissing(geometry)
        geometry = tpfv_geometry(m)
    end
    geom_pairs = (
        Pair(Faces(), [:neighbors, :areas, :normals, :face_centroids]),
        Pair(Cells(), [:cell_centroids, :volumes]),
        Pair(HalfFaces(), [:half_face_cells, :half_face_faces]),
        Pair(BoundaryFaces(), [:boundary_areas, :boundary_centroids, :boundary_normals, :boundary_neighbors])
    )
    for (entity, names) in geom_pairs
        if hasentity(Ω, entity)
            for name in names
                Ω[name, entity] = getproperty(geometry, name)
            end
        end
    end
end

function tpfv_geometry(G::FiniteVolumeMesh)
    D = dim(G)
    nc = number_of_cells(G)
    nf = number_of_faces(G)
    # Cell geometry
    T = float_type(G)
    cell_centroids = zeros(T, D, nc)
    volumes = zeros(T, nc)
    for i in 1:nc
        c, volumes[i] = compute_centroid_and_measure(G, Cells(), i)
        for d in 1:D
            cell_centroids[d, i] = c[d]
        end
    end
    # Face geometry
    N = get_neighborship(G)

    face_areas, face_normals, face_centroids = face_geometry_helper(G, D, nc, nf, Faces())
    boundary_areas, boundary_normals, boundary_centroids = face_geometry_helper(G, D, nc, nf, BoundaryFaces())

    return TwoPointFiniteVolumeGeometry(
        N,
        face_areas,
        volumes,
        face_normals,
        cell_centroids,
        face_centroids,
        boundary_areas = boundary_areas,
        boundary_normals = boundary_normals,
        boundary_centroids = boundary_centroids,
        boundary_neighbors = get_neighborship(G, internal = false)
    )
end

function face_geometry_helper(G, D, nc, nf, e)
    nf = count_entities(G, e)
    T = float_type(G)
    areas = zeros(T, nf)
    normals = zeros(T, D, nf)
    centroids = zeros(T, D, nf)
    for f in 1:nf
        c, a = compute_centroid_and_measure(G, e, f)
        for d in 1:D
            centroids[d, f] = c[d]
        end
        areas[f] = a
        # Assume correct order for normal
        normal = face_normal(G, f, e)

        for d in 1:D
            normals[d, f] = normal[d]
        end
    end
    return (areas, normals, centroids)
end


import Base: promote_rule
promote_rule(::Type{UnstructuredMesh}, ::Type{CartesianMesh}) = UnstructuredMesh

function cellmap_to_posmap(x, num_cells = length(x))
    posmap = Int[1]
    vals = Int[]
    for cell in 1:num_cells
        vals_cell = get(x, cell, [])
        push!(posmap, posmap[end] + length(vals_cell))
        for f in vals_cell
            push!(vals, f)
        end
    end
    return (vals, posmap)
end

include("RadialMeshes/RadialMeshes.jl")
include("EmbeddedMeshes/EmbeddedMeshes.jl")
