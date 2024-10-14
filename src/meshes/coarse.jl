export CoarseMesh

struct CoarseMesh{G, T} <: FiniteVolumeMesh
    parent::G
    partition::Vector{Int}
    partition_lookup::Vector{Vector{Int}}
    coarse_faces_to_fine::T
    coarse_boundary_to_fine::T
    face_neighbors::Vector{Tuple{Int, Int}}
    boundary_cells::Vector{Int}
    tags::MeshEntityTags{Int}
end

"""
    CoarseMesh(G::JutulMesh, p)

Construct a coarse mesh from a given `JutulMesh` that can be converted to an
`UnstructuredMesh` instance. The second argument `p` should be a partition
Vector with one entry per cell in the original grid that assigns that cell to a
coarse block. Should be one-indexed and the numbering should be sequential and
contain at least one fine cell for each coarse index. This is tested by the
function.
"""
function CoarseMesh(G, p)
    G = UnstructuredMesh(G)
    function sorted_pair(left, right)
        @assert left != right
        if left < right
            return (left, right)
        else
            return (right, left)
        end
    end

    nc = number_of_cells(G)
    @assert minimum(p) > 0
    m = maximum(p)
    @assert m <= nc
    for x in 1:m
        @assert count(isequal(x), p) > 0 "Bad partition: Block $x is empty."
    end
    cells_p = map(i -> findall(isequal(i), p), 1:m)
    # Partition internal faces
    coarse_faces = OrderedDict{Tuple{Int, Int}, Vector{Int}}()
    for (faceno, npair) in enumerate(G.faces.neighbors)
        l, r = npair
        p_l = p[l]
        p_r = p[r]
        if p_l != p_r
            lr = sorted_pair(p_l, p_r)
            if !haskey(coarse_faces, lr)
                coarse_faces[lr] = Int[]
            end
            push!(coarse_faces[lr], faceno)
        end
    end
    # Partition boundary faces This is simpler as we do not aggregate these
    # faces together since there is no information about where they overlap.
    # Construct a trivial mapping for the time being, until we have a suitable
    # merging algorithm.
    coarse_boundary_cells = p[G.boundary_faces.neighbors]
    nbf_fine = number_of_boundary_faces(G)
    bval = collect(1:nbf_fine)
    bpos = collect(1:(nbf_fine+1))
    cboundary = IndirectionMap(bval, bpos)

    function to_indir_facemap(coarse_faces)
        pos = Int[1]
        fmap = Int[]
        T = eltype(keys(coarse_faces))
        neigh = T[]
        for (k, v) in coarse_faces
            push!(neigh, k)
            for face in v
                push!(fmap, face)
            end
            n_added = length(v)
            push!(pos, pos[end]+n_added)
        end
        return (IndirectionMap(fmap, pos), neigh)
    end
    cfaces, coarse_neighbors = to_indir_facemap(coarse_faces)
    tags = MeshEntityTags()
    cg = CoarseMesh(G, p, cells_p, cfaces, cboundary, coarse_neighbors, coarse_boundary_cells, tags)
    initialize_entity_tags!(cg)
    return cg
end

function Base.show(io::IO, t::MIME"text/plain", g::CoarseMesh)
    nc = number_of_cells(g)
    nf = number_of_faces(g)
    nb = number_of_boundary_faces(g)
    print(io, "CoarseMesh with $nc cells, $nf faces and $nb boundary faces (on ")
    Base.show(io, t, g.parent)
    print(io, ")")
end

function dim(G::CoarseMesh)
    return dim(G.parent)
end

function mesh_z_is_depth(G::CoarseMesh)
    return mesh_z_is_depth(G.parent)
end

function number_of_cells(G::CoarseMesh)
    return length(G.partition_lookup)
end

function number_of_faces(G::CoarseMesh)
    return length(G.coarse_faces_to_fine)
end

function number_of_boundary_faces(G::CoarseMesh)
    return length(G.coarse_boundary_to_fine)
end

function compute_centroid_and_measure(CG::CoarseMesh, e::Cells, i)
    T = eltype(CG.parent.node_points)
    centroid = zero(T)
    vol = 0.0

    for cell in CG.partition_lookup[i]
        subcentroid, subvol = compute_centroid_and_measure(CG.parent, e, cell)
        centroid += subvol.*subcentroid
        vol += subvol
    end
    return (centroid./vol, vol)
end

function compute_centroid_and_measure(CG::CoarseMesh, e::Union{BoundaryFaces, Faces}, i)
    T = eltype(CG.parent.node_points)
    centroid = zero(T)
    area = 0.0
    get_map(::Faces) = CG.coarse_faces_to_fine
    get_map(::BoundaryFaces) = CG.coarse_boundary_to_fine

    for face in get_map(e)[i]
        subcentroid, subarea = compute_centroid_and_measure(CG.parent, e, face)
        centroid += subarea.*subcentroid
        area += subarea
    end
    return (centroid./area, area)
end

function face_normal(G::CoarseMesh, coarse_face, e = Faces())
    get_faces(::Faces) = (G.coarse_faces_to_fine, G.parent.faces.neighbors)
    get_faces(::BoundaryFaces) = (G.coarse_boundary_to_fine, G.parent.boundary_faces.neighbors)

    fine_faces, N = get_faces(e)
    p = G.partition
    get_block(::Faces) = G.face_neighbors[coarse_face][1]
    get_block(::BoundaryFaces) = G.boundary_cells[coarse_face]

    left_block = get_block(e)

    T = eltype(G.parent.node_points)
    normal = zero(T)
    for face in fine_faces[coarse_face]
        if p[first(N[face])] == left_block
            sgn = 1
        else
            sgn = -1
        end
        normal += sgn*face_normal(G.parent, face, e)
    end
    return normal./norm(normal, 2)
end

function get_neighborship(CG::CoarseMesh; internal = true)
    if internal
        nf = number_of_faces(CG)
        N = zeros(Int, 2, nf)
        for (i, lr) in enumerate(CG.face_neighbors)
            N[1, i] = lr[1]
            N[2, i] = lr[2]
        end
    else
        N = CG.boundary_cells
    end
    return N
end

function triangulate_mesh(m::CoarseMesh; kwarg...)
    points, triangulation, mapper = triangulate_mesh(m.parent; kwarg...)
    # TODO: This overload is a bit primitive. Plotting on the coarse grid will
    # be just as costly as plotting on the fine grid... Could cull some of the
    # interior cells to improve this.
    cell_ix = m.partition[mapper.indices.Cells]
    face_index = missing
    mapper = (
        Cells = (cell_data) -> cell_data[cell_ix],
        Faces = (face_data) -> face_data[face_index],
        indices = (Cells = cell_ix, Faces = face_index)
    )
    return (points = points, triangulation = triangulation, mapper = mapper)
end

function plot_primitives(mesh::CoarseMesh, plot_type; kwarg...)
    if plot_type == :mesh
        out = triangulate_mesh(mesh; kwarg...)
    else
        out = nothing
    end
    return out
end

function cell_dims(cg::CoarseMesh, pos)
    # TODO: This function is inefficient
    index = cell_index(cg, pos)
    g = cg.parent
    T = eltype(g.node_points)
    minv = Inf .+ zero(T)
    maxv = -Inf .+ zero(T)

    for (face, lr) in enumerate(cg.face_neighbors)
        l, r = lr
        if l == index || r == index
            pt, = compute_centroid_and_measure(cg, Faces(), face)
            if any(isnan.(pt))
                continue
            end
            minv = min.(pt, minv)
            maxv = max.(pt, maxv)
        end
    end
    for (bface, bcell) in enumerate(cg.boundary_cells)
        if bcell == index
            pt, = compute_centroid_and_measure(cg, BoundaryFaces(), bface)
            if any(isnan.(pt))
                continue
            end
            minv = min.(pt, minv)
            maxv = max.(pt, maxv)
        end
    end
    Δ = maxv - minv
    @assert all(x -> x > 0, Δ) "Cell dimensions were zero? Computed $Δ = $maxv - $minv for cell $index."
    return Tuple(Δ)
end
