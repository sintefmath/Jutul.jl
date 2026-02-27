"""
Struct that contains mappings for a set of faces that are made up of nodes and
are part of cells.
"""
struct FaceMap{M, N}
    "IndirectionMap that maps cells to faces"
    cells_to_faces::M
    "IndirectionMap that maps faces to nodes"
    faces_to_nodes::M
    "Neighbors for each face"
    neighbors::Vector{N}
    function FaceMap(c2f::M, f2n::M, neigh::Vector{N}; check = true) where {M, N}
        nc = length(c2f)
        nf = length(f2n)
        @assert length(neigh) == nf
        for c in 1:nc
            for face in c2f[c]
                @assert c in neigh[face]
            end
        end
        new{M, N}(c2f, f2n, neigh)
    end
end

struct UnstructuredMesh{D, S, IM, IF, M, F, BM, NM, T} <: FiniteVolumeMesh
    structure::S
    faces::FaceMap{M, Tuple{Int, Int}}
    boundary_faces::FaceMap{M, Int}
    node_points::Vector{SVector{D, F}}
    cell_map::IM
    face_map::IF
    boundary_map::BM
    node_map::NM
    "Tags on cells/faces/nodes"
    tags::MeshEntityTags{T}
    "Interpret z coordinate as depths in plots"
    z_is_depth::Bool
end

function float_type(::UnstructuredMesh{<:Any, <:Any, <:Any, <:Any, <:Any, T, <:Any, <:Any, <:Any}) where T
    return T
end

function convert_coord_points(points::AbstractMatrix{F}) where F
    dim, nn = size(points)
    @assert dim <= 3
    T_xyz = SVector{dim, eltype(points)}
    new_points = Vector{T_xyz}(undef, nn)
    for i in 1:nn
        new_points[i] = T_xyz(points[:, i])
    end
    return (new_points, dim)
end

function convert_coord_points(pts::Vector{SVector{N, F}}) where {N, F}
    return (pts, N)
end

function convert_coord_points(pts::Vector{NTuple{N, F}}) where {N, F}
    T = SVector{N, F}
    nn = length(pts)
    new_points = Vector{T}(undef, nn)
    for i in 1:nn
        new_points[i] = T(pts[i])
    end
    return (new_points, N)
end

function convert_neighborship(N::AbstractMatrix; nf = size(N, 2), nc = maximum(N), kwarg...)
    @assert size(N, 1) == 2 "Expected neighborship of (2,nf), was $(size(N))"

    N_new = Vector{Tuple{Int, Int}}(undef, nf)
    for i in 1:nf
        N_new[i] = (N[1, i], N[2, i])
    end
    return convert_neighborship(N_new; nc = nc, nf = nf, kwarg...)
end

function convert_neighborship(N::Vector{Tuple{Int, Int}}; nc = nothing, nf = nothing, allow_zero = false)
    # Just do asserts
    if !isnothing(nf)
        @assert length(N) == nf
    end
    if !isnothing(nc)
        for (f, t) in enumerate(N)
            for (i, c) in enumerate(t)
                @assert c <= nc "Neighborship exceeded $nc in face $f cell $i: neighborship was $t"
                if allow_zero
                    @assert c >= 0 "Neighborship was negative in face $f cell $i: neighborship was $t"
                else
                    @assert c > 0 "Neighborship was non-positive in face $f cell $i: neighborship was $t"
                end
            end
        end
    end
    return N
end

# Outer constructor: Take MRST format and turn into separate lists for interior and boundary
function UnstructuredMesh(cells_faces, cells_facepos, faces_nodes, faces_nodespos, node_points, face_neighbors::Matrix{Int}; kwarg...)
    nc = length(cells_facepos)-1
    nf = length(faces_nodespos)-1
    node_points, dim = convert_coord_points(node_points)
    nn = length(node_points)

    face_neighbors = convert_neighborship(face_neighbors, nf = nf, nc = nc, allow_zero = true)
    @assert dim <= 3
    @assert dim >= 1
    # @assert maximum(face_neighbors) <= nc
    # @assert minimum(face_neighbors) >= 0
    @assert maximum(faces_nodes) <= nn "Too few nodes provided"

    new_faces_nodes = similar(faces_nodes, 0)
    new_faces_nodespos = [1]

    boundary_faces_nodes = similar(faces_nodes, 0)
    boundary_faces_nodespos = [1]

    faceindex = Int[]
    int_indices = Int[]
    bnd_indices = Int[]

    added_interior = 0
    added_boundary = 0

    for face in 1:nf
        l, r = face_neighbors[face]
        npos = faces_nodespos[face]:(faces_nodespos[face+1]-1)
        n = length(npos)
        bnd = l == 0 || r == 0
        if bnd
            if l == 0
                npos = reverse(npos)
            end

            for i in npos
                push!(boundary_faces_nodes, faces_nodes[i])
            end
            
            push!(boundary_faces_nodespos, boundary_faces_nodespos[end] + n)
            added_boundary += 1
            # Minus sign means boundary index
            push!(faceindex, -added_boundary)
            push!(bnd_indices, face)
        else
            for i in npos
                push!(new_faces_nodes, faces_nodes[i])
            end
            push!(new_faces_nodespos, new_faces_nodespos[end] + n)
            added_interior += 1
            # Positive sign for interior
            push!(faceindex, added_interior)
            push!(int_indices, face)
        end
    end
    @assert added_boundary + added_interior == nf

    new_cells_faces = similar(cells_faces, 0)
    new_cells_facepos = [1]

    boundary_cells_faces = similar(cells_faces, 0)
    boundary_cells_facepos = [1]

    for cell in 1:nc
        bnd_count = 0
        int_count = 0
        for fp in cells_facepos[cell]:(cells_facepos[cell+1]-1)
            face = cells_faces[fp]
            ix = faceindex[face]
            if ix > 0
                # Interior
                push!(new_cells_faces, ix)
                int_count += 1
            else
                # Boundary
                push!(boundary_cells_faces, abs(ix))
                bnd_count += 1
            end
        end
        push!(new_cells_facepos, new_cells_facepos[end] + int_count)
        push!(boundary_cells_facepos, boundary_cells_facepos[end] + bnd_count)
    end

    int_neighbors = face_neighbors[int_indices]
    bnd_cells = Int[]
    for i in bnd_indices
        l, r = face_neighbors[i]
        @assert l == 0 || r == 0
        push!(bnd_cells, l + r)
    end
    return UnstructuredMesh(
        new_cells_faces,
        new_cells_facepos,
        boundary_cells_faces,
        boundary_cells_facepos,
        new_faces_nodes,
        new_faces_nodespos,
        boundary_faces_nodes,
        boundary_faces_nodespos,
        node_points,
        int_neighbors,
        bnd_cells;
        kwarg...,
        face_map = faceindex
        )
end

# Middle constructor, do some checking and convert to real data structures (SVector, tuple neighbors, indirection maps)
function UnstructuredMesh(
        new_cells_faces,
        new_cells_facepos,
        boundary_cells_faces,
        boundary_cells_facepos,
        faces_nodes,
        faces_nodespos,
        boundary_faces_nodes,
        boundary_faces_nodespos,
        node_points,
        int_neighbors,
        bnd_cells;
        kwarg...
    )
    cells_to_faces = IndirectionMap(new_cells_faces, new_cells_facepos)
    cells_to_bnd = IndirectionMap(boundary_cells_faces, boundary_cells_facepos)
    faces_to_nodes = IndirectionMap(faces_nodes, faces_nodespos)
    bnd_to_nodes = IndirectionMap(boundary_faces_nodes, boundary_faces_nodespos)

    nc = length(cells_to_faces)
    nb = length(bnd_to_nodes)
    nf = length(faces_to_nodes)

    node_points, dim = convert_coord_points(node_points)
    nn = length(node_points)

    @assert dim <= 3
    @assert dim >= 1
    int_neighbors = convert_neighborship(int_neighbors)
    @assert length(bnd_cells) == nb
    bnd_cells::AbstractVector
    max_bnd = maximum(bnd_cells, init = 1)
    max_bnd <= nc || throw(ArgumentError("Maximum boundary cell $max_bnd exceeded number of cells $nc."))
    min_bnd = minimum(bnd_cells, init = nc)
    minimum(bnd_cells, init = nc) > 0 || throw(ArgumentError("Minimum boundary cell $min_bnd was less than 1."))

    @assert maximum(faces_to_nodes.vals, init = 0) <= nn "Too few nodes provided"
    return UnstructuredMesh(cells_to_faces, cells_to_bnd, faces_to_nodes, bnd_to_nodes, node_points, int_neighbors, bnd_cells; kwarg...)
end

function UnstructuredMesh(
        cells_to_faces::T,
        cells_to_bnd::T,
        faces_to_nodes::T,
        bnd_to_nodes::T,
        node_points::Vector{SVector{dim, F}},
        face_neighbors::Vector{Tuple{Int, Int}},
        boundary_cells::Vector{Int};
        cell_map::IM = nothing,
        face_map::IF = nothing,
        boundary_map::BM = nothing,
        node_map::NM = nothing,
        structure::S = nothing,
        z_is_depth = false
    ) where {T<:IndirectionMap, IM, IF, dim, F<:Real, BM, NM, S}
    faces = FaceMap(cells_to_faces, faces_to_nodes, face_neighbors)
    bnd = FaceMap(cells_to_bnd, bnd_to_nodes, boundary_cells)
    @assert length(face_neighbors) == length(faces_to_nodes)
    @assert length(boundary_cells) == length(bnd_to_nodes)
    tags = MeshEntityTags()
    g = UnstructuredMesh{dim, S, IM, IF, T, F, BM, NM, Int}(
        structure,
        faces,
        bnd,
        node_points,
        cell_map,
        face_map,
        boundary_map,
        node_map,
        tags,
        z_is_depth
    )
    initialize_entity_tags!(g)
    return g
end

function UnstructuredMesh(G::UnstructuredMesh)
    return G
end

function mesh_z_is_depth(G::UnstructuredMesh)
    return G.z_is_depth
end

"""
    UnstructuredMesh(g::CartesianMesh)

Convert `CartesianMesh` instance to unstructured grid. Note that the mesh must
be 2D and 3D for a 1-to-1 conversion. 1D meshes are implicitly converted to 2D.
"""
function UnstructuredMesh(g::CartesianMesh; warn_1d = true, kwarg...)
    d = dim(g)
    if d == 1
        if warn_1d
            @warn "Conversion from CartesianMesh to UnstructuredMesh is only fully supported for 2D/3D grids. Converting 1D grid to 2D."
        end
        nx = number_of_cells(g)
        dy = 1.0
        dx = only(g.deltas)
        X0 = only(g.origin)
        Y0 = 0.0
        wrap(x::AbstractFloat, n) = fill(x, n)
        wrap(x, n) = x
        g = CartesianMesh(
            (nx, 1),
            (wrap(dx, nx), wrap(1.0, 1)),
            origin = [X0, Y0]
        )
        out = UnstructuredMesh(g)
    else
        out = unstructured_from_cart(g, Val(d); kwarg...)
    end
end

Base.convert(::Type{UnstructuredMesh}, g::CartesianMesh) = UnstructuredMesh(g)

function unstructured_from_cart(g, ::Val{2}; kwarg...)
    d = dim(g)
    @assert d == 2
    nx, ny, = grid_dims_ijk(g)

    X0, Y0 = g.origin

    nc = number_of_cells(g)
    nf = number_of_faces(g)
    nbf = number_of_boundary_faces(g)
    num_nodes_x = nx+1
    num_nodes_y = ny+1
    num_nodes = num_nodes_x*num_nodes_y
    nodeix = reshape(1:num_nodes, num_nodes_x, num_nodes_y)

    node_points = Vector{SVector{2, Float64}}()
    dx, dy = g.deltas
    function get_point(D::T, i) where {T<:Real}
        newpt = (i-1)*D
        return newpt::T
    end
    function get_point(D::Union{NTuple{N, T}, Vector{T}}, i) where {T<:Real, N}
        pt = zero(T)
        for j in 1:(i-1)
            pt += D[j]
        end
        return pt::T
    end
    for j in 1:num_nodes_y
        Y = get_point(dy, j)
        for i in 1:num_nodes_x
            X = get_point(dx, i)
            XY = SVector{2, Float64}(X + X0, Y + Y0)
            push!(node_points, XY)
        end
    end
    @assert length(node_points) == length(nodeix)
    cell_to_faces = Vector{Vector{Int}}()
    sizehint!(cell_to_faces, nc)
    for i in 1:nc
        push!(cell_to_faces, Int[])
    end
    cell_to_boundary = Vector{Vector{Int}}()
    sizehint!(cell_to_boundary, nc)
    for i in 1:nc
        push!(cell_to_boundary, Int[])
    end

    int_neighbors = Vector{Tuple{Int, Int}}()
    sizehint!(int_neighbors, nf)
    # Note: The following loops are arranged to reproduce the MRST ordering.
    function insert_face!(nodes, pos, arg...)
        for node in arg
            push!(nodes, node)
        end
        push!(pos, pos[end] + length(arg))
    end
    function add_internal_neighbor!(t, D)
        x, y = t
        index = cell_index(g, t)
        l = index
        r = cell_index(g, (x + (D == 1), y + (D == 2)))
        push!(int_neighbors, (l, r))
    end
    faces_nodes = Int[]
    faces_nodespos = Int[1]
    sizehint!(faces_nodes, 4*nf)
    sizehint!(faces_nodespos, nf+1)
    # Faces with X-normal > 0
        for y in 1:ny
            for x in 2:nx
                p1 = nodeix[x, y]
                p2 = nodeix[x, y+1]
                insert_face!(faces_nodes, faces_nodespos, p1, p2)
                add_internal_neighbor!((x-1, y), 1)
            end
        end
    # Faces with Y-normal > 0
    for y in 2:ny
        for x in 1:nx
            p1 = nodeix[x, y]
            p2 = nodeix[x+1, y]
            insert_face!(faces_nodes, faces_nodespos, p2, p1)
            add_internal_neighbor!((x, y-1), 2)
        end
    end

    boundary_faces_nodes = Int[]
    boundary_faces_nodespos = Int[1]

    sizehint!(boundary_faces_nodes, 4*nbf)
    sizehint!(boundary_faces_nodespos, nbf+1)

    bnd_cells = Int[]
    sizehint!(bnd_cells, nbf)
    function add_boundary_cell!(t, D)
        index = cell_index(g, t)
        push!(bnd_cells, index)
    end
    for y in 1:ny
        for x in [1, nx+1]
            p1 = nodeix[x, y]
            p2 = nodeix[x, y+1]
            if x == 1
                insert_face!(boundary_faces_nodes, boundary_faces_nodespos, p2, p1)
                add_boundary_cell!((x, y), 1)
            else
                insert_face!(boundary_faces_nodes, boundary_faces_nodespos, p1, p2)
                add_boundary_cell!((x-1, y), 1)
            end
        end
    end
    # Faces with Y-normal > 0
    for x in 1:nx
        for y in [1, ny+1]
            p1 = nodeix[x, y]
            p2 = nodeix[x+1, y]
            if y == 1
                insert_face!(boundary_faces_nodes, boundary_faces_nodespos, p1, p2)
                add_boundary_cell!((x, y), 2)
            else
                insert_face!(boundary_faces_nodes, boundary_faces_nodespos, p2, p1)
                add_boundary_cell!((x, y-1), 2)
            end
        end
    end
    cells_faces, cells_facepos = get_facepos(reinterpret(reshape, Int, int_neighbors), nc)

    for (bf, bc) in enumerate(bnd_cells)
        push!(cell_to_boundary[bc], bf)
    end

    boundary_cells_faces = Int[]
    boundary_cells_facepos = Int[1]
    for bfaces in cell_to_boundary
        n = length(bfaces)
        for bf in bfaces
            push!(boundary_cells_faces, bf)
        end
        push!(boundary_cells_facepos, boundary_cells_facepos[end]+n)
    end

    return UnstructuredMesh(
        cells_faces,
        cells_facepos,
        boundary_cells_faces,
        boundary_cells_facepos,
        faces_nodes,
        faces_nodespos,
        boundary_faces_nodes,
        boundary_faces_nodespos,
        node_points,
        int_neighbors,
        bnd_cells;
        structure = CartesianIndex(nx, ny),
        cell_map = 1:nc,
        kwarg...
    )
end

function unstructured_from_cart(g, ::Val{3}; kwarg...)
    d = dim(g)
    @assert d == 3
    nx, ny, nz = grid_dims_ijk(g)
    X0, Y0, Z0 = g.origin

    nc = number_of_cells(g)
    nf = number_of_faces(g)
    nbf = number_of_boundary_faces(g)
    num_nodes_x = nx+1
    num_nodes_y = ny+1
    num_nodes_z = nz+1
    num_nodes = num_nodes_x*num_nodes_y*num_nodes_z
    nodeix = reshape(1:num_nodes, num_nodes_x, num_nodes_y, num_nodes_z)

    dx, dy, dz = g.deltas
    Float_T = promote_type(eltype(dx), eltype(dy), eltype(dz), eltype(X0), eltype(Y0), eltype(Z0))
    node_points = Vector{SVector{3, Float_T}}()

    sizehint!(node_points, num_nodes_x*num_nodes_y*num_nodes_z)
    xpts = get_cartesian_points(dx, num_nodes_x)
    ypts = get_cartesian_points(dy, num_nodes_y)
    zpts = get_cartesian_points(dz, num_nodes_z)
    for k in 1:num_nodes_z
        Z = zpts[k]
        for j in 1:num_nodes_y
            Y = ypts[j]
            for i in 1:num_nodes_x
                X = xpts[i]
                XYZ = SVector{3, Float_T}(X + X0, Y + Y0, Z + Z0)
                push!(node_points, XYZ)
            end
        end
    end
    @assert length(node_points) == length(nodeix)
    cell_to_faces = Vector{Vector{Int}}()
    sizehint!(cell_to_faces, nc)
    for i in 1:nc
        push!(cell_to_faces, Int[])
    end
    cell_to_boundary = Vector{Vector{Int}}()
    sizehint!(cell_to_boundary, nc)
    for i in 1:nc
        push!(cell_to_boundary, Int[])
    end

    int_neighbors = Vector{Tuple{Int, Int}}()
    sizehint!(int_neighbors, nf)
    # Note: The following loops are arranged to reproduce the MRST ordering.
    function insert_face!(nodes, pos, arg...)
        for node in arg
            push!(nodes, node)
        end
        push!(pos, pos[end] + length(arg))
    end
    function add_internal_neighbor!(t, D)
        x, y, z = t
        index = cell_index(g, t)
        l = index
        r = cell_index(g, (x + (D == 1), y + (D == 2), z + (D == 3)))
        push!(int_neighbors, (l, r))
    end
    faces_nodes = Int[]
    faces_nodespos = Int[1]
    sizehint!(faces_nodes, 4*nf)
    sizehint!(faces_nodespos, nf+1)
    # Faces with X-normal > 0
    for z = 1:nz
        for y in 1:ny
            for x in 2:nx
                p1 = nodeix[x, y, z]
                p2 = nodeix[x, y+1, z]
                p3 = nodeix[x, y+1, z+1]
                p4 = nodeix[x, y, z+1]
                insert_face!(faces_nodes, faces_nodespos, p1, p2, p3, p4)
                add_internal_neighbor!((x-1.0, y, z), 1)
            end
        end
    end
    # Faces with Y-normal > 0
    for y in 2:ny
        for z = 1:nz
            for x in 1:nx
                p1 = nodeix[x, y, z+1]
                p2 = nodeix[x+1, y, z+1]
                p3 = nodeix[x+1, y, z]
                p4 = nodeix[x, y, z]
                insert_face!(faces_nodes, faces_nodespos, p1, p2, p3, p4)
                add_internal_neighbor!((x, y-1.0, z), 2)
            end
        end
    end
    # Faces with Z-normal > 0
    for z = 2:nz
        for y in 1:ny
            for x in 1:nx
                p1 = nodeix[x+1, y, z]
                p2 = nodeix[x+1, y+1, z]
                p3 = nodeix[x, y+1, z]
                p4 = nodeix[x, y, z]
                insert_face!(faces_nodes, faces_nodespos, p1, p2, p3, p4)
                add_internal_neighbor!((x, y, z-1.0), 3)
            end
        end
    end

    boundary_faces_nodes = Int[]
    boundary_faces_nodespos = Int[1]

    sizehint!(boundary_faces_nodes, 4*nbf)
    sizehint!(boundary_faces_nodespos, nbf+1)

    bnd_cells = Int[]
    sizehint!(bnd_cells, nbf)
    function add_boundary_cell!(t, D)
        index = cell_index(g, t)
        push!(bnd_cells, index)
    end
    for y in 1:ny
        for z = 1:nz
            for x in [1, nx+1]
                p1 = nodeix[x, y, z+1]
                p2 = nodeix[x, y+1, z+1]
                p3 = nodeix[x, y+1, z]
                p4 = nodeix[x, y, z]
                if x == 1
                    insert_face!(boundary_faces_nodes, boundary_faces_nodespos, p1, p2, p3, p4)
                    add_boundary_cell!((x, y, z), 1)
                else
                    insert_face!(boundary_faces_nodes, boundary_faces_nodespos, p4, p3, p2, p1)
                    add_boundary_cell!((x-1.0, y, z), 1)
                end
            end
        end
    end
    # Faces with Y-normal > 0
    for x in 1:nx
        for z = 1:nz
            for y in [1, ny+1]
                p1 = nodeix[x, y, z]
                p2 = nodeix[x+1, y, z]
                p3 = nodeix[x+1, y, z+1]
                p4 = nodeix[x, y, z+1]
                if y == 1
                    insert_face!(boundary_faces_nodes, boundary_faces_nodespos, p1, p2, p3, p4)
                    add_boundary_cell!((x, y, z), 2)
                else
                    insert_face!(boundary_faces_nodes, boundary_faces_nodespos, p4, p3, p2, p1)
                    add_boundary_cell!((x, y-1.0, z), 2)
                end
            end
        end
    end
    # Faces with Z-normal > 0
    for x in 1:nx
        for y in 1:ny
            for z = [1, nz+1]
                p1 = nodeix[x, y, z]
                p2 = nodeix[x, y+1, z]
                p3 = nodeix[x+1, y+1, z]
                p4 = nodeix[x+1, y, z]
                if z == 1
                    insert_face!(boundary_faces_nodes, boundary_faces_nodespos, p1, p2, p3, p4)
                    add_boundary_cell!((x, y, z), 3)
                else
                    insert_face!(boundary_faces_nodes, boundary_faces_nodespos, p4, p3, p2, p1)
                    add_boundary_cell!((x, y, z-1.0), 3)
                end
            end
        end
    end
    cells_faces, cells_facepos = get_facepos(reinterpret(reshape, Int, int_neighbors), nc)

    for (bf, bc) in enumerate(bnd_cells)
        push!(cell_to_boundary[bc], bf)
    end

    boundary_cells_faces = Int[]
    boundary_cells_facepos = Int[1]
    for bfaces in cell_to_boundary
        n = length(bfaces)
        for bf in bfaces
            push!(boundary_cells_faces, bf)
        end
        push!(boundary_cells_facepos, boundary_cells_facepos[end]+n)
    end

    return UnstructuredMesh(
        cells_faces,
        cells_facepos,
        boundary_cells_faces,
        boundary_cells_facepos,
        faces_nodes,
        faces_nodespos,
        boundary_faces_nodes,
        boundary_faces_nodespos,
        node_points,
        int_neighbors,
        bnd_cells;
        structure = CartesianIndex(nx, ny, nz),
        cell_map = 1:nc,
        kwarg...
    )
end

function get_cartesian_points(D::T, n) where {T<:Real}
    return map(i -> (i-1)*D, 1:n)
end

function get_cartesian_points(D::Union{NTuple{N, T}, Vector{T}}, n) where {T<:Real, N}
    points = Vector{T}(undef, n)
    pt = zero(T)
    for i in 1:n
        points[i] = pt
        if i < n
            pt += D[i]
        end
    end
    return points
end

function UnstructuredMesh(g::MRSTWrapMesh; kwarg...)
    G_raw = g.data
    faces_raw = Int.(vec(G_raw.cells.faces[:, 1]))
    facePos_raw = Int.(vec(G_raw.cells.facePos[:, 1]))
    nodes_raw = Int.(vec(G_raw.faces.nodes[:, 1]))
    nodePos_raw = Int.(vec(G_raw.faces.nodePos[:, 1]))
    coord = collect(G_raw.nodes.coords')
    N_raw = Int.(G_raw.faces.neighbors')
    return UnstructuredMesh(faces_raw, facePos_raw, nodes_raw, nodePos_raw, coord, N_raw; kwarg...)
end

function UnstructuredMesh(G_raw::AbstractDict; kwarg...)
    faces_raw = Int.(vec(G_raw["cells"]["faces"][:, 1]))
    facePos_raw = Int.(vec(G_raw["cells"]["facePos"][:, 1]))
    nodes_raw = Int.(vec(G_raw["faces"]["nodes"][:, 1]))
    nodePos_raw = Int.(vec(G_raw["faces"]["nodePos"][:, 1]))
    coord = collect(G_raw["nodes"]["coords"]')
    N_raw = Int.(G_raw["faces"]["neighbors"]')
    return UnstructuredMesh(faces_raw, facePos_raw, nodes_raw, nodePos_raw, coord, N_raw; kwarg...)
end

function mesh_linesegments(m;
        cells = nothing,
        faces = nothing,
        boundary_faces = nothing,
        outer = dim(m) == 3 && isnothing(faces) && isnothing(boundary_faces)
    )
    if isnothing(cells)
        cells = 1:number_of_cells(m)
    end
    if isnothing(faces)
        faces = 1:number_of_faces(m)
    end
    if isnothing(boundary_faces)
        boundary_faces = 1:number_of_boundary_faces(m)
    end
    if !(m isa UnstructuredMesh || m isa CoarseMesh)
        m = UnstructuredMesh(m)
    end
    nodes = Vector{Tuple{Int, Int}}()
    for face in faces
        if m isa UnstructuredMesh
            l, r = m.faces.neighbors[face]
        else
            # CoarseMesh
            l, r = m.face_neighbors[face]
        end
        if l > r
            r, l = l, r
        end
        l_in = l in cells
        r_in = r in cells
        a = outer && ((l_in && !r_in) || (r_in && !l_in))
        b = !outer && (l_in || r_in)
        if a || b
            add_mesh_linesegments_for_face!(nodes, m, face, boundary = false)
        end
    end

    for bf in boundary_faces
        if m isa UnstructuredMesh
            c = m.boundary_faces.neighbors[bf]
        else
            # CoarseMesh
            c = m.boundary_cells[bf]
        end
        if c in cells
            add_mesh_linesegments_for_face!(nodes, m, bf; boundary = true)
        end
    end
    if m isa UnstructuredMesh
        node_points = m.node_points
    else
        node_points = m.parent.node_points
    end
    pts = map(x -> (node_points[x[1]], node_points[x[2]]), nodes)
    return pts
end

function add_mesh_linesegments_for_face!(nodes, m::UnstructuredMesh, face::Int; boundary::Bool = false)
    if boundary
        f2n = m.boundary_faces.faces_to_nodes[face]
    else
        f2n = m.faces.faces_to_nodes[face]
    end
    prev_node = missing
    for node in f2n
        if !ismissing(prev_node)
            push!(nodes, (prev_node, node))
        end
        prev_node = node
    end
    push!(nodes, (first(f2n), f2n[end]))
    return nodes
end

function add_mesh_linesegments_for_face!(nodes, m::CoarseMesh, face::Int; boundary::Bool = false)
    if boundary
        faces = m.coarse_boundary_to_fine[face]
    else
        faces = m.coarse_faces_to_fine[face]
    end
    # Segments that only appear once are true edges
    seg_count = Dict{Tuple{Int, Int}, Int}()
    segments = Tuple{Int, Int}[]
    for fine_face in faces
        add_mesh_linesegments_for_face!(segments, m.parent, fine_face, boundary = boundary)
        for (i, seg) in enumerate(segments)
            l, r = seg
            # Sort pairs
            if l < r
                l, r = r, l
            end
            k = (l, r)
            if !haskey(seg_count, k)
                seg_count[k] = 0
            end
            seg_count[k] += 1
        end
        empty!(segments)
    end
    for (k, v) in pairs(seg_count)
        if v == 1
            push!(nodes, k)
        end
    end
    return nodes
end
