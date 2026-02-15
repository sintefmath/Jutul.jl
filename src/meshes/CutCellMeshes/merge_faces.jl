"""
    merge_coplanar_faces(mesh; coplanar_tol=1e-8)

Post-process a mesh by merging coplanar faces that share the same cell pair
(for interior faces) or the same cell (for boundary faces) and share at least
two nodes.  The merged polygon is re-ordered so that its normal points from
left to right cell (interior) or outward from the cell (boundary).

Returns a new `UnstructuredMesh`.
"""
function merge_coplanar_faces(
    mesh::UnstructuredMesh{3};
    coplanar_tol::Real = 1e-8
)
    nc = number_of_cells(mesh)
    nf = number_of_faces(mesh)
    nb = number_of_boundary_faces(mesh)
    node_points = mesh.node_points

    # ------------------------------------------------------------------
    # 1. Merge interior faces
    # ------------------------------------------------------------------
    # Group faces by their (sorted) neighbor pair
    pair_to_faces = Dict{Tuple{Int,Int}, Vector{Int}}()
    for f in 1:nf
        l, r = mesh.faces.neighbors[f]
        key = l < r ? (l, r) : (r, l)
        push!(get!(pair_to_faces, key, Int[]), f)
    end

    merged_face_nodes = Vector{Int}[]          # nodes per merged face
    merged_face_neighbors = Tuple{Int,Int}[]   # (left, right) per merged face

    face_done = falses(nf)
    for (pair, fgroup) in pair_to_faces
        _merge_face_group!(
            merged_face_nodes, merged_face_neighbors,
            fgroup, face_done, mesh, node_points, coplanar_tol, false
        )
    end
    # Any un-merged face is added as-is
    for f in 1:nf
        if !face_done[f]
            push!(merged_face_nodes, collect(mesh.faces.faces_to_nodes[f]))
            push!(merged_face_neighbors, mesh.faces.neighbors[f])
        end
    end

    # ------------------------------------------------------------------
    # 2. Merge boundary faces
    # ------------------------------------------------------------------
    cell_to_bfaces = Dict{Int, Vector{Int}}()
    for bf in 1:nb
        c = mesh.boundary_faces.neighbors[bf]
        push!(get!(cell_to_bfaces, c, Int[]), bf)
    end

    merged_bnd_nodes = Vector{Int}[]
    merged_bnd_cells = Int[]

    bnd_done = falses(nb)
    for (c, bgroup) in cell_to_bfaces
        _merge_face_group!(
            merged_bnd_nodes, merged_bnd_cells,
            bgroup, bnd_done, mesh, node_points, coplanar_tol, true
        )
    end
    for bf in 1:nb
        if !bnd_done[bf]
            push!(merged_bnd_nodes, collect(mesh.boundary_faces.faces_to_nodes[bf]))
            push!(merged_bnd_cells, mesh.boundary_faces.neighbors[bf])
        end
    end

    # ------------------------------------------------------------------
    # 3. Build the new mesh
    # ------------------------------------------------------------------
    return _rebuild_mesh(mesh, node_points, merged_face_nodes, merged_face_neighbors,
                         merged_bnd_nodes, merged_bnd_cells)
end

# ==========================================================================
#  Internal helpers
# ==========================================================================

"""
    _merge_face_group!(out_nodes, out_meta, fgroup, done, mesh, node_points, tol, is_boundary)

Given a group of faces that all belong to the same cell pair (interior) or
same cell (boundary), find clusters of coplanar faces that share nodes and
merge each cluster into a single face.
"""
function _merge_face_group!(
    out_nodes::Vector{Vector{Int}},
    out_meta,
    fgroup::Vector{Int},
    done::BitVector,
    mesh::UnstructuredMesh{3},
    node_points::Vector{SVector{3, T}},
    tol::Real,
    is_boundary::Bool
) where T
    nf = length(fgroup)
    if nf <= 1
        return  # nothing to merge
    end

    # Compute normal for each face
    face_normals = Vector{SVector{3, T}}(undef, nf)
    face_node_sets = Vector{Set{Int}}(undef, nf)
    face_node_lists = Vector{Vector{Int}}(undef, nf)
    for (i, f) in enumerate(fgroup)
        if is_boundary
            fnodes = collect(mesh.boundary_faces.faces_to_nodes[f])
        else
            fnodes = collect(mesh.faces.faces_to_nodes[f])
        end
        face_node_lists[i] = fnodes
        face_node_sets[i] = Set(fnodes)
        pts = [node_points[n] for n in fnodes]
        face_normals[i] = polygon_normal(pts)
    end

    # Union-Find clustering: merge faces that are coplanar AND share ≥ 2 nodes
    parent = collect(1:nf)
    function uf_find(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end
    function uf_unite(a, b)
        ra, rb = uf_find(a), uf_find(b)
        if ra != rb
            parent[ra] = rb
        end
    end

    for i in 1:nf
        for j in (i+1):nf
            shared = length(intersect(face_node_sets[i], face_node_sets[j]))
            if shared >= 2
                # Check coplanarity: normals must be parallel AND faces
                # must lie on the same plane (a shared node's position
                # relative to the other face's plane must be near-zero).
                n1 = face_normals[i]
                n2 = face_normals[j]
                if abs(abs(dot(n1, n2)) - 1) < tol
                    # Verify same-plane: pick a node from face j and
                    # check its distance from face i's plane.  Scale
                    # tolerance by the face diagonal to be independent
                    # of the coordinate system origin.
                    ref_pt = node_points[face_node_lists[i][1]]
                    test_pt = node_points[face_node_lists[j][1]]
                    plane_dist = abs(dot(n1, test_pt - ref_pt))
                    diag = norm(test_pt - ref_pt)
                    char_len = max(one(T), diag)
                    if plane_dist < tol * char_len
                        uf_unite(i, j)
                    end
                end
            end
        end
    end

    # Group faces by cluster root
    clusters = Dict{Int, Vector{Int}}()
    for i in 1:nf
        r = uf_find(i)
        push!(get!(clusters, r, Int[]), i)
    end

    for (_, cluster) in clusters
        if length(cluster) == 1
            continue  # nothing to merge; leave for the fallback loop
        end
        # Collect all nodes from the cluster
        all_nodes = Set{Int}()
        for idx in cluster
            union!(all_nodes, face_node_sets[idx])
        end

        # The boundary of the merged polygon consists of edges that appear
        # exactly once across all faces in the cluster.
        edge_count = Dict{Tuple{Int,Int}, Int}()
        for idx in cluster
            fnodes = face_node_lists[idx]
            nn = length(fnodes)
            for k in 1:nn
                a = fnodes[k]
                b = fnodes[mod1(k + 1, nn)]
                edge = a < b ? (a, b) : (b, a)
                edge_count[edge] = get(edge_count, edge, 0) + 1
            end
        end

        # Boundary edges appear exactly once
        boundary_edges = Tuple{Int,Int}[]
        for (edge, cnt) in edge_count
            if cnt == 1
                push!(boundary_edges, edge)
            end
        end

        # Check that the boundary forms a simple polygon: every boundary
        # node must have exactly two boundary edges.  If any node has a
        # different degree the merged outline is non-simple and cannot be
        # properly chained into a single polygon.
        bnd_deg = Dict{Int, Int}()
        for (a, b) in boundary_edges
            bnd_deg[a] = get(bnd_deg, a, 0) + 1
            bnd_deg[b] = get(bnd_deg, b, 0) + 1
        end
        if any(d != 2 for d in values(bnd_deg))
            continue  # non-simple boundary; leave faces un-merged
        end

        # Extract the boundary polygon by chaining boundary edges.
        merged_nodes = _chain_boundary_edges(boundary_edges)
        if isempty(merged_nodes) || length(merged_nodes) != length(bnd_deg)
            # Chain did not cover all boundary nodes
            continue  # leave faces un-merged
        end

        # Order the polygon nodes correctly
        pts = [node_points[n] for n in merged_nodes]
        avg_normal = normalize(sum(face_normals[idx] for idx in cluster))
        ordered_pts = order_polygon_points(pts, avg_normal)

        # Map ordered points back to node indices using a point→node dictionary
        pt_to_node = Dict{SVector{3, T}, Int}()
        for n in merged_nodes
            pt_to_node[node_points[n]] = n
        end
        ordered_nodes = Int[]
        for pt in ordered_pts
            n = get(pt_to_node, pt, 0)
            if n != 0
                push!(ordered_nodes, n)
            else
                for mn in merged_nodes
                    if node_points[mn] ≈ pt
                        push!(ordered_nodes, mn)
                        break
                    end
                end
            end
        end

        # Only merge if the resulting polygon is convex on its plane
        if !_is_convex_polygon([node_points[n] for n in ordered_nodes], avg_normal)
            continue  # leave faces un-merged for the fallback loop
        end

        # Mark source faces as done
        for idx in cluster
            done[fgroup[idx]] = true
        end

        # Fix orientation for interior faces: normal should point from left to right
        if !is_boundary
            ref_face = fgroup[cluster[1]]
            l, r = mesh.faces.neighbors[ref_face]
            _fix_interior_orientation!(ordered_nodes, node_points, l, r, mesh)
            push!(out_nodes, ordered_nodes)
            push!(out_meta, (l, r))
        else
            ref_face = fgroup[cluster[1]]
            c = mesh.boundary_faces.neighbors[ref_face]
            _fix_boundary_orientation!(ordered_nodes, node_points, c, mesh)
            push!(out_nodes, ordered_nodes)
            push!(out_meta, c)
        end
    end
end

"""
    _chain_boundary_edges(edges) -> Vector{Int}

Given a set of directed/undirected edges, chain them into a single polygon
boundary.  Returns an ordered list of node indices.
"""
function _chain_boundary_edges(edges::Vector{Tuple{Int,Int}})
    if isempty(edges)
        return Int[]
    end
    # Build adjacency: each node connects to its neighbors
    adj = Dict{Int, Vector{Int}}()
    for (a, b) in edges
        push!(get!(adj, a, Int[]), b)
        push!(get!(adj, b, Int[]), a)
    end

    # Chain starting from the first edge's first node
    start = edges[1][1]
    chain = [start]
    visited = Set{Int}([start])

    current = start
    while true
        neighbors = get(adj, current, Int[])
        found = false
        for n in neighbors
            if !(n in visited)
                push!(chain, n)
                push!(visited, n)
                current = n
                found = true
                break
            end
        end
        if !found
            break
        end
    end

    return chain
end

"""
    _fix_interior_orientation!(nodes, node_points, left, right, mesh)

Ensure interior face normal (computed from node ordering) points from cell
`left` toward cell `right`.  If not, reverse the node order.
"""
function _fix_interior_orientation!(
    nodes::Vector{Int},
    node_points::Vector{SVector{3, T}},
    left::Int, right::Int,
    mesh::UnstructuredMesh{3}
) where T
    pts = [node_points[n] for n in nodes]
    face_normal = polygon_normal(pts)
    face_centroid = sum(pts) / length(pts)

    # Approximate cell centroids from their face centroids
    left_centroid = _approx_cell_centroid(mesh, left)
    right_centroid = _approx_cell_centroid(mesh, right)

    # Normal should point from left to right
    dir = right_centroid - left_centroid
    if dot(face_normal, dir) < 0
        reverse!(nodes)
    end
end

"""
    _fix_boundary_orientation!(nodes, node_points, cell, mesh)

Ensure boundary face normal points outward (away from the cell).
"""
function _fix_boundary_orientation!(
    nodes::Vector{Int},
    node_points::Vector{SVector{3, T}},
    cell::Int,
    mesh::UnstructuredMesh{3}
) where T
    pts = [node_points[n] for n in nodes]
    face_normal = polygon_normal(pts)
    face_centroid = sum(pts) / length(pts)

    cell_centroid = _approx_cell_centroid(mesh, cell)

    # Normal should point away from cell (outward)
    dir = face_centroid - cell_centroid
    if dot(face_normal, dir) < 0
        reverse!(nodes)
    end
end

"""
    _approx_cell_centroid(mesh, cell)

Compute an approximate cell centroid by averaging the centroids of all
its faces (interior + boundary).
"""
function _approx_cell_centroid(
    mesh::UnstructuredMesh{3},
    cell::Int
)
    T = eltype(eltype(mesh.node_points))
    centroid = zero(SVector{3, T})
    count = 0

    for f in mesh.faces.cells_to_faces[cell]
        fnodes = collect(mesh.faces.faces_to_nodes[f])
        pts = [mesh.node_points[n] for n in fnodes]
        centroid += sum(pts) / length(pts)
        count += 1
    end
    for bf in mesh.boundary_faces.cells_to_faces[cell]
        fnodes = collect(mesh.boundary_faces.faces_to_nodes[bf])
        pts = [mesh.node_points[n] for n in fnodes]
        centroid += sum(pts) / length(pts)
        count += 1
    end
    if count > 0
        centroid /= count
    end
    return centroid
end

"""
    _rebuild_mesh(mesh, node_points, face_nodes, face_neighbors, bnd_nodes, bnd_cells)

Build a new UnstructuredMesh from merged face data.
"""
function _rebuild_mesh(
    mesh::UnstructuredMesh{3},
    node_points::Vector{SVector{3, T}},
    face_nodes_list::Vector{Vector{Int}},
    face_neighbors::Vector{Tuple{Int,Int}},
    bnd_nodes_list::Vector{Vector{Int}},
    bnd_cells::Vector{Int}
) where T
    nc = number_of_cells(mesh)
    nf_new = length(face_nodes_list)
    nb_new = length(bnd_nodes_list)

    # Build cell-to-face mappings
    cell_int_faces = [Int[] for _ in 1:nc]
    for (fi, (l, r)) in enumerate(face_neighbors)
        push!(cell_int_faces[l], fi)
        push!(cell_int_faces[r], fi)
    end

    cell_bnd_faces = [Int[] for _ in 1:nc]
    for (bi, c) in enumerate(bnd_cells)
        push!(cell_bnd_faces[c], bi)
    end

    # Flatten into indirection-map format
    faces_nodes = Int[]
    faces_nodespos = Int[1]
    for fnodes in face_nodes_list
        append!(faces_nodes, fnodes)
        push!(faces_nodespos, faces_nodespos[end] + length(fnodes))
    end

    bnd_faces_nodes = Int[]
    bnd_faces_nodespos = Int[1]
    for fnodes in bnd_nodes_list
        append!(bnd_faces_nodes, fnodes)
        push!(bnd_faces_nodespos, bnd_faces_nodespos[end] + length(fnodes))
    end

    cells_faces = Int[]
    cells_facepos = Int[1]
    for c in 1:nc
        append!(cells_faces, cell_int_faces[c])
        push!(cells_facepos, cells_facepos[end] + length(cell_int_faces[c]))
    end

    bnd_cells_faces = Int[]
    bnd_cells_facepos = Int[1]
    for c in 1:nc
        append!(bnd_cells_faces, cell_bnd_faces[c])
        push!(bnd_cells_facepos, bnd_cells_facepos[end] + length(cell_bnd_faces[c]))
    end

    return UnstructuredMesh(
        cells_faces,
        cells_facepos,
        bnd_cells_faces,
        bnd_cells_facepos,
        faces_nodes,
        faces_nodespos,
        bnd_faces_nodes,
        bnd_faces_nodespos,
        node_points,
        face_neighbors,
        bnd_cells
    )
end

"""
    _is_convex_polygon(pts, normal; tol=1e-10)

Check whether a 3D planar polygon (given as ordered vertices) is convex when
projected onto its plane.  `normal` is the face normal used to define the
winding direction.  All cross products of consecutive edge pairs must point
in the same direction as `normal` (or be zero for collinear edges).
"""
function _is_convex_polygon(
    pts::Vector{SVector{3, T}},
    normal::SVector{3, T};
    tol::Real = 1e-10
) where T
    n = length(pts)
    if n <= 3
        return true  # triangles are always convex
    end
    for i in 1:n
        j = mod1(i + 1, n)
        k = mod1(i + 2, n)
        e1 = pts[j] - pts[i]
        e2 = pts[k] - pts[j]
        c = cross(e1, e2)
        d = dot(c, normal)
        if d < -tol
            return false
        end
    end
    return true
end
