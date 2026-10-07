function trajectory_to_points(trajectory::Matrix{Float64})
    N = size(trajectory, 2)
    @assert N in (2, 3) "2D/3D matrices are supported."
    return collect(vec(reinterpret(SVector{N, Float64}, collect(trajectory'))))
end

function trajectory_to_points(x::AbstractVector{SVector{N, Float64}}) where {N}
    return x
end


"""
    find_enclosing_cells(G, traj; geometry = tpfv_geometry(G), extra_out = false)

Find the cell indices of cells in the mesh `G` that are intersected by a given
piecewise linear trajectory `traj`. `traj` can be either a matrix with equal
number of columns as dimensions in G (i.e. three columns for 3D) or a `Vector`
of `SVector` instances with the same length.

The cells are returned in the order they are first visited when traversing the
trajectory from the first to the last point.

The search is done by computing the intersections between each segment of the
trajectory and the faces of the mesh. The crossing direction at each face
determines which cell the trajectory enters, so that each part of the
trajectory between two consecutive face intersections can be assigned to a
single cell. A uniform background grid of buckets is used to limit the number of
faces that must be checked for each segment.

The optional argument `geometry` is used to define the centroids and normals
used in the calculations. You can precompute this if you need to perform many
searches.

If `extra_out = true`, the function returns `(cells, extra)` where `extra` is a
`Dict` with the following per-cell entries (ordered as `cells`):
- `:lengths`: Length of the trajectory inside the cell.
- `:direction`: Mean direction of the trajectory inside the cell, scaled so that
  `direction .* lengths` is the vector sum of the subsegments inside the cell.
- `:normed_direction`: The vector sum of the subsegments divided by the
  dimensions of the cell.
- `:centroids`: Length-weighted centroid of the subsegments inside the cell.

`use_boundary` is by default set to `false`. If set to true, the boundary
normals from the geometry are used to determine if the trajectory enters or
exits the mesh at a boundary face. This requires that the boundary normals are
oriented outwards, which is currently not the case for all meshes from
downstream packages. Otherwise, the vector from the cell centroid to the
boundary face centroid is used.

`atol` is the tolerance used when checking if faces are inside the bounding box
of the trajectory.

`cells` can be used to limit the search to a subset of cells in the mesh. By
default all cells are used. Faces between a cell in the subset and a cell
outside it are treated as boundary faces.

The keyword arguments `n` and `limit_box` are retained for backwards
compatibility and have no effect.
"""
function find_enclosing_cells(
        G, traj;
        geometry = missing,
        n = missing,
        use_boundary = false,
        atol = 0.01,
        limit_box = true,
        cells = missing,
        extra_out = false
    )
    G = UnstructuredMesh(G)
    if ismissing(geometry)
        geometry = tpfv_geometry(G)
    end
    pts = trajectory_to_points(traj)
    length(pts) > 1 || throw(ArgumentError("Trajectory must have at least two points."))
    T = eltype(pts)
    length(T) == dim(G) || throw(ArgumentError("Trajectory dimension $(length(T)) does not match mesh dimension $(dim(G))."))
    if ismissing(cells)
        active = fill(true, number_of_cells(G))
    else
        active = fill(false, number_of_cells(G))
        active[cells] .= true
    end
    normals = vec(reinterpret(T, geometry.normals))
    face_centroids = vec(reinterpret(T, geometry.face_centroids))
    cell_centroids = vec(reinterpret(T, geometry.cell_centroids))
    boundary_centroids = vec(reinterpret(T, geometry.boundary_centroids))
    if use_boundary
        boundary_normals = vec(reinterpret(T, geometry.boundary_normals))
    else
        boundary_normals = boundary_centroids .- cell_centroids[G.boundary_faces.neighbors]
        for i in eachindex(boundary_normals)
            boundary_normals[i] /= norm(boundary_normals[i], 2)
        end
    end
    return find_enclosing_cells_impl(G, pts, active,
        normals, face_centroids, cell_centroids, boundary_normals, boundary_centroids;
        atol = atol, extra_out = extra_out
    )
end

function find_enclosing_cells_impl(G::UnstructuredMesh, pts::AbstractVector{SVector{D, T_f}}, active::Vector{Bool},
        normals, face_centroids, cell_centroids, boundary_normals, boundary_centroids;
        atol = 0.01,
        extra_out = false
    ) where {D, T_f}
    T = SVector{D, T_f}
    nf = number_of_faces(G)
    nbf = number_of_boundary_faces(G)
    is_active(c) = active[c]
    all_active = all(active)

    # Faces are numbered with interior faces first, then boundary faces.
    face_nodes(f) = f <= nf ? G.faces.faces_to_nodes[f] : G.boundary_faces.faces_to_nodes[f - nf]
    face_center(f) = f <= nf ? face_centroids[f] : boundary_centroids[f - nf]

    # Cumulative arc length along the trajectory
    nseg = length(pts) - 1
    seg_len = [norm(pts[k + 1] - pts[k], 2) for k in 1:nseg]
    cum_len = zeros(T_f, nseg + 1)
    for k in 1:nseg
        cum_len[k + 1] = cum_len[k] + seg_len[k]
    end
    total_len = cum_len[end]

    # Bounding box of trajectory
    lo_t = reduce((a, b) -> min.(a, b), pts) .- atol
    hi_t = reduce((a, b) -> max.(a, b), pts) .+ atol

    # Find candidate faces that overlap with the bounding box of the trajectory
    cand_faces = Int[]
    cand_lo = T[]
    cand_hi = T[]
    face_diam = zero(T_f)
    for f in 1:(nf + nbf)
        if !all_active
            if f <= nf
                l, r = G.faces.neighbors[f]
                (active[l] || active[r]) || continue
            else
                active[G.boundary_faces.neighbors[f - nf]] || continue
            end
        end
        lo_f, hi_f = trajectory_bounding_box(G.node_points, face_nodes(f))
        if all(lo_f .<= hi_t) && all(hi_f .>= lo_t)
            push!(cand_faces, f)
            push!(cand_lo, lo_f)
            push!(cand_hi, hi_f)
            face_diam += maximum(hi_f - lo_f)
        end
    end
    ncand = length(cand_faces)
    face_diam = ncand > 0 ? face_diam / ncand : one(T_f)
    scale = max(total_len, face_diam)
    tol_s = 1e-8 * scale

    # Bucket grid covering the bounding box of the trajectory. Bucket size is
    # chosen proportional to the mean face size so that each bucket contains
    # a small number of faces.
    h = max(2 * face_diam, eps(T_f))
    nb = ntuple(d -> clamp(ceil(Int, (hi_t[d] - lo_t[d]) / h), 1, 1000), Val(D))
    hb = SVector{D, T_f}(ntuple(d -> (hi_t[d] - lo_t[d]) / nb[d], Val(D)))
    bucket_ix(x, d) = clamp(floor(Int, (x - lo_t[d]) / hb[d]) + 1, 1, nb[d])
    bucket_range(lo, hi) = CartesianIndices(ntuple(d -> bucket_ix(lo[d], d):bucket_ix(hi[d], d), Val(D)))
    lin = LinearIndices(nb)
    nbucket = prod(nb)
    bucket_pos = zeros(Int, nbucket + 1)
    for i in 1:ncand
        for I in bucket_range(cand_lo[i], cand_hi[i])
            bucket_pos[lin[I] + 1] += 1
        end
    end
    bucket_pos[1] = 1
    for i in 1:nbucket
        bucket_pos[i + 1] += bucket_pos[i]
    end
    bucket_faces = zeros(Int, bucket_pos[end] - 1)
    bucket_fill = copy(bucket_pos)
    for i in 1:ncand
        for I in bucket_range(cand_lo[i], cand_hi[i])
            b = lin[I]
            bucket_faces[bucket_fill[b]] = i
            bucket_fill[b] += 1
        end
    end

    # Find all face intersections as (arc length, face, crossing). The
    # crossing is the cosine of the angle between the segment and the face
    # normal at the intersection point, oriented from the first to the second
    # neighbor (interior faces) or outwards (boundary faces).
    events = Tuple{T_f, Int, T_f}[]
    hits = Tuple{T_f, T_f}[]
    stamp = zeros(Int, ncand)
    seg_faces = Int[]
    for k in 1:nseg
        seg_len[k] > 0 || continue
        a = pts[k]
        b = pts[k + 1]
        d = b - a
        # Walk along the segment in pieces no longer than one bucket in each
        # dimension and collect faces in the overlapped buckets.
        npiece = max(1, maximum(ntuple(i -> ceil(Int, abs(d[i]) / hb[i]), Val(D))))
        empty!(seg_faces)
        for j in 1:npiece
            p0 = a + d * ((j - 1) / npiece)
            p1 = a + d * (j / npiece)
            for I in bucket_range(min.(p0, p1) .- atol, max.(p0, p1) .+ atol)
                b_ix = lin[I]
                for pos in bucket_pos[b_ix]:(bucket_pos[b_ix + 1] - 1)
                    i = bucket_faces[pos]
                    if stamp[i] != k
                        stamp[i] = k
                        push!(seg_faces, i)
                    end
                end
            end
        end
        for i in seg_faces
            segment_intersects_box(a, d, cand_lo[i], cand_hi[i], atol) || continue
            f = cand_faces[i]
            nodes = face_nodes(f)
            empty!(hits)
            segment_face_intersections!(hits, a, d, G.node_points, nodes, face_center(f))
            isempty(hits) && continue
            # Orientation of the node ordering relative to the reference
            A = face_area_vector(G.node_points, nodes, face_center(f))
            if f <= nf
                l, r = G.faces.neighbors[f]
                σ = sign(dot(A, cell_centroids[r] - cell_centroids[l]))
            else
                σ = sign(dot(A, boundary_normals[f - nf]))
            end
            for (t, cosθ) in hits
                push!(events, (cum_len[k] + t * seg_len[k], f, σ * cosθ))
            end
        end
    end
    sort!(events, by = first)

    # Group events at the same position and find the cells on either side of
    # each group, using the crossing direction.
    group_s = T_f[]
    group_before = Vector{Int}[]
    group_after = Vector{Int}[]
    function add_unique!(v, c)
        if is_active(c) && !(c in v)
            push!(v, c)
        end
    end
    for (s, f, sd) in events
        if isempty(group_s) || s - group_s[end] > tol_s
            push!(group_s, s)
            push!(group_before, Int[])
            push!(group_after, Int[])
        end
        before = group_before[end]
        after = group_after[end]
        if f <= nf
            l, r = G.faces.neighbors[f]
            if sd > 1e-12
                add_unique!(before, l)
                add_unique!(after, r)
            elseif sd < -1e-12
                add_unique!(before, r)
                add_unique!(after, l)
            else
                for c in (l, r)
                    add_unique!(before, c)
                    add_unique!(after, c)
                end
            end
        else
            bf = f - nf
            c = G.boundary_faces.neighbors[bf]
            if sd >= -1e-12
                add_unique!(before, c)
            end
            if sd <= 1e-12
                add_unique!(after, c)
            end
        end
    end

    # Assign each interval between consecutive groups to a cell
    function point_at(s)
        k = clamp(searchsortedlast(cum_len, s), 1, nseg)
        if seg_len[k] > 0
            return pts[k] + (pts[k + 1] - pts[k]) * ((s - cum_len[k]) / seg_len[k])
        else
            return pts[k]
        end
    end
    function locate(candidates, s0, s1)
        if length(candidates) == 1
            return candidates[1]
        end
        pt = point_at((s0 + s1) / 2)
        c = find_enclosing_cell(G, pt, normals, face_centroids, boundary_normals, boundary_centroids, candidates)
        if isnothing(c)
            c = isempty(candidates) ? 0 : candidates[1]
        end
        return c
    end

    ngroup = length(group_s)
    interval_cells = Int[]
    interval_s = Tuple{T_f, T_f}[]
    if ngroup == 0
        # No faces are crossed: Entire trajectory is inside a single cell, or
        # outside the mesh.
        candidates = cells_inside_bounding_box(G, lo_t, hi_t, atol = 0.0)
        if !all_active
            candidates = filter(is_active, candidates)
        end
        c = find_enclosing_cell(G, point_at(total_len / 2), normals, face_centroids, boundary_normals, boundary_centroids, candidates)
        if !isnothing(c)
            push!(interval_cells, c)
            push!(interval_s, (zero(T_f), total_len))
        end
    else
        for i in 0:ngroup
            s0 = i == 0 ? zero(T_f) : group_s[i]
            s1 = i == ngroup ? total_len : group_s[i + 1]
            s1 - s0 > tol_s || continue
            if i == 0
                candidates = group_before[1]
            elseif i == ngroup
                candidates = group_after[ngroup]
            else
                after = group_after[i]
                before = group_before[i + 1]
                candidates = intersect(after, before)
                if isempty(candidates)
                    # Inconsistent crossings (e.g. a missed face due to
                    # tolerances): Fall back to point location.
                    candidates = union(after, before)
                end
            end
            isempty(candidates) && continue
            c = locate(candidates, s0, s1)
            if !isempty(interval_cells) && interval_cells[end] == c
                interval_s[end] = (interval_s[end][1], s1)
            else
                push!(interval_cells, c)
                push!(interval_s, (s0, s1))
            end
        end
    end

    cell_to_ix = Dict{Int, Int}()
    unique_cells = Int[]
    for c in interval_cells
        if !haskey(cell_to_ix, c)
            push!(unique_cells, c)
            cell_to_ix[c] = length(unique_cells)
        end
    end
    if !extra_out
        return unique_cells
    end
    nu = length(unique_cells)
    lengths = zeros(T_f, nu)
    direction = zeros(T, nu)
    centroids = zeros(T, nu)
    for (c, (s0, s1)) in zip(interval_cells, interval_s)
        ix = cell_to_ix[c]
        # The interval may span several segments of the trajectory
        k0 = clamp(searchsortedlast(cum_len, s0), 1, nseg)
        k1 = clamp(searchsortedfirst(cum_len, s1) - 1, 1, nseg)
        for k in k0:k1
            seg_len[k] > 0 || continue
            sa = max(s0, cum_len[k])
            sb = min(s1, cum_len[k + 1])
            sb > sa || continue
            d = pts[k + 1] - pts[k]
            p0 = pts[k] + d * ((sa - cum_len[k]) / seg_len[k])
            p1 = pts[k] + d * ((sb - cum_len[k]) / seg_len[k])
            l = sb - sa
            lengths[ix] += l
            direction[ix] += p1 - p0
            centroids[ix] += l * (p0 + p1) / 2
        end
    end
    normed_direction = zeros(T, nu)
    for (ix, c) in enumerate(unique_cells)
        normed_direction[ix] = direction[ix] ./ T(cell_dims(G, c))
        if lengths[ix] > 0
            direction[ix] /= lengths[ix]
            centroids[ix] /= lengths[ix]
        else
            centroids[ix] = cell_centroids[c]
        end
    end
    extra = Dict{Symbol, Any}()
    extra[:lengths] = lengths
    extra[:direction] = direction
    extra[:normed_direction] = normed_direction
    extra[:centroids] = centroids
    return (unique_cells, extra)
end

function trajectory_bounding_box(points::AbstractVector{SVector{D, T}}, nodes) where {D, T}
    lo = SVector{D, T}(ntuple(_ -> T(Inf), D))
    hi = SVector{D, T}(ntuple(_ -> T(-Inf), D))
    for node in nodes
        pt = points[node]
        lo = min.(lo, pt)
        hi = max.(hi, pt)
    end
    return (lo, hi)
end

"""
    segment_intersects_box(a, d, lo, hi, atol)

Slab test: Check if the segment `a + t*d` for `t` in `[0, 1]` intersects the
axis-aligned box `[lo - atol, hi + atol]`.
"""
function segment_intersects_box(a::SVector{D, T}, d::SVector{D, T}, lo, hi, atol) where {D, T}
    tmin = zero(T)
    tmax = one(T)
    for i in 1:D
        l = lo[i] - atol
        u = hi[i] + atol
        if abs(d[i]) < eps(T)
            if a[i] < l || a[i] > u
                return false
            end
        else
            t1 = (l - a[i]) / d[i]
            t2 = (u - a[i]) / d[i]
            if t1 > t2
                t1, t2 = t2, t1
            end
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax
                return false
            end
        end
    end
    return true
end

"""
    segment_face_intersections!(hits, a, d, node_points, nodes, center)

Find all intersections between the segment `a + t*d` for `t` in `[0, 1]` and the
face defined by `nodes`, and push `(t, cosθ)` to `hits` where `cosθ` is the
cosine of the angle between `d` and the local face normal at the intersection
(oriented by the node ordering of the face). In 3D, the face is triangulated as
a fan around `center`, and a segment can intersect a non-planar face more than
once. Segments that are parallel to the face are not considered intersecting.
"""
function segment_face_intersections!(hits, a::SVector{3, T}, d::SVector{3, T}, node_points, nodes, center; ϵ = 1e-8) where T
    nn = length(nodes)
    nh = length(hits)
    for i in 1:nn
        p1 = node_points[nodes[i]]
        p2 = node_points[nodes[mod1(i + 1, nn)]]
        # Möller-Trumbore for the triangle (center, p1, p2)
        e1 = p1 - center
        e2 = p2 - center
        p = cross(d, e2)
        det = dot(e1, p)
        if abs(det) <= ϵ * norm(d, 2) * norm(e1, 2) * norm(e2, 2)
            continue
        end
        inv_det = one(T) / det
        s = a - center
        u = dot(s, p) * inv_det
        (u < -ϵ || u > 1 + ϵ) && continue
        q = cross(s, e1)
        v = dot(d, q) * inv_det
        (v < -ϵ || u + v > 1 + ϵ) && continue
        t = dot(e2, q) * inv_det
        (t < -ϵ || t > 1 + ϵ) && continue
        t = clamp(t, zero(T), one(T))
        # Hits on edges shared by two triangles of this face are only counted once
        any(j -> abs(hits[j][1] - t) <= ϵ, (nh + 1):length(hits)) && continue
        n = cross(e1, e2)
        push!(hits, (t, dot(n, d) / (norm(n, 2) * norm(d, 2))))
    end
    return hits
end

function segment_face_intersections!(hits, a::SVector{2, T}, d::SVector{2, T}, node_points, nodes, center; ϵ = 1e-8) where T
    cross2(x, y) = x[1] * y[2] - x[2] * y[1]
    length(nodes) == 2 || throw(ArgumentError("Faces in 2D must have exactly two nodes."))
    p = node_points[nodes[1]]
    e = node_points[nodes[2]] - p
    denom = cross2(d, e)
    if abs(denom) <= ϵ * norm(d, 2) * norm(e, 2)
        return hits
    end
    w = p - a
    t = cross2(w, e) / denom
    u = cross2(w, d) / denom
    if t < -ϵ || t > 1 + ϵ || u < -ϵ || u > 1 + ϵ
        return hits
    end
    n = SVector{2, T}(e[2], -e[1])
    push!(hits, (clamp(t, zero(T), one(T)), dot(n, d) / (norm(n, 2) * norm(d, 2))))
    return hits
end

"""
    face_area_vector(node_points, nodes, center)

Area-weighted normal of a face, oriented by the node ordering (consistent with
the normals used in `segment_face_intersections!`).
"""
function face_area_vector(node_points::AbstractVector{SVector{3, T}}, nodes, center) where T
    A = zero(SVector{3, T})
    nn = length(nodes)
    for i in 1:nn
        A += cross(node_points[nodes[i]] - center, node_points[nodes[mod1(i + 1, nn)]] - center)
    end
    return A / 2
end

function face_area_vector(node_points::AbstractVector{SVector{2, T}}, nodes, center) where T
    e = node_points[nodes[2]] - node_points[nodes[1]]
    return SVector{2, T}(e[2], -e[1])
end

function point_in_bounding_box(pt, low_bb, high_bb; atol::Float64 = 0.01)
    N = length(pt)
    N == length(low_bb) == length(high_bb) || throw(ArgumentError("Dimensions must match."))
    for i in 1:N
        pt_i = pt[i]
        if pt_i < low_bb[i] - atol
            return false
        end
        if pt_i > high_bb[i] + atol
            return false
        end
    end
    return true
end

"""
    find_enclosing_cell(G::UnstructuredMesh{D}, pt::SVector{D, T},
        normals::AbstractVector{SVector{D, T}},
        face_centroids::AbstractVector{SVector{D, T}},
        boundary_normals::AbstractVector{SVector{D, T}},
        boundary_centroids::AbstractVector{SVector{D, T}},
        cells = 1:number_of_cells(G)
    ) where {D, T}

Find enclosing cell of a point. This can be a bit expensive for larger meshes.
Recommended to use the more high level `find_enclosing_cells` instead.
"""
function find_enclosing_cell(
        G::UnstructuredMesh{D}, pt::SVector{D, T},
        normals::AbstractVector{SVector{D, T}},
        face_centroids::AbstractVector{SVector{D, T}},
        boundary_normals::AbstractVector{SVector{D, T}},
        boundary_centroids::AbstractVector{SVector{D, T}},
        cells = 1:number_of_cells(G)
    ) where {D, T}
    inside_normal(pt, normal, centroid) = dot(normal, pt - centroid) <= 0
    found_cell = nothing
    for cell in cells
        inside = true
        for face in G.faces.cells_to_faces[cell]
            if G.faces.neighbors[face][1] == cell
                sgn = 1
            else
                sgn = -1
            end
            normal = sgn * normals[face]
            center = face_centroids[face]
            inside = inside && inside_normal(pt, normal, center)
            if !inside
                break
            end
        end
        if !inside
            continue
        end

        for bface in G.boundary_faces.cells_to_faces[cell]
            normal = boundary_normals[bface]
            center = boundary_centroids[bface]
            inside = inside && inside_normal(pt, normal, center)
            if !inside
                break
            end
        end
        # A final check to see if the point is inside the bounding box of the
        # cell. This is not strictly necessary, but can be useful for some
        # degenerate geometries.
        if inside && point_inside_cell_bounding_box(G, cell, pt)
            found_cell = cell
            break
        end
    end
    return found_cell
end

function point_inside_cell_bounding_box(G::UnstructuredMesh, cell, pt::SVector{D, T}; atol = 0.0) where {D, T}
    bb_low = zero(SVector{D, T}) .+ Inf
    bb_high = zero(SVector{D, T}) .- Inf
    for face in G.faces.cells_to_faces[cell]
        for pt in G.faces.faces_to_nodes[face]
            bb_low = min.(bb_low, G.node_points[pt])
            bb_high = max.(bb_high, G.node_points[pt])
        end
    end
    for bface in G.boundary_faces.cells_to_faces[cell]
        for pt in G.boundary_faces.faces_to_nodes[bface]
            bb_low = min.(bb_low, G.node_points[pt])
            bb_high = max.(bb_high, G.node_points[pt])
        end
    end
    return point_in_bounding_box(pt, bb_low, bb_high, atol = atol)
end

"""
    cells_inside_bounding_box(G::UnstructuredMesh, low_bb, high_bb; algorithm = :box, atol = 0.01)


"""
function cells_inside_bounding_box(G::UnstructuredMesh, low_bb, high_bb; algorithm = :box, atol = 0.01)
    D = dim(G)
    length(low_bb) == length(high_bb) == D || throw(ArgumentError("Dimensions of bounding box must match with grid dimension $D."))
    nodes = G.node_points
    cells = Int[]

    function bb_overlap(A, B)
        Amin, Amax = A
        Bmin, Bmax = B
        return Amax >= Bmin && Bmax >= Amin
    end

    if algorithm == :nodal
        # Check if any node is inside the bounding box
        node_is_active = fill(false, length(nodes))
        for (i, node) in enumerate(nodes)
            node_is_active[i] = point_in_bounding_box(node, low_bb, high_bb, atol = atol)
        end
        active_faces = Int[]
        for face in 1:length(G.faces.faces_to_nodes)
            for node in G.faces.faces_to_nodes[face]
                if node_is_active[node]
                    push!(active_faces, face)
                    break
                end
            end
        end
        for f in active_faces
            l, r = G.faces.neighbors[f]
            push!(cells, l, r)
        end
    elseif algorithm == :box
        # Check intersection of bounding boxes of cells with the provided bounding box
        low_bb_cell = zeros(D)
        high_bb_cell = zeros(D)
        for cell in 1:number_of_cells(G)
            @. low_bb_cell = Inf
            @. high_bb_cell = -Inf
            for face in G.faces.cells_to_faces[cell]
                for nodeix in G.faces.faces_to_nodes[face]
                    node = nodes[nodeix]
                    low_bb_cell = min.(low_bb_cell, node)
                    high_bb_cell = max.(high_bb_cell, node)
                end
            end
            for face in G.boundary_faces.cells_to_faces[cell]
                for nodeix in G.boundary_faces.faces_to_nodes[face]
                    node = nodes[nodeix]
                    low_bb_cell = min.(low_bb_cell, node)
                    high_bb_cell = max.(high_bb_cell, node)
                end
            end
            inside = true
            for d in 1:D
                dim_overlap = bb_overlap(
                    (low_bb_cell[d], high_bb_cell[d]),
                    (low_bb[d], high_bb[d])
                )
                inside = inside && dim_overlap
            end
            if inside
                push!(cells, cell)
            end
        end
    else
        throw(ArgumentError("Unknown algorithm $algorithm."))

    end
    return unique!(sort!(cells))
end
