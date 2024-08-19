function get_half_face_normal(G, cell, face, normals, areas)
    if G.faces.neighbors[face][2] == cell
        sgn = -1
    else
        sgn = 1
    end
    return sgn*normals[face]*areas[face]
end

function ntpfa_decompose_half_face(G::UnstructuredMesh{D}, cell, face, K, cell_centroids, face_centroids, normals, areas, bnd_face_centroids, bnd_normals, bnd_areas) where D
    # Vector we are going to decompose
    normal = get_half_face_normal(G, cell, face, normals, areas)
    AKn = K[cell]*normal
    # Get local set of HAPs + weights
    cells = Int[]
    weights = Tuple{Float64, Float64}[]
    points = SVector{D, Float64}[]
    K_self = K[cell]
    x_self = cell_centroids[cell]
    for f in G.faces.cells_to_faces[cell]
        l, r = G.faces.neighbors[f]
        # Don't use left and right, use cell and other.
        if l == cell
            other = r
            sgn = 1.0
        else
            other = l
            sgn = -1.0
        end
        x_f = face_centroids[f]
        n_f = sgn*normals[f]
        K_other = K[other]
        x_other = cell_centroids[other]
        hp, w = find_harmonic_average_point(K_self, x_self, K_other, x_other, x_f, n_f)
        # @info "Harmonic point found" hp w
        push!(cells, other)
        push!(points, hp)
        push!(weights, w)
    end
    for bf in G.boundary_faces.cells_to_faces[cell]
        # TODO: Something a bit smarter here.
        push!(cells, cell)
        push!(points, bnd_face_centroids[bf])
        push!(weights, (0.5, 0.5))
    end
    # Next, figure out which ones we are going to keep.
    x_t = cell_centroids[cell]

    trip, trip_w = find_minimizing_basis(x_t, AKn, points, throw = false)
    if any(isequal(0), trip)
        out = nothing
    else
        l_r = NFVM.reconstruct_l(trip, trip_w, x_t, points)
        # @assert norm(l_r - AKn)/norm(AKn) < 1e-8 "Mismatch in reconstruction, $l_r != $AKn"

        active_weights = map(x -> weights[x], trip)
        out = (
            self = cell,
            self_weights = map(first, active_weights), # Weights for self
            other_cells_weights = map(last, active_weights), # Weights for other cells
            other_cells = map(x -> cells[x], trip), # Other cell for each HAP
            harmonic_average_points = map(x -> points[x], trip),
            triplet_weights = trip_w,
            Kn = AKn
        )
    end
    return out
end

function remainder_trans(decomp, l, r, sgn = 1)
    out = Tuple{Int, Float64}[]
    for (i, c) in enumerate(decomp.other_cells)
        if c != l && c != r
            tw_i = decomp.triplet_weights[i]
            cw_i = decomp.other_cells_weights[i]
            w_i = sgn*tw_i*cw_i
            push!(out, (c, w_i))
        end
    end
    return out
end

function two_point_trans(decomp, cell)
    # @warn "Decomposing $cell"
    for (k, v) in pairs(decomp)
        # @info "$k" v
    end
    T = 0.0
    if decomp.self == cell
        T += sum(decomp.self_weights.*decomp.triplet_weights)
    end
    for (i, c) in enumerate(decomp.other_cells)
        if c == cell
            tw_i = decomp.triplet_weights[i]
            cw_i = decomp.other_cells_weights[i]
            # @info "Found self in other $cell" c i tw_i cw_i
            T += tw_i*cw_i
        end
    end
    # @info "Final T = $T"
    return T
end

function NFVMLinearDiscretization(t_tpfa::Real; left, right)
    # Fallback constructor - essentially just a two-point flux with extra steps.
    t_r = t_tpfa
    t_l = -t_tpfa
    t_mpfa = Tuple{Int, Float64}[]
    return NFVMLinearDiscretization(left, right, t_l, t_r, t_mpfa)
end

function NFVMLinearDiscretization(decomp; left, right)
    t_l = two_point_trans(decomp, left)
    t_r = two_point_trans(decomp, right)

    w_tot = -sum(decomp.triplet_weights)
    if decomp.self == left
        sgn = 1
        t_l += w_tot
    else
        sgn = -1
        t_r += w_tot
    end
    t_mpfa = remainder_trans(decomp, left, right, sgn)
    t_l *= sgn
    t_r *= sgn

    # @info "Computed trans $t_r p_r ($r) $t_l p_l ($l)" t_mpfa
    return NFVMLinearDiscretization(left, right, t_l, t_r, t_mpfa)
end

function ntpfa_decompose_faces(G::UnstructuredMesh{D}, perm, scheme::Symbol = :avgmpfa;
        faces = 1:number_of_faces(G),
        tpfa_trans = missing,
        extra_out = false
    ) where D
    geo = tpfv_geometry(G)
    areas = geo.areas
    Vec_t = SVector{D, Float64}
    possible_schemes = (:mpfa, :avgmpfa, :ntpfa, :nmpfa, :test_tpfa)
    scheme in possible_schemes || throw(ArgumentError("Scheme must be one of $possible_schemes, was :$scheme"))

    normals = reinterpret(Vec_t, geo.normals)
    cell_centroids = reinterpret(Vec_t, geo.cell_centroids)
    face_centroids = reinterpret(Vec_t, geo.face_centroids)

    bnd_normals = reinterpret(Vec_t, geo.boundary_normals)
    bnd_areas = geo.boundary_areas
    bnd_face_centroids = reinterpret(Vec_t, geo.boundary_centroids)

    if perm isa AbstractMatrix
        K = SMatrix{D, D, Float64, D*D}[]
        for i in axes(perm, 2)
            push!(K, Jutul.expand_perm(perm[:, i], Val(D)))
        end
    else
        perm::AbstractVector
        K = perm
    end

    nf = number_of_faces(G)
    function ntpfa_trans_for_face(f)
        @assert f <= nf && f > 0 "Face $f not in range 1:$nf"
        l, r = G.faces.neighbors[f]
        left_decompose = ntpfa_decompose_half_face(G, l, f, K, cell_centroids, face_centroids, normals, areas, bnd_face_centroids, bnd_normals, bnd_areas)
        right_decompose = ntpfa_decompose_half_face(G, r, f, K, cell_centroids, face_centroids, normals, areas, bnd_face_centroids, bnd_normals, bnd_areas)
        if isnothing(left_decompose) || isnothing(right_decompose) || scheme == :test_tpfa
            # A branch for handling fallback to TPFA scheme if something has
            # gone wrong in the decomposition.
            if ismissing(tpfa_trans)
                error("Unable to use fallback transmissibility if tpfa_trans keyword argument is defaulted.")
            end
            T_fallback = tpfa_trans[f]
            l_trans = NFVMLinearDiscretization(T_fallback, left = l, right = r)
            r_trans = NFVMLinearDiscretization(T_fallback, left = l, right = r)
        else
            l_trans = NFVMLinearDiscretization(left_decompose, left = l, right = r)
            r_trans = NFVMLinearDiscretization(right_decompose, left = l, right = r)
        end
        if scheme == :avgmpfa || scheme == :mpfa || scheme == :test_tpfa
            disc = merge_to_avgmpfa(l_trans, r_trans)
        else
            disc = NFVMNonLinearDiscretization(l_trans, r_trans, scheme)
        end
        if extra_out
            out = (disc, left_decompose, right_decompose)
        else
            out = disc
        end
        return out
    end
    disc = map(ntpfa_trans_for_face, faces)
    return disc
end
