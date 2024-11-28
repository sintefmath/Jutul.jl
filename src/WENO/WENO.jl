module WENO
    using Jutul, StaticArrays, LinearAlgebra

    struct WENOHalfFaceDiscretization{D, N, R} <: Jutul.UpwindDiscretization
        cell::Int64
        distance::SVector{D, R}
        gradient::Vector{
            @NamedTuple{
                grad::SVector{N, R},
                area::R,
                cells::NTuple{N, Int64}
            }
        }
    end

    struct WENOFaceDiscretization{D, N, R}
        left::WENOHalfFaceDiscretization{D, N, R}
        right::WENOHalfFaceDiscretization{D, N, R}
        do_clamp::Bool
        threshold::Float64
        epsilon::Float64
        function WENOFaceDiscretization(
                l::WENOHalfFaceDiscretization{D, N, R},
                r::WENOHalfFaceDiscretization{D, N, R};
                do_clamp = false,
                threshold = 0.0,
                epsilon = 1e-10
            ) where {D, N, R}
            @assert D == N-1
            return new{D, N, R}(l, r, do_clamp, threshold, epsilon)
        end
    end

    function Jutul.cell_pair(weno::WENOFaceDiscretization)
        return (weno.left.cell, weno.right.cell)
    end

    function Jutul.discretization_stencil(d::WENOFaceDiscretization, ::Cells)
        cells = Int[]
        for hf in (d.left, d.right)
            for g in hf.gradient
                for c in g.cells
                    push!(cells, c)
                end
            end
        end
        return unique!(cells)
    end

    function weno_upwind(upw::WENOFaceDiscretization, X::AbstractVector, q)
        return weno_upwind(upw, c -> X[c], q)
    end

    function weno_upwind(upw::WENOFaceDiscretization, X, q)
        flag = q < zero(q)
        if flag
            up = upw.right
            other = upw.left.cell
        else
            up = upw.left
            other = upw.right.cell
        end
        return interpolate_weno(up, other, X, upw.do_clamp, upw.threshold, upw.epsilon)
    end

    function weno_discretize(domain::DataDomain; kwarg...)
        weno_cells = weno_discretize_cells(domain)
        g = physical_representation(domain)
        g::UnstructuredMesh
        N = g.faces.neighbors
        D = dim(g)
        Point_t = SVector{D, Float64}
        fc = reinterpret(Point_t, domain[:face_centroids])

        # Now condense this into face discretizations:
        # For each face, we find two pairs corresponding to the two cells, and need to store:
        # 1. Distance to the face centroid for each cell center
        # 2. The stencil in terms of cells.
        # 3. The gradient basis for each stencil.
        return map(
            f -> weno_discretize_face(weno_cells, N[f], fc[f]; kwarg...),
            1:length(N)
        )
    end

    function weno_discretize_face(weno_cells, N, face_centroid; kwarg...)
        l, r = N
        lcell = weno_cells[l]
        rcell = weno_cells[r]
        ldisc = weno_discretize_half_face(l, weno_cells[l], face_centroid)
        rdisc = weno_discretize_half_face(r, weno_cells[r], face_centroid)
        return WENOFaceDiscretization(ldisc, rdisc; kwarg...)
    end

    function weno_discretize_half_face(cell, wenodisc, fc::SVector{D, R}) where {D, R}
        function half_face_impl(planar_set, grad, V)
            cells = map(i -> wenodisc.stencil[i].cell, planar_set)
            points = map(i -> convert(SVector{D, R}, wenodisc.stencil[i].point), planar_set)
            # Now that we know the delta to the face centroid, we can collapse
            # the gradient basis to a single basis for the set of support cells.
            new_grad = grad[1]*V[1]
            for i in 2:length(grad)
                new_grad += grad[i]*V[i]
            end
            if all(isfinite, new_grad)
                area = area_from_points(points)
            else
                area = Inf
            end
            return (grad = new_grad, area = area, cells = cells)
        end
        V = wenodisc.S*(fc - wenodisc.center)
        # For each planar set, we need to store:
        # - Gradient basis
        # - Area/volume
        # - List of cells
        # ∇u = dot(B1, [u1, u2, u3]) + dot(B2, [u1, u2, u3])
        grad_disc = map(
            (planar_set, grad) -> half_face_impl(planar_set, grad, V),
            wenodisc.planar_set,
            wenodisc.gradients
        )
        filter!(
            x -> x.area > 1e-10 && isfinite(x.area),
            grad_disc
        )
        if length(grad_disc) == 0
            jutul_message("WENO", "No valid gradients found. Falling back to single-point-upwind", color = :yellow)
            if D == 2
                cells = (cell, cell, cell)
                fake_grad = SVector{3, R}(0.0, 0.0, 0.0)
            else
                @assert D == 3
                cells = (cell, cell, cell, cell)
                fake_grad = SVector{4, R}(0.0, 0.0, 0.0, 0.0)
            end
            grad_disc = [
                (grad = fake_grad, area = 1.0, cells = cells)
            ]
        end
        return WENOHalfFaceDiscretization(cell, V, vec(grad_disc))
    end

    function area_from_points(points)
        N = length(points)
        D = length(first(points))
        if D == 2
            @assert N == 3
            u, v, w = points
            x1, y1 = u
            x2, y2 = v
            x3, y3 = w
            area = 0.5*abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
        else
            @assert D == 3
            @assert N == 4
            u, v, w, l = points
            U = u - w
            V = v - w
            L = l - w
            area = 1/6*abs(dot(U, cross(V, L)))
        end
        return area
    end

    function weno_discretize_cells(domain::DataDomain)
        g = physical_representation(domain)
        D = dim(g)
        Point_t = SVector{D, Float64}
        cc = reinterpret(Point_t, domain[:cell_centroids])
        fc = reinterpret(Point_t, domain[:face_centroids])
        bc = reinterpret(Point_t, domain[:boundary_centroids])
        nc = length(cc)
        return map(c -> weno_discretize_cell(g, cc, fc, bc, c), 1:nc)
    end

    function weno_discretize_cell(g::UnstructuredMesh, cc, fc, bc, c)
        stencil = find_weno_stencil(g, cc, fc, bc, c)
        planar_set = find_weno_planar_sets(g, stencil)
        grad = planar_set_gradients(stencil, planar_set)
        @assert length(grad) == length(planar_set)
        return (
            center = stencil.center,
            stencil = stencil.stencil,
            S = stencil.S,            # Transformation basis
            gradients = grad,         # Basis for the gradients, equal length as planar_set
            planar_set = planar_set   # Triplets/quadruplets
        )
    end

    function find_weno_stencil(g, cc, fc, bc, c::Int)
        center = cc[c]
        function new_neighbor(point, cell::Int, face::Int = 0, is_bnd::Bool = false)
            return Dict(
                :point => point - center,
                :cell => cell,
                :face => face,
                :face_is_boundary => is_bnd,
                :is_center => cell == c
            )
        end
        # Cell itself
        self = new_neighbor(center, c)
        out = [self]
        # Interior faces
        for face in g.faces.cells_to_faces[c]
            l, r = g.faces.neighbors[face]
            if l == c
                other = r
            else
                other = l
            end
            ccent = cc[other]
            if all(isfinite, ccent)
                next = new_neighbor(ccent, other, face)
                push!(out, next)
            end
        end
        # Boundary faces
        for bface in g.boundary_faces.cells_to_faces[c]
            bcent = bc[bface]
            if all(isfinite, bcent)
                next = new_neighbor(bcent, c, bface, true)
                push!(out, next)
            end
        end
        pts = map(x -> x[:point], out)
        S = point_set_transformation_basis(pts)
        function convert_neighbor(i)
            pt = out[i][:point]
            T = typeof(pt)
            out[i][:point] = convert(T, S*pt)
            return (; pairs(out[i])...)
        end
        return (center = center, S = S, stencil = map(convert_neighbor, eachindex(out)))
    end

    function point_set_transformation_basis(pts::Vector{SVector{griddim, Float64}}) where griddim
        M = hcat(pts...)'
        if griddim == 1
            MT = SMatrix{1, 1, Float64, 1}
        elseif griddim == 2
            MT = SMatrix{2, 2, Float64, 4}
        else
            @assert griddim == 3
            MT = SMatrix{3, 3, Float64, 9}
        end
        out = missing
        try
            UDV = svd(M)
            U = UDV.U
            D = UDV.S
            Vt = UDV.Vt
            Dbar = Diagonal(D[1:griddim])
            out = Dbar*inv(Vt')
        catch
            @warn "Failure to decompose point set, using identity" M
            if griddim == 1
                out = @SMatrix [1.0]
            elseif griddim == 2
                out = @SMatrix [1.0 0.0; 0.0 1.0]
            else
                @assert griddim == 3
                out = @SMatrix [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
            end
        end
        return convert(MT, out)
    end

    function cell_to_node_indices(g::UnstructuredMesh, c)
        nodes = Int[]
        for face in g.faces.cells_to_faces[c]
            for node in g.faces.faces_to_nodes[face]
                push!(nodes, node)
            end
        end
        for bface in g.boundary_faces.cells_to_faces[c]
            for node in g.boundary_faces.faces_to_nodes[bface]
                push!(nodes, node)
            end
        end
        return nodes
    end

    function find_weno_planar_sets(g::UnstructuredMesh, weno_decomposition)
        wset = weno_decomposition.stencil
        function wset_nodes(i)
            ws = wset[i]
            f = ws.face
            if ws.face_is_boundary
                nodes = g.boundary_faces.faces_to_nodes[f]
            else
                nodes = g.faces.faces_to_nodes[f]
            end
        end
        D = dim(g)
        self = wset[1]
        @assert self.is_center
        c = self.cell
        self_nodes = cell_to_node_indices(g, c)
        n = length(wset)
        out = NTuple{D+1, Int}[]
        if D == 2
            # Triplets with at least one shared node
            for i in 2:n
                nodes_i = wset_nodes(i)
                nodes_overlap = intersect(nodes_i, self_nodes)
                @assert length(nodes_overlap) > 1
                for j in (i+1):n
                    nodes_j = wset_nodes(j)
                    common_nodes = intersect(nodes_overlap, wset_nodes(j))
                    if length(common_nodes) > 0
                        push!(out, (1, i, j))
                    end
                end
            end
        else
            for i in 2:n
                nodes_i = wset_nodes(i)
                nodes_overlap_outer = intersect(nodes_i, self_nodes)
                @assert length(nodes_overlap_outer) > 1
                for j in (i+1):n
                    nodes_j = wset_nodes(j)
                    nodes_overlap = intersect(nodes_overlap_outer, wset_nodes(j))
                    for k in (j+1):n
                        common_nodes = intersect(nodes_overlap, wset_nodes(k))
                        if length(common_nodes) > 0
                            push!(out, (1, i, j, k))
                        end
                    end
                end
            end
        end
        return out
    end

    function planar_set_gradients(decomp, triplets)
        wsets = decomp.stencil
        function get_gradient(triplet, D)
            if D == 2
                i, j, k = triplet
                p1 = wsets[i].point
                p2 = wsets[j].point
                p3 = wsets[k].point
                C = @SMatrix [
                    p1[1] p2[1] p3[1];
                    p1[2] p2[2] p3[2];
                    1.0 1.0 1.0
                ]
                invC = inv(C)
                bad = !all(isfinite, invC)
                B1 = invC[:, 1]
                B2 = invC[:, 2]
                return (B1, B2)
            else
                i, j, k, l = triplet
                p1 = wsets[i].point
                p2 = wsets[j].point
                p3 = wsets[k].point
                p4 = wsets[l].point
                C = @SMatrix [
                    p1[1] p2[1] p3[1] p4[1];
                    p1[2] p2[2] p3[2] p4[2];
                    p1[3] p2[3] p3[3] p4[3];
                    1.0 1.0 1.0 1.0
                ]
                invC = inv(C)
                bad = !all(isfinite, invC)
                B1 = invC[:, 1]
                B2 = invC[:, 2]
                B3 = invC[:, 3]
                return (B1, B2, B3)
            end
            if bad
                error("Failure in basis inversion: C = $C")
            end
        end
        center = decomp.center
        D = length(center)
        return map(t -> get_gradient(t, D), triplets)
    end

    function interpolate_weno(upw::WENOHalfFaceDiscretization, other_cell, U, do_clamp, threshold, ϵ)
        has_threshold = !isnan(threshold)
        cell = upw.cell
        u_c = U(cell)
        if has_threshold || do_clamp
            # If we have a threshold, we need to compare with the other cell.
            # Similarily for the clamp option.
            u_other = U(other_cell)
        else
            u_other = NaN
        end
        if has_threshold
            val_u_c = value(u_c)
            val_u_other = value(u_other)
            scale = max(abs(val_u_c) + abs(val_u_other), 1e-12)
            Δ_f = abs(val_u_c - val_u_other)/scale
            if Δ_f < threshold
                return u_c
            end
        end
        T = typeof(u_c)
        Δu = zero(T)
        β_tot = zero(T)
        @simd for i in eachindex(upw.gradient)
            @inbounds g = upw.gradient[i]
            Ω_i = g.area
            # This assert should hold since the self cell is always first
            # Disabled because @assert can't be compile time toggled in Julia
            # @assert g.cells[1] == cell
            ∇u = evaluate_gradient(g.grad, g.cells, u_c, U)
            # Assume linear weights are areas
            γ_i = Ω_i
            # Find (unscaled) weight and add to total delta
            ∇u_squared_sum = zero(T)
            for ∇u_i in ∇u
                ∇u_squared_sum += ∇u_i*∇u_i
            end
            denom = (ϵ + ∇u_squared_sum*Ω_i)
            β_i = γ_i / (denom*denom)
            Δu += β_i*∇u
            # Keep track of total since we need to divide by it afterwards
            β_tot += β_i
        end
        u_f = u_c + Δu/β_tot
        if do_clamp
            if u_c > u_other
                lo, hi = u_other, u_c
            else
                lo, hi = u_c, u_other
            end
            u_f = clamp(u_f, lo, hi)
        end
        return u_f
    end

    @inline function evaluate_gradient(grad::SVector{N, R}, cells, u_c, U) where {N, R}
        @inbounds ∇g = grad[1]*u_c
        @inbounds for i in 2:N
            c = cells[i]
            ∇g += grad[i]*U(c)
        end
        return ∇g
    end
end
