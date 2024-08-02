export ConservationLaw, ConservationLawTPFAStorage, conserved_symbol

number_of_equations_per_entity(model::SimulationModel, ::ConservationLaw{<:Any, <:Any, <:Any, N}) where N = N

flux_vector_type(::ConservationLaw{<:Any, <:Any, <:Any, N}, ::Val{T}) where {N, T} = SVector{N, T}

conserved_symbol(::ConservationLaw{C, <:Any}) where C = C

flux_type(c::ConservationLaw) = c.flux_type

discretization(e::ConservationLaw) = e.flow_discretization


function update_equation_in_entity!(eq_buf::AbstractVector{T_e}, self_cell, state, state0, eq::ConservationLaw, model, Δt, ldisc = local_discretization(eq, self_cell)) where T_e
    # Compute accumulation term
    conserved = conserved_symbol(eq)
    M₀ = state0[conserved]
    M = state[conserved]
    # Compute ∇⋅V
    disc = eq.flow_discretization
    flux(face) = face_flux(face, eq, state, model, Δt, disc, ldisc, Val(T_e))
    div_v = ldisc.div(flux)
    for i in eachindex(div_v)
        ∂M∂t = accumulation_term(M, M₀, Δt, i, self_cell)
        @inbounds eq_buf[i] = ∂M∂t + div_v[i]
    end
end

@inline function accumulation_term(M, M₀, dt, i, e)
    return @inbounds (M[i, e] - M₀[i, e])/dt
end

@inline function accumulation_term(M::AbstractVector, M₀, dt, i, e)
    return @inbounds (M[e] - M₀[e])/dt
end

struct ConservationLawTPFAStorage{A, HC, HF, S}
    accumulation::A
    accumulation_symbol::Symbol
    half_face_flux_cells::HC
    half_face_flux_faces::HF
    sources::S
end

function ConservationLawTPFAStorage(model, eq::ConservationLaw; kwarg...)
    number_of_equations = number_of_equations_per_entity(model, eq)
    D, ctx = model.domain, model.context
    cell_entity = Cells()
    face_entity = Faces()
    nc = count_active_entities(D, cell_entity, for_variables = false)
    nf = count_active_entities(D, face_entity, for_variables = false)
    nhf = number_of_half_faces(eq.flow_discretization)
    face_partials = degrees_of_freedom_per_entity(model, face_entity)
    alloc = (n, entity, n_entities_pos) -> CompactAutoDiffCache(number_of_equations, n, model,
                                                                                entity = entity, n_entities_pos = n_entities_pos, 
                                                                                context = ctx; kwarg...)
    # Accumulation terms
    acc = alloc(nc, cell_entity, nc)
    # Source terms - as sparse matrix
    t_acc = eltype(acc.entries)
    src = sparse(zeros(0), zeros(0), zeros(t_acc, 0), size(acc.entries)...)
    # Half face fluxes - differentiated with respect to pairs of cells
    hf_cells = alloc(nhf, cell_entity, nhf)
    # Half face fluxes - differentiated with respect to the faces
    if face_partials > 0
        hf_faces = alloc(nf, face_entity, nhf)
    else
        hf_faces = nothing
    end
    return ConservationLawTPFAStorage(acc, conserved_symbol(eq), hf_cells, hf_faces, src)
end

function setup_equation_storage(model, eq::ConservationLaw{<:Any, <:TwoPointPotentialFlowHardCoded, <:Any, <:Any}, storage; extra_sparsity = nothing, kwarg...)
    # Maybe check that the sparsity matches the default?
    return ConservationLawTPFAStorage(model, eq; kwarg...)
end

"Update positions of law's derivatives in global Jacobian"
function align_to_jacobian!(eq_s::ConservationLawTPFAStorage, eq::ConservationLaw, jac, model, u::Cells; equation_offset = 0, variable_offset = 0)
    fd = eq.flow_discretization
    M = global_map(model.domain)

    acc = eq_s.accumulation
    hflux_cells = eq_s.half_face_flux_cells
    diagonal_alignment!(acc, eq, jac, u, model.context, target_offset = equation_offset, source_offset = variable_offset)
    half_face_flux_cells_alignment!(hflux_cells, acc, jac, model.context, M, fd, target_offset = equation_offset, source_offset = variable_offset)
end

function half_face_flux_cells_alignment!(face_cache, acc_cache, jac, context, global_map, flow_disc; target_offset = 0, source_offset = 0, dims = ad_dims(acc_cache))
    facepos = flow_disc.conn_pos
    nc = length(facepos) - 1
    cd = flow_disc.conn_data
    for cell in 1:nc
        @inbounds for f_ix in facepos[cell]:(facepos[cell + 1] - 1)
            align_half_face_cells(face_cache, jac, cd, f_ix, cell, dims, context, global_map, target_offset, source_offset)
        end
    end
end

function align_half_face_cells(face_cache, jac, cd, f_ix, active_cell_i, dims, context, global_map, target_offset, source_offset)
    nu, ne, np = dims
    cd_f = cd[f_ix]
    other = cd_f.other
    cell = full_cell(active_cell_i, global_map)
    @assert cell == cd_f.self "Expected $cell, was $(cd_f.self) for conn $cd_f"
    other_i = interior_cell(other, global_map)
    if isnothing(other_i) || isnothing(active_cell_i)
        # Either of the two cells is inactive - we set to zero.
        for e in 1:ne
            for d = 1:np
                set_jacobian_pos!(face_cache, f_ix, e, d, 0)
            end
        end
    else
        for e in 1:ne
            for d = 1:np
                pos = find_jac_position(
                    jac,
                    other_i, active_cell_i,
                    0, 0, # These should be zero, I think.
                    target_offset, source_offset,
                    e, d,
                    nu, nu,
                    ne, np,
                    context
                )
                set_jacobian_pos!(face_cache, f_ix, e, d, pos)
            end
        end
    end
end

# function half_face_flux_cells_alignment!(face_cache, acc_cache, jac, context::SingleCUDAContext, map, flow_disc; target_offset = 0, source_offset = 0)
#     dims = ad_dims(acc_cache)
#     nu, ne, np = dims
#     # error()
#     facepos = flow_disc.conn_pos
#     nc = length(facepos) - 1
#     cd = flow_disc.conn_data

#     # 
#     layout = matrix_layout(context)
#     fpos = face_cache.jacobian_positions


#     @kernel function algn(fpos, @Const(cd), @Const(rows), @Const(cols), nu, ne, np, target_offset, source_offset, layout)
#         cell, e, d = @index(Global, NTuple)
#         for f_ix in facepos[cell]:(facepos[cell + 1] - 1)
#             cd_f = cd[f_ix]
#             f = cd_f.face
#             other = cd_f.other
#             row, col = row_col_sparse(other + target_offset, cell + source_offset, e, d, 
#             nu, nu,
#             ne, np,
#             layout)

#             ix = zero(eltype(cols))
#             for pos = cols[col]:cols[col+1]-1
#                 if rows[pos] == row
#                     ix = pos
#                     break
#                 end
#             end
#             # ix = find_sparse_position_CSC(rows, cols, row, col)
#             fpos[jacobian_cart_ix(f_ix, e, d, np)] = ix
#         end
#     end
#     # nf = size(N, 2)
#     dims = (nc, ne, np)
#     kernel = algn(context.device, context.block_size)

#     rows = Int64.(jac.rowVal)
#     cols = Int64.(jac.colPtr)
#     event_jac = kernel(fpos, cd, rows, cols, nu, ne, np, target_offset, source_offset, layout, ndrange = dims)
#     wait(event_jac)
# end


function align_to_jacobian!(eq_s::ConservationLawTPFAStorage, law::ConservationLaw, jac, model, ::Faces; equation_offset = 0, variable_offset = 0)
    fd = law.flow_discretization
    pr = physical_representation(model.domain)
    neighborship = get_neighborship(pr)

    hflux_faces = eq_s.half_face_flux_faces
    if !isnothing(hflux_faces)
        half_face_flux_faces_alignment!(hflux_faces, jac, model.context, neighborship, fd, target_offset = equation_offset, source_offset = variable_offset)
    end
end

function align_to_jacobian!(eq_s::ConservationLawTPFAStorage, law::ConservationLaw, jac, model, entity::Any; kwarg...)
    # Assume that this is ok. Warning already issued.
end

function half_face_flux_faces_alignment!(face_cache, jac, context, N, flow_disc; target_offset = 0, source_offset = 0)
    nf, ne, np = ad_dims(face_cache)
    nhf = size(face_cache.jacobian_positions, 2)
    @assert nhf/2 == nf
    facepos = flow_disc.conn_pos
    nc = length(facepos) - 1
    for cell in 1:nc
        for f_ix in facepos[cell]:(facepos[cell + 1] - 1)
            face = flow_disc.conn_data[f_ix].face
            for e in 1:ne
                for d = 1:np
                    pos = find_jac_position(jac, cell, face, 0, 0, target_offset, source_offset, e, d, nc, nf, ne, np, context)
                    set_jacobian_pos!(face_cache, f_ix, e, d, pos)
                end
            end
        end
    end
end

function update_linearized_system_equation!(nz, r, model, law::ConservationLaw, eq_s::ConservationLawTPFAStorage)
    acc = eq_s.accumulation
    cell_flux = eq_s.half_face_flux_cells
    face_flux = eq_s.half_face_flux_faces
    src = eq_s.sources
    cpos = law.flow_discretization.conn_pos
    ctx = model.context
    update_linearized_system_subset_conservation_accumulation!(nz, r, model, acc, cell_flux, cpos, ctx)
    if use_sparse_sources(law)
        update_linearized_system_subset_conservation_sources!(nz, r, model, acc, src)
    end
    if !isnothing(face_flux)
        conn_data = law.flow_discretization.conn_data
        update_linearized_system_subset_face_flux!(nz, model, face_flux, cpos, conn_data)
    end
end

# function update_linearized_system_subset_conservation_accumulation!(nz, r, model, acc::CompactAutoDiffCache, cell_flux::CompactAutoDiffCache, conn_pos, context::SingleCUDAContext)
#     nc, ne, np = ad_dims(acc)
#     dims = (nc, ne, np)
#     CUDA.synchronize()
#     # kdims = dims .+ (0, 0, 1)

#     centries = acc.entries
#     fentries = cell_flux.entries
#     cp = acc.jacobian_positions
#     fp = cell_flux.jacobian_positions

#     @kernel function cu_fill(nz, @Const(r), @Const(conn_pos), @Const(centries), @Const(fentries), cp, fp, np)
#         cell, e, d = @index(Global, NTuple)
#         # diag_entry = get_entry(acc, cell, e, centries)
#         diag_entry = centries[e, cell].partials[d]
#         @inbounds for i = conn_pos[cell]:(conn_pos[cell + 1] - 1)
#             # q = get_entry(cell_flux, i, e, fentries)
#             @inbounds q = fentries[e, i].partials[d]
#             fpos = get_jacobian_pos(np, i, e, d, fp)
#             @inbounds nz[fpos] = q
#             diag_entry -= q
#         end
    
#         apos = get_jacobian_pos(np, cell, e, d, cp)
#         @inbounds nz[apos] = diag_entry
#     end
#     kernel = cu_fill(context.device, context.block_size)
#     event_jac = kernel(nz, r, conn_pos, centries, fentries, cp, fp, np, ndrange = dims)

#     @kernel function cu_fill_r(r, @Const(conn_pos), @Const(centries), @Const(fentries))
#         cell, e = @index(Global, NTuple)

#         @inbounds diag_entry = centries[e, cell].value
#         @inbounds for i = conn_pos[cell]:(conn_pos[cell + 1] - 1)
#             @inbounds diag_entry -= fentries[e, i].value
#         end
#         @inbounds r[e, cell] = diag_entry
#     end
#     rdims = (nc, ne)
#     kernel_r = cu_fill_r(context.device, context.block_size)
#     event_r = kernel_r(r, conn_pos, centries, fentries, ndrange = rdims)
#     wait(event_r)
#     wait(event_jac)
#     CUDA.synchronize()
# end

function update_linearized_system_subset_conservation_accumulation!(nz, r, model, acc::CompactAutoDiffCache, cell_flux::CompactAutoDiffCache, conn_pos, context)
    nc, ne, np = ad_dims(acc)
    threaded_fill_conservation_eq!(nz, r, context, acc, cell_flux, conn_pos, nc, Val(ne), Val(np))
end

function threaded_fill_conservation_eq!(nz, r, context, acc, cell_flux, conn_pos, nc, ne, np)
    tb = minbatch(context, nc)
    @batch minbatch=tb for cell = 1:nc
        fill_conservation_eq!(nz, r, cell, acc, cell_flux, conn_pos, np, ne)
    end
end

function fill_conservation_eq!(nz, r, cell, acc, cell_flux, conn_pos, ::Val{Np}, ::Val{Ne}) where {Np, Ne}
    # These should be elided by the compiler
    acc_partials = zeros(SMatrix{Ne, Np})
    acc_r = zeros(SVector{Ne})
    # Accumulation
    for e in 1:Ne
        v = get_entry_val(acc, cell, e)
        acc_r = setindex(acc_r, v, e)
    end
    @simd for e in 1:Ne
        for p in 1:Np
            dv = get_entry(acc, cell, e, p)
            acc_partials = setindex(acc_partials, dv, e, p)
        end
    end
    # Fluxes
    start = @inbounds conn_pos[cell]
    stop = @inbounds conn_pos[cell + 1] - 1
    if Np > 0
        for i = start:stop
            for e in 1:Ne
                v = acc_r[e] + get_entry_val(cell_flux, i, e)
                acc_r = setindex(acc_r, v, e)
            end
            fpos_outer = get_jacobian_pos(cell_flux, i, 1, 1)
            is_inner = fpos_outer > 0
            if is_inner
                @simd for p in 1:Np
                    @inbounds for e in 1:Ne
                        ∂ = get_entry(cell_flux, i, e, p)
                        f_v = acc_partials[e, p] + ∂
                        acc_partials = setindex(acc_partials, f_v, e, p)
                        fpos = get_jacobian_pos(cell_flux, i, e, p)
                        update_jacobian_inner!(nz, fpos, -∂)
                    end
                end
            else
                @simd for p in 1:Np
                    @inbounds for e in 1:Ne
                        ∂ = get_entry(cell_flux, i, e, p)
                        f_v = acc_partials[e, p] + ∂
                        acc_partials = setindex(acc_partials, f_v, e, p)
                    end
                end
            end
        end
    end
    @inbounds for e in 1:Ne
        r[e, cell] = acc_r[e]
    end
    @simd for p in 1:Np
        for e in 1:Ne
            @inbounds ∂ = acc_partials[e, p]
            apos = get_jacobian_pos(acc, cell, e, p)
            Jutul.update_jacobian_inner!(nz, apos, ∂)
        end
    end
end

function update_linearized_system_subset_conservation_sources!(nz, r, model, acc, src)
    dims = ad_dims(acc)
    rv_src = rowvals(src)
    nz_src = nonzeros(src)
    cp = acc.jacobian_positions
    update_lsys_sources_theaded!(nz, r, acc, src, rv_src, nz_src, cp, model.context, dims)
end

function update_lsys_sources_theaded!(nz, r, acc, src, rv_src, nz_src, cp, context, dims)
    nc, _, np = dims
    tb = minbatch(context, nc)
    @batch minbatch=tb for cell = 1:nc
        @inbounds for rp in nzrange(src, cell)
            e = rv_src[rp]
            v = nz_src[rp]
            # Value
            r[e, cell] += v.value
            # Partials
            for d = 1:np
                pos = get_jacobian_pos(acc, cell, e, d, cp)
                @inbounds nz[pos] += v.partials[d]
            end
        end
    end
end

function update_linearized_system_subset_face_flux!(Jz, model, face_flux, conn_pos, conn_data)
    dims = ad_dims(face_flux)
    fentries = face_flux.entries
    fp = face_flux.jacobian_positions
    update_lsys_face_flux_threaded!(Jz, face_flux, conn_pos, conn_data, fentries, fp, model.context, dims)
end

function update_lsys_face_flux_threaded!(Jz, face_flux, conn_pos, conn_data, fentries, fp, context, dims)
    _, ne, np = dims
    nc = length(conn_pos) - 1
    tb = minbatch(context, nc)
    @batch minbatch=tb for cell = 1:nc
        @inbounds for i = conn_pos[cell]:(conn_pos[cell + 1] - 1)
            @inbounds for e in 1:ne
                c = conn_data[i]
                face = c.face
                sgn = c.face_sign
                f = sgn*get_entry(face_flux, face, e)
                @inbounds for d = 1:np
                    df_di = f.partials[d]
                    fpos = get_jacobian_pos(face_flux, i, e, d, fp)
                    @inbounds Jz[fpos] = df_di
                end
            end
        end
    end
end

function declare_pattern(model, eq::ConservationLaw, e_s::ConservationLawTPFAStorage, entity::Cells)
    df = eq.flow_discretization
    hfd = Array(df.conn_data)
    n = number_of_entities(model, eq)
    # Diagonals
    D = [i for i in 1:n]
    if length(hfd) > 0
        # Fluxes
        I = map(x -> x.self, hfd)
        J = map(x -> x.other, hfd)
        I, J = map_ij_to_active(I, J, model.domain, entity)
        I = vcat(I, D)
        J = vcat(J, D)
    else
        I = collect(D)
        J = collect(D)
    end

    return (I, J)
end

function declare_pattern(model, e::ConservationLaw, e_s::ConservationLawTPFAStorage, entity::Faces)
    df = e.flow_discretization
    cd = df.conn_data
    I = map(x -> x.self, cd)
    J = map(x -> x.face, cd)
    I, J = map_ij_to_active(I, J, model.domain, entity)

    return (I, J)
end

function declare_pattern(model, e::ConservationLaw, e_s::ConservationLawTPFAStorage, entity)
    @warn "Using hard-coded conservation law for entity $entity may give incorrect Jacobian. Assuming no dependence upon this entity for conservation law."
    I = Vector{Int64}()
    J = Vector{Int64}()
    return (I, J)
end

function half_face_flux_sparse_pos!(fluxpos, jac, nc, conn_data, neq, nder, equation_offset = 0, variable_offset = 0)
    n = size(fluxpos, 1)
    nf = size(fluxpos, 2)
    @assert nder == n / neq

    for i in 1:nf
        # Off diagonal positions
        cdi = conn_data[i]
        self = cdi.self
        other = cdi.other

        for col = 1:nder
            # Diagonal positions
            col_pos = (col - 1) * nc + other + variable_offset
            pos = jac.colptr[col_pos]:jac.colptr[col_pos + 1] - 1
            rowval = jac.rowval[pos]
            for row = 1:neq
                row_ix = self + (row - 1) * nc + equation_offset
                fluxpos[(row - 1) * nder + col, i] = pos[rowval .== row_ix][1]
                # @printf("Matching %d %d to %d\n", row_ix, col_pos, pos[rowval .== row_ix][1])
            end
        end
    end
end

function state_pair(storage, conserved, model)
    m0 = storage.state0[conserved]
    m = storage.state[conserved]
    M = global_map(model.domain)
    v = x -> active_view(x, M)
    return (v(m0), v(m))
end

# Update of discretization terms
function update_accumulation!(eq_s, law, storage, model, dt)
    conserved = eq_s.accumulation_symbol
    acc = get_entries(eq_s.accumulation)
    m0, m = state_pair(storage, conserved, model)
    if m isa AbstractVector
        @tullio acc[i] = (m[i] - m0[i])/dt
    else
        @tullio acc[c, i] = (m[c, i] - m0[c, i])/dt
    end
    return acc
end

export update_half_face_flux!, update_accumulation!, update_equation!, get_diagonal_entries

function update_equation!(eq_s::ConservationLawTPFAStorage, law::ConservationLaw, storage, model, dt)
    # Zero out any sparse indices
    reset_sources!(eq_s)
    # Next, update accumulation, "intrinsic" sources and fluxes
    @tic "accumulation" update_accumulation!(eq_s, law, storage, model, dt)
    @tic "fluxes" update_half_face_flux!(eq_s, law, storage, model, dt)
end

function update_half_face_flux!(eq_s::ConservationLawTPFAStorage, law::ConservationLaw, storage, model, dt)
    fd = law.flow_discretization
    state = storage.state
    update_half_face_flux!(eq_s, law, state, model, dt, fd)
end

function update_half_face_flux!(eq_s::ConservationLawTPFAStorage, law::ConservationLaw, state, model, dt, flow_disc)
    flux_c = get_entries(eq_s.half_face_flux_cells)

    N, M = size(flux_c)
    T = eltype(flux_c)
    # flux_static = reinterpret(SVector{N, T}, flux_c)
    flux_static = unsafe_reinterpret(SVector{N, T}, flux_c, M)
    state_c = local_ad(state, 1, T)
    update_half_face_flux_tpfa!(flux_static, law, state_c, model, dt, flow_disc, Cells())

    hf_face = eq_s.half_face_flux_faces
    if !isnothing(hf_face)
        flux_v = get_entries(hf_face)
        F = eltype(flux_v)
        face_flux_static = reinterpret(SVector{N, F}, flux_v)
        state_f = local_ad(state, 1, F)
        update_half_face_flux_tpfa!(face_flux_static, law, state_f, model, dt, flow_disc, Faces())
    end
end

function update_half_face_flux_tpfa!(hf_cells::Union{AbstractArray{SVector{N, T}}, AbstractVector{T}}, eq, state::S, model, dt, flow_disc, ::Cells) where {T, N, S<:LocalStateAD}
    conn_data = flow_disc.conn_data
    conn_pos = flow_disc.conn_pos
    M = global_map(model.domain)
    nc = length(conn_pos)-1
    tb = minbatch(model.context, nc)
    @tic "flux (cells)" @batch minbatch=tb for c in 1:nc
        self = full_cell(c, M)
        state_c = new_entity_index(state, self)
        update_half_face_flux_tpfa_internal!(hf_cells, eq, state_c, model, dt, flow_disc, conn_pos, conn_data, c)
    end
end

function update_half_face_flux_tpfa_internal!(hf_cells::AbstractArray{T}, eq, state, model, dt, flow_disc, conn_pos, conn_data, c) where T
    start = @inbounds conn_pos[c]
    stop = @inbounds conn_pos[c+1]-1
    for i in start:stop
        (; self, other, face, face_sign) = @inbounds conn_data[i]
        @inbounds hf_cells[i] = face_flux!(zero(T), self, other, face, face_sign, eq, state, model, dt, flow_disc)
    end
end

function update_half_face_flux_tpfa!(hf_faces::AbstractArray{SVector{N, T}}, eq, state, model, dt, flow_disc, ::Faces) where {T, N}
    nf = number_of_faces(model.domain)
    pr = physical_representation(model.domain)
    neighbors = get_neighborship(pr)
    tb = minbatch(model.context, nf)
    @tic "flux (faces)" @batch minbatch = tb for f in 1:nf
        state_f = new_entity_index(state, f)
        @inbounds left = neighbors[1, f]
        @inbounds right = neighbors[2, f]
        @inbounds hf_faces[f] = face_flux!(hf_faces[f], left, right, f, 1, eq, state_f, model, dt, flow_disc)
    end
end


function face_flux!(entry, l, r, f, face_sign, eq, state, model, dt, disc)
    error("Not specialized for $eq")
end

@inline function face_flux(l, r, f, face_sign, eq::ConservationLaw{<:Any, <:Any, <:Any, N}, state, model, dt, disc, T = Val(Float64)) where N
    out = zero(flux_vector_type(eq, T))
    return face_flux!(out, l, r, f, face_sign, eq, state, model, dt, disc)
end

function face_flux!(entry, face, eq, state, model, dt, disc, local_disc)
    error("Not specialized for $eq")
end

@inline function face_flux(face, eq::ConservationLaw{<:Any, <:Any, <:Any, N}, state, model, dt, disc, ldisc, T = Float64) where N
    V_t = flux_vector_type(eq, T)
    out = zero(V_t)
    return face_flux!(out, face, eq, state, model, dt, disc, ldisc)::V_t
end

function reset_sources!(eq_s::ConservationLawTPFAStorage)
    if use_sparse_sources(eq_s)
        @. eq_s.sources = 0
    end
end

@inline function get_diagonal_entries(eq::ConservationLaw, eq_s::ConservationLawTPFAStorage)
    if use_sparse_sources(eq_s)
        return eq_s.sources
    else
        # Hack.
        return eq_s.accumulation.entries
    end
end

# is_cuda_eq(eq::ConservationLawTPFAStorage) = isa(eq.accumulation.entries, CuArray)
use_sparse_sources(eq) = false#!is_cuda_eq(eq)
