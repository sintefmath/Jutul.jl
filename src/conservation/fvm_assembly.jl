struct ConservationLawFiniteVolumeStorage{A, HF, HA}
    accumulation::A
    accumulation_symbol::Symbol
    face_flux_cells::HF
    face_flux_extra_alignment::HA
    unique_alignment::Vector{Int}
    neighbors::Vector{Tuple{Int, Int}}
end

function setup_equation_storage(model, eq::ConservationLaw{conserved, PotentialFlow{:fvm, A, B, C}, <:Any, <:Any}, storage; kwarg...) where {conserved, A, B, C}
    return ConservationLawFiniteVolumeStorage(model, eq, storage; kwarg...)
end

function ConservationLawFiniteVolumeStorage(model, eq, storage; extra_sparsity = nothing, kwarg...)
    disc = eq.flow_discretization
    conserved = conserved_symbol(eq)

    function F!(out, state, state0, face)
        face_disc = (face) -> (kgrad = disc.kgrad[face], upwind = disc.upwind[face])
        local_disc = (face_disc = face_disc,)
        tmp = face_disc(face)
        dt = 1.0
        N = length(out)
        T = eltype(out)
        val = @SVector zeros(T, N)
        q = face_flux!(val, face, eq, state, model, dt, disc, local_disc)
        @. out = q
        return out
    end

    # N = number_of_entities(model, eq)
    N = number_of_faces(model.domain)
    n = number_of_equations_per_entity(model, eq)
    e = Faces()
    nt = count_active_entities(model.domain, e)
    caches = create_equation_caches(model, n, N, storage, F!, nt; self_entity = e, kwarg...)
    face_cache = caches.Cells
    extra_alignment = similar(face_cache.jacobian_positions)
    to_zero_pos = Int[]
    # Accumulation term
    nca = count_active_entities(model.domain, Cells())
    acc_cache = CompactAutoDiffCache(n, nca, model, entity = Cells())
    # TODO: Store unique(extra_alignment and face_cache.jacobian_positions)
    @assert only(get_primary_variable_ordered_entities(model)) == Cells()
    N = get_neighborship(model.domain.representation)
    neighbors = Tuple{Int, Int}[]
    @assert size(N, 2) == nt
    for i in 1:nt
        push!(neighbors, (N[1, i], N[2, i]))
    end

    return ConservationLawFiniteVolumeStorage(acc_cache, conserved, face_cache, extra_alignment, to_zero_pos, neighbors)
end

function declare_pattern(model, e::ConservationLaw, e_s::ConservationLawFiniteVolumeStorage, entity::Cells)
    nc_active = count_active_entities(model.domain, Cells())
    hf_map = e.flow_discretization.half_face_map
    face_cache = e_s.face_flux_cells
    vpos = face_cache.vpos
    vars = face_cache.variables
    IJ = Vector{Tuple{Int, Int}}()
    (; face_pos, cells, faces) = hf_map
    N = e_s.neighbors
    for c in 1:nc_active
        push!(IJ, (c, c))
    end
    disc = e.flow_discretization
    N = e_s.neighbors
    for face in eachindex(N)
        lc, rc = N[face]
        for fpos in vrange(face_cache, face)
            cell = vars[fpos]
            push!(IJ, (lc, cell))
            push!(IJ, (rc, cell))
        end
        for facedisc in (disc.kgrad[face], disc.upwind[face])
            for c in discretization_stencil(facedisc, entity)
                push!(IJ, (lc, c))
                push!(IJ, (rc, c))
                # Assume symmetry - bit of a hack but makes life easier when
                # setting up linear solvers.
                push!(IJ, (c, lc))
                push!(IJ, (c, rc))
            end
        end
    end
    IJ = unique!(IJ)
    return (map(first, IJ), map(last, IJ))
end

function declare_pattern(model, e::ConservationLaw, e_s::ConservationLawFiniteVolumeStorage, entity)
    @warn "Using hard-coded conservation law for entity $entity may give incorrect Jacobian. Assuming no dependence upon this entity for conservation law."
    I = Vector{Int64}()
    J = Vector{Int64}()
    return (I, J)
end

function align_to_jacobian!(eq_s::ConservationLawFiniteVolumeStorage, eq::ConservationLaw, jac, model, u::Cells; equation_offset = 0, variable_offset = 0)
    fd = eq.flow_discretization
    M = global_map(model.domain)

    acc = eq_s.accumulation
    diagonal_alignment!(acc, eq, jac, u, model.context, target_offset = equation_offset, source_offset = variable_offset)
    nf = number_of_faces(model.domain)
    face_cache = eq_s.face_flux_cells
    vpos = face_cache.vpos
    vars = face_cache.variables
    nc, _, _ = ad_dims(acc)
    nu, ne, np = ad_dims(face_cache)
    @assert nu == nf
    left_facepos = face_cache.jacobian_positions
    right_facepos = eq_s.face_flux_extra_alignment
    N = eq_s.neighbors
    for face in 1:nf
        l, r = N[face]
        for fpos in vpos[face]:(vpos[face+1]-1)
            cell = vars[fpos]
            for e in 1:ne
                for d = 1:np
                    pos = find_jac_position(
                        jac,
                        l, cell,
                        0, 0,
                        equation_offset, variable_offset,
                        e, d,
                        nc, nc,
                        ne, np,
                        model.context
                    )
                    set_jacobian_pos!(left_facepos, fpos, e, d, np, pos)
                    pos = find_jac_position(
                        jac,
                        r, cell,
                        0, 0,
                        equation_offset, variable_offset,
                        e, d,
                        nc, nc,
                        ne, np,
                        model.context
                    )
                    set_jacobian_pos!(right_facepos, fpos, e, d, np, pos)
                end
            end
        end
    end
    # Store all touched elements that need to be reset to zero before assembly.
    ua = eq_s.unique_alignment
    for i in acc.jacobian_positions
        push!(ua, i)
    end
    for i in left_facepos
        push!(ua, i)
    end
    for i in right_facepos
        push!(ua, i)
    end
    unique!(ua)
    @assert minimum(ua) > 0
    return eq_s
end

function update_equation!(eq_s::ConservationLawFiniteVolumeStorage, law::ConservationLaw, storage, model, dt)
    for i in 1:number_of_entities(model, law)
        prepare_equation_in_entity!(i, law, eq_s, storage.state, storage.state0, model, dt)
    end
    @tic "accumulation" update_accumulation!(eq_s, law, storage, model, dt)
    @tic "fluxes" fvm_update_face_fluxes!(eq_s, law, storage, model, dt)
end

function fvm_update_face_fluxes!(eq_s, law, storage, model, dt)
    disc = law.flow_discretization
    @inline @inbounds function face_disc(face)
        return (
        kgrad = disc.kgrad[face],
        upwind = disc.upwind[face]
        )
    end
    local_disc = (face_disc = face_disc,)

    face_cache = eq_s.face_flux_cells
    nu, ne, np = ad_dims(face_cache)
    T = eltype(face_cache)
    val = @SVector zeros(T, ne)
    local_state = local_ad(storage.state, 1, T)
    vars = face_cache.variables
    fvm_update_face_fluxes_inner!(face_cache, model, law, disc, local_disc, dt, vars, local_state, nu, val)
end

function fvm_update_face_fluxes_inner!(face_cache, model, law, disc, local_disc, dt, vars, local_state, nu, val::T) where T
    for face in 1:nu
        @inbounds for j in vrange(face_cache, face)
            v_i = @views face_cache.entries[:, j]
            var = vars[j]

            state_i = new_entity_index(local_state, var)
            flux = face_flux!(val, face, law, state_i, model, dt, disc, local_disc)
            flux::T
            for i in eachindex(flux)
                @inbounds v_i[i] = flux[i]
            end
        end
    end
end

@inline function get_diagonal_entries(eq::ConservationLaw, eq_s::ConservationLawFiniteVolumeStorage)
    return eq_s.accumulation.entries
end

function update_linearized_system_equation!(nz, r, model, eq::ConservationLaw, eq_s::ConservationLawFiniteVolumeStorage)
    # Zero out the buffers
    zero_ix = eq_s.unique_alignment
    @inbounds for i in zero_ix
        nz[i] = 0.0
    end
    # Accumulation term
    if false
        # TODO: Something wrong with expected transpose of r in old cache for residuals.
        fill_equation_entries!(nz, r, model, eq_s.accumulation)
    else
        acc = eq_s.accumulation
        nc, ne, np = ad_dims(acc)
        for i in 1:nc
            @inbounds for e in 1:ne
                a = get_entry(acc, i, e)
                r[e, i] = a.value
                for d = 1:np
                    update_jacobian_entry!(nz, acc, i, e, d, a.partials[d])
                end
            end
        end
    end
    # Fill fluxes
    face_cache = eq_s.face_flux_cells
    face_fluxes = face_cache.entries
    left_facepos = face_cache.jacobian_positions
    right_facepos = eq_s.face_flux_extra_alignment
    nu, ne, np = ad_dims(face_cache)
    vpos = face_cache.vpos
    vars = face_cache.variables

    nc = number_of_cells(model.domain)::Int
    @assert size(r, 1) == ne
    @assert vpos[end]-1 == size(face_fluxes, 2)
    N = eq_s.neighbors
    fvm_face_assembly!(r, nz, vpos, face_cache, left_facepos, right_facepos, N, nu, Val(ne), Val(np))
end

function fvm_face_assembly!(r, nz, vpos, face_cache, left_facepos, right_facepos, N, nu, ::Val{ne}, ::Val{np}) where {ne, np}
    @inbounds for face in 1:nu
        lc, rc = N[face]
        start = vpos[face]
        stop = vpos[face+1]-1
        if start == stop
            # No sparsity? A bit odd but guard against it.
            continue
        end
        @inbounds for e in 1:ne
            qval = get_entry_val(face_cache, start, e)
            r[e, lc] += qval
            r[e, rc] -= qval
        end
        for fpos in start:stop
            for e in 1:ne
                # Flux (with derivatives with respect to some cell)
                q = get_entry(face_cache, fpos, e)
                @inbounds for d in 1:np
                    ∂q = q.partials[d]
                    # Flux for left cell (l -> r)
                    lc_i = get_jacobian_pos(np, fpos, e, d, left_facepos)
                    nz[lc_i] += ∂q
                    # Flux for right cell (r -> l)
                    rc_i = get_jacobian_pos(np, fpos, e, d, right_facepos)
                    nz[rc_i] -= ∂q
                end
            end
        end
    end
end
