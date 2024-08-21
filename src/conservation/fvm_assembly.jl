struct ConservationLawFiniteVolumeStorage{A, HF, HA}
    accumulation::A
    accumulation_symbol::Symbol
    face_flux_cells::HF
    face_flux_extra_alignment::HA
end

function setup_equation_storage(model, eq::ConservationLaw{conserved, PotentialFlow{:fvm, A, B, C}, <:Any, <:Any}, storage; extra_sparsity = nothing, kwarg...) where {conserved, A, B, C}
    # F!(out, state, state0, i) = face_flux!(out, i, state, state0, eq, model, 1.0)
    disc = eq.flow_discretization

    function F!(out, state, state0, face)
        local_disc = nothing# local_discretization(eq, self_cell)
        # kgrad, upw = ldisc.face_disc(face)
        face_disc = (face) -> (kgrad = disc.kgrad[face], upwind = disc.upwind[face])
        local_disc = (face_disc = face_disc,)
        dt = 1.0
        N = length(out)
        T = eltype(out)
        val = @SVector zeros(T, N)
        face_flux!(val, face, eq, state, model, dt, disc, local_disc)
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
    # Accumulation term
    nca = count_active_entities(model.domain, Cells())
    acc_cache = CompactAutoDiffCache(n, nca, model, entity = Cells())
    @assert only(get_primary_variable_ordered_entities(model)) == Cells()
    return ConservationLawFiniteVolumeStorage(acc_cache, conserved, face_cache, extra_alignment)
end

function declare_pattern(model, e::ConservationLaw, e_s::ConservationLawFiniteVolumeStorage, entity::Cells)
    nc_active = count_active_entities(model.domain, Cells())
    hf_map = e.flow_discretization.half_face_map
    face_cache = e_s.face_flux_cells
    vpos = face_cache.vpos
    vars = face_cache.variables
    IJ = Vector{Tuple{Int, Int}}()
    (; face_pos, cells, faces) = hf_map
    for c in 1:nc_active
        push!(IJ, (c, c))
        for i in face_pos[c]:(face_pos[c+1]-1)
            face = faces[i]
            fcell = cells[i]
            for pos in vpos[face]:(vpos[face+1]-1)
                dcell = vars[pos]
                push!(IJ, (fcell, dcell))
                push!(IJ, (dcell, fcell))
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
    # hflux_cells = eq_s.half_face_flux_cells
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
    N = get_neighborship(model.domain.representation)
    for face in 1:nf
        l, r = N[:, face]
        for pos in vpos[face]:(vpos[face+1]-1)
            cell = vars[pos]
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
                    set_jacobian_pos!(left_facepos, face, e, d, np, pos)
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
                    set_jacobian_pos!(right_facepos, face, e, d, np, pos)
                end
            end
        end
    end
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
    face_disc = (face) -> (kgrad = disc.kgrad[face], upwind = disc.upwind[face])
    local_disc = (face_disc = face_disc,)

    face_cache = eq_s.face_flux_cells
    nu, ne, np = ad_dims(face_cache)
    T = eltype(face_cache)
    val = @SVector zeros(T, np)
    local_state = local_ad(storage.state, 1, T)
    vars = face_cache.variables
    for face in 1:nu
        for j in vrange(face_cache, face)
            v_i = @views face_cache.entries[:, j]
            var = vars[j]

            state_i = new_entity_index(local_state, var)
            flux = face_flux!(val, face, law, state_i, model, dt, disc, local_disc)
            @. v_i = flux
        end
    end
end

@inline function get_diagonal_entries(eq::ConservationLaw, eq_s::ConservationLawFiniteVolumeStorage)
    return eq_s.accumulation.entries
end
