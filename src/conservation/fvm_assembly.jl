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
    faces = e_s.face_flux_cells
    nc_active = count_active_entities(model.domain, Cells())
    hf_map = e.flow_discretization.half_face_map
    I = Vector{Int64}()
    J = Vector{Int64}()
    (; face_pos, cells) = hf_map
    for c in 1:nc_active
        push!(I, c)
        push!(J, c)
        for i in face_pos[c]:(face_pos[c+1]-1)
            fcell = cells[i]
            if fcell != c
                push!(I, c)
                push!(J, fcell)
            end
        end
    end
    return (I, J)
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
    nu, ne, np = ad_dims(face_cache)
    @assert nu == nf
    left_facepos = face_cache.jacobian_positions
    right_facepos = eq_s.face_flux_extra_alignment
    N = get_neighborship(model.domain.representation)
    for face in 1:nf
        for pos in vpos[face]:(vpos[face+1]-1)
            cell = vars[pos]
            l, r = N[face, :]
            for e in 1:ne
                for d = 1:np
                    pos = find_jac_position(
                        jac,
                        l, cell,
                        0, 0,
                        equation_offset, variable_offset,
                        e, d,
                        nu, nu,
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
                        nu, nu,
                        ne, np,
                        model.context
                    )
                    set_jacobian_pos!(right_facepos, face, e, d, np, pos)
                end
            end
        end
    end
    error()
end