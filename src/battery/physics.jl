using Terv

export half_face_two_point_grad

@inline function half_face_two_point_grad(
    conn_data::NamedTuple, p::AbstractArray
    )
    half_face_two_point_grad(
        conn_data.self, conn_data.other, conn_data.T, p
        )
end

@inline function half_face_two_point_grad(
    c_self::I, c_other::I, T, phi::AbstractArray{R}
    ) where {R<:Real, I<:Integer}
    return -T * (phi[c_self] - value(phi[c_other]))
end

function update_equation!(law::Conservation, storage, model, dt)
    update_accumulation!(law, storage, model, dt)
    update_half_face_flux!(law, storage, model, dt)
end

# Update of discretization terms
function update_accumulation!(law, storage, model::ChargeConservation, dt)
    acc = get_entries(law.accumulation)
    acc .= 0  # Assume no accumulation
    return acc
end

function update_half_face_flux!(
    law::Conservation, storage, model, dt
    )
    fd = law.flow_discretization
    update_half_face_flux!(law, storage, model, dt, fd)
end


function update_half_face_flux!(
    law, storage, model, dt, flowd::TwoPointPotentialFlow{U, K, T}
    ) where {U,K,T<:ECFlow}

    pot = storage.state.TPFlux  # ?WHy is this named pot?
    flux = get_entries(law.half_face_flux_cells)
    @tullio flux[i] = pot[i]
end

@inline function get_diagonal_cache(eq::Conservation)
    return eq.accumulation
end


#######################
# Boundary conditions #
#######################

# Called from uppdate_state_dependents
function apply_forces_to_equation!(
    storage, model::SimulationModel{D, S}, eq::Conservation, force
    ) where {D<:Any, S<:CurrentCollector}
    acc = get_entries(eq.accumulation)
    insert_sources(acc, force, storage)
end

function insert_sources(acc, source::vonNeumannBC, storage)
    for (i, v) in enumerate(source.values)
        c = source.cells[i]
        @inbounds acc[c] -= v
    end
end

function insert_sources(acc, source::DirichletBC, storage)
    T = source.half_face_Ts
    phi_ext = source.values
    phi = storage.primary_variables.Phi
    for (i, c) in enumerate(source.cells)
        # This loop is used insted of @tullio due to possibility of 
        # two sources at the different faces, but the same cell
        @inbounds acc[c] += - T[i]*(phi_ext[i] - phi[c])
    end
end

function insert_sources(acc, source, storage) end

