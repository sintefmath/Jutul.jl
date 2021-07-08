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
function update_accumulation!(law::Conservation{ChargeAcc}, storage, model, dt)
    acc = get_entries(law.accumulation)
    acc .= 0  # Assume no accumulation
    return acc
end

function update_accumulation!(law::Conservation, storage, model, dt)
    conserved = law.accumulation_symbol
    acc = get_entries(law.accumulation)
    m = storage.state[conserved]
    m0 = storage.state0[conserved]
    @tullio acc[c] = (m[c] - m0[c])/dt
    return acc
end

function update_half_face_flux!(
    law::Conservation, storage, model, dt
    )
    fd = law.flow_discretization
    update_half_face_flux!(law, storage, model, dt, fd)
end


# TODO: Theis should happen via intermediate types

function update_half_face_flux!(
    law::Conservation{MassAcc}, storage, model, dt, 
    flowd::TwoPointPotentialFlow{U, K, T}
    ) where {U,K,T<:ECFlow}

    grad_C = storage.state.TPFlux_C
    flux = get_entries(law.half_face_flux_cells)
    @tullio flux[i] = grad_C[i]
end

function update_half_face_flux!(
    law::Conservation{ChargeAcc}, storage, model, dt, 
    flowd::TwoPointPotentialFlow{U, K, T}
    ) where {U,K,T<:ECFlow}

    flux = storage.state.TPFlux_Phi
    f = get_entries(law.half_face_flux_cells)
    @tullio f[i] = flux[i]
end

function update_half_face_flux!(
    law::Conservation{EnergyAcc}, storage, model, dt, 
    flowd::TwoPointPotentialFlow{U, K, T}
    ) where {U,K,T<:ECFlow}

    flux = storage.state.TPFlux_T
    flux = get_entries(law.half_face_flux_cells)
    @tullio flux[i] = flux[i]
end


@inline function get_diagonal_cache(eq::Conservation)
    return eq.accumulation
end


#######################
# Boundary conditions #
#######################


# Helper functio to find 
# TODO: There has to be a readymade implementation for this

function corr_type(::vonNeumannBC{ChargeAcc}) return ChargeAcc() end
function corr_type(::vonNeumannBC{MassAcc}) return MassAcc() end
function corr_type(::vonNeumannBC{EnergyAcc}) return EnergyAcc() end
function corr_type(::Conservation{ChargeAcc}) return ChargeAcc() end
function corr_type(::Conservation{MassAcc}) return MassAcc() end
function corr_type(::Conservation{EnergyAcc}) return EnergyAcc() end

# !Temporary hack, a potential does not necesessary corr to 
# !a single potential.
# TODO: mak boundary condition to play nice with non-diag Onsager matrix
function corr_type(::DirichletBC{Phi}) return ChargeAcc() end
function corr_type(::DirichletBC{C}) return MassAcc() end
function corr_type(::DirichletBC{T}) return EnergyAcc() end

function get_potential_vals(storage, ::MassAcc)
    return storage.primary_variables.C
end
function get_potential_vals(storage, ::ChargeAcc)
    return storage.primary_variables.Phi
end
function get_potential_vals(storage, ::EnergyAcc)
    return storage.primary_variables.T
end

# Called from uppdate_state_dependents
function apply_forces_to_equation!(
    storage, model::SimulationModel{D, S}, eq::Conservation{T}, force
    ) where {D<:Any, S<:ElectroChemicalComponent, T}
    if corr_type(force) == corr_type(eq)
        acc = get_entries(eq.accumulation)
        insert_sources(acc, force, storage)
    end
end

function insert_sources(acc, source::vonNeumannBC, storage)
    for (i, v) in enumerate(source.values)
        c = source.cells[i]
        @inbounds acc[c] -= v
    end
end


# TODO: Include resistivity / other types of factors/ constants

function insert_sources(acc, source::DirichletBC, storage)
    T = source.half_face_Ts
    pot_ext = source.values
    pot = get_potential_vals(storage, corr_type(source))
    for (i, c) in enumerate(source.cells)
        # This loop is used insted of @tullio due to possibility of 
        # two sources at the different faces, but the same cell
        @inbounds acc[c] += - T[i]*(pot_ext[i] - pot[c])
    end
end

function insert_sources(acc, source, storage) end

