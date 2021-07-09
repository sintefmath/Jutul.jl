using Terv

export half_face_two_point_kgrad


#####################
# Gradient operator #
#####################

@inline function half_face_two_point_kgrad(
    conn_data::NamedTuple, p::AbstractArray, k::AbstractArray
    )
    half_face_two_point_kgrad(
        conn_data.self, conn_data.other, conn_data.T, p, k
        )
end

@inline function harm_av(
    c_self::I, c_other::I, T::R, k::AbstractArray{R}
    ) where {R<:Real, I<:Integer}
    return T * (k[c_self]^-1 + k[c_other]^-1)^-1
end

@inline function grad(c_self, c_other, p::AbstractArray)
    return   p[c_self] - value(p[c_other])
end

@inline function half_face_two_point_kgrad(
    c_self::I, c_other::I, T::R, phi::AbstractArray, k::AbstractArray{R}
    ) where {R<:Real, I<:Integer}
    return - harm_av(c_self, c_other, T, k) * grad(c_self, c_other, phi)
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

    flux = storage.state.TPkGrad_C
    f = get_entries(law.half_face_flux_cells)
    @tullio f[i] = flux[i]
end

function update_half_face_flux!(
    law::Conservation{ChargeAcc}, storage, model, dt, 
    flowd::TwoPointPotentialFlow{U, K, T}
    ) where {U,K,T<:ECFlow}

    flux = storage.state.TPkGrad_Phi
    f = get_entries(law.half_face_flux_cells)
    @tullio f[i] = flux[i]
end

function update_half_face_flux!(
    law::Conservation{EnergyAcc}, storage, model, dt, 
    flowd::TwoPointPotentialFlow{U, K, T}
    ) where {U,K,T<:ECFlow}

    flux = storage.state.TPkGrad_T
    f = get_entries(law.half_face_flux_cells)
    @tullio f[i] = flux[i]
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

