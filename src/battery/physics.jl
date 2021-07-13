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
    c_self::I, c_other::I, T::R, k::AbstractArray
    ) where {R<:Real, I<:Integer}
    return T * (k[c_self]^-1 + k[c_other]^-1)^-1
end

@inline function grad(c_self, c_other, p::AbstractArray)
    return p[c_self] - value(p[c_other])
end

@inline function half_face_two_point_kgrad(
    c_self::I, c_other::I, T::R, phi::AbstractArray, k::AbstractArray
    ) where {R<:Real, I<:Integer}
    return  harm_av(c_self, c_other, T, k) * grad(c_self, c_other, phi)
end

function update_equation!(law::Conservation, storage, model, dt)
    update_accumulation!(law, storage, model, dt)
    update_half_face_flux!(law, storage, model, dt)
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


function get_flux(storage,  model::SimulationModel{D, S, F, Con}, 
    law::Conservation{ChargeAcc}) where {D, S <: ElectroChemicalComponent, F, Con}
    return - storage.state.TPkGrad_Phi
end

function get_flux(storage,  model::SimulationModel{D, S, F, Con}, 
    law::Conservation{MassAcc}) where {D, S <: ElectroChemicalComponent, F, Con}
    return - storage.state.TPkGrad_C
end

function get_flux(storage,  model::SimulationModel{D, S, F, Con}, 
    law::Conservation{EnergyAcc}) where {D, S <: ElectroChemicalComponent, F, Con}
    return - storage.state.TPkGrad_T
end

function update_half_face_flux!(
    law::Conservation, storage, model, dt, 
    flow::TwoPointPotentialFlow{U, K, T}
    ) where {U,K,T<:ECFlow}

    flux = get_flux(storage, model, law)
    f = get_entries(law.half_face_flux_cells)
    @tullio f[i] = flux[i]
end


#######################
# Boundary conditions #
#######################


# Called from uppdate_state_dependents
function apply_boundary_conditions!(
    storage, model::SimulationModel{A, B, C, D}
    ) where {A, B<:ElectroChemicalComponent, C, D}
    equations = storage.equations
    for key in keys(equations)
        eq = equations[key]
        apply_bc_to_equation!(storage, model, eq)
    end
end


function apply_bc_to_equation!(storage, model, eq::Conservation{ChargeAcc})
    # values
    Phi = storage.state[:Phi]
    BoundaryPhi = storage.state[:BoundaryPhi]
    σ = storage.state[:Conductivity]

    # Type
    bp = model.secondary_variables[:BoundaryPhi]
    T = bp.T_half_face

    acc = get_entries(eq.accumulation)
    for (i, c) in enumerate(bp.cells)
        @inbounds acc[c] -= - σ[c]*T[i]*(Phi[c] - BoundaryPhi[i])
    end
end

