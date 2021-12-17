
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
    return +(p[c_self] - value(p[c_other]))
end

@inline function half_face_two_point_kgrad(
    c_self::I, c_other::I, T::R, phi::AbstractArray, k::AbstractArray
    ) where {R<:Real, I<:Integer}
    return  harm_av(c_self, c_other, T, k) * grad(c_self, c_other, phi)
end

function update_equation!(law::Conservation, storage, model, dt)
    update_accumulation!(law, storage, model, dt)
    update_half_face_flux!(law, storage, model, dt)
    update_density!(law, storage, model)
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


function get_flux(storage,  model::ECModel, law::Conservation{Charge})
    return - storage.state.TPkGrad_Phi
end

function get_flux(storage,  model::ECModel, law::Conservation{Mass})
    return - storage.state.TPkGrad_C
end

function get_flux(storage, model::ECModel, law::Conservation{Energy})
    return - storage.state.TPkGrad_T
end

function update_half_face_flux!(law::Conservation, storage, model, dt, flow::TPFlow)

    flux = get_flux(storage, model, law)
    f = get_entries(law.half_face_flux_cells)
    @tullio f[i] = flux[i]
end

function update_density!(law::Conservation, storage, model)
    nothing
end


#######################
# Boundary conditions #
#######################
# TODO: Add possibilites for different potentials to have different boundary cells

function corr_type(::Conservation{T}) return T() end


# Called from uppdate_state_dependents
function apply_boundary_conditions!(storage, parameters, model::ECModel)
    equations = storage.equations
    for eq in equations
        apply_bc_to_equation!(storage, parameters, model, eq)
    end
end


function apply_boundary_potential!(
    acc, state, parameters, model, eq::Conservation{Charge}
    )
    # values
    Phi = state[:Phi]
    BoundaryPhi = state[:BoundaryPhi]
    κ = state[:Conductivity]

    bc = model.domain.grid.boundary_cells
    T_hf = model.domain.grid.boundary_T_hf

    for (i, c) in enumerate(bc)
        @inbounds acc[c] -= - κ[c]*T_hf[i]*(Phi[c] - BoundaryPhi[i])
    end
end

function apply_boundary_potential!(
    acc, state, parameters, model, eq::Conservation{Mass}
    )
    # values
    C = state[:C]
    BoundaryC = state[:BoundaryC]
    D = state[:Diffusivity]

    # Type
    bc = model.domain.grid.boundary_cells
    T_hf = model.domain.grid.boundary_T_hf

    for (i, c) in enumerate(bc)
        @inbounds acc[c] += - D[c]*T_hf[i]*(C[c] - BoundaryC[i])
    end
end

function apply_boundary_potential!(
    acc, state, parameters, model, eq::Conservation{Energy}
    )
    # values
    T = state[:T]
    BoundaryT = state[:BoundaryT]
    λ = state[:ThermalConductivity]

    bc = model.domain.grid.boundary_cells
    T_hf = model.domain.grid.boundary_T_hf

    for (i, c) in enumerate(bc)
        @inbounds acc[c] += - λ[c]*T_hf[i]*(T[c] - BoundaryT[i])
    end
end


function apply_bc_to_equation!(storage, parameters, model, eq::Conservation)
    acc = get_entries(eq.accumulation)
    state = storage.state

    apply_boundary_potential!(acc, state, parameters, model, eq)

    jkey = BOUNDARY_CURRENT[corr_type(eq)]
    if haskey(state, jkey)
        apply_boundary_current!(acc, state, jkey, model, eq)
    end
end

function apply_boundary_current!(acc, state, jkey, model, eq::Conservation)
    J = state[jkey]

    jb = model.secondary_variables[jkey]
    for (i, c) in enumerate(jb.cells)
        @inbounds acc[c] -= J[i]
    end
end
