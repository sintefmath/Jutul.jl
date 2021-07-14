using Terv, Polynomials

export Electrolyte, TestElyte
export p1, p2, p3, cnst

abstract type Electrolyte <: ElectroChemicalComponent end
struct TestElyte <: Electrolyte end

struct TPDGrad{T} <: KGrad{T} end
# Is it necesessary with a new struxt for all these?
struct DmuDc <: ScalarVariable end
struct ConsCoeff <: ScalarVariable end

abstract type Current <: ScalarVariable end
struct TotalCurrent <: Current end
struct ChargeCarrierFlux <: Current end
struct EnergyFlux <: Current end

function number_of_units(model, pv::Current)
    return 2*count_units(model.domain, Faces())
end

function select_equations_system!(
    eqs, domain, system::Electrolyte, formulation
    )
    charge_cons = (arg...; kwarg...) -> Conservation(
        ChargeAcc(), arg...; kwarg...
        )
    mass_cons = (arg...; kwarg...) -> Conservation(
        MassAcc(), arg...; kwarg...
        )
    energy_cons = (arg...; kwarg...) -> Conservation(
        EnergyAcc(), arg...; kwarg...
        )
    
    eqs[:charge_conservation] = (charge_cons, 1)
    eqs[:mass_conservation] = (mass_cons, 1)
    eqs[:energy_conservation] = (energy_cons, 1)
end

function select_primary_variables_system!(
    S, domain, system::Electrolyte, formulation
    )
    S[:Phi] = Phi()
    S[:C] = C()
    S[:T] = T()
end


function select_secondary_variables_system!(
    S, domain, system::Electrolyte, formulation
    )
    S[:TPkGrad_Phi] = TPkGrad{Phi}()
    S[:TPkGrad_C] = TPkGrad{C}()
    S[:TPDGrad_C] = TPDGrad{C}()
    S[:TPkGrad_T] = TPkGrad{T}()
    
    S[:Conductivity] = Conductivity()
    S[:ThermalConductivity] = Conductivity()
    S[:Diffusivity] = Diffusivity()
    S[:DmuDc] = DmuDc()
    S[:ConsCoeff] = ConsCoeff()

    S[:TotalCurrent] = TotalCurrent()
    S[:ChargeCarrierFlux] = ChargeCarrierFlux()
    
    S[:ChargeAcc] = ChargeAcc()
    S[:MassAcc] = MassAcc()
    S[:EnergyAcc] = EnergyAcc()
end

# Must be available to evaluate time derivatives
function minimum_output_variables(
    system::Electrolyte, primary_variables
    )
    [:ChargeAcc, :MassAcc, :EnergyAcc, :Conductivity, :Diffusivity]
end

function setup_parameters(
    ::SimulationModel{<:Any, <:Electrolyte, <:Any, <:Any}
    )
    d = Dict{Symbol, Any}()
    d[:tolerances] = Dict{Symbol, Any}()
    d[:tolerances][:default] = 1e-3
    d[:t] = 1.
    d[:z] = 1.
    return d
end

const poly_param = [
    -10.5       0.074       -6.96e-5    ;
    0.668e-3    -1.78e-5    2.80e-8     ;
    0.494e-6    -8.86e-10   0           ;
]
const p1 = Polynomial(poly_param[1:end, 1])
const p2 = Polynomial(poly_param[1:end, 2])
const p3 = Polynomial(poly_param[1:end, 3])

@inline function cond(T::Real, C::Real, ::Electrolyte)
    fact = 1e-4
    return fact * C * (p1(C) + p2(C) * T + p3(C) * T^2)^2
end

const diff_params = [
    -4.43   -54 ;
    -0.22   0.0 ;
]
const Tgi = [229 5.0]

@inline function diffusivity(T::Real, C::Real, ::Electrolyte)
    # ?Should these be parameters?
    return (
        1e-4 * 10 ^ ( 
            diff_params[1,1] + 
            diff_params[1,2] / ( T - Tgi[1] - Tgi[2] * C * 1e-3) + 
            diff_params[2,1] * C * 1e-3
            )
        )
end

@terv_secondary function update_as_secondary!(
    dmudc, sv::DmuDc, model, param, T, C
    )
    R = GAS_CONSTANT
    @tullio dmudc[i] = R * (T[i] / C[i])
end


@terv_secondary function update_as_secondary!(
    con, tv::Conductivity, model::SimulationModel{<:Any, <:Electrolyte, <:Any,<:Any}, param, T, C
    )
    s = model.system
    @tullio con[i] = cond(T[i], C[i], s)
end

@terv_secondary function update_as_secondary!(
    D, sv::Diffusivity, model::SimulationModel{<:Any, <:Electrolyte, <:Any,<:Any},
     param, C, T
    )
    s = model.system
    @tullio D[i] = diffusivity(T[i], C[i], s)
end


@terv_secondary function update_as_secondary!(
    coeff, tv::ConsCoeff, model::SimulationModel{<:Any, <:Electrolyte, <:Any, <:Any}, 
    param, Conductivity, DmuDc
    )
    t = param.t
    z = param.z
    F = FARADAY_CONST
    @tullio coeff[i] = Conductivity[i]*DmuDc[i] * t/(F*z)
end

@terv_secondary function update_as_secondary!(
    kGrad_C, tv::TPkGrad{C}, model::SimulationModel{<:Any, <:Electrolyte, <:Any, <:Any}, param, C, ConsCoeff
    )
    mf = model.domain.discretizations.charge_flow
    conn_data = mf.conn_data
    @tullio kGrad_C[i] = half_face_two_point_kgrad(conn_data[i], C, ConsCoeff)
end

@terv_secondary function update_as_secondary!(
    DGrad_C, tv::TPDGrad{C}, model::SimulationModel{Dom, S, F, Con}, 
    param, C, Diffusivity
    ) where {Dom, S <: Electrolyte, F, Con}
    mf = model.domain.discretizations.charge_flow
    conn_data = mf.conn_data
    @tullio DGrad_C[i] = half_face_two_point_kgrad(conn_data[i], C, Diffusivity)
end


@terv_secondary function update_as_secondary!(
    j, tv::TotalCurrent, model, param, 
    TPkGrad_C, TPkGrad_Phi
    )
    @tullio j[i] =  - TPkGrad_C[i] - TPkGrad_Phi[i]
end

@terv_secondary function update_as_secondary!(
    N, tv::ChargeCarrierFlux, model, param, 
    TPDGrad_C, TotalCurrent
    )
    t = param.t
    z = param.z
    F = FARADAY_CONST
    @tullio N[i] =  - TPDGrad_C[i] + t / (F * z) * TotalCurrent[i]
end

function get_flux(storage,  model::SimulationModel{D, S, F, Con}, 
    law::Conservation{ChargeAcc}) where {D, S <: Electrolyte, F, Con}
    return storage.state.TotalCurrent
end

function get_flux(storage,  model::SimulationModel{D, S, F, Con}, 
    law::Conservation{MassAcc}) where {D, S <: Electrolyte, F, Con}
    return storage.state.ChargeCarrierFlux
end

function get_flux(storage,  model::SimulationModel{D, S, F, Con}, 
    law::Conservation{EnergyAcc}) where {D, S <: Electrolyte, F, Con}
    return - storage.state.TPkGrad_T
end


function apply_boundary_potential!(
    acc, state, parameters, model::SimulationModel{<:Any,<:Electrolyte,<:Any,<:Any}, 
    eq::Conservation{ChargeAcc}
    )
    # values
    Phi = state[:Phi]
    C = state[:C]
    dmudc = state[:DmuDc]
    κ = state[:Conductivity]

    F = FARADAY_CONST
    z = parameters.z
    t = parameters.t

    BoundaryPhi = state[:BoundaryPhi]
    BoundaryC = state[:BoundaryC]

    # Type
    # TODO: What if potential is defined on different cells
    bp = model.secondary_variables[:BoundaryPhi]
    T_hf = bp.T_half_face
    for (i, c) in enumerate(bp.cells)
        @inbounds acc[c] -= (
            - dmudc[c] * t/(F*z) * κ[c] * T_hf[i] * (C[c] - BoundaryC[i])
            - κ[c] * T_hf[i] * (Phi[c] - BoundaryPhi[i])
        )
    end
end

function update_accumulation!(law::Conservation{EnergyAcc}, storage, model, dt)
    conserved = law.accumulation_symbol
    acc = get_entries(law.accumulation)
    m = storage.state[conserved]
    m0 = storage.state0[conserved]
    # TODO: Add energy density from J^2 and GradC^2
    @tullio acc[c] = (m[c] - m0[c])/dt
    return acc
end


function apply_boundary_potential!(
    acc, state, model::SimulationModel{<:Any,<:Electrolyte,<:Any,<:Any}, 
    eq::Conservation{MassAcc}
    )
    # values
    Phi = state[:Phi]
    C = state[:C]
    dmudc = state[:DmuDc]
    κ = state[:Conductivity]
    D = state[:Diffusivity]

    F = FARADAY_CONST
    z = parameters.z
    t = parameters.t

    BoundaryPhi = state[:BoundaryPhi]
    BoundaryC = state[:BoundaryC]

    # Type
    bp = model.secondary_variables[:BoundaryC]
    T_hf = bp.T_half_face

    for (i, c) in enumerate(bp.cells)
        @inbounds j = (
            - dmudc[c] * t/(F*z) * κ[c] * T_hf[i] * (C[c] - BoundaryC[i])
            - κ[c] * T_hf[i] * (Phi[c] - BoundaryPhi[i])
        )
        @inbounds acc[c] -= (
            - D[c] * T_hf[i](C[c] - BoundaryC[i])
            + t / (F * z) * j
        )
    end
end

function apply_boundary_potential!(
    acc, state, model::SimulationModel{<:Any,<:Electrolyte,<:Any,<:Any}, 
    eq::Conservation{EnergyAcc}
    )
    # values
    T = state[:T]
    λ = state[:ThermalConductivity]

    # Type
    bp = model.secondary_variables[:BoundaryC]
    T = bp.T_half_face

    for (i, c) in enumerate(bp.cells)
        # TODO: Add influence of boundary on energy density
        @inbounds acc[c] -= - λ[c] * Thf[i] * (T[c] - BoundaryT[i])
    end
end



