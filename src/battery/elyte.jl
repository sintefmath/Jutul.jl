using Terv

export Electrolyte, TestElyte

abstract type Electrolyte <: ElectroChemicalComponent end
struct TestElyte <: Electrolyte end

struct TPDGrad{T} <: KGrad{T} end
# Is it necesessary with a new struxt for all these?
struct Conductivity <: ScalarVariable end
struct Diffusivity <: ScalarVariable end
struct DmuDc <: ScalarVariable end
struct ConsCoeff <: ScalarVariable end

abstract type Current <: ScalarVariable end
struct TotalCurrent <: Current end
struct ChargeCarrierFlux <: Current end
struct ChemicalCurrent <: Current end



function number_of_units(model, pv::Current)
    """ Two fluxes per face """
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
    eqs[:charge_conservation] = (charge_cons, 1)
    eqs[:mass_conservation] = (mass_cons, 1)

end

function select_primary_variables_system!(
    S, domain, system::Electrolyte, formulation
    )
    S[:Phi] = Phi()
    S[:C] = C()
end


function select_secondary_variables_system!(
    S, domain, system::Electrolyte, formulation
    )
    S[:TPkGrad_Phi] = TPkGrad{Phi}()
    S[:TPkGrad_C] = TPkGrad{C}()
    S[:TPDGrad_C] = TPDGrad{C}()
    
    S[:T] = T()
    S[:Conductivity] = Conductivity()
    S[:DmuDc] = DmuDc()
    S[:Diffusivity] = Diffusivity()
    S[:ConsCoeff] = ConsCoeff()

    S[:TotalCurrent] = TotalCurrent()
    S[:ChargeCarrierFlux] = ChargeCarrierFlux()
    
    S[:ChargeAcc] = ChargeAcc()
    S[:MassAcc] = MassAcc()
end

# Must be available to evaluate time derivatives
function minimum_output_variables(
    system::Electrolyte, primary_variables
    )
    [:ChargeAcc, :MassAcc, :Conductivity, :Diffusivity]
end

function setup_parameters(
    ::SimulationModel{<:Any, <:Electrolyte, <:Any, <:Any}
    )
    d = Dict{Symbol, Any}()
    d[:tolerances] = Dict{Symbol, Any}()
    d[:tolerances][:default] = 1e-3
    d[:t] = 1
    d[:z] = 1
    return d
end


@inline function cond(T::Real, C::Real, ::Electrolyte)
    return 4. + 3.5 * C + (2.)*T + 0.1 * C * T # Arbitrary for now
end

@inline function diffusivity(T::Real, C::Real, ::Electrolyte)
    # ?Should these be parameters?
    cnst = [[-4.43, -54] [-0.22, 0.0 ]]
    Tgi = [229, 5.0]
    K = 1e9 # Martins "make it big" constant
    return K * (
        1e-4 * 10 ^ ( 
            cnst[1,1] + 
            cnst[1,2] / ( T - Tgi[1] - Tgi[2] * C * 1e-3) + 
            cnst[2,1] * C * 1e-3
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
    con, tv::Conductivity, model, param, T, C
    )
    s = model.system
    @tullio con[i] = cond(T[i], C[i], s)
end


@terv_secondary function update_as_secondary!(
    D, sv::Diffusivity, model, param, C, T
    )
    s = model.system
    @tullio D[i] = diffusivity(T[i], C[i], s)
end
    
@terv_secondary function update_as_secondary!(
    kgrad_phi, tv::TPkGrad{Phi}, model::SimulationModel{D, S, F, C}, param, 
    Phi, Conductivity
    ) where {D, S <: Electrolyte, F, C}
    mf = model.domain.discretizations.charge_flow
    conn_data = mf.conn_data
    @tullio kgrad_phi[i] = half_face_two_point_kgrad(conn_data[i], Phi, Conductivity)
end


@terv_secondary function update_as_secondary!(
    coeff, tv::ConsCoeff, model::SimulationModel{<:Any, <:Electrolyte, <:Any, <:Any}, 
    param, Conductivity, DmuDc
    )
    t = param.t
    z = param.z
    F = FARADAY_CONST
    @tullio coeff[i] := Conductivity[i]*DmuDc[i] * t/(F*z)
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
