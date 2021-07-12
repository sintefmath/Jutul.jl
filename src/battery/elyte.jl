using Terv

export Electrolyte, TestElyte

abstract type Electrolyte <: ElectroChemicalComponent end
struct TestElyte <: Electrolyte end

struct TPDGrad{T} <: KGrad{T} end
# Is it necesessary with a new struxt for all these?
struct Conductivity <: ScalarVariable end
struct Diffusivity <: ScalarVariable end
struct DmuDc <: ScalarVariable end

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
    S[:TPkGrad_T] = TPkGrad{T}()
    
    S[:T] = T()
    S[:Conductivity] = Conductivity()
    S[:DmuDc] = DmuDc()
    S[:Diffusivity] = Diffusivity()

    S[:TotalCurrent] = TotalCurrent()
    S[:ChargeCarrierFlux] = ChargeCarrierFlux()
    
    S[:ChargeAcc] = ChargeAcc()
    S[:MassAcc] = MassAcc()

    # Should find a way to avoid having 1 per cell
    t = 1; z = 1
    S[:t] = ConstantVariables([t])
    S[:z] = ConstantVariables([z])
end

# Must be available to evaluate time derivatives
function minimum_output_variables(
    system::Electrolyte, primary_variables
    )
    [:ChargeAcc, :MassAcc, :Conductivity, :Diffusivity] # :EnergyAcc
end

@terv_secondary function update_as_secondary!(
    dmudc, sv::DmuDc, model, param, T, C
    )
    # TODO: Find a better way to hadle constants
    R = 1 # Gas constant in proper units
    @tullio dmudc[i] = R * (T[i] / C[i])
end


@inline function cond(
    T::Real, C::Real, ::Electrolyte
    )
    return 4. + 3.5 * C + (2.)*T + 0.1 * C * T # Arbitrary for now
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
    cnst = [[-4.43, -54] [-0.22, 0.0 ]]
    Tgi = [229, 5.0]

    @tullio D[i] = (
        1e-4 * 10 ^ ( 
            cnst[1,1] + 
            cnst[1,2] / ( T[i] - Tgi[1] - Tgi[2] * C[i] * 1e-3) + 
            cnst[2,1] * C[i] * 1e-3
            )
        )
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
    pot, tv::TPkGrad{C}, model::SimulationModel{D, S, F, Con}, param, C,
    Conductivity, DmuDc, t, z
    ) where {D, S <: Electrolyte, F, Con}
    mf = model.domain.discretizations.charge_flow
    conn_data = mf.conn_data
    Far = 1 # Faradays constant in proper units
    # Should this be its own variable
    @tullio k[i] := Conductivity[i] * DmuDc[i] * t[i] / (Far * z[i])
    @tullio pot[i] = half_face_two_point_kgrad(conn_data[i], C, k)
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
    @tullio j[i] =  -TPkGrad_C[i] + TPkGrad_Phi[i]
end

@terv_secondary function update_as_secondary!(
    N, tv::ChargeCarrierFlux, model, param, 
    TPDGrad_C, TotalCurrent, t, z
    )
    F = 1 # Faraday const in proper units
    a = t[1] / (F * z[1]) # A hack, for now
    @tullio N[i] =  -TPDGrad_C[i] + a * TotalCurrent[i]
end


function get_flux(storage,  model::SimulationModel{D, S, F, Con}, 
    law::Conservation{ChargeAcc}) where {D, S <: Electrolyte, F, Con}
    return storage.state.TotalCurrent
end

function get_flux(storage,  model::SimulationModel{D, S, F, Con}, 
    law::Conservation{MassAcc}) where {D, S <: Electrolyte, F, Con}
    return storage.state.ChargeCarrierFlux
end
