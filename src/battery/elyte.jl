using Terv

export Electrolyte, TestElyte

abstract type Electrolyte <: ElectroChemicalComponent end
struct TestElyte <: Electrolyte end

struct ChemicalCurrent <: ScalarVariable end
struct TotalCurrent <: ScalarVariable end
struct Conductivity <: ScalarVariable end
struct IonicConductivity <: ScalarVariable end
struct Diffusivity <: ScalarVariable end

function number_of_units(model, pv::TotalCurrent)
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

# Should not use only flow type
function select_secondary_variables_system!(
    S, domain, system::Electrolyte, formulation
    )
    S[:TPkGrad_Phi] = TPkGrad{Phi}()
    S[:TPkGrad_C] = TPkGrad{C}()
    S[:TPkGrad_T] = TPkGrad{T}()
    
    S[:Conductivity] = Conductivity()
    S[:IonicConductivity] = Conductivity() # Kappa
    S[:TotalCurrent] = TotalCurrent()
    
    S[:ChargeAcc] = ChargeAcc()
    S[:MassAcc] = MassAcc()
    S[:EnergyAcc] = EnergyAcc()
end

# Must be available to evaluate time derivatives
function minimum_output_variables(
    system::Electrolyte, primary_variables
    )
    [:ChargeAcc, :MassAcc, :EnergyAcc, :Conductivity]
end

function cond(
    C, T, model::SimulationModel{D, S, F, Con},
    ) where {D, S <: ElectroChemicalComponent, F, Con}
    return 4. + 3.5*C + (2.)*T + 0.1 * C * T # Arbitrary for now
end

@terv_secondary function update_as_secondary!(
    κ, sv::Conductivity, model, param, T, C
    )
    @tullio κ[i] = cond(T[i], C[i], model)
end

function cond(
    κ, T, model::SimulationModel{D, S, F, Con},
    ) where {D, S <: ElectroChemicalComponent, F, Con}
    # Standin for κ dμds * etc..
    return 1
    return κ * T
end

@terv_secondary function update_as_secondary!(
    κ_I, sv::IonicConductivity, model, param, Conductivity
    )
    @tullio κ_I[i] = ionic_conductivity(Conductivity[i], T[i], model)
end
    
@terv_secondary function update_as_secondary!(
    pot, tv::TPkGrad{Phi}, model::SimulationModel{D, S, F, C}, param, 
    Phi, Conductivity ) where {D, S <: ElectroChemicalComponent, F, C}
    mf = model.domain.discretizations.charge_flow
    conn_data = mf.conn_data
    @tullio pot[i] = half_face_two_point_kgrad(conn_data[i], Phi,Conductivity)
end


function get_alpha(
    model::SimulationModel{D, S, F, Con}
    ) where {D, S <: ElectroChemicalComponent, F, Con}
    return ones(number_of_units(model, C()))
end

@terv_secondary function update_as_secondary!(
    pot, tv::TPkGrad{C}, model::SimulationModel{D, S, F, Con}, param, C
    ) where {D, S <: ElectroChemicalComponent, F, Con}
    mf = model.domain.discretizations.charge_flow
    conn_data = mf.conn_data
    α = get_alpha(model)
    @tullio pot[i] = half_face_two_point_kgrad(conn_data[i], C, α)
end

@terv_secondary function update_as_secondary!(
    j, tv::TotalCurrent, model, param, 
    TPkGrad_C, TPkGrad_Phi
    )
    @tullio j[i] =  TPkGrad_C[i] + TPkGrad_Phi[i]
end


function get_flux(storage,  model::SimulationModel{D, S, F, Con}, 
    law::Conservation{MassAcc}) where {D, S <: Electrolyte, F, Con}
    return storage.state.TotalCurrent
end
