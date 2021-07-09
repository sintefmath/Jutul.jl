using Terv

export Electrolyte, TestElyte

abstract type Electrolyte <: ElectroChemicalComponent end
struct TestElyte <: Electrolyte end

struct ChemicalCurrent <: ScalarVariable end
struct TotalCurrent <: ScalarVariable end

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
    
    S[:TotalCurrent] = TotalCurrent()
    
    S[:ChargeAcc] = ChargeAcc()
    S[:MassAcc] = MassAcc()
    S[:EnergyAcc] = EnergyAcc()
end

# Must be available to evaluate time derivatives
function minimum_output_variables(
    system::Electrolyte, primary_variables
    )
    [:ChargeAcc, :MassAcc, :EnergyAcc]
end


function get_current_coeff_c(
    model::SimulationModel{D, S, F, Cons}, C, T
    ) where {D, S<:Electrolyte, F, Cons}
    return ones(number_of_units(model, TotalCurrent()))
end
function get_current_coeff_phi(
    model::SimulationModel{D, S, F, Cons}, C, T
    ) where {D, S<:Electrolyte, F, Cons}
    return ones(number_of_units(model, TotalCurrent()))
end

@terv_secondary function update_as_secondary!(
    j, tv::TotalCurrent, model, param, 
    TPkGrad_C, TPkGrad_Phi, C, T
    )
    # Should have one coefficient for each, probably
    coeff_phi = get_current_coeff_phi(model, C, T)
    coeff_c = get_current_coeff_c(model, C, T)
    @tullio j[i] =  coeff_c[i]*TPkGrad_C[i] + coeff_phi[i]*TPkGrad_Phi[i]
end


function update_half_face_flux!(
    law::Conservation{MassAcc}, storage, model, dt, 
    flowd::TwoPointPotentialFlow{U, K, T}
    ) where {U,K,T<:ECFlow}

    j = storage.state.TotalCurrent
    f = get_entries(law.half_face_flux_cells)
    @tullio f[i] = j[i]
end
