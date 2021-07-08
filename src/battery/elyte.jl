using Terv

export Electrolyte, TestElyte

abstract type Electrolyte <: ElectroChemicalComponent end
struct TestElyte <: Electrolyte end

struct ChemicalCurrent <: ScalarVariable end
struct TotalCurrent <: ScalarVariable end

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
    S[:TPFlux_Phi] = TPFlux{Phi}()
    S[:TPFlux_C] = TPFlux{C}()
    S[:TPFlux_T] = TPFlux{T}()
    
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
    return ones(size(C))
end
function get_current_coeff_phi(
    model::SimulationModel{D, S, F, Cons}, C, T
    ) where {D, S<:Electrolyte, F, Cons}
    return ones(size(C))
end

@terv_secondary function update_as_secondary!(
    j, tv::EnergyAcc, model, param, 
    TPFlux_C, TPFlux_Phi, C, T
    )
    # Should have one coefficient for each, probably
    coeff_phi = get_current_coeff_phi(model, C, T)
    coeff_c = get_current_coeff_c(model, C, T)
    @tullio j[i] =  coeff_c[i]*TPFlux_C[i] + coeff_phi[i]*TPFlux_Phi[i]
end
