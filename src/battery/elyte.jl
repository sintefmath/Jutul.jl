using Terv

export Electrolyte, TestElyte

abstract type Electrolyte <: ElectroChemicalComponent end
struct TestElyte <: Electrolyte end

struct ChemicalCurrent <: ScalarVariable end
struct TotalCurrent <: ScalarVariable end

function select_equations_system!(
    eqs, domain, system::Electrolyte, formulation
    )
    charge_cons = (arg...; kwarg...) -> Conservation(Phi(), arg...; kwarg...)
    mass_cons = (arg...; kwarg...) -> Conservation(C(), arg...; kwarg...)
    eqs[:charge_conservation] = (charge_cons, 1)
    eqs[:mass_conservation] = (mass_cons, 1)
end

function select_primary_variables_system!(
    S, domain, system::Electrolyte, formulation
    )
    S[:Phi] = Phi()
    S[:C] = C()
end

# Should not use only flow type
function select_secondary_variables_system!!(
    S, domain, system::Electrolyte, formulation, flow_type
    )
    S[:TPFlux_Phi] = TPFlux{Phi}()
    S[:TPFlux_C] = TPFlux{C}()
    S[:TotalCurrent] = TotalCurrent()
    S[:TotalCharge] = TotalCharge()
    S[:TotalConcentration] = TotalConcentration()

end

function get_current_coeff(
    model::SimulationModel{D, S, F, C}, C, Phi
    ) where {D, S::TestElyte, F, C}
    return ones(size(C))
end

function get_current_coeff(
    model::SimulationModel{D, S, F, C}, C, Phi
    ) where {D, S::Electrolyte, F, C}
    error("current coeff not implemented for abstract electrolyte")
end

@terv_secondary function update_as_secondary!(
    j, tv::TotalCurrent, model, param, 
    TPFlux_C, TPFlux_Phi, C, Phi # T to be added
    )
    # Should have one coefficient for each, probably
    coeff = get_current_coeff(model, C, Phi)
    @tullio j[i] = coeff[i] * (TPFlux_C[i] + TPFlux_Phi[i])
end

