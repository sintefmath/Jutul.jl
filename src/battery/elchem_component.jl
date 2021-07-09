using Terv

export ECComponent

struct ECComponent <: ElectroChemicalComponent end # Not a good name

function minimum_output_variables(
    system::ECComponent, primary_variables
    )
    [:ChargeAcc, :MassAcc]
end

function select_primary_variables_system!(
    S, domain, system::ECComponent, formulation
    )
    S[:Phi] = Phi()
    S[:C] = C()
end

function select_secondary_variables_system!(
    S, domain, system::ECComponent, formulation
    )
    S[:TPkGrad_Phi] = TPkGrad{Phi}()
    S[:TPkGrad_C] = TPkGrad{C}()
    S[:ChargeAcc] = ChargeAcc()
    S[:MassAcc] = MassAcc()
end

function select_equations_system!(
    eqs, domain, system::ECComponent, formulation
    )
    charge_cons = (arg...; kwarg...) -> Conservation(ChargeAcc(), arg...; kwarg...)
    mass_cons = (arg...; kwarg...) -> Conservation(MassAcc(), arg...; kwarg...)
    eqs[:charge_conservation] = (charge_cons, 1)
    eqs[:mass_conservation] = (mass_cons, 1)
end
