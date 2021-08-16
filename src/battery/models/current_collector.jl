using Terv
export CurrentCollector

struct CurrentCollector <: ElectroChemicalComponent end

function minimum_output_variables(
    system::CurrentCollector, primary_variables
    )
    [:TPkGrad_Phi, :Charge]
end

function select_primary_variables_system!(
    S, domain, system::CurrentCollector, formulation
    )
    S[:Phi] = Phi()
end

function select_secondary_variables_system!(
    S, domain, system::CurrentCollector, formulation
    )
    S[:TPkGrad_Phi] = TPkGrad{Phi}()
    S[:Charge] = Charge()
    S[:Conductivity] = Conductivity()
end

function select_equations_system!(
    eqs, domain, system::CurrentCollector, formulation    )
    charge_cons = (arg...; kwarg...) -> Conservation(Charge(), arg...; kwarg...)
    eqs[:charge_conservation] = (charge_cons, 1)
end
