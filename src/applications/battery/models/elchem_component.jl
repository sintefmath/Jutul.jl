export ECComponent

struct ECComponent <: ElectroChemicalComponent end # Not a good name

function minimum_output_variables(
    system::ElectroChemicalComponent, primary_variables
    )
    [:Charge, :Mass, :Energy]
end

function select_primary_variables_system!(
    S, domain, system::ElectroChemicalComponent, formulation
    )
    S[:Phi] = Phi()
    S[:C] = C()
    S[:T] = T()
end

function select_secondary_variables_system!(
    S, domain, system::ElectroChemicalComponent, formulation
    )
    S[:TPkGrad_Phi] = TPkGrad{Phi}()
    S[:TPkGrad_C] = TPkGrad{C}()
    S[:TPkGrad_T] = TPkGrad{T}()
    
    S[:Charge] = Charge()
    S[:Mass] = Mass()
    S[:Energy] = Mass()

    S[:Conductivity] = Conductivity()
    S[:Diffusivity] = Diffusivity()
    S[:ThermalConductivity] = ThermalConductivity()
end

function select_equations_system!(
    eqs, domain, system::ElectroChemicalComponent, formulation
    )
    charge_cons = (arg...; kwarg...) -> Conservation(Charge(), arg...; kwarg...)
    mass_cons = (arg...; kwarg...) -> Conservation(Mass(), arg...; kwarg...)
    energy_cons = (arg...; kwarg...) -> Conservation(Energy(), arg...; kwarg...)
    eqs[:charge_conservation] = (charge_cons, 1)
    eqs[:mass_conservation] = (mass_cons, 1)
    eqs[:energy_conservation] = (energy_cons, 1)
end

