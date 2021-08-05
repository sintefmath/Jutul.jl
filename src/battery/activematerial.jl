using Terv, Polynomials

export ActiveMaterial, ACMaterial

abstract type ActiveMaterial <: ElectroChemicalComponent end
struct ACMaterial <: ActiveMaterial end
struct Ocd <: ScalarVariable end
const ActiveMaterialModel = SimulationModel{<:Any, <:ActiveMaterial, <:Any, <:Any}
function minimum_output_variables(
    system::ActiveMaterial, primary_variables
    )
    [:ChargeAcc, :MassAcc, :EnergyAcc, :Ocd]
end

function select_primary_variables_system!(
    S, domain, system::ActiveMaterial, formulation
    )
    S[:Phi] = Phi()
    S[:C] = C()
    # S[:T] = T()
end

function select_secondary_variables_system!(
    S, domain, system::ActiveMaterial, formulation
    )
    S[:TPkGrad_Phi] = TPkGrad{Phi}()
    S[:TPkGrad_C] = TPkGrad{C}()
    # S[:TPkGrad_T] = TPkGrad{T}()
    
    S[:ChargeAcc] = ChargeAcc()
    S[:MassAcc] = MassAcc()
    S[:EnergyAcc] = MassAcc()

    S[:Conductivity] = Conductivity()
    S[:Diffusivity] = Diffusivity()
    # S[:ThermalConductivity] = ThermalConductivity()
    S[:Ocd] = Ocd()
end

function select_equations_system!(
    eqs, domain, system::ActiveMaterial, formulation
    )
    charge_cons = (arg...; kwarg...) -> Conservation(ChargeAcc(), arg...; kwarg...)
    mass_cons = (arg...; kwarg...) -> Conservation(MassAcc(), arg...; kwarg...)
    # energy_cons = (arg...; kwarg...) -> Conservation(EnergyAcc(), arg...; kwarg...)
    eqs[:charge_conservation] = (charge_cons, 1)
    eqs[:mass_conservation] = (mass_cons, 1)
    # eqs[:energy_conservation] = (energy_cons, 1)
end

# ? Does this maybe look better ?
@terv_secondary(
function update_as_secondary!(
    vocd, tv::Ocd, model::ActiveMaterialModel, param, C
    )
    s = model.system
    # @tullio vocd[i] = ocd(T[i], C[i], s)
    @tullio vocd[i] = ocd(300.0, C[i], s)
end
)
