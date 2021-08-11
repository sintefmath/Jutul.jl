#=
A simple model of electrolyte without energy conservation
=#

using Terv
export SimpleElyte

struct SimpleElyte <: Electrolyte end
const SimpleElyteModel = SimulationModel{<:Any, <:SimpleElyte, <:Any, <:Any}


function select_primary_variables_system!(S, domain, system::SimpleElyte, formulation)
    S[:Phi] = Phi()
    S[:C] = C()
end

function select_equations_system!(eqs, domain, system::SimpleElyte, formulation)
    charge_cons = (arg...; kwarg...) -> Conservation(ChargeAcc(), arg...; kwarg...)
    mass_cons = (arg...; kwarg...) -> Conservation(MassAcc(), arg...; kwarg...)
    
    eqs[:charge_conservation] = (charge_cons, 1)
    eqs[:mass_conservation] = (mass_cons, 1)
end


function select_secondary_variables_system!(S, domain, system::SimpleElyte, formulation)
    S[:TPkGrad_Phi] = TPkGrad{Phi}()
    S[:TPkGrad_C] = TPkGrad{C}()
    S[:TPDGrad_C] = TPDGrad{C}()

    S[:T] = T()
    S[:Conductivity] = Conductivity()
    S[:Diffusivity] = Diffusivity()
    S[:DmuDc] = DmuDc()
    S[:ConsCoeff] = ConsCoeff()

    S[:TotalCurrent] = TotalCurrent()
    S[:ChargeCarrierFlux] = ChargeCarrierFlux()

    S[:ChargeAcc] = ChargeAcc()
    S[:MassAcc] = MassAcc()
end

function minimum_output_variables(system::SimpleElyte, primary_variables)
    return [
        :ChargeAcc, :MassAcc, :Conductivity, :Diffusivity, :TPkGrad_Phi, :TPkGrad_C, 
        :TotalCurrent, :ChargeCarrierFlux
        ]
end
