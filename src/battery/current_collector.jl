using Terv

export CurrentCollector

struct CurrentCollector <: ElectroChemicalComponent end

function minimum_output_variables(
    system::CurrentCollector, primary_variables
    )
    [:ChargeAcc]
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
    S[:ChargeAcc] = ChargeAcc()
    # S[:Conductivity] = ConstantVariables()
end


function select_equations_system!(
    eqs, domain, system::CurrentCollector, formulation
    )
    charge_cons = (arg...; kwarg...) -> Conservation(ChargeAcc(), arg...; kwarg...)
    eqs[:charge_conservation] = (charge_cons, 1)
end


# @terv_secondary function update_as_secondary!(
#     pot, tv::TPkGrad{Phi}, model::SimulationModel{D, S, F, C}, param, 
#     Phi, Conductivty ) where {D, S <: ElectroChemicalComponent, F, C}
#     mf = model.domain.discretizations.charge_flow
#     conn_data = mf.conn_data
#     σ = Conductivty
#     @tullio pot[i] = half_face_two_point_kgrad(conn_data[i], Phi, σ)
# end
