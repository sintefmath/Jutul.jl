using Terv
export CurrentCollector

struct CurrentCollector <: ElectroChemicalComponent end

function number_of_units(model, BP::BoundaryPotential)
    return size(BP.cells)[1]
end

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


    μ = 2.1 # Why not?
    S[:Conductivity] = ConstantVariables([μ,])
end


function select_equations_system!(
    eqs, domain, system::CurrentCollector, formulation
    )
    charge_cons = (arg...; kwarg...) -> Conservation(ChargeAcc(), arg...; kwarg...)
    eqs[:charge_conservation] = (charge_cons, 1)
end


@terv_secondary function update_as_secondary!(
    kGrad, sv::TPkGrad{Phi}, model::SimulationModel{D, S, F, C}, param, 
    Phi, Conductivity ) where {D, S <: CurrentCollector, F, C}
    mf = model.domain.discretizations.charge_flow
    conn_data = mf.conn_data
    σ = Conductivity
    @tullio kGrad[i] = half_face_two_point_kgrad(conn_data[i], Phi, σ)
end
