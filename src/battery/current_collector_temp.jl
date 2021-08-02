using Terv
export CurrentCollectorT

struct CurrentCollectorT <: ElectroChemicalComponent end

function minimum_output_variables(
    system::CurrentCollector, primary_variables
    )
    [:ChargeAcc]
    [:EnergyAcc]
end

function select_primary_variables_system!(
    S, domain, system::CurrentCollector, formulation
    )
    S[:Phi] = Phi()
    S[:T] = T
end

function select_secondary_variables_system!(
    S, domain, system::CurrentCollector, formulation
    )
    S[:TPkGrad_Phi] = TPkGrad{Phi}()
    S[:TPkGrad_T] = TPkGrad{T}()

    S[:ChargeAcc] = ChargeAcc()
    S[:EnergyAcc] = EnergyAcc()

    S[:Conductivity] = Conductivity()
    S[:ThermalConductivity] = ThermalConductivity()
end


function select_equations_system!(
    eqs, domain, system::CurrentCollector, formulation
    )
    charge_cons = (arg...; kwarg...) -> Conservation(ChargeAcc(), arg...; kwarg...)
    energy_cons = (arg...; kwarg...) -> Conservation(EnergyAcc(), arg...; kwarg...)
    eqs[:charge_conservation] = (charge_cons, 1)
    eqs[:energy_conservation] = (energy_cons, 1)
end

@terv_secondary(
function update_as_secondary!(j_cell, sc::JCell, model, param, TotalCurrent)

    P = model.domain.grid.P
    J = TotalCurrent
    mf = model.domain.discretizations.charge_flow
    ccv = model.domain.grid.cellcellvectbl
    conn_data = mf.conn_data

    j_cell .= 0 # ? Is this necesessary ?
    for c in 1:number_of_cells(model.domain)
        face_to_cell!(j_cell, J, c, P, ccv, conn_data)
    end
end
)

@terv_secondary(
function update_as_secondary!(jsq, sc::JSq, model, param, JCell)

    S = model.domain.grid.S
    mf = model.domain.discretizations.charge_flow
    conn_data = mf.conn_data
    cctbl = model.domain.grid.cellcelltbl
    ccv = model.domain.grid.cellcellvectbl

    jsq .= 0
    for c in 1:number_of_cells(model.domain)
        vec_to_scalar(jsq, JCell, c, S, ccv, cctbl, conn_data)
    end
end
)
