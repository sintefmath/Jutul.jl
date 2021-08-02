using Terv
export CurrentCollectorT

struct CurrentCollectorT <: ElectroChemicalComponent end
const CCT = SimulationModel{<:Any, <:CurrentCollectorT, <:Any, <:Any}

struct kGradPhiCell <: CellVector end
struct kGradPhiSq <: ScalarNonDiagVaraible end
struct kGradPhiSqDiag <: ScalarVariable end

function minimum_output_variables(
    system::CurrentCollectorT, primary_variables
    )
    [:ChargeAcc, :EnergyAcc]
end

function select_primary_variables_system!(
    S, domain, system::CurrentCollectorT, formulation
    )
    S[:Phi] = Phi()
    S[:T] = T()
end

function select_secondary_variables_system!(
    S, domain, system::CurrentCollectorT, formulation
    )
    S[:TPkGrad_Phi] = TPkGrad{Phi}()
    S[:TPkGrad_T] = TPkGrad{T}()

    S[:ChargeAcc] = ChargeAcc()
    S[:EnergyAcc] = EnergyAcc()

    S[:Conductivity] = Conductivity()
    S[:ThermalConductivity] = ThermalConductivity()

    S[:BoundaryPhi] = BoundaryPotential{Phi}()
    S[:BoundaryT] = BoundaryPotential{T}()
    
    S[:kGradPhiCell] = kGradPhiCell()    
    S[:kGradPhiSq] = kGradPhiSq()
    S[:kGradPhiSqDiag] = kGradPhiSqDiag()
end


function select_equations_system!(
    eqs, domain, system::CurrentCollectorT, formulation
    )
    charge_cons = (arg...; kwarg...) -> Conservation(ChargeAcc(), arg...; kwarg...)
    energy_cons = (arg...; kwarg...) -> Conservation(EnergyAcc(), arg...; kwarg...)
    eqs[:charge_conservation] = (charge_cons, 1)
    eqs[:energy_conservation] = (energy_cons, 1)
end


function update_density!(law::Conservation, storage, model::CCT)
    ρ = storage.state.kGradPhiSq # ! Should devide on κ
    ρ_law = get_entries(law.density)
    @tullio ρ[i] = ρ_law[i]
end


@terv_secondary(
function update_as_secondary!(j_cell, sc::kGradPhiCell, model, param, TPkGrad_Phi)

    P = model.domain.grid.P
    J = TPkGrad_Phi
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
function update_as_secondary!(jsq, sc::kGradPhiSq, model, param, kGradPhiCell)

    S = model.domain.grid.S
    mf = model.domain.discretizations.charge_flow
    conn_data = mf.conn_data
    cctbl = model.domain.grid.cellcelltbl
    ccv = model.domain.grid.cellcellvectbl

    jsq .= 0
    for c in 1:number_of_cells(model.domain)
        vec_to_scalar!(jsq, kGradPhiCell, c, S, ccv, cctbl, conn_data)
    end
end
)

@terv_secondary(
function update_as_secondary!(jsq_diag, sc::kGradPhiSqDiag, model, param, kGradPhiSq)
    """ Carries the diagonal velues of JSq """

    cctbl = model.domain.grid.cellcelltbl
    cc = i -> get_cell_index_scalar(i, i, cctbl)
    for i in 1:number_of_cells(model.domain)
        jsq_diag[i] = kGradPhiSq[cc(i)]
    end
end
)
