using Terv
export CurrentCollectorT

struct CurrentCollectorT <: ElectroChemicalComponent end
const CCT = SimulationModel{<:Any, <:CurrentCollectorT, <:Any, <:Any}

struct kGradPhiCell <: CellVector end
struct EDensity <: ScalarNonDiagVaraible end
struct EDensityDiag <: ScalarVariable end

function minimum_output_variables(
    system::CurrentCollectorT, primary_variables
    )
    [:ChargeAcc, :EnergyAcc, :EDensity, :EDensityDiag]
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
    S[:EDensity] = EDensity()
    S[:EDensityDiag] = EDensityDiag() # For plotting
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
    ρ = storage.state.EDensity # ! Should divide on κ
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
        face_to_cell!(j_cell, J, c, model)
    end
end
)


@terv_secondary(
function update_as_secondary!(ρ, sc::EDensity, model, param, kGradPhiCell, Conductivity)

    S = model.domain.grid.S
    mf = model.domain.discretizations.charge_flow
    conn_data = mf.conn_data

    cctbl = model.domain.grid.cellcelltbl
    ccv = model.domain.grid.cellcellvectbl
    κ = Conductivity

    ρ .= 0
    for c in 1:number_of_cells(model.domain)
        vec_to_scalar!(ρ, kGradPhiCell, c, model)
    end

    cc = (c, n) -> get_cell_index_scalar(c, n, cctbl)
    nc = number_of_cells(model.domain)
    for c = 1:nc
        ρ[cc(c, c)] *= 1 / κ[c]
        κc = value(κ[c])
        @inbounds for neigh in get_neigh(c, model)
            n = neigh.other
            ρ[cc(c, n)] *= 1 / κc
        end
    end
end
)

@terv_secondary(
function update_as_secondary!(ρ_diag, sc::EDensityDiag, model, param, EDensity)
    """ Carries the diagonal velues of ρ """

    cctbl = model.domain.grid.cellcelltbl
    cc = i -> get_cell_index_scalar(i, i, cctbl)
    for i in 1:number_of_cells(model.domain)
        ρ_diag[i] = EDensity[cc(i)]
    end
end
)
