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

function update_linearized_system_equation!(
    nz, r, model::CCT, law::Conservation{EnergyAcc}
    )
    
    acc = get_diagonal_cache(law)
    cell_flux = law.half_face_flux_cells
    cpos = law.flow_discretization.conn_pos
    density = law.density

    fill_jac_flux_and_acc!(nz, r, model, acc, cell_flux, cpos)
    fill_jac_density!(nz, r, model, density)
end


function update_density!(law::Conservation{EnergyAcc}, storage, model::CCT)
    ρ = storage.state.EDensity
    ρ_law = get_entries(law.density)
    @tullio ρ[i] = ρ_law[i]
end


@terv_secondary(
function update_as_secondary!(j_cell, sc::kGradPhiCell, model, param, TPkGrad_Phi)
    nc = number_of_cells(model.domain)
    for c in 1:nc
        face_to_cell!(j_cell, TPkGrad_Phi, c, model)
    end
end
)

@terv_secondary(
function update_as_secondary!(ρ, sc::EDensity, model, param, kGradPhiCell, Conductivity)
    cctbl = model.domain.discretizations.charge_flow.cellcell.tbl
    κ = Conductivity

    nc = number_of_cells(model.domain)
    for c in 1:nc
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

    cctbl = model.domain.discretizations.charge_flow.cellcell.tbl
    cc = i -> get_cell_index_scalar(i, i, cctbl)
    for i in 1:number_of_cells(model.domain)
        ρ_diag[i] = EDensity[cc(i)]
    end
end
)
