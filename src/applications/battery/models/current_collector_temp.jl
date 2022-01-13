export CurrentCollectorT

struct CurrentCollectorT <: ElectroChemicalComponent end
const CCT = SimulationModel{<:Any, <:CurrentCollectorT, <:Any, <:Any}

struct kGradPhiCell <: CellVector end
struct EDensity <: ScalarNonDiagVaraible end
struct EDensityDiag <: ScalarVariable end

function minimum_output_variables(
    system::CurrentCollectorT, primary_variables
    )
    [:Charge, :Energy, :EDensityDiag]
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

    S[:Charge] = Charge()
    S[:Energy] = Energy()

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
    charge_cons = (arg...; kwarg...) -> Conservation(Charge(), arg...; kwarg...)
    energy_cons = (arg...; kwarg...) -> Conservation(Energy(), arg...; kwarg...)
    eqs[:charge_conservation] = (charge_cons, 1)
    eqs[:energy_conservation] = (energy_cons, 1)
end

function update_linearized_system_equation!(
    nz, r, model::CCT, law::Conservation{Energy}
    )
    
    acc = get_diagonal_cache(law)
    cell_flux = law.half_face_flux_cells
    cpos = law.flow_discretization.conn_pos
    density = law.density

    fill_jac_flux_and_acc!(nz, r, model, acc, cell_flux, cpos)
    fill_jac_density!(nz, r, model, density)
end


function update_density!(law::Conservation{Energy}, storage, model::CCT)
    ρ = storage.state.EDensity
    ρ_law = get_entries(law.density)
    @tullio ρ_law[i] = ρ[i]
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

    mf = model.domain.discretizations.charge_flow
    cc = mf.cellcell
    for c = 1:nc
        for cn in cc.pos[c]:(cc.pos[c+1]-1)
            c, n = cc.tbl[cn]
            κc = (c == n) ? κ[c] : value(κ[c])
            ρ[cn] *= 1 / κc
        end
    end
end
)

@terv_secondary(
function update_as_secondary!(ρ_diag, sc::EDensityDiag, model, param, EDensity)
    """ Carries the diagonal velues of ρ """
    mf = model.domain.discretizations.charge_flow
    cc = mf.cellcell
    nc = number_of_cells(model.domain)
    for cn in cc.pos[1:end-1] # The diagonal elements
        c, n = cc.tbl[cn]
        @assert c == n
        ρ_diag[c] = EDensity[cn]
    end
end
)

function align_to_jacobian!(
    law::Conservation, jac, model::CCT, u::Cells; equation_offset = 0, 
    variable_offset = 0
    )
    fd = law.flow_discretization
    M = global_map(model.domain)

    acc = law.accumulation
    hflux_cells = law.half_face_flux_cells
    density = law.density

    diagonal_alignment!(
        acc, jac, u, model.context;
        target_offset = equation_offset, source_offset = variable_offset
        )
    half_face_flux_cells_alignment!(
        hflux_cells, acc, jac, model.context, M, fd, 
        target_offset = equation_offset, source_offset = variable_offset
        )
    density_alignment!(
        density, acc, jac, model.context, fd;
        target_offset = equation_offset, source_offset = variable_offset
        )
end