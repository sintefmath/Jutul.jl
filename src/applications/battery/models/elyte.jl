export Electrolyte, TestElyte, DmuDc, ConsCoeff
export p1, p2, p3, cnst, diffusivity

abstract type Electrolyte <: ElectroChemicalComponent end
struct TestElyte <: Electrolyte end

# Alias for convinience
const ElectrolyteModel = SimulationModel{<:Any, <:Electrolyte, <:Any, <:Any}
const TestElyteModel = SimulationModel{<:Any, <:TestElyte, <:Any, <:Any}

struct TPDGrad{T} <: KGrad{T} end
# Is it necesessary with a new struxt for all these?
struct DmuDc <: ScalarVariable end
struct ConsCoeff <: ScalarVariable end

struct DGradCCell <: CellVector end
struct DGradCSq <: ScalarNonDiagVaraible end
struct DGradCSqDiag <: ScalarVariable end
struct EnergyDensity <: ScalarNonDiagVaraible end
struct EDDiag <: ScalarVariable end

function select_equations_system!(eqs, domain, system::Electrolyte, formulation)
    charge_cons = (arg...; kwarg...) -> Conservation(Charge(), arg...; kwarg...)
    mass_cons = (arg...; kwarg...) -> Conservation(Mass(), arg...; kwarg...)
    energy_cons = (arg...; kwarg...) -> Conservation(Energy(), arg...; kwarg...)
    
    eqs[:charge_conservation] = (charge_cons, 1)
    eqs[:mass_conservation] = (mass_cons, 1)
    eqs[:energy_conservation] = (energy_cons, 1)
end

function select_primary_variables_system!(S, domain, system::Electrolyte, formulation)
    S[:Phi] = Phi()
    S[:C] = C()
    S[:T] = T()
end


function select_secondary_variables_system!(S, domain, system::Electrolyte, formulation)
    S[:TPkGrad_Phi] = TPkGrad{Phi}()
    S[:TPkGrad_C] = TPkGrad{C}()
    S[:TPDGrad_C] = TPDGrad{C}()
    S[:TPkGrad_T] = TPkGrad{T}()
    
    S[:Conductivity] = Conductivity()
    S[:ThermalConductivity] = Conductivity()
    S[:Diffusivity] = Diffusivity()
    S[:DmuDc] = DmuDc()
    S[:ConsCoeff] = ConsCoeff()

    S[:TotalCurrent] = TotalCurrent()
    S[:ChargeCarrierFlux] = ChargeCarrierFlux()

    S[:JCell] = JCell()
    S[:JSq] = JSq()
    S[:DGradCCell] = DGradCCell()
    S[:DGradCSq] = DGradCSq()

    S[:EnergyDensity] = EnergyDensity()

    S[:Charge] = Charge()
    S[:Mass] = Mass()
    S[:Energy] = Energy()

    # Variables for plotting
    S[:DGradCSqDiag] = DGradCSqDiag()
    S[:JSqDiag] = JSqDiag()
    S[:EDDiag] = EDDiag()
end

# Must be available to evaluate time derivatives
function minimum_output_variables(system::Electrolyte, primary_variables)
    [
        :Charge, :Mass, :Energy, :Conductivity, :Diffusivity,
        :TotalCurrent, :DGradCSqDiag, :JSqDiag, :EnergyDensity
    ]
end


function update_linearized_system_equation!(
    nz, r, model::TestElyteModel, law::Conservation{Energy}
    )
    
    acc = get_diagonal_cache(law)
    cell_flux = law.half_face_flux_cells
    cpos = law.flow_discretization.conn_pos
    density = law.density

    @sync begin
        @async fill_jac_flux_and_acc!(nz, r, model, acc, cell_flux, cpos)
        @async fill_jac_density!(nz, r, model, density)
    end
end


#######################
# Secondary Variables #
#######################

function setup_parameters(::ElectrolyteModel)
    d = Dict{Symbol, Any}()
    d[:tolerances] = Dict{Symbol, Any}()
    d[:tolerances][:default] = 1e-3
    d[:t] = 1.
    d[:z] = 1.
    return d
end

const poly_param = [
    -10.5       0.074       -6.96e-5    ;
    0.668e-3    -1.78e-5    2.80e-8     ;
    0.494e-6    -8.86e-10   0           ;
]
const p1 = Polynomial(poly_param[1:end, 1])
const p2 = Polynomial(poly_param[1:end, 2])
const p3 = Polynomial(poly_param[1:end, 3])

@inline function cond(T::Real, C::Real, ::Electrolyte)
    fact = 1e-4  # * 500 # fudge factor
    return fact * C * (p1(C) + p2(C) * T + p3(C) * T^2)^2
end

const diff_params = [
    -4.43   -54 ;
    -0.22   0.0 ;
]
const Tgi = [229 5.0]

@inline function diffusivity(T::Real, C::Real, ::Electrolyte)
    return (
        1e-4 * 10 ^ ( 
            diff_params[1,1] + 
            diff_params[1,2] / ( T - Tgi[1] - Tgi[2] * C * 1e-3) + 
            diff_params[2,1] * C * 1e-3
            )
        )
end


@terv_secondary(
function update_as_secondary!(dmudc, sv::DmuDc, model, param, T, C)
    R = GAS_CONSTANT
    @tullio dmudc[i] = R * (T[i] / C[i])
end
)

# ? Does this maybe look better ?
@terv_secondary(
function update_as_secondary!(
    con, tv::Conductivity, model::ElectrolyteModel, param, T, C
    )
    s = model.system
    vf = model.domain.grid.vol_frac
    @tullio con[i] = cond(T[i], C[i], s) * vf[i]^1.5
end
)

@terv_secondary function update_as_secondary!(
    D, sv::Diffusivity, model::ElectrolyteModel, param, C, T
    )
    s = model.system
    vf = model.domain.grid.vol_frac
    @tullio D[i] = diffusivity(T[i], C[i], s)  * vf[i]^1.5
end


@terv_secondary function update_as_secondary!(
    coeff, tv::ConsCoeff, model::ElectrolyteModel, param, Conductivity, DmuDc
    )
    t = param.t
    z = param.z
    F = FARADAY_CONST
    @tullio coeff[i] = Conductivity[i]*DmuDc[i] * t/(F*z)
end

@terv_secondary function update_as_secondary!(
    kGrad_C, tv::TPkGrad{C}, model::ElectrolyteModel, param, C, ConsCoeff
    )
    mf = model.domain.discretizations.charge_flow
    conn_data = mf.conn_data
    @tullio kGrad_C[i] = half_face_two_point_kgrad(conn_data[i], C, ConsCoeff)
end

@terv_secondary function update_as_secondary!(
    DGrad_C, tv::TPDGrad{C}, model::ElectrolyteModel, param, C, Diffusivity
    )
    mf = model.domain.discretizations.charge_flow
    conn_data = mf.conn_data
    @tullio DGrad_C[i] = half_face_two_point_kgrad(conn_data[i], C, Diffusivity)
end


@terv_secondary function update_as_secondary!(
    j, tv::TotalCurrent, model, param, TPkGrad_C, TPkGrad_Phi
    )
    @tullio j[i] =  - TPkGrad_C[i] - TPkGrad_Phi[i]
end

@terv_secondary function update_as_secondary!(
    N, tv::ChargeCarrierFlux, model, param, TPDGrad_C, TotalCurrent
    )
    t = param.t
    z = param.z
    F = FARADAY_CONST
    @tullio N[i] =  + TPDGrad_C[i] + t / (F * z) * TotalCurrent[i]
end

@terv_secondary(
function update_as_secondary!(j_cell, sc::JCell, model, param, TotalCurrent)
    for c in 1:number_of_cells(model.domain)
        face_to_cell!(j_cell, TotalCurrent, c, model)
    end
end
)

@terv_secondary(
function update_as_secondary!(jsq, sc::JSq, model, param, JCell)
    for c in 1:number_of_cells(model.domain)
        vec_to_scalar!(jsq, JCell, c, model)
    end
end
)

@terv_secondary(
function update_as_secondary!(j_cell, sc::DGradCCell, model, param, TPDGrad_C)
    for c in 1:number_of_cells(model.domain)
        face_to_cell!(j_cell, TPDGrad_C, c, model)
    end
end
)

@terv_secondary(
function update_as_secondary!(jsq, sc::DGradCSq, model, param, DGradCCell)
    for c in 1:number_of_cells(model.domain)
        vec_to_scalar!(jsq, DGradCCell, c, model)
    end
end
)

@terv_secondary(
function update_as_secondary!(
    ρ, sc::EnergyDensity, model, param, DGradCSq, JSq, Diffusivity, Conductivity, DmuDc
    )
    κ = Conductivity
    D = Diffusivity

    mf = model.domain.discretizations.charge_flow
    cc = mf.cellcell
    nc = number_of_cells(model.domain)
    v = (a, c, n) -> (c == n) ? a[c] : value(a[c])
    for c = 1:nc
        for cn in cc.pos[c]:(cc.pos[c+1]-1)
            c, n = cc.tbl[cn]
            ρ[cn] = JSq[cn] / v(κ, c, n) + v(DmuDc, c, n) * DGradCSq[cn] / v(D, c, n)
        end
    end
end
)

@terv_secondary(
function update_as_secondary!(jsq_diag, sc::JSqDiag, model, param, JSq)
    """ Carries the diagonal velues of JSq """
    mf = model.domain.discretizations.charge_flow
    cc = mf.cellcell
    nc = number_of_cells(model.domain)
    for cn in cc.pos[1:end-1] # The diagonal elements
        c, n = cc.tbl[cn]
        @assert c == n
        jsq_diag[c] = JSq[cn]
    end
end
)

@terv_secondary(
function update_as_secondary!(jsq_diag, sc::DGradCSqDiag, model, param, DGradCSq)
    mf = model.domain.discretizations.charge_flow
    cc = mf.cellcell
    nc = number_of_cells(model.domain)
    for cn in cc.pos[1:end-1] # The diagonal elements
        c, n = cc.tbl[cn]
        @assert c == n
        jsq_diag[c] = DGradCSq[cn]
    end
end
)

@terv_secondary(
function update_as_secondary!(ρ_diag, sc::EDDiag, model, param, EnergyDensity)
    mf = model.domain.discretizations.charge_flow
    cc = mf.cellcell
    nc = number_of_cells(model.domain)
    for cn in cc.pos[1:end-1] # The diagonal elements
        c, n = cc.tbl[cn]
        @assert c == n
        ρ_diag[c] = EnergyDensity[cn]
    end
end
)

function update_density!(law::Conservation{Energy}, storage, model::ElectrolyteModel)
    ρ = storage.state.EnergyDensity
    ρ_law = get_entries(law.density)
    @tullio ρ[c] = ρ_law[i]
end

function get_flux(
    storage,  model::ElectrolyteModel, law::Conservation{Charge}
    )
    return storage.state.TotalCurrent
end

function get_flux(
    storage,  model::ElectrolyteModel, law::Conservation{Mass}
    )
    return storage.state.ChargeCarrierFlux
end

function get_flux(
    storage,  model::ElectrolyteModel, law::Conservation{Energy}
    )
    return - storage.state.TPkGrad_T
end

function align_to_jacobian!(
    law::Conservation, jac, model::TestElyteModel, u::Cells; equation_offset = 0, 
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


function apply_boundary_potential!(
    acc, state, parameters, model::ElectrolyteModel, eq::Conservation{Charge}
    )
    # values
    Phi = state[:Phi]
    C = state[:C]
    κ = state[:Conductivity]
    coeff = state[:ConsCoeff]

    BPhi = state[:BoundaryPhi]
    BC = state[:BoundaryC]

    bc = model.domain.grid.boundary_cells
    T_hf = model.domain.grid.boundary_T_hf

    for (i, c) in enumerate(bc)
        @inbounds acc[c] -= (
            - coeff[c] * T_hf[i] * (C[c] - BC[i])
            - κ[c] * T_hf[i] * (Phi[c] - BPhi[i])
        )
    end
end


function apply_boundary_potential!(
    acc, state, parameters, model::ElectrolyteModel, eq::Conservation{Mass}
    )
    # values
    Phi = state[:Phi]
    C = state[:C]
    κ = state[:Conductivity]
    D = state[:Diffusivity]

    F = FARADAY_CONST
    z = parameters.z
    t = parameters.t
    coeff = state[:ConsCoeff]

    BPhi = state[:BoundaryPhi]
    BC = state[:BoundaryC]

    bc = model.domain.grid.boundary_cells
    T_hf = model.domain.grid.boundary_T_hf

    for (i, c) in enumerate(bc)
        @inbounds j = (
            - coeff[c] * T_hf[i] * (C[c] - BC[i])
            - κ[c] * T_hf[i] * (Phi[c] - BPhi[i])
        )
        @inbounds acc[c] -= (
            - D[c] * T_hf[i] * (C[c] - BC[i])
            + t / (F * z) * j
        )
    end
end

function apply_boundary_potential!(
    acc, state, parameters, model::ElectrolyteModel, eq::Conservation{Energy}
    )
    # values
    T = state[:T]
    λ = state[:ThermalConductivity]
    BT = state[:BoundaryT]


    # Type
    bc = model.domain.grid.boundary_cells
    T_hf = model.domain.grid.boundary_T_hf

    for (i, c) in enumerate(bc)
        # TODO: Add influence of boundary on energy density
        @inbounds acc[c] -= - λ[c] * T_hf[i] * (T[c] - BT[i])
    end
end

