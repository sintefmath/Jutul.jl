using Terv, Polynomials

export Electrolyte, TestElyte
export p1, p2, p3, cnst

abstract type Electrolyte <: ElectroChemicalComponent end
struct TestElyte <: Electrolyte end

# Alias for convinience
const ElectrolyteModel = SimulationModel{<:Any, <:Electrolyte, <:Any, <:Any}

struct TPDGrad{T} <: KGrad{T} end
# Is it necesessary with a new struxt for all these?
struct DmuDc <: ScalarVariable end
struct ConsCoeff <: ScalarVariable end

abstract type Current <: ScalarVariable end
struct TotalCurrent <: Current end
struct ChargeCarrierFlux <: Current end
struct EnergyFlux <: Current end

abstract type NonDiagCellVariables <: TervVariables end

# Abstract type of a vector that is defined on a cell, from face flux
abstract type CellVector <: NonDiagCellVariables end
struct JCell <: CellVector end
struct DGradCCell <: CellVector end

abstract type ScalarNonDiagVaraible <: NonDiagCellVariables end
struct JSq <: ScalarNonDiagVaraible end
struct DGradCSq <: ScalarNonDiagVaraible end

struct JSqDiag <: ScalarVariable end
struct DGradCSqDiag <: ScalarVariable end

struct EnergyDensity <: ScalarNonDiagVaraible end
struct EDDiag <: ScalarVariable end

function initialize_variable_value(
    model, pvar::NonDiagCellVariables, val; perform_copy=true
    )
    nu = number_of_units(model, pvar)
    nv = values_per_unit(model, pvar)
    
    @assert length(val) == nu * nv "Expected val length $(nu*nv), got $(length(val))"
    val::AbstractVector

    if perform_copy
        val = deepcopy(val)
    end
    return transfer(model.context, val)
end

function initialize_variable_value!(state, model, pvar::NonDiagCellVariables, symb::Symbol, val::Number)
    num_val = number_of_units(model, pvar)*values_per_unit(model, pvar)
    V = repeat([val], num_val)
    return initialize_variable_value!(state, model, pvar, symb, V)
end

function number_of_units(model, pv::NonDiagCellVariables)
    """ Each value depends on a cell and all its neighbours """
    return size(model.domain.grid.cellcelltbl, 1)
end

function values_per_unit(model, u::CellVector)
    return 2
end

function values_per_unit(model, u::ScalarNonDiagVaraible)
    return 1
end

function degrees_of_freedom_per_unit(model, sf::NonDiagCellVariables)
    return values_per_unit(model, sf) 
end

function number_of_units(model, pv::Current)
    return 2*count_units(model.domain, Faces())
end

function select_equations_system!(
    eqs, domain, system::Electrolyte, formulation
    )
    charge_cons = (arg...; kwarg...) -> Conservation(
        ChargeAcc(), arg...; kwarg...
        )
    mass_cons = (arg...; kwarg...) -> Conservation(
        MassAcc(), arg...; kwarg...
        )
    energy_cons = (arg...; kwarg...) -> Conservation(
        EnergyAcc(), arg...; kwarg...
        )
    
    eqs[:charge_conservation] = (charge_cons, 1)
    eqs[:mass_conservation] = (mass_cons, 1)
    eqs[:energy_conservation] = (energy_cons, 1)
end

function select_primary_variables_system!(
    S, domain, system::Electrolyte, formulation
    )
    S[:Phi] = Phi()
    S[:C] = C()
    S[:T] = T()
end


function select_secondary_variables_system!(
    S, domain, system::Electrolyte, formulation
    )
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

    S[:ChargeAcc] = ChargeAcc()
    S[:MassAcc] = MassAcc()
    S[:EnergyAcc] = EnergyAcc()

    # Variables for plotting
    S[:DGradCSqDiag] = DGradCSqDiag()
    S[:JSqDiag] = JSqDiag()
    S[:EDDiag] = EDDiag()
end

# Must be available to evaluate time derivatives
function minimum_output_variables(
    system::Electrolyte, primary_variables
    )
    [
        :ChargeAcc, :MassAcc, :EnergyAcc, :Conductivity, :Diffusivity,
        :TotalCurrent, :DGradCSqDiag, :JSqDiag, :EnergyDensity
    ]
end


##############
# Tensormaps #
##############
# TODO: These should all be computed initially, to avoid serach later
# TODO: Use standard way to loop through neigh. (as fill_jac_entries!)

function get_cell_index_vec(c, n, i, tbl)
    """ 
    Returns cni, the index of a vector defined on cells, with dependence
    on neighbouring cells. v[cni] is the i'th component of the field in
    cell c, dependent on cell n (for partial derivatives).
    """
    bool = map(x -> x.cell == c && x.cell_dep == n && x.vec == i, tbl)
    indx = findall(bool)
    @assert size(indx) == (1,) "Invalid or duplicate face cell combo, size = $(size(indx)) for (c, n, i) = ($c, $n, $i)"
    return indx[1] # cni
end

function get_face_index(f, c, conn_data)
    """
    Retuns i, b. i is the index of a vector defined on faces. v[i] is the
    value on face f. b is true if f is a face of the cell c, and false else
    """
    bool = map(x -> [x.face == f x.self == c], conn_data)
    indx = findall(x -> x[1] == 1, bool)
    @assert size(indx) == (2,) "Invalid or duplicate face cell combo, size $(size(indx)) for (f, c) = ($f, $c)"
    #! Very ugly, can this be done better?
    if bool[indx[1]][2]
        return indx[1], true
    elseif bool[indx[2]][2]
        return indx[2], true
    else   
        return indx[1][1], false
    end
end

function get_cell_index_scalar(c, n, tbl)
    """
    Returns cn, the index of a scalar defined on cells, which also depends
    on neighbouring cells. v[cn] is the value at cell c, dependent on c.
    """
    bool = map(x -> x.cell == c && x.cell_dep == n, tbl)
    indx = findall(bool)
    @assert size(indx) == (1,) "Invalid or duplicate face cell combo, size = $(size(indx)) for (c, n, i) = ($c, $n)"
    return indx[1]
end


# ! Boundary current is not included
function face_to_cell!(j_cell, J, c, P, ccv, conn_data)
    """
    Take a field defined on faces, J, finds its value at the cell c.
    j_cell is a vector that has 2 coponents per cell
    P maps between values defined on the face, and vectors defined in cells
    P_[c, f, i] = P_[2*(c-1) + i, f], (c=cell, i=space, f=face)
    j_c[c, c', i] = P_[c, f, i] * J_[f, c'] (c'=cell dependence)
    """

    # TODO: Combine the loop over neigh_self with the self
    cell_mask = map(x -> x.self==c, conn_data)
    neigh_self = conn_data[cell_mask] # ? is this the best way??
    for neigh in neigh_self
        f = neigh.face
        for i in 1:2 #! Only valid in 2D for now
            cic = get_cell_index_vec(c, c, i, ccv)
            fc, bool = get_face_index(f, c, conn_data)
            @assert bool
            ci = 2*(c-1) + i
            j_cell[cic] += P[ci, f] * J[fc]
        end
    end

    # what is the best order to loop through?
    for neigh in neigh_self
        n = neigh.other
        for neigh2 in neigh_self
            f = neigh2.face
            for i in 1:2
                cin = get_cell_index_vec(c, n, i, ccv)
                fn, bool = get_face_index(f, n, conn_data)

                # The value should only depend on cell n
                if bool
                    Jfn = J[fn]
                else
                    Jfn = value(J[fn])
                end

                j_cell[cin] += P[2*(c-1) + i, f] * Jfn
            end
        end
    end # the end is near
end

function vec_to_scalar(jsq, j, c, S, ccv, cctbl, conn_data)
    """
    Takes in vector valued field defined on the cell, and returns the
    modulus square
    jsq[c, c'] = S[c, 2*(c-1) + i] * j[c, c', i]^2
    """
    cell_mask = map(x -> x.self==c, conn_data)
    neigh_self = conn_data[cell_mask]
    cc = get_cell_index_scalar(c, c, cctbl)
    for i in 1:2
        cci = get_cell_index_vec(c, c, i, ccv)
        jsq[cc] += S[c, 2*(c-1) + i] * j[cci]^2
    end
    for neigh in neigh_self
        n = neigh.other
        cn = get_cell_index_scalar(c, n, cctbl)
        for i in 1:2
            cni = get_cell_index_vec(c, n, i, ccv)
            jsq[cn] += S[c, 2*(c-1) + i] * j[cni]^2
        end
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
    fact = 1e-4
    return fact * C * (p1(C) + p2(C) * T + p3(C) * T^2)^2
end

const diff_params = [
    -4.43   -54 ;
    -0.22   0.0 ;
]
const Tgi = [229 5.0]

@inline function diffusivity(T::Real, C::Real, ::Electrolyte)
    # ?Should these be parameters?
    return (
        1e-4 * 10 ^ ( 
            diff_params[1,1] + 
            diff_params[1,2] / ( T - Tgi[1] - Tgi[2] * C * 1e-3) + 
            diff_params[2,1] * C * 1e-3
            )
        )
end

@terv_secondary function update_as_secondary!(
    dmudc, sv::DmuDc, model, param, T, C
    )
    R = GAS_CONSTANT
    @tullio dmudc[i] = R * (T[i] / C[i])
end

# ? Does this maybe look better ?
@terv_secondary(
function update_as_secondary!(
    con, tv::Conductivity, model::ElectrolyteModel, param, T, C
    )
    s = model.system
    @tullio con[i] = cond(T[i], C[i], s)
end
)

@terv_secondary function update_as_secondary!(
    D, sv::Diffusivity, model::ElectrolyteModel, param, C, T
    )
    s = model.system
    @tullio D[i] = diffusivity(T[i], C[i], s)
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
    j, tv::TotalCurrent, model, param, 
    TPkGrad_C, TPkGrad_Phi
    )
    @tullio j[i] =  - TPkGrad_C[i] - TPkGrad_Phi[i]
end

@terv_secondary function update_as_secondary!(
    N, tv::ChargeCarrierFlux, model, param, 
    TPDGrad_C, TotalCurrent
    )
    t = param.t
    z = param.z
    F = FARADAY_CONST
    @tullio N[i] =  - TPDGrad_C[i] + t / (F * z) * TotalCurrent[i]
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


@terv_secondary(
function update_as_secondary!(j_cell, sc::DGradCCell, model, param, TPDGrad_C)

    P = model.domain.grid.P
    J = TPDGrad_C
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
function update_as_secondary!(jsq, sc::DGradCSq, model, param, DGradCCell)

    S = model.domain.grid.S
    mf = model.domain.discretizations.charge_flow
    conn_data = mf.conn_data
    cctbl = model.domain.grid.cellcelltbl
    ccv = model.domain.grid.cellcellvectbl

    jsq .= 0
    for c in 1:number_of_cells(model.domain)
        vec_to_scalar(jsq, DGradCCell, c, S, ccv, cctbl, conn_data)
    end
end
)


@terv_secondary(
function update_as_secondary!(
    ρ, sc::EnergyDensity, model, param, DGradCSq, JSq, Diffusivity, Conductivity, DmuDc
    )
    
    mf = model.domain.discretizations.charge_flow
    conn_data = mf.conn_data
    cctbl = model.domain.grid.cellcelltbl

    κ = Conductivity
    D = Diffusivity

    for c in 1:number_of_cells(model.domain)

        cell_mask = map(x -> x.self==c, conn_data)
        neigh_self = conn_data[cell_mask]
        cc = get_cell_index_scalar(c, c, cctbl)

        # TODO: This should be combinded
        ρ[cc] = JSq[cc] / κ[c] + DmuDc[c] * DGradCSq[cc] / D[c]

        for neigh in neigh_self
            n = neigh.other
            cn = get_cell_index_scalar(c, n, cctbl)
            ρ[cn] = JSq[cn] / value(κ[c]) + value(DmuDc[c]) * DGradCSq[cn] / value(D[c])
        end
    end
end
)

function update_density!(law::Conservation, storage, model::ElectrolyteModel)
    ρ = storage.state.EnergyDensity
    ρ_law = get_entries(law.density)
    @tullio ρ[i] = ρ_law[i]
end


@terv_secondary(
function update_as_secondary!(jsq_diag, sc::JSqDiag, model, param, JSq)
    """ Carries the diagonal velues of JSq """

    cctbl = model.domain.grid.cellcelltbl
    cc = i -> get_cell_index_scalar(i, i, cctbl)
    for i in 1:number_of_cells(model.domain)
        jsq_diag[i] = JSq[cc(i)]
    end
end
)


@terv_secondary(
function update_as_secondary!(jsq_diag, sc::DGradCSqDiag, model, param, DGradCSq)
    cctbl = model.domain.grid.cellcelltbl
    cc = i -> get_cell_index_scalar(i, i, cctbl)
    for i in 1:number_of_cells(model.domain)
        jsq_diag[i] = DGradCSq[cc(i)]
    end
end
)


@terv_secondary(
function update_as_secondary!(ρ_diag, sc::EDDiag, model, param, EnergyDensity)
    cctbl = model.domain.grid.cellcelltbl
    cc = i -> get_cell_index_scalar(i, i, cctbl)
    for i in 1:number_of_cells(model.domain)
        ρ_diag[i] = EnergyDensity[cc(i)]
    end
end
)


function get_flux(
    storage,  model::ElectrolyteModel, law::Conservation{ChargeAcc}
    )
    return storage.state.TotalCurrent
end

function get_flux(
    storage,  model::ElectrolyteModel, law::Conservation{MassAcc}
    )
    return storage.state.ChargeCarrierFlux
end

function get_flux(
    storage,  model::ElectrolyteModel, law::Conservation{EnergyAcc}
    )
    return - storage.state.TPkGrad_T
end

function update_accumulation!(law::Conservation{EnergyAcc}, storage, model, dt)
    conserved = law.accumulation_symbol
    acc = get_entries(law.accumulation)
    m = storage.state[conserved]
    m0 = storage.state0[conserved]
    @tullio acc[c] = (m[c] - m0[c])/dt
    return acc
end

function apply_boundary_potential!(
    acc, state, parameters, model::ElectrolyteModel, eq::Conservation{ChargeAcc}
    )
    # values
    Phi = state[:Phi]
    C = state[:C]
    dmudc = state[:DmuDc]
    κ = state[:Conductivity]

    F = FARADAY_CONST
    z = parameters.z
    t = parameters.t

    BPhi = state[:BoundaryPhi]
    BC = state[:BoundaryC]

    bc = model.domain.grid.boundary_cells
    T_hf = model.domain.grid.boundary_T_hf

    for (i, c) in enumerate(bc)
        @inbounds acc[c] -= (
            - dmudc[c] * t/(F*z) * κ[c] * T_hf[i] * (C[c] - BC[i])
            - κ[c] * T_hf[i] * (Phi[c] - BPhi[i])
        )
    end
end


function apply_boundary_potential!(
    acc, state, model::ElectrolyteModel, eq::Conservation{MassAcc}
    )
    # values
    Phi = state[:Phi]
    C = state[:C]
    dmudc = state[:DmuDc]
    κ = state[:Conductivity]
    D = state[:Diffusivity]

    F = FARADAY_CONST
    z = parameters.z
    t = parameters.t

    BPhi = state[:BoundaryPhi]
    BC = state[:BoundaryC]

    bc = model.domain.grid.boundary_cells
    T_hf = model.domain.grid.boundary_T_hf

    for (i, c) in enumerate(bc)
        @inbounds j = (
            - dmudc[c] * t/(F*z) * κ[c] * T_hf[i] * (C[c] - BC[i])
            - κ[c] * T_hf[i] * (Phi[c] - BPhi[i])
        )
        @inbounds acc[c] -= (
            - D[c] * T_hf[i](C[c] - BC[i])
            + t / (F * z) * j
        )
    end
end

function apply_boundary_potential!(
    acc, state, model::ElectrolyteModel, eq::Conservation{EnergyAcc}
    )
    # values
    T = state[:T]
    λ = state[:ThermalConductivity]
    BT = state[:BoundaryT]


    # Type
    bp = model.secondary_variables[:BoundaryC]
    bc = model.domain.grid.boundary_cells
    T_hf = model.domain.grid.boundary_T_hf

    for (i, c) in enumerate(bc)
        # TODO: Add influence of boundary on energy density
        @inbounds acc[c] -= - λ[c] * T_hf[i] * (T[c] - BT[i])
    end
end

