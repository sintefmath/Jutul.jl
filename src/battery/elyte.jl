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

# Abstract type of a vector that is defined on a cell, from face flux
abstract type CellVector end
struct JCell <: CellVector end
struct DGradCCell <: CellVector end

struct ScalarNonDiagVaraible <: ScalarVariable end
struct JSq <: ScalarVariable end
struct DGradCSq <: ScalarVariable end

function number_of_units(model, pv::ScalarNonDiagVaraible)
    """ Each value depends on a cell and all its neighbours """
    # TODO: Return number of cells + number of neighbours
end

function number_of_units(model, pv::CellVector)
    """ 2 components per cell """
    return 2*count_units(model.domain, Cells())
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

    S[:ChargeAcc] = ChargeAcc()
    S[:MassAcc] = MassAcc()
    S[:EnergyAcc] = EnergyAcc()
end

# Must be available to evaluate time derivatives
function minimum_output_variables(
    system::Electrolyte, primary_variables
    )
    [:ChargeAcc, :MassAcc, :EnergyAcc, :Conductivity, :Diffusivity, :TotalCurrent, :JSq]
end

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


function get_cell_index(c, k, diff_C)
    # TODO: Should return the index of j_cell that holds the value of j, in cell c, with the derivative w.r.t cell diff_c
end

function get_face_index(f, diff_c)
    # TODO: Return the value at face f, that depends on cell diff_c
end

function get_face(c, model)
    #TODO: get all the faces of C
end

function get_neighbors(c, model)
    # TODO: Return a list of the neighbors of c
end

@terv_secondary(
function update_as_secondary!(j_cell, sc::JCell, model, param, TotalCurrent)
    """
    j_cell is a vector that has 2 coponents per cell
    P maps between values defined on the face, and vectors defined in cells
    P_[c, i, f] = P_[2*c + i, f], (c=cell, i=space, f=face)
    j_c[c, i, c'] = P_[c, i, f] * J_[f, c'] (c'=cell dependence)
    """
    P = model.domain.grid.P
    J = TotalCurrent    
    mf = model.domain.discretizations.charge_flow
    conn_data = mf.conn_data
    # TODO: Make map from face-valued flux to cell-valued vector
    for c in 1:number_of_cells(model.domain.discretization)
        for f in get_faces(c, model)
            cic = get_cell_index(c, i, c)
            fc = get_face_index(f, c)
            j_cell[cic] = P[2*c + i, f] * J[fc]
        end
        for n in get_neighbours(c, model)
            for f in get_faces(c, model)
                # TODO: Make so that J only depends on n
                cin = get_cell_index(c, i, n)
                get_cell_index(c, i, n)
                fc = get_face_index(f, n)
                fc = get_face_index(f, c)
                j_cell[cin] = P[2*c + i, f] * J[fc]
            end
        end
    end
end
)

@terv_secondary(
function update_as_secondary!(j_sq, sc::JSq, model, param, JCell)
    # TODO: Make map from cell-valued vector to scalar, include dep.
    nothing
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
    # TODO: Add energy density from J^2 and GradC^2
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



