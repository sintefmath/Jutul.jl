using Terv

# TODO: These are not all needed
export ElectroChemicalComponent, CurrentCollector, Electectrolyte, TestElyte
export vonNeumannBC, DirichletBC, BoundaryCondition, MinimalECTPFAGrid
export ChargeFlow, MixedFlow, Conservation, BoundaryPotential, BoundaryCurrent
export Phi, C, T, ChargeAcc, MassAcc, EnergyAcc, KGrad
export BOUNDARY_CURRENT, corr_type

###########
# Classes #
###########

abstract type ElectroChemicalComponent <: TervSystem end
# Alias for a genereal Electro Chemical Model
const ECModel = SimulationModel{<:Any, <:ElectroChemicalComponent, <:Any, <:Any}

abstract type ElectroChemicalGrid <: TervGrid end

# Potentials
# ? introduce abstract type Potential?
struct Phi <: ScalarVariable end
struct C <: ScalarVariable end
struct T <: ScalarVariable end

struct Conductivity <: ScalarVariable end
struct Diffusivity <: ScalarVariable end
struct ThermalConductivity <: ScalarVariable end

struct Conservation{T} <: TervEquation
    accumulation::TervAutoDiffCache
    accumulation_symbol::Symbol
    half_face_flux_cells::TervAutoDiffCache
    density::TervAutoDiffCache
    flow_discretization::FlowDiscretization
end

# Accumulation variables
abstract type AccumulationVariable <: ScalarVariable end
struct ChargeAcc <: AccumulationVariable end
struct MassAcc <: AccumulationVariable end
struct EnergyAcc <: AccumulationVariable end

# Currents corresponding to a accumulation type
const BOUNDARY_CURRENT = Dict(
    ChargeAcc() => :BCCharge,
    MassAcc()   => :BCMass,
    EnergyAcc() => :BCEnergy,
)
# TODO: Can this not be automated????
function corr_type(::Conservation{ChargeAcc}) ChargeAcc() end
function corr_type(::Conservation{MassAcc}) MassAcc() end
function corr_type(::Conservation{EnergyAcc}) EnergyAcc() end

# Represents k∇T, where k is a tensor, T a potential
abstract type KGrad{T} <: ScalarVariable end
struct TPkGrad{T} <: KGrad{T} end

abstract type ECFlow <: FlowType end
struct ChargeFlow <: ECFlow end
struct MixedFlow <: ECFlow end

struct BoundaryPotential{T} <: ScalarVariable end
struct BoundaryCurrent{T} <: ScalarVariable 
    cells
end

abstract type Current <: ScalarVariable end
struct TotalCurrent <: Current end
struct ChargeCarrierFlux <: Current end
struct EnergyFlux <: Current end

function number_of_units(model, pv::Current)
    return 2*count_units(model.domain, Faces())
end

abstract type NonDiagCellVariables <: TervVariables end

# Abstract type of a vector that is defined on a cell, from face flux
abstract type CellVector <: NonDiagCellVariables end
struct JCell <: CellVector end

abstract type ScalarNonDiagVaraible <: NonDiagCellVariables end
struct JSq <: ScalarNonDiagVaraible end

struct JSqDiag <: ScalarVariable end

struct MinimalECTPFAGrid{R<:AbstractFloat, I<:Integer} <: ElectroChemicalGrid
    """
    Simple grid for a electro chemical component
    """
    volumes::AbstractVector{R}
    neighborship::AbstractArray{I}
    boundary_cells::AbstractArray{I}
    boundary_T_hf::AbstractArray{R}
    P::AbstractArray{R} # Tensor to map from cells to faces
    S::AbstractArray{R} # Tensor map cell vector to cell scalar
end

struct TPFlow{F} <: FlowDiscretization
    conn_pos
    conn_data
    cellfacecellvec
    cellcellvec
end

function get_neighborship(grid)
    return grid.neighborship
end

function TPFlow(grid::TervGrid, T)
    N = get_neighborship(grid)
    faces, face_pos = get_facepos(N)

    nhf = length(faces)
    nc = length(face_pos) - 1
    if !isnothing(T)
        @assert length(T) == nhf ÷ 2
    end
    get_el = (face, cell) -> get_connection(face, cell, faces, N, T, nothing, nothing, false)
    el_type = typeof(get_el(1, 1))
    
    conn_data = Vector{el_type}(undef, nhf)
    @threads for cell = 1:nc
        @inbounds for fpos = face_pos[cell]:(face_pos[cell+1]-1)
            conn_data[fpos] = get_el(faces[fpos], cell)
        end
    end

    cfcv = get_cellfacecellvec_tbl(N, face_pos, conn_data, faces)
    cc = get_cellcellvec_tbl(N, face_pos, conn_data)

    TPFlow{ChargeFlow}(face_pos, conn_data, cfcv, cc)
end


################
# Constructors #
################


function MinimalECTPFAGrid(pv, N, bc=[], T_hf=[], P=[], S=[])
    nc = length(pv)
    pv::AbstractVector
    @assert size(N, 1) == 2
    if length(N) > 0
        @assert minimum(N) > 0
        @assert maximum(N) <= nc
    end
    @assert all(pv .> 0)
    @assert size(bc) == size(T_hf)
    

    MinimalECTPFAGrid{eltype(pv), eltype(N)}(pv, N, bc, T_hf, P, S)
end

function Conservation(
    acc_type, model, number_of_equations;
    flow_discretization = model.domain.discretizations[1], kwarg...
    )
    """
    A conservation law corresponding to the underlying potential pvar
    """

    accumulation_symbol = acc_symbol(acc_type)

    D = model.domain
    cell_unit = Cells()
    face_unit = Faces()
    nc = count_units(D, cell_unit)
    nf = count_units(D, face_unit)
    nhf = 2 * nf
    nn = size(get_neighborship(D.grid), 2)
    n_tot = nc + 2 * nn

    alloc = (n, unit, n_units_pos) -> CompactAutoDiffCache(
        number_of_equations, n, model, unit = unit, n_units_pos = n_units_pos,
        context = model.context; kwarg...
    )

    acc = alloc(nc, cell_unit, nc)
    hf_cells = alloc(nhf, cell_unit, nhf)
    density = alloc(n_tot, cell_unit, n_tot)

    Conservation{typeof(acc_type)}(
        acc, accumulation_symbol, hf_cells, density, flow_discretization
    )
end


function acc_symbol(::ChargeAcc)
    return :ChargeAcc
end

function acc_symbol(::MassAcc)
    return :MassAcc
end

function acc_symbol(::EnergyAcc)
    return :EnergyAcc
end
