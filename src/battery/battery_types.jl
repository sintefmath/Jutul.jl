using Terv

export ElectroChemicalComponent, CurrentCollector, Electectrolyte, TestElyte
export vonNeumannBC, DirichletBC, BoundaryCondition, MinimalECTPFAGrid
export ChargeFlow, MixedFlow, Conservation, BoundaryPotential, BoundaryCurrent
export Phi, C, T, ChargeAcc, MassAcc, EnergyAcc, KGrad
export BOUNDARY_CURRENT, corr_type
###########
# Classes #
###########

abstract type ElectroChemicalComponent <: TervSystem end

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
    half_face_flux_faces::Union{TervAutoDiffCache,Nothing}
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


struct MinimalECTPFAGrid{R<:AbstractFloat, I<:Integer} <: ElectroChemicalGrid
    """
    Simple grid for a electro chemical component
    """
    volumes::AbstractVector{R}
    neighborship::AbstractArray{I}
    boundary_cells::AbstractArray{I}
    boundary_T_hf::AbstractArray{I}

    function MinimalECTPFAGrid(pv, N, bc=[], T_hf=[])
        nc = length(pv)
        pv::AbstractVector
        @assert size(N, 1) == 2
        if length(N) > 0
            @assert minimum(N) > 0
            @assert maximum(N) <= nc
        end
        @assert all(pv .> 0)
        @assert size(bc) == size(T_hf)
        new{eltype(pv), eltype(N)}(pv, N, bc, T_hf)
    end
end


################
# Constructors #
################


function acc_symbol(::ChargeAcc)
    return :ChargeAcc
end

function acc_symbol(::MassAcc)
    return :MassAcc
end

function acc_symbol(::EnergyAcc)
    return :EnergyAcc
end

function Conservation(
    acc_type, model, number_of_equations;
    flow_discretization = nothing, kwarg...
    )
    """
    A conservation law corresponding to the underlying potential pvar
    """
    if isnothing(flow_discretization)
        flow_discretization = model.domain.discretizations[1]
    end
    accumulation_symbol = acc_symbol(acc_type)

    D = model.domain
    cell_unit = Cells()
    face_unit = Faces()
    nc = count_units(D, cell_unit)
    nf = count_units(D, face_unit)
    nhf = 2 * nf
    face_partials = degrees_of_freedom_per_unit(model, face_unit)

    alloc = (n, unit, n_units_pos) -> CompactAutoDiffCache(
    number_of_equations, n, model, unit = unit, n_units_pos = n_units_pos,
    context = model.context; kwarg...
    )

    acc = alloc(nc, cell_unit, nc)
    hf_cells = alloc(nhf, cell_unit, nhf)

    if face_partials > 0
        hf_faces = alloc(nf, face_unit, nhf)
    else
        hf_faces = nothing
    end

    Conservation{typeof(acc_type)}(
        acc, accumulation_symbol, hf_cells, hf_faces, flow_discretization
    )
end

