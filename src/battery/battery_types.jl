using Terv
export ElectroChemicalComponent, CurrentCollector, Electectrolyte, TestElyte
export vonNeumannBC, DirichletBC, BoundaryCondition, MinimalECTPFAGrid
export ChargeFlow, MixedFlow
export Phi, C, T, ChargeAcc, MassAcc, EnergyAcc

###########
# Classes #
###########

abstract type ElectroChemicalComponent <: TervSystem end
struct CurrentCollector <: ElectroChemicalComponent end
struct ECComponent <: ElectroChemicalComponent end # Not a good name

abstract type ElectroChemicalGrid <: TervGrid end

# Potentials
# ? introduce abstract type Potential?
struct Phi <: ScalarVariable end
struct C <: ScalarVariable end
struct T <: ScalarVariable end
# Accumulation variables

abstract type AccumulationVariable <: ScalarVariable end
struct ChargeAcc <: AccumulationVariable end
struct MassAcc <: AccumulationVariable end
struct EnergyAcc <: AccumulationVariable end

# ? Should this be named TPGrad{} instead ???
struct TPFlux{T} <: ScalarVariable end

abstract type ECFlow <: FlowType end
struct ChargeFlow <: ECFlow end
struct MixedFlow <: ECFlow end


abstract type BoundaryCondition{T} <: TervForce end
struct vonNeumannBC{T} <: BoundaryCondition{T} 
    cells
    values
end

struct DirichletBC{T} <: BoundaryCondition{T}
    cells
    values
    half_face_Ts
end

struct Conservation{T} <: TervEquation 
    accumulation::TervAutoDiffCache
    accumulation_symbol::Symbol
    half_face_flux_cells::TervAutoDiffCache
    half_face_flux_faces::Union{TervAutoDiffCache,Nothing}
    flow_discretization::FlowDiscretization
end


struct MinimalECTPFAGrid{R<:AbstractFloat, I<:Integer} <: ElectroChemicalGrid
    volumes::AbstractVector{R}
    neighborship::AbstractArray{I}
    function MinimalECTPFAGrid(pv, N)
        nc = length(pv)
        pv::AbstractVector
        @assert size(N, 1) == 2
        if length(N) > 0
            @assert minimum(N) > 0
            @assert maximum(N) <= nc
        end
        @assert all(pv .> 0)
        new{eltype(pv), eltype(N)}(pv, N)
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

