using Terv
export ElectroChemicalComponent, CurrentCollector
export vonNeumannBC, DirichletBC, BoundaryCondition, MinimalECTPFAGrid
export ChargeFlow, MixedFlow

###########
# Classes #
###########

abstract type ElectroChemicalComponent <: TervSystem end
struct CurrentCollector <: ElectroChemicalComponent end
struct ECComponent <: ElectroChemicalComponent end # Not a good name

abstract type ElectroChemicalGrid <: TervGrid end
struct Phi <: ScalarVariable end
struct C <: ScalarVariable end
struct TotalCharge <: GroupedVariables end # should be scalar
struct TotalConcentration <: ScalarVariable end
struct TPFlux{T} <: GroupedVariables end

abstract type ECFlow <: FlowType end
struct ChargeFlow <: ECFlow end
struct MixedFlow <: ECFlow end


# Todo: allow for several variables in BC's
abstract type BoundaryCondition <: TervForce end
struct vonNeumannBC <: BoundaryCondition 
    cells
    values
end

struct DirichletBC <: BoundaryCondition 
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


function acc_symbol(p::Phi)
    return :TotalCharge
end

function acc_symbol(p::C)
    return :TotalConcentration
end


# ? There has to be a way to not copy these
function Conservation(
    pvar, model, number_of_equations;
    flow_discretization = nothing, kwarg...
    )
    """
    A conservation law corresponding to the underlying potential pvar
    """
    if isnothing(flow_discretization)
        flow_discretization = model.domain.discretizations[1]
    end
    accumulation_symbol = acc_symbol(pvar)


    # Todo: This is copy-pasted, what is necessary??
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

    Conservation{typeof(pvar)}(
        acc, accumulation_symbol, hf_cells, hf_faces, flow_discretization
    )
end

