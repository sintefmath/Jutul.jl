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
    boundary_T_hf::AbstractArray{R}
    P::AbstractArray{R} # Tensor to map from cells to faces
    S::AbstractArray{R} # Tensor map cell vector to cell scalar
    cellcellvectbl
    cellcelltbl
end


function get_cellcellvec_map(neigh)
    """ Creates cellcellvectbl """
    dim = 2
    # Must have 2 copies of each neighbourship
    # This took some time to figure out 😰
    neigh = [
        [neigh[1, i] neigh[2, i]; neigh[2, i] neigh[1, i]] 
        for i in 1:size(neigh, 2)
        ]
    neigh = reduce(vcat, neigh)'
    cell1 = [repeat([i], dim) for i in neigh[1, :]]
    cell2 = [repeat([i], dim) for i in neigh[2, :]]
    num_neig = size(cell1)[1]
    vec = [1:dim for i in 1:num_neig]
    cell, cell_dep, vec = map(x -> reduce(vcat, x), [cell1, cell2, vec])
    # must add self dependence
    for i in 1:maximum(cell) #! Probably not the best way to find nc
        push!(cell, i); push!(cell, i)
        push!(cell_dep, i); push!(cell_dep, i)
        push!(vec, 1); push!(vec, 2)
    end
    # ? Should these be sorted in som way ?

    tbl = [
        (cell = cell[i], cell_dep = cell_dep[i], vec = vec[i]) 
            for i in 1:size(cell, 1)
        ]
    return tbl
end

function get_cellcell_map(neigh)
    """ Creates cellcelltbl """
    neigh = [
        [neigh[1, i] neigh[2, i]; neigh[2, i] neigh[1, i]] 
        for i in 1:size(neigh, 2)
        ]
    neigh = reduce(vcat, neigh)'
    cell1 = neigh[1, :]
    cell2 = neigh[2, :]
    num_neig = size(cell1)[1]
    cell, cell_dep = map(x -> reduce(vcat, x), [cell1, cell2])
    for i in 1:maximum(cell) #! Probably not the best way to find nc
        push!(cell, i);
        push!(cell_dep, i);
    end
    tbl = [
        (cell = cell[i], cell_dep = cell_dep[i])
            for i in 1:size(cell, 1)
        ]
    return tbl
end


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
    
    cellcellvectbl = get_cellcellvec_map(N)
    cellcelltbl  = get_cellcell_map(N)

    MinimalECTPFAGrid{eltype(pv), eltype(N)}(
        pv, N, bc, T_hf, P, S, cellcellvectbl, cellcelltbl
        )
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

