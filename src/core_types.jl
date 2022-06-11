export JutulSystem, JutulDomain, JutulVariables, JutulGrid, JutulContext
export SimulationModel, JutulVariables, JutulFormulation, JutulEquation
export setup_parameters, JutulForce
export Cells, Nodes, Faces, declare_entities
export ConstantVariables, ScalarVariable, GroupedVariables, FractionVariables

export SingleCUDAContext, DefaultContext
export BlockMajorLayout, EquationMajorLayout, UnitMajorLayout

export transfer, allocate_array

export JutulStorage

import Base: show, size, setindex!, getindex, ndims

# Physical system
abstract type JutulSystem end

# Discretization - currently unused
abstract type JutulDiscretization end
# struct DefaultDiscretization <: JutulDiscretization end

# Primary/secondary variables
abstract type JutulVariables end
abstract type ScalarVariable <: JutulVariables end
abstract type GroupedVariables <: JutulVariables end
abstract type FractionVariables <: GroupedVariables end

# Driving forces
abstract type JutulForce end

# Context
abstract type JutulContext end
abstract type GPUJutulContext <: JutulContext end
abstract type CPUJutulContext <: JutulContext end

# Traits etc for matrix ordering
abstract type JutulMatrixLayout end
"""
Equations are stored sequentially in rows, derivatives of same type in columns:
"""
struct EquationMajorLayout <: JutulMatrixLayout
    as_adjoint::Bool
end
EquationMajorLayout() = EquationMajorLayout(false)
is_cell_major(::EquationMajorLayout) = false
"""
Domain entities sequentially in rows:
"""
struct UnitMajorLayout <: JutulMatrixLayout
    as_adjoint::Bool
end
UnitMajorLayout() = UnitMajorLayout(false)
is_cell_major(::UnitMajorLayout) = true

"""
Same as UnitMajorLayout, but the nzval is a matrix
"""
struct BlockMajorLayout <: JutulMatrixLayout
    as_adjoint::Bool
end
BlockMajorLayout() = BlockMajorLayout(false)
is_cell_major(::BlockMajorLayout) = true

matrix_layout(::Nothing) = EquationMajorLayout(false)
represented_as_adjoint(layout) = layout.as_adjoint

struct SparsePattern{L}
    I
    J
    n
    m
    block_n
    block_m
    layout::L
    function SparsePattern(I, J, n::T, m::T, layout, block_n::T = 1, block_m::T = block_n) where {T <: Integer}
        if isa(I, Integer)
            @assert isa(J, Integer)
            I = vec(I)
            J = vec(J)
        end
        if length(I) > 0
            I::AbstractVector{T}
            J::AbstractVector{T}
            @assert length(I) == length(J)
            @assert maximum(I) <= n "Maximum row index $(maximum(I)) exceeded $n"
            @assert minimum(I) > 0  "Minimum row index $(minimum(I)) was below 1"
            @assert maximum(J) <= m "Maximum column index $(maximum(J)) exceeded $m"
            @assert minimum(J) > 0  "Minimum column index $(minimum(J)) was below 1"
        else
            # Empty vectors might be Any, and the asserts above
            # cannot be used.
            I = Vector{T}()
            J = Vector{T}()
        end
        @assert n > 0
        @assert m > 0
        new{typeof(layout)}(I, J, n, m, block_n, block_m, layout)
    end
end

ijnm(p::SparsePattern) = (p.I, p.J, p.n, p.m)
block_size(p::SparsePattern) = (p.block_n, p.block_m)

abstract type JutulPartitioner end

include("contexts/interface.jl")
include("contexts/csr.jl")
include("contexts/default.jl")
include("contexts/cuda.jl")

# Domains
abstract type JutulDomain end

export DiscretizedDomain
struct DiscretizedDomain{G} <: JutulDomain
    grid::G
    discretizations
    entities
    global_map
end
function Base.show(io::IO, d::DiscretizedDomain)
    print(io, "DiscretizedDomain with $(d.grid) and discretizations for $(join(keys(d.discretizations), ", "))\n")
end

function DiscretizedDomain(grid, disc = nothing; global_map = TrivialGlobalMap())
    entities = declare_entities(grid)
    u = Dict{Any, Int64}() # Is this a good definition?
    for entity in entities
        num = entity.count
        @assert num >= 0 "Units must have non-negative counts."
        u[entity.entity] = num
    end
    DiscretizedDomain(grid, disc, u, global_map)
end

function transfer(context::SingleCUDAContext, domain::DiscretizedDomain)
    F = context.float_t
    I = context.index_t
    t = (x) -> transfer(context, x)

    g = t(domain.grid)
    d_cpu = domain.discretizations

    k = keys(d_cpu)
    val = map(t, values(d_cpu))
    d = (;zip(k, val)...)
    u = domain.entities
    return DiscretizedDomain(g, d, u, domain.global_map)
end


# Formulation
abstract type JutulFormulation end
struct FullyImplicit <: JutulFormulation end

# Equations
abstract type JutulEquation end
abstract type DiagonalEquation <: JutulEquation end

# Models
export JutulModel, FullyImplicit, SimulationModel, JutulEquation, JutulFormulation

abstract type JutulModel end
abstract type AbstractSimulationModel <: JutulModel end

struct SimulationModel{O<:JutulDomain,
                       S<:JutulSystem,
                       F<:JutulFormulation,
                       C<:JutulContext} <: AbstractSimulationModel
    domain::O
    system::S
    context::C
    formulation::F
    primary_variables::OrderedDict{Symbol, JutulVariables}
    secondary_variables::OrderedDict{Symbol, JutulVariables}
    equations::OrderedDict{Symbol, Tuple{DataType, Int64}}
    output_variables::Vector{Symbol}
    function SimulationModel(domain, system;
                                            formulation = FullyImplicit(),
                                            context = DefaultContext(),
                                            output_level = :primary_variables
                                            )
        context = initialize_context!(context, domain, system, formulation)
        domain = transfer(context, domain)
        primary = select_primary_variables(domain, system, formulation)
        primary = transfer(context, primary)
        function check_prim(pvar)
            a = map(associated_entity, values(pvar))
            for u in unique(a)
                ut = typeof(u)
                deltas =  diff(findall(typeof.(a) .== ut))
                if any(deltas .!= 1)
                    error("All primary variables of the same type must come sequentially: Error ocurred for $ut:\nPrimary: $pvar\nTypes: $a")
                end
            end
        end
        check_prim(primary)
        secondary = select_secondary_variables(domain, system, formulation)
        secondary = transfer(context, secondary)

        equations = select_equations(domain, system, formulation)
        outputs = select_output_variables(domain, system, formulation, primary, secondary, output_level)

        D = typeof(domain)
        S = typeof(system)
        F = typeof(formulation)
        C = typeof(context)
        new{D, S, F, C}(domain, system, context, formulation, primary, secondary, equations, outputs)
    end
end

function Base.show(io::IO, t::MIME"text/plain", model::SimulationModel)
    println("SimulationModel:")
    for f in fieldnames(typeof(model))
        p = getfield(model, f)
        print(io, "  $f:\n")
        if f == :primary_variables || f == :secondary_variables
            ctr = 1
            for (key, pvar) in p
                nv = degrees_of_freedom_per_entity(model, pvar)
                nu = number_of_entities(model, pvar)
                u = associated_entity(pvar)
                if f == :secondary_variables
                    print(io, "   $ctr $key ← $(typeof(pvar))) (")
                else
                    print(io, "   $ctr) $key (")
                end
                if nv > 1
                    print(io, "$nv×")
                end
                print(io, "$nu")

                print(io, " ∈ $(typeof(u)))\n")
                ctr += 1
            end
            print(io, "\n")
        elseif f == :domain
            print(io, "    ")
            print(io, p)
            print(io, "\n")
        elseif f == :equations
            ctr = 1
            for (key, eq) in p
                println(io, "   $ctr) $key implemented as $(eq[2]) × $(eq[1])")
                ctr += 1
            end
            print(io, "\n")
        elseif f == :output_variables
            print(io, "    $(join(p, ", "))")
        else
            print(io, "    ")
            print(io, p)
            print(io, "\n\n")
        end
    end
end

# Grids etc
export JutulUnit, Cells, Faces, Nodes
## Grid
abstract type JutulGrid end

## Discretized entities
abstract type JutulUnit end

struct Cells <: JutulUnit end
struct Faces <: JutulUnit end
struct Nodes <: JutulUnit end

# Sim model

function SimulationModel(g::JutulGrid, system; discretization = nothing, kwarg...)
    # Simple constructor that assumes
    d = DiscretizedDomain(g, discretization)
    SimulationModel(d, system; kwarg...)
end

"""
A set of constants, repeated over the entire set of Cells or some other entity
"""
struct ConstantVariables <: GroupedVariables
    constants
    entity::JutulUnit
    single_entity::Bool
    function ConstantVariables(constants, entity = Cells(); single_entity = nothing)
        if !isa(constants, AbstractArray)
            @assert length(constants) == 1
            constants = [constants]
        end
        if isnothing(single_entity)
            # Single entity means that we have one (or more) values that are given for all entities
            # by a single representative entity
            single_entity = isa(constants, AbstractVector)
        end
        if isa(constants, CuArray) && single_entity
            @warn "Single entity constants have led to crashes on CUDA/Tullio kernels!" maxlog = 5
        end
        new(constants, entity, single_entity)
    end
end

import Base: getindex, @propagate_inbounds, parent, size, axes

struct ConstantWrapper{R}
    data::Vector{R}
    nrows::Integer
    function ConstantWrapper(data, n)
        new{eltype(data)}(data, n)
    end
end
Base.eltype(c::ConstantWrapper{T}) where T = T
Base.length(c::ConstantWrapper) = length(c.data)
Base.size(c::ConstantWrapper) = (length(c.data), c.nrows)
Base.size(c::ConstantWrapper, i) = i == 1 ? length(c.data) : c.nrows
Base.@propagate_inbounds Base.getindex(c::ConstantWrapper{R}, i, j) where R = c.data[i]::R
Base.@propagate_inbounds Base.getindex(c::ConstantWrapper{R}, i) where R = c.data[1]::R
Base.setindex!(c::ConstantWrapper, arg...) = setindex!(c.data, arg...)
Base.ndims(c::ConstantWrapper) = 2
Base.view(c::ConstantWrapper, ::Colon, i) = c.data

function Base.axes(c::ConstantWrapper, d)
    if d == 1
        return Base.OneTo(length(c.data))
    else
        return Base.OneTo(c.nrows)
    end
end


struct JutulStorage{K}
    data::Union{Dict{Symbol, <:Any}, NamedTuple}
    function JutulStorage(S = Dict{Symbol, Any}())
        if isa(S, Dict)
            K = Nothing
        else
            @assert isa(S, NamedTuple)
            K = keys(S)
            K::Tuple
        end
        return new{K}(S)
    end
end


function convert_to_immutable_storage(S::JutulStorage)
    tup = convert_to_immutable_storage(data(S))
    return JutulStorage(tup)
end

function Base.getproperty(S::JutulStorage, name::Symbol)
    Base.getproperty(data(S), name)
end

data(S::JutulStorage{Nothing}) = getfield(S, :data)
data(S::JutulStorage) = getfield(S, :data)::NamedTuple

function Base.setproperty!(S::JutulStorage, name::Symbol, x)
    Base.setproperty!(data(S), name, x)
end

function Base.setindex!(S::JutulStorage, x, name::Symbol)
    Base.setindex!(data(S), x, name)
end

function Base.getindex(S::JutulStorage, name::Symbol)
    Base.getindex(data(S), name)
end

function Base.haskey(S::JutulStorage{Nothing}, name::Symbol)
    return Base.haskey(data(S), name)
end

function Base.keys(S::JutulStorage{Nothing})
    return Base.keys(data(S))
end


function Base.haskey(S::JutulStorage{K}, name::Symbol) where K
    return name in K
end

function Base.keys(S::JutulStorage{K}) where K
    return K
end

function Base.show(io::IO, t::MIME"text/plain", storage::JutulStorage)
    D = data(storage)
    if isa(D, AbstractDict)
        println("JutulStorage (mutable) with fields:")
    else
        println("JutulStorage (immutable) with fields:")
    end
    for key in keys(D)
        println("  $key: $(typeof(D[key]))")
    end
end

abstract type AbstractGlobalMap end
struct TrivialGlobalMap end

struct FiniteVolumeGlobalMap{T} <: AbstractGlobalMap
    # Full set -> global set
    cells::Vector{T}
    # Inner set -> full set
    inner_to_full_cells::Vector{T}
    # Full set -> inner set
    full_to_inner_cells::Vector{T}
    faces::Vector{T}
    cell_is_boundary::Vector{Bool}
    variables_always_active::Bool
    function FiniteVolumeGlobalMap(cells, faces, is_boundary = nothing; variables_always_active = false)
        n = length(cells)
        if isnothing(is_boundary)
            is_boundary = repeat([false], length(cells))
        end
        @assert length(is_boundary) == length(cells)
        inner_to_full_cells = findall(is_boundary .== false)
        full_to_inner_cells = Vector{Integer}(undef, n)
        for i = 1:n
            v = only(indexin(i, inner_to_full_cells))
            if isnothing(v)
                v = 0
            end
            @assert v >= 0 && v <= n
            full_to_inner_cells[i] = v
        end
        for i in inner_to_full_cells
            @assert i > 0 && i <= n
        end
        new{eltype(cells)}(cells, inner_to_full_cells, full_to_inner_cells, faces, is_boundary, variables_always_active)
    end
end
