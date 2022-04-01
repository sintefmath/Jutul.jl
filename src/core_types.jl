export JutulSystem, JutulDomain, JutulVariables, JutulGrid, JutulContext
export SimulationModel, JutulVariables, JutulFormulation, JutulEquation
export setup_parameters, JutulForce
export Cells, Nodes, Faces
export ConstantVariables, ScalarVariable, GroupedVariables

export SingleCUDAContext, SharedMemoryContext, DefaultContext
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

# CUDA context - everything on the single CUDA device attached to machine
struct SingleCUDAContext <: GPUJutulContext
    float_t::Type
    index_t::Type
    block_size
    device
    matrix_layout
    function SingleCUDAContext(float_t::Type = Float32, index_t::Type = Int64, block_size = 256, layout = EquationMajorLayout())
        @assert CUDA.functional() "CUDA must be functional for this context."
        return new(float_t, index_t, block_size, CUDADevice(), layout)
    end
end
matrix_layout(c::SingleCUDAContext) = c.matrix_layout

"Context that uses KernelAbstractions for GPU parallelization"
struct SharedMemoryKernelContext <: CPUJutulContext
    block_size
    device
    function SharedMemoryKernelContext(block_size = Threads.nthreads())
        # Remark: No idea what block_size means here.
        return new(block_size, CPU())
    end
end

"Context that uses threads etc to accelerate loops"
struct SharedMemoryContext <: CPUJutulContext

end

thread_batch(::Any) = 1000

"Default context"
struct DefaultContext <: CPUJutulContext
    matrix_layout
    minbatch::Int64
    function DefaultContext(; matrix_layout = EquationMajorLayout(), minbatch = 1000)
        new(matrix_layout, minbatch)
    end
end

thread_batch(c::DefaultContext) = c.minbatch

matrix_layout(c::DefaultContext) = c.matrix_layout

function jacobian_eltype(context, layout, block_size)
    return float_type(context)
end

function r_eltype(context, layout, block_size)
    return float_type(context)
end

function jacobian_eltype(context::CPUJutulContext, layout::BlockMajorLayout, block_size)
    return SMatrix{block_size..., float_type(context), prod(block_size)}
end

function r_eltype(context::CPUJutulContext, layout::BlockMajorLayout, block_size)
    return SVector{block_size[1], float_type(context)}
end


# Domains
abstract type JutulDomain end

struct DiscretizedDomain{G} <: JutulDomain
    grid::G
    discretizations
    entities
    global_map
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
    primary_variables
    secondary_variables
    equations
    output_variables
    function SimulationModel(domain, system;
                                            formulation = FullyImplicit(),
                                            context = DefaultContext(),
                                            output_level = :primary_variables
                                            )
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
        print("  $f:\n")
        if f == :primary_variables || f == :secondary_variables
            ctr = 1
            for (key, pvar) in p
                nv = degrees_of_freedom_per_entity(model, pvar)
                nu = number_of_entities(model, pvar)
                u = associated_entity(pvar)
                print("   $ctr) $key (")
                if nv > 1
                    print("$nv×")
                end
                print("$nu")

                print(" ∈ $(typeof(u)))\n")
                ctr += 1
            end
            print("\n")
        elseif f == :domain
            if hasproperty(p, :grid)
                g = p.grid
                print("    grid: $(typeof(g))")
            else

            end
            print("\n\n")
        elseif f == :equations
            ctr = 1
            for (key, eq) in p
                println("   $ctr) $key implemented as $(eq[2]) × $(eq[1])")
                ctr += 1
            end
            print("\n")
        else
            println("    $p\n")
        end
    end
end

# Grids etc

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


struct JutulStorage
    data::Union{Dict{Symbol, Any}, NamedTuple}
    function JutulStorage(S = Dict{Symbol, Any}())
        new(S)
    end
end

function convert_to_immutable_storage(S::JutulStorage)
    tup = convert_to_immutable_storage(data(S))
    return JutulStorage(tup)
end

function Base.getproperty(S::JutulStorage, name::Symbol)
    Base.getproperty(data(S), name)
end

data(S::JutulStorage) = getfield(S, :data)

function Base.setproperty!(S::JutulStorage, name::Symbol, x)
    Base.setproperty!(data(S), name, x)
end

function Base.setindex!(S::JutulStorage, x, name::Symbol)
    setindex!(data(S), x, name)
end

function Base.getindex(S::JutulStorage, name::Symbol)
    getindex(data(S), name)
end

function Base.haskey(S::JutulStorage, name::Symbol)
    return Base.haskey(data(S), name)
end

function Base.keys(S::JutulStorage)
    return Base.keys(data(S))
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

function Base.show(io::IO, t::JutulStorage, storage::JutulStorage)
    data = storage.data
    if isa(data, AbstractDict)
        println("JutulStorage (mutable) with fields:")
    else
        println("JutulStorage (immutable) with fields:")
    end
    for key in keys(data)
        println("  $key: $(typeof(data[key]))")
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
    function FiniteVolumeGlobalMap(cells, faces, is_boundary = nothing)
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
        new{eltype(cells)}(cells, inner_to_full_cells, full_to_inner_cells, faces, is_boundary)
    end
end
