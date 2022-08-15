export JutulSystem, JutulDomain, JutulVariables, AbstractJutulMesh, JutulContext
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

"Ask discretization for entry i for specific entity"
(D::JutulDiscretization)(i, entity = Cells()) = nothing

discretization(eq) = eq.discretization

function local_discretization(eq, i)
    return discretization(eq)(i, associated_entity(eq))
end

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

function Base.adjoint(ctx::T) where T<: JutulContext
    return T(matrix_layout = adjoint(ctx.matrix_layout))
end

function Base.adjoint(layout::T) where T <: JutulMatrixLayout
    return T(true)
end

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
struct DiscretizedDomain{G, D, E, M} <: JutulDomain
    grid::G
    discretizations::D
    entities::E
    global_map::M
end
function Base.show(io::IO, d::DiscretizedDomain)
    disc = d.discretizations
    if isnothing(disc)
        print(io, "DiscretizedDomain with $(d.grid)\n")
    else
        print(io, "DiscretizedDomain with $(d.grid) and discretizations for $(join(keys(d.discretizations), ", "))\n")
    end
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
    parameters::OrderedDict{Symbol, JutulVariables}
    equations::OrderedDict{Symbol, JutulEquation}
    output_variables::Vector{Symbol}
end

function SimulationModel(domain, system;
                            formulation=FullyImplicit(),
                            context=DefaultContext(),
                            output_level=:primary_variables
                        )
    context = initialize_context!(context, domain, system, formulation)
    domain = transfer(context, domain)

    T = OrderedDict{Symbol,JutulVariables}
    primary = T()
    secondary = T()
    parameters = T()
    equations = OrderedDict{Symbol,JutulEquation}()
    outputs = Vector{Symbol}()
    D = typeof(domain)
    S = typeof(system)
    F = typeof(formulation)
    C = typeof(context)
    model = SimulationModel{D,S,F,C}(domain, system, context, formulation, primary, secondary, parameters, equations, outputs)
    select_primary_variables!(model)
    select_secondary_variables!(model)
    select_parameters!(model)
    select_equations!(model)
    function check_prim(pvar)
        a = map(associated_entity, values(pvar))
        for u in unique(a)
            ut = typeof(u)
            deltas = diff(findall(typeof.(a) .== ut))
            if any(deltas .!= 1)
                error("All primary variables of the same type must come sequentially: Error ocurred for $ut:\nPrimary: $pvar\nTypes: $a")
            end
        end
    end
    check_prim(primary)
    select_output_variables!(model, output_level)
    return model
end

function Base.copy(m::SimulationModel{O, S, C, F}) where {O, S, C, F}
    pvar = copy(m.primary_variables)
    svar = copy(m.secondary_variables)
    outputs = copy(m.output_variables)
    prm = copy(m.parameters)
    eqs = m.equations
    return SimulationModel{O, S, C, F}(m.domain, m.system, m.context, m.formulation, pvar, svar, prm, eqs, outputs)
end

function Base.show(io::IO, t::MIME"text/plain", model::SimulationModel)
    println("SimulationModel:")
    for f in fieldnames(typeof(model))
        p = getfield(model, f)
        print(io, "  $f:\n")
        if f == :primary_variables || f == :secondary_variables || f == :parameters
            ctr = 1
            if length(keys(p)) == 0
                print(io, "   None.\n")
            else
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
            end
            print(io, "\n")
        elseif f == :domain
            print(io, "    ")
            if !isnothing(p)
                print(io, p)
            end
            print(io, "\n")
        elseif f == :equations
            ctr = 1
            for (key, eq) in p
                println(io, "   $ctr) $key")#implemented as $(eq[2]) × $(eq[1])")
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
export JutulEntity, Cells, Faces, Nodes
## Grid
abstract type AbstractJutulMesh end

## Discretized entities
abstract type JutulEntity end

struct Cells <: JutulEntity end
struct Faces <: JutulEntity end
struct Nodes <: JutulEntity end

# Sim model

function SimulationModel(g::AbstractJutulMesh, system; discretization = nothing, kwarg...)
    # Simple constructor that assumes
    d = DiscretizedDomain(g, discretization)
    SimulationModel(d, system; kwarg...)
end

"""
A set of constants, repeated over the entire set of Cells or some other entity
"""
struct ConstantVariables <: GroupedVariables
    constants
    entity::JutulEntity
    single_entity::Bool
    function ConstantVariables(constants, entity = Cells(); single_entity = nothing)
        error("Disabled, use parameters instead")
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
as_value(c::ConstantWrapper) = c

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


"""
An AutoDiffCache is a type that holds both a set of AD values and a map into some
global Jacobian.
"""
abstract type JutulAutoDiffCache end
"""
Cache that holds an AD vector/matrix together with their positions.
"""
struct CompactAutoDiffCache{I, ∂x, E, P} <: JutulAutoDiffCache where {I <: Integer, ∂x <: Real}
    entries::E
    entity
    jacobian_positions::P
    equations_per_entity::I
    number_of_entities::I
    npartials::I
    function CompactAutoDiffCache(equations_per_entity, n_entities, npartials_or_model = 1; 
                                                        entity = Cells(),
                                                        context = DefaultContext(),
                                                        tag = nothing,
                                                        n_entities_pos = nothing,
                                                        kwarg...)
        if isa(npartials_or_model, JutulModel)
            model = npartials_or_model
            npartials = degrees_of_freedom_per_entity(model, entity)
        else
            npartials = npartials_or_model
        end
        npartials::Integer

        I = index_type(context)
        # Storage for AD variables
        t = get_entity_tag(tag, entity)
        entries = allocate_array_ad(equations_per_entity, n_entities, context = context, npartials = npartials, tag = t; kwarg...)
        D = eltype(entries)
        # Position in sparse matrix - only allocated, then filled in later.
        # Since partials are all fetched together with the value, we make partials the fastest index.
        if isnothing(n_entities_pos)
            # This can be overriden - if a custom assembly is planned.
            n_entities_pos = n_entities
        end
        I_t = nzval_index_type(context)
        pos = Array{I_t, 2}(undef, equations_per_entity*npartials, n_entities_pos)
        pos = transfer(context, pos)
        new{I, D, typeof(entries), typeof(pos)}(entries, entity, pos, equations_per_entity, n_entities, npartials)
    end
end

struct GenericAutoDiffCache{N, E, ∂x, A, P, M, D} <: JutulAutoDiffCache where {∂x <: Real}
    # N - number of equations per entity
    entries::A
    vpos::P               # Variable positions (CSR-like, length N + 1 for N entities)
    variables::P          # Indirection-mapped variable list of same length as entries
    jacobian_positions::M
    diagonal_positions::D
    number_of_entities_target::Integer
    number_of_entities_source::Integer
    function GenericAutoDiffCache(T, nvalues_per_entity::I, entity::JutulEntity, sparsity::Vector{Vector{I}}, nt, ns; has_diagonal = true) where I
        @assert nt > 0
        @assert ns > 0
        counts = map(length, sparsity)
        # Create value storage with AD type
        num_entities_touched = sum(counts)
        v = zeros(T, nvalues_per_entity, num_entities_touched)
        A = typeof(v)
        # Create various index mappings + alignment from sparsity
        variables = vcat(sparsity...)
        pos = cumsum(vcat(1, counts))
        algn = zeros(I, nvalues_per_entity*number_of_partials(T), num_entities_touched)
        if has_diagonal
            # Create indices into the self-diagonal part if requested, asserting that the diagonal is present
            m = length(sparsity)
            diag_ix = zeros(I, m)    
            for i = 1:m
                found = false
                for j = pos[i]:(pos[i+1]-1)
                    if variables[j] == i
                        diag_ix[i] = j
                        found = true
                    end
                end
                @assert found "Diagonal must be present in sparsity pattern. Entry $i/$m was missing the diagonal."
            end
        else
            diag_ix = nothing
        end
        P = typeof(pos)
        variables::P
        return new{nvalues_per_entity, entity, T, A, P, typeof(algn), typeof(diag_ix)}(v, pos, variables, algn, diag_ix, nt, ns)
    end
end
