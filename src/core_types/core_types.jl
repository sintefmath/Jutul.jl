export JutulSystem, JutulDomain, JutulVariables, JutulMesh, JutulContext
export SimulationModel, JutulVariables, JutulFormulation, JutulEquation
export setup_parameters, JutulForce
export Cells, Nodes, Faces, declare_entities
export ScalarVariable, VectorVariables, FractionVariables

export SingleCUDAContext, DefaultContext
export BlockMajorLayout, EquationMajorLayout, EntityMajorLayout

export transfer, allocate_array

export JutulStorage

import Base: show, size, setindex!, getindex, ndims

"""
Abstract type for the physical system to be solved.
"""
abstract type JutulSystem end

"""
Abstract type for a Jutul discretization
"""
abstract type JutulDiscretization end

"""
    d = disc(i, Cells())

Ask discretization for entry `i` when discretizing some equation on the 
chosen entity (e.g. [`Cells`](@ref))
"""
function (D::JutulDiscretization)(i, entity = Cells())
    return nothing
end

discretization(eq) = eq.discretization

function local_discretization(eq, i)
    return discretization(eq)(i, associated_entity(eq))
end

# struct DefaultDiscretization <: JutulDiscretization end

# Primary/secondary variables
"""
Abstract type for all variables in Jutul.

A variable is associated with a [`JutulEntity`](@ref) through the
[`associated_entity`](@ref) function. A variable is local to that entity, and
cannot depend on other entities. Variables are used by models to define:
- primary variables: Sometimes referred to as degrees of freedom, primary
  unknowns or solution variables
- parameters: Static quantities that impact the solution
- secondary variables: Can be computed from a combination of other primary and
  secondary variables and parameters.
"""
abstract type JutulVariables end

"""
Abstract type for scalar variables (one entry per entity, e.g. pressure or
temperature in each cell of a model)
"""
abstract type ScalarVariable <: JutulVariables end
"""
Abstract type for vector variables (more than one entry per entity, for example
saturations or displacements)
"""
abstract type VectorVariables <: JutulVariables end
"""
Abstract type for fraction variables (vector variables that sum up to unity over
each entity).

By default, these are limited to the [0, 1] range through
[`maximum_value`](@ref) and [`minimum_value`](@ref) default implementations.
"""
abstract type FractionVariables <: VectorVariables end

"""
Abstract type for driving forces
"""
abstract type JutulForce end

"""
Abstract type for the context Jutul should execute in (matrix formats, memory allocation, etc.)
"""
abstract type JutulContext end
abstract type GPUJutulContext <: JutulContext end
abstract type CPUJutulContext <: JutulContext end

# Traits etc for matrix ordering
"""
Abstract type for matrix layouts. A layout determines how primary variables and
equations are ordered in a sparse matrix representation. Note that this is
different from the matrix format itself as it concerns the ordering itself: For
example, if all equations for a single cell come in sequence, or if a single
equation is given for all entities before the next equation is written.

Different layouts does not change the solution of the system, but different
linear solvers support different layouts.
"""
abstract type JutulMatrixLayout end
"""
Equations are stored sequentially in rows, derivatives of same type in columns:

For a test system with primary variables P, S and equations E1, E2 and two cells
this will give the following ordering on the diagonal:
(∂E1/∂p)₁, (∂E1/∂p)₂, (∂E2/∂S)₁, (∂E2/∂S)₂
"""
struct EquationMajorLayout <: JutulMatrixLayout
    as_adjoint::Bool
end
EquationMajorLayout() = EquationMajorLayout(false)
is_cell_major(::EquationMajorLayout) = false

"""
Equations are grouped by entity, listing all equations and derivatives for
entity 1 before proceeding to entity 2 etc.

For a test system with primary variables P, S and equations E1, E2 and two cells
this will give the following ordering on the diagonal:
(∂E1/∂p)₁, (∂E2/∂S)₁, (∂E1/∂p)₂, (∂E2/∂S)₂
"""
struct EntityMajorLayout <: JutulMatrixLayout
    as_adjoint::Bool
end
EntityMajorLayout() = EntityMajorLayout(false)
is_cell_major(::EntityMajorLayout) = true

const ScalarLayout = Union{EquationMajorLayout, EntityMajorLayout}

"""
Same as [`EntityMajorLayout`](@ref), but the system is a sparse matrix where
each entry is a small dense matrix.

For a test system with primary variables P, S and equations E1, E2 and two cells
this will give a diagonal of length 2:
[(∂E1/∂p)₁ (∂E1/∂S)₁ ; (∂E2/∂p)₁ (∂E2/∂S)₁]
[(∂E1/∂p)₂ (∂E1/∂S)₂ ; (∂E2/∂p)₂ (∂E2/∂S)₂]
"""
struct BlockMajorLayout <: JutulMatrixLayout
    as_adjoint::Bool
end
BlockMajorLayout() = BlockMajorLayout(false)
is_cell_major(::BlockMajorLayout) = true

matrix_layout(::Nothing) = EquationMajorLayout(false)
represented_as_adjoint(layout) = layout.as_adjoint

scalarize_layout(layout, other_layout) = layout
function scalarize_layout(layout::BlockMajorLayout, other_layout::ScalarLayout)
    slayout = EntityMajorLayout()
    if represented_as_adjoint(other_layout)
        @assert represented_as_adjoint(layout)
        slayout = adjoint(slayout)
    end
    return slayout
end

function Base.adjoint(ctx::T) where T<: JutulContext
    return T(matrix_layout = adjoint(ctx.matrix_layout))
end

function Base.adjoint(layout::T) where T <: JutulMatrixLayout
    return T(true)
end

struct SparsePattern{L_r, L_c}
    I
    J
    n
    m
    block_n
    block_m
    layout_row::L_r
    layout_col::L_c
    function SparsePattern(I, J, n::T, m::T, layout_row::LR, layout_col::LC, block_n::T = 1, block_m::T = block_n) where {T <: Integer, LR, LC}
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
        if n == 0
            @debug "Pattern has zero rows?" n m I J
        end
        if m == 0
            @debug "Pattern has zero columns?" n m I J
        end
        new{LR, LC}(I, J, n, m, block_n, block_m, layout_row, layout_col)
    end
end

function Base.adjoint(p::SparsePattern)
    # Note: We only permute the outer pattern, not the inner.
    return SparsePattern(p.J, p.I, p.m, p.n, p.layout_row, p.layout_col, p.block_n, p.block_m)
end

ijnm(p::SparsePattern) = (p.I, p.J, p.n, p.m)
block_size(p::SparsePattern) = (p.block_n, p.block_m)

abstract type JutulPartitioner end

include("contexts/interface.jl")
include("contexts/csr.jl")
include("contexts/default.jl")
include("contexts/cuda.jl")

# Domains
include("domains.jl")

# Formulation
abstract type JutulFormulation end
struct FullyImplicitFormulation <: JutulFormulation end

# Equations
"""
Abstract type for all residual equations
"""
abstract type JutulEquation end
abstract type DiagonalEquation <: JutulEquation end

# Models
export JutulModel, FullyImplicitFormulation, SimulationModel, JutulEquation, JutulFormulation

abstract type JutulModel end
abstract type AbstractSimulationModel <: JutulModel end

struct SimulationModel{O<:JutulDomain,
                       S<:JutulSystem,
                       F<:JutulFormulation,
                       C<:JutulContext
                       } <: AbstractSimulationModel
    domain::O
    system::S
    context::C
    formulation::F
    data_domain
    primary_variables::OrderedDict{Symbol, Any}
    secondary_variables::OrderedDict{Symbol, Any}
    parameters::OrderedDict{Symbol, Any}
    equations::OrderedDict{Symbol, Any}
    output_variables::Vector{Symbol}
    extra::OrderedDict{Symbol, Any}
    optimization_level::Int
end

"""
    SimulationModel(domain, system; <kwarg>)

Instantiate a model for a given `system` discretized on the `domain`.
"""
function SimulationModel(domain, system;
                            formulation=FullyImplicitFormulation(),
                            context=DefaultContext(),
                            output_level=:primary_variables,
                            data_domain = missing,
                            extra = OrderedDict{Symbol, Any}(),
                            plot_mesh = missing,
                            primary_variables = missing,
                            secondary_variables = missing,
                            parameters = missing,
                            equations = missing,
                            optimization_level = 1,
                            outputs = Vector{Symbol}(),
                            kwarg...
                        )
    context = initialize_context!(context, domain, system, formulation)
    if ismissing(data_domain)
        if domain isa DataDomain
            data_domain = domain
        else
            data_domain = DataDomain(physical_representation(domain))
        end
    end
    domain = discretize_domain(domain, system; kwarg...)
    domain = transfer(context, domain)
    if !ismissing(plot_mesh)
        error("plot_mesh argument is deprecated.")
    end

    T = OrderedDict{Symbol,JutulVariables}
    need_primary = ismissing(primary_variables)
    if need_primary
        primary_variables = T()
    end
    need_secondary = ismissing(secondary_variables)
    if need_secondary
        secondary_variables = T()
    end
    need_parameters = ismissing(parameters)
    if need_parameters
        parameters = T()
    end
    need_equations = ismissing(equations)
    if need_equations
        equations = OrderedDict{Symbol,JutulEquation}()
    end
    D = typeof(domain)
    S = typeof(system)
    F = typeof(formulation)
    C = typeof(context)
    model = SimulationModel{D,S,F,C}(
        domain,
        system,
        context,
        formulation,
        data_domain,
        primary_variables,
        secondary_variables,
        parameters,
        equations,
        outputs,
        extra,
        optimization_level
    )
    update_model_pre_selection!(model)
    if need_primary
        select_primary_variables!(model)
    end
    if need_secondary
        select_secondary_variables!(model)
    end
    if need_parameters
        select_parameters!(model)
    end
    if need_equations
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
    end
    if need_primary
        check_prim(primary_variables)
    end
    select_output_variables!(model, output_level)
    update_model_post_selection!(model)
    return model
end

function SimulationModel{D,S,F,C}(
        domain,
        system,
        context,
        formulation,
        data_domain,
        primary_variables,
        secondary_variables,
        parameters,
        equations,
        outputs,
        extra
    ) where {D,S,F,C}
    # Backward compatibility constructor
    return SimulationModel{D,S,F,C}(
        domain,
        system,
        context,
        formulation,
        data_domain,
        primary_variables,
        secondary_variables,
        parameters,
        equations,
        outputs,
        extra,
        1
    )
end

"""
    physical_representation(m::SimulationModel)

Get the underlying physical representation for the model (domain or mesh)
"""
physical_representation(m::SimulationModel) = physical_representation(m.domain)
physical_representation(m::JutulModel) = missing

function update_model_pre_selection!(model)
    return model
end

function update_model_post_selection!(model)
    return model
end

import Base: copy
function Base.copy(m::SimulationModel{O, S, C, F}) where {O, S, C, F}
    pvar = copy(m.primary_variables)
    svar = copy(m.secondary_variables)
    outputs = copy(m.output_variables)
    prm = copy(m.parameters)
    eqs = m.equations
    return SimulationModel{O, S, C, F}(m.domain, m.system, m.context, m.formulation, m.plot_mesh, pvar, svar, prm, eqs, outputs)
end

function Base.getindex(model::SimulationModel, s::Symbol)
    return get_variables_and_parameters(model)[s]
end

function Base.show(io::IO, t::MIME"text/plain", @nospecialize(model::SimulationModel))
    println(io, "SimulationModel:\n")
    ne = number_of_equations(model)
    np = number_of_degrees_of_freedom(model)
    nprm = number_of_parameters(model)
    print(io, "  Model with $np degrees of freedom, $ne equations and $nprm parameters\n\n")
    for f in fieldnames(typeof(model))
        p = getfield(model, f)
        print(io, "  $f:\n")
        if f == :primary_variables || f == :secondary_variables || f == :parameters
            ctr = 1
            if length(keys(p)) == 0
                print(io, "   None.\n")
            else
                maxv = 0
                for (key, pvar) in p
                    maxv = max(length(String(key)), maxv)
                end
                for (key, pvar) in p
                    nval = values_per_entity(model, pvar)
                    nu = number_of_entities(model, pvar)
                    u = associated_entity(pvar)
                    N = length(String(key))
                    pad = repeat(" ", maxv - N)
                    print(io, "   $ctr) $key$pad ")
                    # if !isa(pvar, ScalarVariable)#nval > 1 || (nval != nv && f == :primary_variables)
                    print(io, "∈ $nu $(typeof(u)): ")
                    if f == :primary_variables
                        ndof = degrees_of_freedom_per_entity(model, pvar)
                        if ndof != nval
                            print(io, "$ndof dof, $nval values each")
                        else
                            print(io, "$ndof dof each")
                        end
                    else
                        if isa(pvar, ScalarVariable)
                            print(io, "Scalar")
                        else
                            print(io, "$nval values each")
                        end
                    end
                    if f == :secondary_variables
                        print(io, "\n")
                        is_pair = pvar isa Pair
                        if is_pair
                            pl, pvar = pvar
                            pl = " ($pl)"
                        else
                            pl = ""
                        end
                        print_t = Base.typename(typeof(pvar)).wrapper
                        print(io, "      -> $print_t as evaluator$pl")
                    end
                    print(io, "\n")
                    #end
                    # print(io, "$nu")

                    ctr += 1
                end
            end
            print(io, "\n")
        elseif f == :domain || f == :data_domain
            print(io, "    ")
            if !isnothing(p)
                show(io, t, p)
            end
            print(io, "\n")
        elseif f == :equations
            ctr = 1
            for (key, eq) in p

                n = number_of_equations_per_entity(model, eq)
                m = number_of_entities(model, eq)
                e = associated_entity(eq)
                println(io, "   $ctr) $key ∈ $m $(typeof(e)): $n values each\n      -> $(typeof(eq))")
                ctr += 1
            end
            print(io, "\n")
        elseif f == :output_variables
            print(io, "    $(join(p, ", "))\n\n")
        elseif f == :extra
            print(io, "    $(typeof(p)) with keys: $(keys(p))")
            print(io, "\n")
        else
            print(io, "    ")
            print(io, p)
            print(io, "\n\n")
        end
    end
end

# Grids etc
export JutulEntity, Cells, Faces, HalfFaces, BoundaryFaces, Nodes, NoEntity
## Grid
"""
A mesh is a type of domain that has been discretized. Abstract subtype.
"""
abstract type JutulMesh <: JutulDomain end

## Discretized entities
"""
Super-type for all entities where [`JutulVariables`](@ref) can be defined.
"""
abstract type JutulEntity end

"""
Entity for Cells (closed volumes with averaged properties for a finite-volume solver)
"""
struct Cells <: JutulEntity end

"""
Entity for Faces (intersection between pairs of [`Cells`](@ref))
"""
struct Faces <: JutulEntity end

"""
Entity for faces on the boundary (faces that are only connected to a single [`Cells`](@ref))
"""
struct BoundaryFaces <: JutulEntity end

"""
Entity for half-faces (face associated with a single [`Cells`](@ref))
"""
struct HalfFaces <: JutulEntity end

"""
Entity for Nodes (intersection between multiple [`Faces`](@ref))
"""
struct Nodes <: JutulEntity end

"""
An entity for something that isn't associated with an entity
"""
struct NoEntity <: JutulEntity end


# Sim model

"""
    SimulationModel(g::JutulMesh, system; discretization = nothing, kwarg...)

Type that defines a simulation model - everything needed to solve the discrete
equations.

The minimal setup requires a [`JutulMesh`](@ref) that defines topology
together with a [`JutulSystem`](@ref) that imposes physical laws.
"""
function SimulationModel(g::JutulMesh, system; discretization = nothing, kwarg...)
    # Simple constructor that assumes
    d = DiscretizedDomain(g, discretization)
    SimulationModel(d, system; kwarg...)
end


import Base: getindex, @propagate_inbounds, parent, size, axes

struct JutulStorage{K}
    data::Union{JUTUL_OUTPUT_TYPE, K}
    always_mutable::Bool
    function JutulStorage(S = JUTUL_OUTPUT_TYPE(); always_mutable = false, kwarg...)
        if isa(S, AbstractDict)
            K = Nothing
            for (k, v) in kwarg
                S[k] = v
            end
        elseif S isa JutulStorage
            @assert length(kwarg) == 0
            return S
        else
            @assert isa(S, NamedTuple)
            K = typeof(S)
            @assert length(kwarg) == 0
        end
        return new{K}(S, always_mutable)
    end
end

function convert_to_immutable_storage(S::JutulStorage)
    if getfield(S, :always_mutable)
        return S
    end
    tup = convert_to_immutable_storage(data(S))
    return JutulStorage(tup)
end

function convert_to_immutable_storage(S::NamedTuple)
    return S
end

function Base.getindex(S::JutulStorage, i::Int)
    d = data(S)
    if d isa OrderedDict
        for (j, v) in enumerate(values(d))
            if j == i
                return v
            end
        end
        throw(BoundsError("Out of bounds $i for JutulStorage."))
    else
        return d[i]
    end
end
Base.length(S::JutulStorage, arg...) = Base.length(values(S), arg...)
Base.iterate(S::JutulStorage, arg...) = Base.iterate(values(S), arg...)
function Base.map(f, S::JutulStorage)
    d = data(S)
    if d isa OrderedDict
        d = NamedTuple(d)
    end
    return Base.map(f, d)
end
Base.pairs(S::JutulStorage) = Base.pairs(data(S))
Base.values(S::JutulStorage) = Base.values(data(S))

function Base.getproperty(S::JutulStorage{Nothing}, name::Symbol)
    Base.getindex(data(S), name)
end

function Base.getproperty(S::JutulStorage, name::Symbol)
    Base.getproperty(data(S), name)
end

Base.propertynames(S::JutulStorage) = keys(getfield(S, :data))

data(S::JutulStorage{Nothing}) = getfield(S, :data)
data(S::JutulStorage{T}) where T = getfield(S, :data)::T

function Base.setproperty!(S::JutulStorage, name::Symbol, x)
    Base.setproperty!(data(S), name, x)
end

function Base.setindex!(S::JutulStorage, x, name::Symbol)
    Base.setindex!(data(S), x, name)
end

function Base.getindex(S::JutulStorage, name::Symbol)
    Base.getindex(data(S), name)
end

function Base.getindex(S::JutulStorage, name::Pair)
    # This is hacked in for CompositeSystem
    return S[last(name)]
end

function Base.haskey(S::JutulStorage{Nothing}, name::Symbol)
    return Base.haskey(data(S), name)
end

function Base.keys(S::JutulStorage{Nothing})
    return Tuple(keys(data(S)))
end


function Base.haskey(S::JutulStorage{NamedTuple{K, V}}, name::Symbol) where {K, V}
    return name in K
end

function Base.keys(S::JutulStorage{NamedTuple{K, V}}) where {K, V}
    return K
end

function Base.show(io::IO, t::MIME"text/plain", @nospecialize(storage::JutulStorage))
    D = data(storage)
    if isa(D, AbstractDict)
        println(io, "JutulStorage (mutable) with fields:")
    else
        println(io, "JutulStorage (immutable) with fields:")
    end
    for key in keys(D)
        println(io, "  $key: $(typeof(D[key]))")
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
    global_to_local::Dict{Int, Int}
    function FiniteVolumeGlobalMap(cells, faces, is_boundary = nothing; variables_always_active = false)
        n = length(cells)
        # @assert issorted(cells)
        if isnothing(is_boundary)
            is_boundary = repeat([false], length(cells))
        end
        @assert length(is_boundary) == length(cells)
        inner_to_full_cells = findall(is_boundary .== false)
        full_to_inner_cells = Vector{Integer}(undef, n)
        inverse_inner_to_full_cells = Dict{Int, Int}()
        for (i, v) in enumerate(inner_to_full_cells)
            inverse_inner_to_full_cells[v] = i
        end
        for i = 1:n
            v = get(inverse_inner_to_full_cells, i, 0)
            @assert v >= 0 && v <= n
            full_to_inner_cells[i] = v
        end
        for i in inner_to_full_cells
            @assert i > 0 && i <= n
        end
        g2l = Dict{Int, Int}()
        for (i, c) in enumerate(cells)
            g2l[c] = i
        end
        new{eltype(cells)}(cells, inner_to_full_cells, full_to_inner_cells, faces, is_boundary, variables_always_active, g2l)
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

struct GenericAutoDiffCache{N, E, ∂x, A, P, M, D, VM} <: JutulAutoDiffCache where {∂x <: Real}
    # N - number of equations per entity
    entries::A
    vpos::P               # Variable positions (CSR-like, length N + 1 for N entities)
    variables::P          # Indirection-mapped variable list of same length as entries
    jacobian_positions::M
    diagonal_positions::D
    number_of_entities_target::Int
    number_of_entities_source::Int
    variable_map::VM
    function GenericAutoDiffCache(T, nvalues_per_entity::I, entity::JutulEntity, sparsity::Vector{Vector{I}}, nt, ns; has_diagonal = true, global_map = TrivialGlobalMap()) where I
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
        P = typeof(pos)
        variables = convert(P, variables)
        algn = zeros(I, nvalues_per_entity*number_of_partials(T), num_entities_touched)
        if has_diagonal
            # Create indices into the self-diagonal part if requested, asserting that the diagonal is present
            m = length(sparsity)
            diag_ix = zeros(I, m)
            ok = true
            for i = 1:m
                found = false
                for j = pos[i]:(pos[i+1]-1)
                    if variables[j] == i
                        diag_ix[i] = j
                        found = true
                    end
                end
                if !found
                    ok = false
                    @debug "Diagonal must be present in sparsity pattern. Entry $i/$m was missing the diagonal."
                    break
                end
            end
            if !ok
                diag_ix = nothing
            end
        else
            diag_ix = nothing
        end
        variables::P
        M_t = typeof(global_map)
        return new{nvalues_per_entity, entity, T, A, P, typeof(algn), typeof(diag_ix), M_t}(v, pos, variables, algn, diag_ix, nt, ns, global_map)
    end
end

Base.eltype(::GenericAutoDiffCache{N, E, ∂x, A, P, M, D, VM}) where {N, E, ∂x, A, P, M, D, VM} = ∂x

"Discretization of kgradp + upwind"
abstract type FlowDiscretization <: JutulDiscretization end

abstract type FluxType end

struct DefaultFlux <: FluxType end
struct ConservationLaw{C, T<:FlowDiscretization, FT<:FluxType, N} <: JutulEquation
    flow_discretization::T
    flux_type::FT
    function ConservationLaw(disc::T, conserved::Symbol = :TotalMasses, N::Integer = 1; flux = DefaultFlux()) where T
        return new{conserved, T, typeof(flux), N}(disc, flux)
    end
end

export CompositeSystem
struct CompositeSystem{label, T} <: JutulSystem
    systems::T
end

function Base.show(io::IO, t::CompositeSystem)
    print(io, "CompositeSystem:\n")
    for (name, sys) in pairs(t.systems)
        print(io, "($name => $sys)\n")
    end
end
function CompositeSystem(label::Symbol = :composite; kwarg...)
    tup = NamedTuple(pairs(kwarg))
    T = typeof(tup)
    return CompositeSystem{label, T}(tup)
end

const CompositeModel = SimulationModel{<:JutulDomain, <:CompositeSystem, <:JutulFormulation, <:JutulContext}

struct JutulLinePlotData
    xdata
    ydata
    datalabels
    title
    xlabel
    ylabel
end

export line_plot_data
function line_plot_data(model::SimulationModel, ::Any)
    return nothing
end

function JutulLinePlotData(x, y; labels = nothing, title = "", xlabel = "", ylabel = "")
    if eltype(x)<:AbstractFloat
        x = [x]
    end
    if eltype(y)<:AbstractFloat
        y = [y]
    end
    if labels isa String
        labels = [labels]
    end
    if isnothing(labels)
        labels = ["" for i in eachindex(x)]
    end
    @assert length(x) == length(y) == length(labels)

    return JutulLinePlotData(x, y, labels, title, xlabel, ylabel)
end


export JutulLinePlotData

"""
Abstract type for termination criteria for steady-state time-stepping simulations.

Subtypes should implement the method

    timestepping_is_done(C::AbstractTerminationCriterion, simulator, states, substates, reports, solve_recorder)

that returns a Bool indicating whether the time-stepping should terminate.
"""
abstract type AbstractTerminationCriterion end

"""
    timestepping_is_done(C::AbstractTerminationCriterion, simulator, states, substates, reports, solve_recorder)

Check whether time-stepping should terminate according to the criterion `C`.
Subtypes of [`AbstractTerminationCriterion`](@ref) should implement this method.

The arguements are:
- `simulator`: The simulator executing the time-stepping. The simulator holds
  the current state after a converged time step (can be accessed in the standard
  output format by `get_output_state(simulator)`)
- `states`: The current states of the simulation (up to the current report step)
- `substates`: The current substates of the simulation for the current report
  step. May be missing if `output_substates` is false.
- `reports`: The reports so far in the simulation.
- `solve_recorder`: The solve recorder holding information about the current
  time, etc. You can get the current time with `get_current_time(solve_recorder,
  :global)`.
""" 
function timestepping_is_done(C::AbstractTerminationCriterion, simulator, states, substates, reports, solve_recorder)
    error("should_terminate not implemented for criterion of type $(typeof(C))")
end

export JutulCase
struct JutulCase
    model::JutulModel
    dt::AbstractVector{<:AbstractFloat}
    forces
    state0
    parameters
    input_data
    termination_criterion::AbstractTerminationCriterion
    start_date::Union{Nothing, DateTime}
end

"""
    JutulCase(model, dt = [1.0], forces = setup_forces(model); state0 = nothing, parameters = nothing, kwarg...)

Set up a structure that holds the complete specification of a simulation case.
"""
function JutulCase(model::JutulModel, dt = [1.0], forces = setup_forces(model);
        state0 = nothing,
        parameters = nothing,
        input_data = nothing,
        termination_criterion = NoTerminationCriterion(),
        start_date = nothing,
        kwarg...
    )
    for (k, v) in kwarg
        if k == :forces
            error("forces was passed as kwarg. It is a positional argument.")
        end
    end
    if isnothing(state0) && isnothing(parameters)
        state0, parameters = setup_state_and_parameters(model, kwarg...)
    elseif isnothing(state0)
        state0 = setup_state(model, kwarg...)
    elseif isnothing(parameters)
        parameters = setup_parameters(model, kwarg...)
    end
    if dt isa Real
        dt = [dt]
    end
    if forces isa AbstractVector
        nf = length(forces)
        nt = length(dt)
        @assert nt == nf "If forces is a vector, the length (=$nf) must match the number of time steps (=$nt)."
    end
    all(dt .> 0) || throw(ArgumentError("All time-steps must be positive, was $dt"))
    return JutulCase(model, dt, forces, state0, parameters, input_data, termination_criterion, start_date)
end

function JutulCase(model, dt, forces, state0, parameters, input_data)
    # This is a backwards compatibility constructor
    return JutulCase(model, dt, forces; state0 = state0, parameters = parameters, input_data = input_data)
end

function Base.show(io::IO, t::MIME"text/plain", case::JutulCase)
    if case.forces isa AbstractVector
        ctrl_type = "forces for each step"
    else
        ctrl_type = "constant forces for all steps"
    end
    nstep = length(case.dt)
    println(io, "Jutul case with $nstep time-steps ($(get_tstr(sum(case.dt)))) and $ctrl_type.\n\nModel:\n")
    Base.show(io, t, case.model)
end

function duplicate(case::JutulCase; copy_model = false)
    # Make copies of everything but the model
    (; model, dt, forces, state0, parameters, input_data) = case
    if copy_model
        model = deepcopy(model)
    end
    return JutulCase(model, deepcopy(dt), deepcopy(forces), deepcopy(state0), deepcopy(parameters), deepcopy(input_data), case.termination_criterion, case.start_date)
end

function Base.getindex(case::JutulCase, ix::Int; kwarg...)
    return Base.getindex(case, ix:ix; kwarg...)
end

function Base.lastindex(case::JutulCase)
    return length(case.dt)
end

function Base.lastindex(case::JutulCase, d::Int)
    d == 1 || throw(ArgumentError("JutulCase is 1D."))
    return Base.lastindex(case)
end

function Base.getindex(case::JutulCase, ix; check::Bool = true)
    (; model, dt, forces, state0, parameters, input_data, termination_criterion, start_date) = case
    if forces isa AbstractVector
        @assert length(forces) == length(dt)
        forces = forces[ix]
    end
    dt = dt[ix]
    if ix isa AbstractRange || ix isa AbstractVector || ix isa Int
        if check && first(ix) != 1
            jutul_message("JutulCase", "Subsetting case beyond starting index does not update state0 and start_date.")
        end
    end
    return JutulCase(model, dt, forces, state0, parameters, input_data, termination_criterion, start_date)
end

export NoRelaxation, SimpleRelaxation

abstract type NonLinearRelaxation end

struct NoRelaxation <: NonLinearRelaxation end

struct SimpleRelaxation <: NonLinearRelaxation
    tol::Float64
    w_min::Float64
    w_max::Float64
    dw_decrease::Float64
    dw_increase::Float64
end

function SimpleRelaxation(; tol = 0.01, w_min = 0.25, dw = 0.2, dw_increase = nothing, dw_decrease = nothing, w_max = 1.0)
    if isnothing(dw_increase)
        dw_increase = dw/2
    end
    if isnothing(dw_decrease)
        dw_decrease = dw
    end
    return SimpleRelaxation(tol, w_min, w_max, dw_decrease, dw_increase)
end

abstract type CrossTerm end

struct CrossTermPair
    target::Symbol
    source::Symbol
    target_equation::Union{Symbol, Pair{Symbol, Symbol}}
    source_equation::Union{Symbol, Pair{Symbol, Symbol}}
    cross_term::CrossTerm
end

function CrossTermPair(target, source, equation, cross_term::CrossTerm; source_equation = equation)
    CrossTermPair(target, source, equation, source_equation, cross_term)
end

Base.transpose(c::CrossTermPair) = CrossTermPair(c.source, c.target, c.source_equation, c.target_equation, c.cross_term,)

abstract type AbstractMultiModel{label} <: JutulModel end
multimodel_label(::AbstractMultiModel{L}) where L = L

"""
    MultiModel(models)
    MultiModel(models, :SomeLabel)

A model variant that is made up of many named submodels, each a fully realized [`SimulationModel`](@ref).

`models` should be a `NamedTuple` or `Dict{Symbol, JutulModel}`.
"""
struct MultiModel{label, T, CT, G, C, GL} <: AbstractMultiModel{label}
    models::T
    cross_terms::CT
    groups::G
    context::C
    reduction::Union{Symbol, Nothing}
    specialize_ad::Bool
    group_lookup::GL
end

function MultiModel(models, label::Union{Nothing, Symbol} = nothing;
        cross_terms = Vector{CrossTermPair}(),
        groups = nothing,
        context = nothing,
        reduction = missing,
        specialize = false,
        specialize_ad = false
    )
    if isnothing(context)
        context = models[first(keys(models))].context
    end
    group_lookup = Dict{Symbol, Int}()
    if isnothing(groups)
        num_groups = 1
        for k in keys(models)
            group_lookup[k] = 1
        end
        if ismissing(reduction)
            reduction = nothing
        end
    else
        if ismissing(reduction)
            reduction = :schur_apply
        end
        groups::Vector{Int}
        nm = length(models)
        num_groups = length(unique(groups))
        @assert maximum(groups) <= nm
        @assert minimum(groups) > 0
        @assert length(groups) == nm
        @assert maximum(groups) == num_groups "Groups must be ordered from 1 to n, was $(unique(groups))"
        if !issorted(groups)
            # If the groups aren't grouped sequentially, re-sort them so they are
            # since parts of the multimodel code depends on this ordering
            ix = sortperm(groups)
            new_models = OrderedDict{Symbol, Any}()
            old_keys = keys(models)
            for i in ix
                k = old_keys[i]
                new_models[k] = models[k]
            end
            models = new_models
            groups = groups[ix]
        end
        for (k, g) in zip(keys(models), groups)
            group_lookup[k] = g
        end
    end
    if isa(models, AbstractDict)
        models = JutulStorage(models)
    elseif models isa NamedTuple
        models_new = JutulStorage()
        for (k, v) in pairs(models)
            models_new[k] = v
        end
        models = models_new
    else
        models::JutulStorage
    end
    if reduction == :schur_apply
        if length(groups) == 1
            reduction = nothing
        end
    end
    if isnothing(groups) && !isnothing(context)
        for (i, m) in enumerate(models)
            if matrix_layout(m.context) != matrix_layout(context)
                error("No groups provided, but the outer context does not match the inner context for model $i")
            end
        end
    end
    if specialize
        models = convert_to_immutable_storage(models)
    end
    T = typeof(models)
    CT = typeof(cross_terms)
    G = typeof(groups)
    C = typeof(context)
    GL = typeof(group_lookup)
    return MultiModel{label, T, CT, G, C, GL}(models, cross_terms, groups, context, reduction, specialize_ad, group_lookup)
end
                              
function MultiModel(models, ::Val{label}; kwarg...) where label
    # BattMo compatability, support ::Val for symbol
    return MultiModel(models, label; kwarg...)                              
end

function convert_to_immutable_storage(model::MultiModel)
    (; models, cross_terms, groups, context, reduction, specialize_ad, group_lookup) = model
    models = convert_to_immutable_storage(models)
    cross_terms = Tuple(cross_terms)
    group_lookup = convert_to_immutable_storage(group_lookup)
    label = multimodel_label(model)
    T = typeof(models)
    CT = typeof(cross_terms)
    G = typeof(groups)
    C = typeof(context)
    GL = typeof(group_lookup)
    return MultiModel{label, T, CT, G, C, GL}(models, cross_terms, groups, context, reduction, specialize_ad, group_lookup)
end

"""
IndirectionMap(vals::Vector{V}, pos::Vector{Int}) where V

Create a indirection map that encodes a variable length dense vector.

`pos` is assumed to be a Vector{Int} of length n+1 where n is the number of
dense vectors that is encoded. The `vals` array holds the entries for vector i
in the range `pos[i]:(pos[i+1]-1)` for fast lookup. Indexing into the
indirection map with index `k` will give a view into the values for vector `k`.
"""
struct IndirectionMap{V}
    vals::Vector{V}
    pos::Vector{Int}
    function IndirectionMap(vals::Vector{V}, pos::Vector{Int}) where V
        lastpos = pos[end]
        @assert length(vals) == lastpos - 1 "Expected vals to have length lastpos - 1 = $(lastpos-1), was $(length(vals))"
        @assert pos[1] == 1
        new{V}(vals, pos)
    end
end

"""
    imap = IndirectionMap(vec_of_vec::Vector{V}) where V

Create indirection map for a variable length dense vector that is represented as
a Vector of Vectors.
"""
function IndirectionMap(vec_of_vec::Vector{Vector{T}}) where T
    vals = T[]
    pos = Int[1]
    for subvec in vec_of_vec
        for v in subvec
            push!(vals, v)
        end
        push!(pos, pos[end] + length(subvec))
    end
    return IndirectionMap(vals, pos)
end

struct IndexRenumerator{T}
    indices::Dict{T, Int}
end

function IndexRenumerator(T = Int)
    d = Dict{T, Int}()
    return IndexRenumerator{T}(d)
end

function IndexRenumerator(x::AbstractArray{T}) where T
    ir = IndexRenumerator(T)
    for i in x
        ir[i]
    end
    return ir
end


function Base.length(m::IndexRenumerator)
    return length(keys(m.indices))
end

function Base.getindex(m::IndexRenumerator{T}, ix::T) where T
    indices = m.indices
    if !(ix in m)
        n = length(m)+1
        indices[ix] = n
    end
    return indices[ix]
end

function (m::IndexRenumerator)(ix)
    Base.getindex(m, ix)
end

function Base.in(ix::T, m::IndexRenumerator{T}) where T
    return haskey(m.indices, ix)
end

function indices(im::IndexRenumerator{T}) where T
    n = length(im)
    out = Vector{T}(undef, n)
    for (k, v) in im.indices
        out[v] = k
    end
    return out
end

function renumber(x, im::IndexRenumerator)
    renumber!(similar(x), im)
end

function renumber!(x, im::IndexRenumerator)
    for (i, v) in enumerate(x)
        x[i] = im[v]
    end
end

struct EntityTags{T}
    tags::Dict{Symbol, Dict{Symbol, Vector{T}}}
    count::Int
end

function EntityTags(n; tag_type = Int)
    data = Dict{Symbol, Dict{Symbol, Vector{tag_type}}}()
    return EntityTags{tag_type}(data, n)
end

Base.haskey(et::EntityTags, k) = Base.haskey(et.tags, k)
Base.keys(et::EntityTags) = Base.keys(et.tags)
Base.getindex(et::EntityTags, arg...) = Base.getindex(et.tags, arg...)
Base.setindex!(et::EntityTags, arg...) = Base.setindex!(et.tags, arg...)

export set_mesh_entity_tag!, get_mesh_entity_tag, mesh_entity_has_tag
struct MeshEntityTags{T}
    tags::Dict{JutulEntity, EntityTags{T}}
end

Base.haskey(et::MeshEntityTags, k) = Base.haskey(et.tags, k)
Base.keys(et::MeshEntityTags) = Base.keys(et.tags)
Base.getindex(et::MeshEntityTags, arg...) = Base.getindex(et.tags, arg...)
Base.setindex!(et::MeshEntityTags, arg...) = Base.setindex!(et.tags, arg...)

function Base.show(io::IO, t::MIME"text/plain", options::MeshEntityTags{T}) where T
    println(io, "MeshEntityTags stored as $T:")
    for (k, v) in pairs(options.tags)
        kv = keys(v)
        if length(kv) == 0
            kv = "<no tags>"
        else
            s = map(x -> "$x $(keys(v[x]))", collect(kv))
            kv = join(s, ",\n\t")
        end
        println(io, "    $k:\n\t$(kv)")
    end
end

function MeshEntityTags(g::JutulMesh; kwarg...)
    return MeshEntityTags(declare_entities(g); kwarg...)
end

function MeshEntityTags(entities = missing; tag_type = Int)
    tags = Dict{JutulEntity, EntityTags{tag_type}}()
    if !ismissing(entities)
        for epair in entities
            tags[epair.entity] = EntityTags(epair.count, tag_type = tag_type)
        end
    end
    return MeshEntityTags{tag_type}(tags)
end

function initialize_entity_tags!(g::JutulMesh)
    tags = mesh_entity_tags(g)
    for epair in declare_entities(g)
        tags[epair.entity] = EntityTags(epair.count)
    end
    return g
end

function mesh_entity_tags(x)
    return x.tags
end

function set_mesh_entity_tag!(m::JutulMesh, arg...; kwarg...)
    set_mesh_entity_tag!(mesh_entity_tags(m), arg...; kwarg...)
    return m
end

function set_mesh_entity_tag!(met::MeshEntityTags{T}, entity::JutulEntity, tag_group::Symbol, tag_value::Symbol, ix::Vector{T}; allow_merge = true, allow_new = true) where T
    tags = met.tags[entity]
    tags::EntityTags{T}
    if !haskey(tags, tag_group)
        @assert allow_new "allow_new = false and tag group $tag_group for entity $entity already exists."
        tags[tag_group] = Dict{Symbol, Vector{T}}()
    end
    if length(ix) > 0
        @assert maximum(ix, init = zero(T)) <= tags.count "Tag value must not exceed $(tags.count) for $entity"
        @assert minimum(ix, init = tags.count) > 0 "Tags must have positive indices."
    end
    tg = tags[tag_group]
    if haskey(tg, tag_value)
        @assert allow_merge "allow_merge = false and tag $tag_value in group $tag_group for entity $entity already exists."
        vals = tg[tag_value]
        for i in ix
            push!(vals, i)
        end
    else
        tg[tag_value] = copy(ix)
    end
    vals = tg[tag_value]
    sort!(vals)
    unique!(vals)
    return met
end


"""
    get_mesh_entity_tag(met::JutulMesh, entity::JutulEntity, tag_group::Symbol, tag_value = missing; throw = true)

Get the indices tagged for `entity` in group `tag_group`, optionally for the
specific `tag_value`. If `ismissing(tag_value)`, the Dict containing the tag
group will be returned.
"""
function get_mesh_entity_tag(m::JutulMesh, arg...; kwarg...)
    return get_mesh_entity_tag(mesh_entity_tags(m), arg...; kwarg...)
end

function get_mesh_entity_tag(met::MeshEntityTags, entity::JutulEntity, tag_group::Symbol, tag_value = missing; throw = true)
    out = missing
    tags = met.tags[entity]
    if haskey(tags, tag_group)
        tg = tags[tag_group]
        if ismissing(tag_value)
            out = tg
        elseif haskey(tg, tag_value)
            tag_value::Symbol
            out = tg[tag_value]
        end
    end
    if ismissing(out) && throw
        if ismissing(tag_value)
            error("Tag $tag_group not found in $entity.")
        else
            error("Tag $tag_group.$tag_value not found in $entity.")
        end
    end
    return out
end

function mesh_entity_has_tag(m::JutulMesh, arg...; kwarg...)
    return mesh_entity_has_tag(mesh_entity_tags(m), arg...; kwarg...)
end

function mesh_entity_has_tag(met::MeshEntityTags, entity::JutulEntity, tag_group::Symbol, tag_value::Symbol, ix)
    tag = get_mesh_entity_tag(met, entity, tag_group, tag_value)
    pos = searchsortedfirst(tag, ix)
    if pos > length(tag)
        out = false
    else
        out = tag[pos] == ix
    end
    return out
end

struct SimResult
    states::AbstractVector
    reports::AbstractVector
    start_timestamp::DateTime
    end_timestamp::DateTime
    function SimResult(states, reports, start_time)
        nr = length(reports)
        ns = length(states)
        @assert ns == nr || ns == nr-1 || ns == 0 "Recieved $ns or $ns - 1 states different from $nr reports"
        return new(states, reports, start_time, now())
    end
end

mutable struct AdjointPackedResult
    step_infos
    states
    forces
    state0
    input_data
    Nstep::Int
    function AdjointPackedResult(states::Vector{JutulStorage{T}}, step_infos::Vector, Nstep::Int, forces::Union{Vector, Missing}; state0 = missing, input_data = missing) where T
        if length(states) != length(step_infos)
            error("States and step_infos must have the same length, was $(length(states)) and $(length(step_infos))")
        end
        if !ismissing(forces) && length(forces) != length(step_infos)
            error("Forces and step_infos must have the same length, was $(length(forces)) and $(length(step_infos))")
        end
        new(step_infos, states, forces, state0, input_data, Nstep)
    end
end

function Base.getindex(r::AdjointPackedResult, i::Int)
    if ismissing(r.forces)
        f_i = missing
    else
        f_i = r.forces[i]
    end
    return (state = r.states[i], step_info = r.step_infos[i], forces = f_i)
end

function Base.length(r::AdjointPackedResult)
    return length(r.states)
end

function AdjointPackedResult(r::SimResult, case::JutulCase)
    return AdjointPackedResult(r, case.forces)
end

function AdjointPackedResult(r::SimResult, forces)
    states, dt, report_step_ix = expand_to_ministeps(r)
    return AdjointPackedResult(states, dt, forces, report_step_ix)
end

function AdjointPackedResult(states, dt::Vector{Float64}, forces)
    return AdjointPackedResult(states, dt, forces, eachindex(dt, states))
end

function AdjointPackedResult(states, reports, forces)
    states, dt, report_step_ix = expand_to_ministeps(states, reports)
    return AdjointPackedResult(states, dt, forces, report_step_ix)
end

function AdjointPackedResult(states, dt::Vector{Float64}, forces, step_index)
    N_report_step = maximum(step_index)
    N_mini_step = length(dt)
    if forces isa Vector
        N_forces = length(forces)
        if N_forces != N_mini_step
            if N_forces == N_report_step
                forces = forces[step_index]
            else
                error("Forces must have the same length as either states ($(N_mini_step)) or report steps ($(N_report_step)), was $(N_forces).")
            end
        end
    end
    length(states) == length(dt) || error("States and timesteps must have the same length, was $(length(states)) and $(length(dt))")
    total_time = sum(dt)
    # Set
    time = 0.0
    ministep_ix = 1
    prev_ministep = 0
    step_infos = missing
    for (i, dt_i) in enumerate(dt)
        time += dt_i
        step_ix = step_index[i]
        if step_ix != prev_ministep
            ministep_ix = 1
            prev_ministep = step_ix
        else
            ministep_ix += 1
        end
        step_info = optimization_step_info(step_ix, time, dt_i,
            Nstep = N_report_step,
            substep = ministep_ix,
            substep_global = i,
            Nsubstep_global = length(dt),
            total_time = total_time
        )
        if i == 1
            step_infos = [step_info]
        else
            push!(step_infos, step_info)
        end
    end
    if !ismissing(forces)
        forces = map(i -> forces_for_timestep(nothing, forces, dt, i), step_index)
    end
    function convert_state_to_jutul_storage(x::JutulStorage)
        return x
    end
    function convert_state_to_jutul_storage(x::Any)
        return x
    end
    function convert_state_to_jutul_storage(x::JUTUL_OUTPUT_TYPE)
        s = JutulStorage()
        for (k, v) in pairs(x)
            if k == :substates
                continue
            end
            s[k] = convert_state_to_jutul_storage(v)
        end
        return s
    end
    states = map(s -> convert_state_to_jutul_storage(s), states)
    return AdjointPackedResult(states, step_infos, maximum(step_index), forces)
end

"""
Abstract type for Jutul objectives.
"""
abstract type AbstractJutulObjective end

objective_depends_on_crossterms(F) = false
objective_depends_on_parameters(F) = true

"""
Abstract type for objective as a sum of function values on the form:

    F(model, state, dt, step_info, forces)

evaluated for each step. This means that the objective is a sum of all of these
values. If you want to only depend on a single step, you can look up
"""
abstract type AbstractSumObjective <: AbstractJutulObjective end

"""
Abstract type for objective as a global objective function on the form:

    F(model, state0, states, step_infos, forces, input_data)

Note that if substeps are enabled by setting `output_substates=true` in the
simulator setup, the length of forces and states will be dynamic and of equal
length to `step_infos`. If you want to recover the state at a specific step, you
can reason about `:step` field of the corresponding entry in the `step_infos`
array.
"""
abstract type AbstractGlobalObjective <: AbstractJutulObjective end

"""
    WrappedSumObjective(objective)

An objective that is a sum of function values, evaluated for each step, defined
by a function. This type automatically wraps functions/callable structs that are
passed to the optimization interface if they have the sum signature (five
arguments).
"""
struct WrappedSumObjective{T} <: AbstractSumObjective
    objective::T
    depends_on_crossterms::Bool
    depends_on_parameters::Bool
    function WrappedSumObjective(objective::T; depends_on_crossterms = false, depends_on_parameters = true) where T
        return new{T}(objective, depends_on_crossterms, depends_on_parameters)
    end
end

"""
    WrappedGlobalObjective(objective)

A global (in time) objective that wraps a function/callable. This type
automatically wraps functions/callable structs that are passed to the
optimization interface if they have the sum signature (five arguments).
"""
struct WrappedGlobalObjective{T} <: AbstractGlobalObjective
    objective::T
    depends_on_crossterms::Bool
    depends_on_parameters::Bool
    function WrappedGlobalObjective(objective::T; depends_on_crossterms = false, depends_on_parameters = true) where T
        return new{T}(objective, depends_on_crossterms, depends_on_parameters)
    end
end

function objective_depends_on_crossterms(O::Union{WrappedGlobalObjective, WrappedSumObjective})
    return O.depends_on_crossterms
end

function objective_depends_on_parameters(O::Union{WrappedGlobalObjective, WrappedSumObjective})
    return O.depends_on_parameters
end

"""
    NoTerminationCriterion()

A termination criterion that never terminates prematurely, leaving all timesteps to be executed.
"""
struct NoTerminationCriterion <: AbstractTerminationCriterion end

function timestepping_is_done(C::NoTerminationCriterion, simulator, states, substates, reports, solve_recorder)
    return false
end

"""
    EndTimeTerminationCriterion(end_time)

A termination criterion that stops time-stepping when the global time reaches `end_time`.
"""
struct EndTimeTerminationCriterion <: AbstractTerminationCriterion
    end_time::Float64
end

function timestepping_is_done(C::EndTimeTerminationCriterion, simulator, states, substates, reports, solve_recorder)
    now = recorder_current_time(solve_recorder, :global)
    return now >= C.end_time
end

