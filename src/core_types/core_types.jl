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
    primary = T()
    secondary = T()
    parameters = T()
    equations = OrderedDict{Symbol,JutulEquation}()
    outputs = Vector{Symbol}()
    D = typeof(domain)
    S = typeof(system)
    F = typeof(formulation)
    C = typeof(context)
    model = SimulationModel{D,S,F,C}(domain, system, context, formulation, data_domain, primary, secondary, parameters, equations, outputs, extra)
    update_model_pre_selection!(model)
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
    update_model_post_selection!(model)
    return model
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

function Base.copy(m::SimulationModel{O, S, C, F}) where {O, S, C, F}
    pvar = copy(m.primary_variables)
    svar = copy(m.secondary_variables)
    outputs = copy(m.output_variables)
    prm = copy(m.parameters)
    eqs = m.equations
    return SimulationModel{O, S, C, F}(m.domain, m.system, m.context, m.formulation, m.plot_mesh, pvar, svar, prm, eqs, outputs)
end

function Base.getindex(model::SimulationModel, s::Symbol)
    return get_variables(model)[s]
end

function Base.show(io::IO, t::MIME"text/plain", model::SimulationModel)
    println(io, "SimulationModel:")
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
                        print_t = Base.typename(typeof(pvar)).wrapper
                        print(io, "      -> $print_t as evaluator")
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
        else
            print(io, "    ")
            print(io, p)
            print(io, "\n\n")
        end
    end
end

# Grids etc
export JutulEntity, Cells, Faces, Nodes, NoEntity
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

Base.iterate(S::JutulStorage) = Base.iterate(data(S))
Base.pairs(S::JutulStorage) = Base.pairs(data(S))

function Base.getproperty(S::JutulStorage{Nothing}, name::Symbol)
    Base.getindex(data(S), name)
end

function Base.getproperty(S::JutulStorage, name::Symbol)
    Base.getproperty(data(S), name)
end

Base.propertynames(S::JutulStorage) = keys(getfield(S, :data))

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
struct CompositeSystem{T} <: JutulSystem
    systems::T
end

function Base.show(io::IO, t::CompositeSystem)
    print(io, "CompositeSystem:\n")
    for (name, sys) in pairs(t.systems)
        print(io, "($name => $sys)\n")
    end
end
function CompositeSystem(; kwarg...)
    return CompositeSystem(NamedTuple(pairs(kwarg)))
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

export JutulCase
struct JutulCase
    model::JutulModel
    dt::AbstractVector{<:AbstractFloat}
    forces
    state0
    parameters
    input_data
end

"""
    JutulCase(model, dt = [1.0], forces = setup_forces(model); state0 = nothing, parameters = nothing, kwarg...)

Set up a structure that holds the complete specification of a simulation case.
"""
function JutulCase(model, dt = [1.0], forces = setup_forces(model); state0 = nothing, parameters = nothing, input_data = nothing, kwarg...)
    if isnothing(state0) && isnothing(parameters)
        state0, parameters = setup_state_and_parameters(model, kwarg...)
    elseif isnothing(state0)
        state0 = setup_state(model, kwarg...)
    elseif isnothing(parameters)
        parameters = setup_parameters(model, kwarg...)
    end
    if forces isa AbstractVector
        nf = length(forces)
        nt = length(dt)
        @assert nt == nf "If forces is a vector, the length (=$nf) must match the number of time steps (=$nt)."
    end
    @assert all(dt .> 0)
    return JutulCase(model, dt, forces, state0, parameters, input_data)
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
    return JutulCase(model, deepcopy(dt), deepcopy(forces), deepcopy(state0), deepcopy(parameters), deepcopy(input_data))
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


"""
    MultiModel(models)

A model variant that is made up of many named submodels, each a fully realized [`SimulationModel`](@ref).

`models` should be a `NamedTuple` or `Dict{Symbol, JutulModel}`.
"""
struct MultiModel{T} <: JutulModel
    models::NamedTuple
    cross_terms::Vector{CrossTermPair}
    groups::Union{Vector, Nothing}
    context::Union{JutulContext, Nothing}
    reduction::Union{Symbol, Nothing}
    specialize_ad::Bool
function MultiModel(models; cross_terms = Vector{CrossTermPair}(), groups = nothing, context = nothing, reduction = nothing, specialize = false, specialize_ad = false)
        if isnothing(groups)
            num_groups = 1
        else
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
        end
        if isa(models, AbstractDict)
            models = convert_to_immutable_storage(models)
        end
        if reduction == :schur_apply
            if length(groups) > 1
                # @assert num_groups == 2
            else
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
            T = typeof(models)
        else
            T = nothing
        end
        new{T}(models, cross_terms, groups, context, reduction, specialize_ad)
    end
end
