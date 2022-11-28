export JutulSystem, JutulDomain, JutulVariables, AbstractJutulMesh, JutulContext
export SimulationModel, JutulVariables, JutulFormulation, JutulEquation
export setup_parameters, JutulForce
export Cells, Nodes, Faces, declare_entities
export ConstantVariables, ScalarVariable, VectorVariables, FractionVariables

export SingleCUDAContext, DefaultContext
export BlockMajorLayout, EquationMajorLayout, EntityMajorLayout

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
abstract type VectorVariables <: JutulVariables end
abstract type FractionVariables <: VectorVariables end

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
struct EntityMajorLayout <: JutulMatrixLayout
    as_adjoint::Bool
end
EntityMajorLayout() = EntityMajorLayout(false)
is_cell_major(::EntityMajorLayout) = true

const ScalarLayout = Union{EquationMajorLayout, EntityMajorLayout}

"""
Same as EntityMajorLayout, but the nzval is a matrix
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
                       C<:JutulContext
                       } <: AbstractSimulationModel
    domain::O
    system::S
    context::C
    formulation::F
    plot_mesh
    primary_variables::OrderedDict{Symbol, Any}
    secondary_variables::OrderedDict{Symbol, Any}
    parameters::OrderedDict{Symbol, Any}
    equations::OrderedDict{Symbol, Any}
    output_variables::Vector{Symbol}
    extra::OrderedDict{Symbol, Any}
end

function SimulationModel(domain, system;
                            formulation=FullyImplicit(),
                            context=DefaultContext(),
                            plot_mesh = nothing,
                            output_level=:primary_variables,
                            extra = OrderedDict{Symbol, Any}()
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
    model = SimulationModel{D,S,F,C}(domain, system, context, formulation, plot_mesh, primary, secondary, parameters, equations, outputs, extra)
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
        elseif f == :domain
            print(io, "    ")
            if !isnothing(p)
                print(io, p)
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
struct ConstantVariables <: VectorVariables
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
        # if isa(constants, CuArray) && single_entity
        #    @warn "Single entity constants have led to crashes on CUDA/Tullio kernels!" maxlog = 5
        # end
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

struct GenericAutoDiffCache{N, E, ∂x, A, P, M, D, VM} <: JutulAutoDiffCache where {∂x <: Real}
    # N - number of equations per entity
    entries::A
    vpos::P               # Variable positions (CSR-like, length N + 1 for N entities)
    variables::P          # Indirection-mapped variable list of same length as entries
    jacobian_positions::M
    diagonal_positions::D
    number_of_entities_target::Integer
    number_of_entities_source::Integer
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

struct ConservationLaw{C, T<:FlowDiscretization, N} <: JutulEquation
    flow_discretization::T
    function ConservationLaw(disc::T, conserved::Symbol = :TotalMasses, N::Integer = 1) where T
        return new{conserved, T, N}(disc)
    end
end

export CompositeSystem
struct CompositeSystem{T} <: JutulSystem
    systems::T
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
    if eltype(labels)<:AbstractString
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
end

function JutulCase(model, dt = [1.0], forces = setup_forces(model); state0 = nothing, parameters = nothing, kwarg...)
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
    return JutulCase(model, dt, forces, state0, parameters)
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
