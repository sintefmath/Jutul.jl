const KASimulationModel = SimulationModel{<:Any, <:Any, <:Any, <:KernelAbstractionsContext}

function update_values!(v::AbstractArray{T}, next::AbstractArray{S},
        context::KernelAbstractionsContext) where {T<:Real, S<:Real}
    # Inputs supplied to reset_variables! commonly live in CPU memory. Move
    # them to the execution backend before launching the value-preserving
    # update kernel; GPU kernels cannot index the host array directly.
    next_backend = Adapt.adapt(context, next)
    preserve_partials = Val(eltype(v) <: ForwardDiff.Dual &&
        eltype(next_backend) <: Real && eltype(v) !== eltype(next_backend) &&
        unpack_tag(v) isa JutulEntity)
    strip_partials = Val(eltype(v) <: AbstractFloat &&
        eltype(next_backend) <: ForwardDiff.Dual)
    strip_partials isa Val{true} && (unpack_tag(next_backend)::JutulEntity)
    function update(i)
        @inbounds old = v[i]
        @inbounds new = next_backend[i]
        new = updated_state_value(old, new, preserve_partials, strip_partials)
        @inbounds v[i] = new
    end
    threaded_loop_minbatch(update, length(v), context)
    return v
end

function replace_values!(old, updated, context::KernelAbstractionsContext)
    for field in keys(old)
        if haskey(updated, field)
            next = Adapt.adapt(context, updated[field])
            update_values!(old[field], next, context)
        end
    end
end

# Adapt uses the context as the adaptation target. Backend packages define how
# their own backend converts an Array, while Jutul supplies the structural rules.
Adapt.adapt_storage(ctx::KernelAbstractionsContext, a::AbstractArray) = Adapt.adapt(ctx.backend, a)
Adapt.adapt_storage(::KernelAbstractionsContext, a::AbstractArray{Symbol}) = Tuple(a)
transfer(ctx::KernelAbstractionsContext, x::AbstractArray) = Adapt.adapt(ctx, x)
backend_to_host(::KernelAbstractionsContext, x) = Adapt.adapt(Array, x)

function Adapt.adapt_structure(to, interpolant::LinearInterpolant)
    return LinearInterpolant(
        Adapt.adapt(to, interpolant.X),
        Adapt.adapt(to, interpolant.F),
        Adapt.adapt(to, interpolant.lookup)
    )
end

function Adapt.adapt_structure(to, flow::PotentialFlow{AD}) where AD
    return PotentialFlow(
        Adapt.adapt(to, flow.kgrad),
        Adapt.adapt(to, flow.upwind),
        Adapt.adapt(to, flow.half_face_map);
        ad = AD
    )
end

function Adapt.adapt_structure(to, g::CartesianMesh)
    # Tags are setup-time metadata backed by dictionaries. Numerical kernels
    # only need the dimensions, cell sizes and origin.
    return CartesianMesh(
        g.dims,
        Adapt.adapt(to, g.deltas),
        Adapt.adapt(to, g.origin),
        nothing
    )
end

function Adapt.adapt_structure(to, d::DiscretizedDomain)
    entities = d.entities isa EntityCounter ? d.entities : EntityCounter(d.entities)
    return DiscretizedDomain(
        Adapt.adapt(to, d.representation),
        Adapt.adapt(to, d.discretizations),
        entities,
        Adapt.adapt(to, d.global_map)
    )
end

function Adapt.adapt_structure(to, d::TwoPointPotentialFlowHardCoded)
    return TwoPointPotentialFlowHardCoded(
        d.gravity,
        Adapt.adapt(to, d.conn_pos),
        Adapt.adapt(to, d.conn_data)
    )
end

function Adapt.adapt_structure(to, eq::ConservationLaw{C, T, FT, N}) where {C, T, FT, N}
    return ConservationLaw(Adapt.adapt(to, eq.flow_discretization), C, N;
        flux = Adapt.adapt(to, eq.flux_type))
end

function Adapt.adapt_structure(to, c::CompactAutoDiffCache)
    entries = Adapt.adapt(to, c.entries)
    positions = Adapt.adapt(to, c.jacobian_positions)
    return CompactAutoDiffCache{typeof(c.equations_per_entity), eltype(entries)}(
        entries, c.entity, positions,
        c.equations_per_entity, c.number_of_entities, c.npartials
    )
end

function Adapt.adapt_structure(to,
        c::GenericAutoDiffCache{N, E, T}) where {N, E, T}
    entries = Adapt.adapt(to, c.entries)
    vpos = Adapt.adapt(to, c.vpos)
    variables = Adapt.adapt(to, c.variables)
    positions = Adapt.adapt(to, c.jacobian_positions)
    diagonal = Adapt.adapt(to, c.diagonal_positions)
    variable_map = Adapt.adapt(to, c.variable_map)
    return GenericAutoDiffCache{N, E, T}(
        entries, vpos, variables, positions, diagonal,
      c.number_of_entities_target, c.number_of_entities_source, variable_map)
end

function Adapt.adapt_structure(to, s::ConservationLawTPFAStorage)
    return ConservationLawTPFAStorage(
        Adapt.adapt(to, s.accumulation),
        s.accumulation_symbol,
        Adapt.adapt(to, s.half_face_flux_cells),
        Adapt.adapt(to, s.half_face_flux_faces),
        nothing
    )
end

function Adapt.adapt_structure(to, state::LocalStateAD{T, I, E}) where {T, I, E}
    adapted = Adapt.adapt(to, getfield(state, :data))
    return LocalStateAD{typeof(adapted), I, E}(getfield(state, :index), adapted)
end

function Adapt.adapt_structure(to, state::ValueStateAD)
    return ValueStateAD(Adapt.adapt(to, getfield(state, :data)))
end

function Adapt.adapt_structure(to, state::MultiModelLocalStateAD{T, I, E}) where {T, I, E}
    adapted = Adapt.adapt(to, getfield(state, :data))
    return MultiModelLocalStateAD{typeof(adapted), I, E}(
        getfield(state, :symbol), getfield(state, :index), adapted)
end

function Adapt.adapt_structure(to, perspective::LocalPerspectiveAD)
    return LocalPerspectiveAD(
        Adapt.adapt(to, getfield(perspective, :data)),
        getfield(perspective, :index)
    )
end

function Adapt.adapt_structure(ctx::KernelAbstractionsContext, A::StaticSparsityMatrixCSR)
    nzval = Adapt.adapt(ctx, nonzeros(A))
    colval = Adapt.adapt(ctx, colvals(A))
    rowptr = Adapt.adapt(ctx, A.rowptr)
    return StaticSparsityMatrixCSR(nzval, colval, rowptr, size(A, 1), size(A, 2), ctx.backend;
        nthreads = 1, minbatch = 1, thread_type = :serial)
end

function Adapt.adapt_structure(ctx::KernelAbstractionsContext, lsys::LinearizedSystem)
    lsys.jac isa StaticSparsityMatrixCSR || throw(ArgumentError(
        "KernelAbstractions transfer requires a CPU simulator built with ParallelCSRContext"
    ))
    jac = Adapt.adapt(ctx, lsys.jac)
    r_buffer = Adapt.adapt(ctx, lsys.r_buffer)
    dx_buffer = Adapt.adapt(ctx, lsys.dx_buffer)
    r = lsys.r === lsys.r_buffer ? r_buffer : Adapt.adapt(ctx, lsys.r)
    dx = lsys.dx === lsys.dx_buffer ? dx_buffer : Adapt.adapt(ctx, lsys.dx)
    function backend_jacobian_buffer()
        nz = nonzeros(jac)
        original_buffer = lsys.jac_buffer
        if eltype(original_buffer) == eltype(nz)
            if size(original_buffer) == size(nz)
                return nz
            else
                return reshape(nz, size(original_buffer))
            end
        end
        buffer = reinterpret(reshape, eltype(original_buffer), nz)
        return reshape(buffer, size(original_buffer))
    end
    jac_buffer = backend_jacobian_buffer()
    return LinearizedSystem(jac, r, dx, jac_buffer, r_buffer, dx_buffer, lsys.matrix_layout)
end

function Adapt.adapt_structure(ctx::KernelAbstractionsContext,
        block::LinearizedBlock{R, C}) where {R, C}
    jac = Adapt.adapt(ctx, block.jac)
    nz = nonzeros(jac)
    if eltype(nz) == eltype(block.jac_buffer) && length(nz) == length(block.jac_buffer)
        jac_buffer = nz
    else
        throw(ArgumentError(
            "KernelAbstractions transfer does not yet support block-valued off-diagonal Jacobians"))
    end
    return LinearizedBlock(
        jac, jac_buffer, block.rowcol_block_size, R(), C(), Val(:assembled))
end

function Adapt.adapt_structure(ctx::KernelAbstractionsContext, lsys::MultiLinearizedSystem)
    function backend_vector_alias(original, original_buffer, adapted_buffer)
        buffer = reshape(adapted_buffer, size(original_buffer))
        if eltype(original) == eltype(original_buffer)
            return reshape(buffer, size(original))
        else
            return reinterpret(reshape, eltype(original), buffer)
        end
    end

    r_buffer = Adapt.adapt(ctx, lsys.r_buffer)
    dx_buffer = Adapt.adapt(ctx, lsys.dx_buffer)
    r = backend_vector_alias(lsys.r, lsys.r_buffer, r_buffer)
    dx = backend_vector_alias(lsys.dx, lsys.dx_buffer, dx_buffer)

    nsystems = size(lsys.subsystems)
    subsystems = Matrix{LinearizedType}(undef, nsystems)
    r_offset = dx_offset = 0
    for col in axes(lsys.subsystems, 2), row in axes(lsys.subsystems, 1)
        old = lsys.subsystems[row, col]
        if row == col
            nr = length(old.r_buffer)
            ndx = length(old.dx_buffer)
            r_range = (r_offset + 1):(r_offset + nr)
            dx_range = (dx_offset + 1):(dx_offset + ndx)
            r_i_buffer = reshape(view(vec(r_buffer), r_range), size(old.r_buffer))
            dx_i_buffer = reshape(view(vec(dx_buffer), dx_range), size(old.dx_buffer))
            r_i = backend_vector_alias(old.r, old.r_buffer, r_i_buffer)
            dx_i = backend_vector_alias(old.dx, old.dx_buffer, dx_i_buffer)
            adapted_old = Adapt.adapt(ctx, old)
            jac = adapted_old.jac
            jac_buffer = adapted_old.jac_buffer
            subsystems[row, col] = LinearizedSystem(
                jac, r_i, dx_i, jac_buffer, r_i_buffer, dx_i_buffer,
                old.matrix_layout
            )
            r_offset += nr
            dx_offset += ndx
        else
            subsystems[row, col] = Adapt.adapt(ctx, old)
        end
    end
    @assert r_offset == length(r_buffer)
    @assert dx_offset == length(dx_buffer)
    schur_buffer = adapt_backend_value(ctx, lsys.schur_buffer)
    return MultiLinearizedSystem{typeof(lsys.matrix_layout)}(
        subsystems, r, dx, r_buffer, dx_buffer, lsys.reduction,
        FactorStore(), lsys.matrix_layout, schur_buffer
    )
end

adapt_backend_value(ctx, x::NamedTuple) = map(v -> adapt_backend_value(ctx, v), x)
adapt_backend_value(ctx, x::Tuple) = map(v -> adapt_backend_value(ctx, v), x)
function adapt_backend_value(ctx, x::AbstractDict)
    return (; (Symbol(k) => adapt_backend_value(ctx, v) for (k, v) in pairs(x))...)
end
adapt_backend_value(ctx, x::JutulStorage) = JutulStorage(adapt_backend_value(ctx, data(x)))
adapt_backend_value(ctx, x::AbstractVector{<:JutulStorage}) =
    tuple((adapt_backend_value(ctx, v) for v in x)...)
adapt_backend_value(ctx, x) = Adapt.adapt(ctx, x)

"""
    secondary_variable_evaluation_plan(model, secondary)

Build a CPU-side execution schedule for secondary variables. Each vector in
the returned vector is one dependency level and contains
`symbol => entity_count` entries. Variables in one level are independent and
may be launched without an intermediate synchronization; the next level starts
only after the backend has finished the preceding level.

Symbols are stored as values in ordinary vectors rather than tuple parameters,
so the plan type does not specialize on the model's property names. The plan is
created while simulator storage is transferred and remains on the CPU.
"""
function secondary_variable_evaluation_plan(
        model, secondary = model.secondary_variables)
    nodes, dependencies = build_variable_graph(
        model, model.primary_variables, model.secondary_variables,
        model.parameters)
    order = sort_symbols(nodes, dependencies)
    positions = Dict(symbol => index for (index, symbol) in enumerate(nodes))
    number_of_roots = length(model.primary_variables) + length(model.parameters)
    node_levels = zeros(Int, length(nodes))
    for index in order
        index <= number_of_roots && continue
        level = 1
        for dependency in dependencies[index]
            level = max(level, node_levels[positions[dependency]] + 1)
        end
        node_levels[index] = level
    end
    levels_by_symbol = Dict(
        nodes[index] => node_levels[index]
        for index in (number_of_roots + 1):length(nodes))

    symbols = collect(keys(secondary))
    isempty(symbols) && return Vector{Vector{Pair{Symbol, Int}}}()
    levels = map(symbol -> levels_by_symbol[symbol], symbols)
    active_levels = sort!(unique(levels))
    return map(active_levels) do level
        entries = Pair{Symbol, Int}[]
        for symbol in symbols
            if levels_by_symbol[symbol] == level
                variable = secondary[symbol]
                push!(entries, symbol => number_of_entities(model, variable))
            end
        end
        entries
    end
end

function update_secondary_variables_state!(state, model, vars,
        plan::AbstractVector)
    context = model.context
    for level in plan
        for (symbol, batch_count) in level
            target = state[symbol]
            variable = vars[symbol]
            function update(batch)
                indices = entity_eachindex(target, batch, batch_count)
                update_secondary_variable!(
                    target, variable, model, state, indices)
            end
            launch_threaded_loop(update, batch_count, context)
        end
        synchronize(context)
    end
    return state
end

# Forces are created together with the CPU simulator. Move only force values
# through this recursive interface: schedule and model containers stay on the
# host, while arrays of actual force objects are transferred in one operation.
struct BackendForces{H, D}
    host::H
    device::D
end

forces_for_host(forces::BackendForces) = forces.host
forces_for_backend(forces::BackendForces) = forces.device

function preprocess_forces(sim, ctx::KernelAbstractionsContext, forces)
    device_forces = transfer_forces_to_backend(ctx, forces)
    storage = get_simulator_storage(sim)
    if get_simulator_model(sim) isa MultiModel && haskey(storage, :host_evaluation)
        return BackendForces(forces, device_forces)
    else
        return device_forces
    end
end

function forces_for_timestep(sim, forces::BackendForces, timesteps,
        step_index; per_step = false)
    return forces
end

Base.getindex(forces::BackendForces, key) = forces.device[key]
Base.getproperty(forces::BackendForces, name::Symbol) =
    name === :host || name === :device ? getfield(forces, name) :
        getproperty(getfield(forces, :device), name)
Base.keys(forces::BackendForces) = keys(forces.device)
Base.values(forces::BackendForces) = values(forces.device)
Base.pairs(forces::BackendForces) = pairs(forces.device)
Base.haskey(forces::BackendForces, key) = haskey(forces.device, key)
Base.length(forces::BackendForces) = length(forces.device)
Base.iterate(forces::BackendForces, state...) = iterate(forces.device, state...)

transfer_forces_to_backend(ctx::KernelAbstractionsContext, ::Nothing) = nothing
transfer_forces_to_backend(ctx::KernelAbstractionsContext, force::JutulForce) =
    Adapt.adapt(ctx, force)
transfer_forces_to_backend(ctx::KernelAbstractionsContext,
    forces::AbstractVector{<:JutulForce}) = Adapt.adapt(ctx, forces)
transfer_forces_to_backend(ctx::KernelAbstractionsContext, forces::NamedTuple) =
    map(force -> transfer_forces_to_backend(ctx, force), forces)
transfer_forces_to_backend(ctx::KernelAbstractionsContext, forces::Tuple) =
    map(force -> transfer_forces_to_backend(ctx, force), forces)
function transfer_forces_to_backend(ctx::KernelAbstractionsContext,
        forces::AbstractDict)
    out = copy(forces)
    for (key, force) in pairs(forces)
        out[key] = transfer_forces_to_backend(ctx, force)
    end
    return out
end
transfer_forces_to_backend(::KernelAbstractionsContext, force) = force

function Adapt.adapt_structure(ctx::KernelAbstractionsContext, model::SimulationModel)
    primary = adapt_backend_value(ctx, model.primary_variables)
    secondary = adapt_backend_value(ctx, model.secondary_variables)
    parameters = adapt_backend_value(ctx, model.parameters)
    equations = adapt_backend_value(ctx, model.equations)
    return SimulationModel(
        Adapt.adapt(ctx, model.domain),
        Adapt.adapt(ctx, model.system),
        ctx,
        Adapt.adapt(ctx, model.formulation),
        missing,
        primary,
        secondary,
        parameters,
        equations,
        Tuple(model.output_variables),
        nothing,
        model.optimization_level
    )
end

function Adapt.adapt_structure(to, model::KASimulationModel)
    # Property and equation definitions are passed to their kernels directly.
    # Keep the model argument lean so unrelated metadata with abstract/Union
    # fields cannot make an otherwise compatible kernel argument non-isbits.
    return SimulationModel(
        Adapt.adapt(to, model.domain),
        Adapt.adapt(to, model.system),
        Adapt.adapt(to, model.context),
        Adapt.adapt(to, model.formulation),
        missing,
        NamedTuple(),
        NamedTuple(),
        NamedTuple(),
        NamedTuple(),
        (),
        nothing,
        model.optimization_level
    )
end

function Adapt.adapt_structure(ctx::KernelAbstractionsContext, ctp::CrossTermPair)
    return CrossTermPair(
        ctp.target, ctp.source, ctp.target_equation, ctp.source_equation,
        Adapt.adapt(ctx, ctp.cross_term)
    )
end

function cpu_csr_model(model::SimulationModel)
    source = model.context
    context = ParallelCSRContext(1;
        matrix_layout = matrix_layout(source), thread_type = :serial)
    return SimulationModel(
        model.domain, model.system, context, model.formulation,
        model.data_domain, model.primary_variables, model.secondary_variables,
        model.parameters, model.equations, model.output_variables, model.extra,
        model.optimization_level
    )
end

function cpu_csr_model(model::MultiModel)
    models = (; (key => cpu_csr_model(submodel)
        for (key, submodel) in pairs(model.models))...)
    outer = ParallelCSRContext(1;
        matrix_layout = matrix_layout(model.context), thread_type = :serial)
    return MultiModel(models, multimodel_label(model);
        cross_terms = model.cross_terms,
        groups = isnothing(model.groups) ? nothing : copy(model.groups),
        group_execution = model.group_execution,
        context = outer,
        reduction = model.reduction,
        specialize = false,
        specialize_ad = model.specialize_ad
    )
end

function Adapt.adapt_structure(ctx::KernelAbstractionsContext, model::MultiModel)
    function backend_subcontext(submodel)
        source = submodel.context
        return KernelAbstractionsContext(ctx.backend;
            float_type = float_type(source),
            index_type = index_type(source),
            matrix_layout = matrix_layout(source),
            workgroupsize = ctx.workgroupsize
        )
    end
    models = (; (key => Adapt.adapt(backend_subcontext(submodel), submodel)
        for (key, submodel) in pairs(model.models))...)
    cross_terms = [Adapt.adapt(ctx, ct) for ct in model.cross_terms]
    groups = isnothing(model.groups) ? nothing : copy(model.groups)
    label = multimodel_label(model)
    return MultiModel(models, label;
        cross_terms = cross_terms,
        groups = groups,
        group_execution = model.group_execution,
        context = ctx,
        reduction = model.reduction,
        specialize = false,
        specialize_ad = model.specialize_ad
    )
end

struct HostEvaluationStorage{M, S, K}
    model::M
    storage::S
    keys::K
end

"""
    prepare_backend_transfer!(storage, model)

Application hook invoked immediately before a host-evaluated submodel is
copied into its backend mirror. It can refresh preallocated numeric state that
replaces host-only control or metadata objects in device kernels.
"""
function backend_copyto!(destination::AbstractArray, source::AbstractArray)
    length(destination) == length(source) || throw(DimensionMismatch(
        "backend copy requires equal lengths, got $(length(destination)) and $(length(source))"))
    isempty(destination) && return destination
    backend = KernelAbstractions.get_backend(destination)
    if applicable(KernelAbstractions.copyto!, backend, destination, source)
        # Host-evaluated submodels own their source buffers for the duration of
        # the simulation. Queue their copies on the backend so a structured
        # state transfer requires one synchronization instead of one per field.
        KernelAbstractions.copyto!(backend, destination, source)
    else
        copyto!(destination, source)
    end
    return destination
end

# Equation-major views are represented as adjoints of reshaped slices. Peel
# identical structural wrappers before copying so CPU/backend transfers use a
# contiguous bulk copy instead of LinearAlgebra's scalar transpose routine.
function backend_copyto!(destination::LinearAlgebra.Adjoint,
        source::LinearAlgebra.Adjoint)
    backend_copyto!(parent(destination), parent(source))
    return destination
end

function backend_copyto!(destination::LinearAlgebra.Transpose,
        source::LinearAlgebra.Transpose)
    backend_copyto!(parent(destination), parent(source))
    return destination
end

function backend_copyto!(destination::Base.ReshapedArray,
        source::AbstractArray)
    backend_copyto!(parent(destination), vec(source))
    return destination
end

function backend_copyto!(destination::SubArray{T, 1, P, I, true},
        source::AbstractArray) where {T, P, I}
    indices = parentindices(destination)
    if length(indices) == 1 && only(indices) isa AbstractUnitRange
        range = only(indices)
        linear_source = vec(source)
        copyto!(parent(destination), first(range), linear_source,
            firstindex(linear_source), length(destination))
    else
        copyto!(destination, source)
    end
    return destination
end

function backend_copyto!(destination::JutulStorage, source::JutulStorage)
    for key in keys(destination)
        haskey(source, key) || continue
        backend_copyto!(destination[key], source[key])
    end
    return destination
end

function backend_copyto!(destination::NamedTuple, source)
    for key in keys(destination)
        haskey(source, key) || continue
        backend_copyto!(destination[key], source[key])
    end
    return destination
end

function backend_copyto!(destination::GenericAutoDiffCache, source::GenericAutoDiffCache)
    backend_copyto!(destination.entries, source.entries)
    return destination
end

function backend_copyto!(destination::CompactAutoDiffCache, source::CompactAutoDiffCache)
    backend_copyto!(destination.entries, source.entries)
    return destination
end

function backend_copyto!(destination::ConservationLawTPFAStorage,
        source::ConservationLawTPFAStorage)
    backend_copyto!(destination.accumulation, source.accumulation)
    backend_copyto!(destination.half_face_flux_cells, source.half_face_flux_cells)
    backend_copyto!(destination.half_face_flux_faces, source.half_face_flux_faces)
    if !isnothing(destination.source) && !isnothing(source.source)
        backend_copyto!(destination.source, source.source)
    end
    return destination
end

function adapt_simulation_storage(ctx::KernelAbstractionsContext, storage_cpu,
        model, lsys = nothing)
    function adapt_variable_definitions(definitions)
        adapted = adapt_backend_value(ctx, definitions)
        plan = secondary_variable_evaluation_plan(
            model, adapted.secondary_variables)
        contents = data(adapted)
        if contents isa NamedTuple
            contents = merge(contents,
                (secondary_variable_evaluation_plan = plan,))
        else
            contents = copy(contents)
            contents[:secondary_variable_evaluation_plan] = plan
        end
        return JutulStorage(contents)
    end

    function state_references(state, definitions)
        return (; (key => state[key] for key in keys(definitions))...)
    end

    state = adapt_backend_value(ctx, storage_cpu.state)
    state0 = adapt_backend_value(ctx, storage_cpu.state0)
    equations = adapt_backend_value(ctx, storage_cpu.equations)
    variable_definitions = adapt_variable_definitions(
        storage_cpu.variable_definitions)

    converted = OrderedDict{Symbol, Any}()
    for (key, value) in pairs(data(storage_cpu))
        if key in (:state, :state0, :LinearizedSystem, :equations,
                   :variable_definitions, :primary_variables, :parameters, :views)
            continue
        end
        converted[key] = adapt_backend_value(ctx, value)
    end
    converted[:state] = state
    converted[:state0] = state0
    converted[:equations] = equations
    converted[:variable_definitions] = variable_definitions
    converted[:primary_variables] = state_references(
        state, variable_definitions.primary_variables)
    converted[:parameters] = state_references(
        state, variable_definitions.parameters)
    if !isnothing(lsys)
        converted[:LinearizedSystem] = lsys
    end
    return JutulStorage(converted)
end

"""
    transfer_to_backend(simulator, backend; kwarg...)
    transfer_to_backend(simulator, context::KernelAbstractionsContext)

Adapt a fully initialized CPU simulator to a KernelAbstractions backend.
[`SimulationModel`](@ref) and [`MultiModel`](@ref) are supported. Sparsity
discovery and Jacobian/cross-term alignment finish on the CPU before the CSR
arrays are moved. Array aliases used by primary variables, parameters,
residual views and Jacobian buffers are rebuilt against the adapted root
arrays. Submodels marked [`AssembleOnDevice`](@ref) retain their CPU model and
storage and copy into preallocated backend mirrors after evaluation.
"""
function transfer_to_backend(sim::Simulator, backend;
        group_execution = missing, kwarg...)
    model = sim.model
    model isa Union{SimulationModel, MultiModel} || throw(ArgumentError(
        "KernelAbstractions transfer supports SimulationModel and MultiModel simulators"))
    ctx = KernelAbstractionsContext(backend;
        float_type = float_type(model.context),
        index_type = index_type(model.context),
        matrix_layout = matrix_layout(model.context),
        kwarg...)
    return transfer_to_backend(sim, ctx; group_execution = group_execution)
end

Base.@noinline function transfer_to_backend(sim::Simulator,
        ctx::KernelAbstractionsContext;
        group_execution = missing)
    Base.@nospecialize sim
    if !ismissing(group_execution)
        sim = set_transfer_group_execution(sim, group_execution)
    end
    model_cpu = sim.model
    model_cpu isa SimulationModel || return transfer_multimodel_to_backend(sim, ctx)
    storage_cpu = sim.storage
    model = Adapt.adapt(ctx, model_cpu)

    lsys = Adapt.adapt(ctx, storage_cpu.LinearizedSystem)
    storage = adapt_simulation_storage(ctx, storage_cpu, model, lsys)
    data(storage)[:views] = setup_equations_and_primary_variable_views(
        storage, model, lsys.r_buffer, lsys.dx_buffer
    )
    storage = convert_to_immutable_storage(storage)
    synchronize(ctx)
    return Simulator(sim.executor, model, storage)
end

Base.@noinline function set_transfer_group_execution(sim::Simulator, policy)
    Base.@nospecialize sim policy
    function execution_for_submodel(key, submodel)
        mode = if policy isa Function
            policy(key, submodel)
        elseif policy isa AbstractDict || policy isa NamedTuple
            get(policy, key, get(policy, :default, SolveFullyOnDevice))
        else
            throw(ArgumentError(
                "group_execution must be a function or keyed collection"))
        end
        mode isa DeviceExecutionMode || throw(ArgumentError(
            "Execution policy for $key must be a DeviceExecutionMode, " *
            "got $(typeof(mode))"))
        return mode
    end

    model = sim.model
    model isa MultiModel || throw(ArgumentError(
        "Per-model execution policies require a MultiModel simulator"))
    keys_m = collect(submodels_symbols(model))
    modes_by_key = map(keys_m) do key
        execution_for_submodel(key, model[key])
    end
    rebuilt = MultiModel(model.models, multimodel_label(model);
        cross_terms = model.cross_terms,
        groups = isnothing(model.groups) ? nothing : copy(model.groups),
        group_execution = modes_by_key,
        context = model.context,
        reduction = model.reduction,
        specialize = false,
        specialize_ad = model.specialize_ad)
    return Simulator(sim.executor, rebuilt, sim.storage)
end

function transfer_multimodel_to_backend(sim::Simulator,
        ctx::KernelAbstractionsContext)
    model_cpu = sim.model
    model_cpu isa MultiModel || throw(ArgumentError(
        "KernelAbstractions transfer supports SimulationModel and MultiModel simulators"))
    storage_cpu = sim.storage

    modes = model_cpu.group_execution
    all(==(NothingOnDevice), modes) && return sim
    any(==(NothingOnDevice), modes) && throw(ArgumentError(
        "Backend transfer cannot combine all-NothingOnDevice groups with device-resident groups"))
    storage_cpu = prepare_backend_transfer!(storage_cpu, model_cpu)
    host_keys = tuple((key for key in submodels_symbols(model_cpu)
        if group_execution_mode(model_cpu, key) == AssembleOnDevice)...)
    for key in host_keys
        prepare_backend_transfer!(storage_cpu[key], model_cpu[key])
    end

    # Multimodel applications often use CSC for their tiny well/facility
    # groups even when the reservoir already uses CSR. Rebuild only the
    # sparsity/alignment copy on the CPU so every backend block has static CSR
    # ordering before any arrays are transferred.
    model_setup = cpu_csr_model(model_cpu)
    setup_data = OrderedDict{Symbol, Any}(pairs(data(deepcopy(storage_cpu))))
    storage_setup = JutulStorage(setup_data)
    setup_linearized_system!(storage_setup, model_setup)
    align_equations_to_linearized_system!(storage_setup, model_setup)
    align_cross_terms_to_linearized_system!(storage_setup, model_setup)

    model = Adapt.adapt(ctx, model_setup)
    lsys = Adapt.adapt(ctx, storage_setup.LinearizedSystem)
    converted = OrderedDict{Symbol, Any}()

    ignored = (
        :state, :state0, :LinearizedSystem, :cross_terms,
        :cross_term_targets_no_symmetry, :cross_term_targets_with_symmetry,
        :multi_model_maps, :eq_maps
    )
    model_keys = submodels_symbols(model_cpu)
    for (key, value) in pairs(data(storage_setup))
        if key in model_keys || key in ignored
            continue
        end
        converted[key] = adapt_backend_value(ctx, value)
    end

    state = JutulStorage()
    state0 = JutulStorage()
    for key in model_keys
        submodel = model[key]
        subcontext = submodel.context
        substorage = adapt_simulation_storage(
            subcontext, storage_setup[key], submodel)
        converted[key] = substorage
        state[key] = substorage.state
        state0[key] = substorage.state0
    end
    converted[:state] = state
    converted[:state0] = state0
    converted[:LinearizedSystem] = lsys
    converted[:cross_terms] = [adapt_backend_value(ctx, ct_s)
        for ct_s in storage_setup.cross_terms]
    if !isempty(host_keys)
        converted[:host_evaluation] = HostEvaluationStorage(
            model_cpu, storage_cpu, host_keys)
    end

    storage = JutulStorage(converted)
    setup_multimodel_maps!(storage, model)
    setup_equations_and_primary_variable_views!(storage, model, lsys)
    # Keep the outer multimodel storage dynamic: specializing it would encode
    # every submodel name and every well/cross-term storage type in one giant
    # compiler type. Individual submodel fields remain immutable and concrete
    # when dispatched to their kernels.
    storage = specialize_simulator_storage(storage, model, false)
    synchronize(ctx)
    return Simulator(sim.executor, model, storage)
end

function transfer_adjoint_simulator(simulator,
        execution_model::KASimulationModel)
    return transfer_to_backend(simulator, adjoint(execution_model.context))
end

function transfer_adjoint_simulator(simulator, execution_model::MultiModel)
    context = execution_model.context
    context isa KernelAbstractionsContext || return simulator
    policy = Dict{Symbol, DeviceExecutionMode}()
    for key in submodels_symbols(execution_model)
        policy[key] = group_execution_mode(execution_model, key)
    end
    return transfer_to_backend(simulator, adjoint(context);
        group_execution = policy)
end

@kernel function adjoint_block_order_kernel!(destination, source, n, bz,
        to_canonical)
    index = @index(Global)
    if index <= n*bz
        block = (index - 1) ÷ n + 1
        entity = (index - 1) % n + 1
        block_major = (entity - 1)*bz + block
        if to_canonical
            @inbounds destination[index] = source[block_major]
        else
            @inbounds destination[block_major] = source[index]
        end
    end
end

function adjoint_transfer_canonical_order_inner!(destination::AbstractArray,
        source::AbstractArray, model::KASimulationModel, ::BlockMajorLayout,
        to_canonical)
    bz = 0
    for entity in get_primary_variable_ordered_entities(model)
        bz == 0 || error("Assumed that block major has a single entity group")
        bz = degrees_of_freedom_per_entity(model, entity)::Int
    end
    n = length(source) ÷ bz
    context = model.context
    kernel! = adjoint_block_order_kernel!(context.backend)
    event = kernel!(destination, source, n, bz, to_canonical;
        ndrange = length(source))
    isnothing(event) || wait(event)
    return destination
end

@kernel function ka_csr_mul_kernel!(y, nzval, colval, rowptr, x, alpha, beta)
    row = @index(Global)
    value_row = zero(eltype(y))
    @inbounds for pos in rowptr[row]:(rowptr[row + 1] - 1)
        value_row += nzval[pos]*x[colval[pos]]
    end
    @inbounds y[row] = alpha*value_row + beta*y[row]
end

function LinearAlgebra.mul!(y::AbstractVector,
        A::StaticSparsityMatrixCSR{Tv, Ti, V, I, R, B},
        x::AbstractVector, alpha::Number, beta::Number) where {
            Tv, Ti<:Integer, V, I, R, B<:KernelAbstractions.Backend}
    kernel! = ka_csr_mul_kernel!(A.backend)
    event = kernel!(y, A.nzval, A.colval, A.rowptr, x, alpha, beta; ndrange = size(A, 1))
    isnothing(event) || wait(event)
    return y
end

mutable struct HostBackendFactorization{F, V}
    factorization::F
    right_hand_side::V
    solution::V
end

function host_backend_factorization(matrix::StaticSparsityMatrixCSR)
    host_matrix = KAPreconditioners.sparse_matrix(matrix)
    factorization = lu(host_matrix)
    scalar_type = eltype(host_matrix)
    right_hand_side = Vector{scalar_type}(undef, size(host_matrix, 1))
    solution = similar(right_hand_side)
    return HostBackendFactorization(
        factorization, right_hand_side, solution)
end

function factorize_linear_system(constructor,
        matrix::StaticSparsityMatrixCSR{
            Tv, Ti, V, I, R, B}) where {
            Tv, Ti<:Integer, V, I, R, B<:KernelAbstractions.Backend}
    return host_backend_factorization(matrix)
end

function refactorize_linear_system!(update!,
        factorization::HostBackendFactorization,
        matrix::StaticSparsityMatrixCSR{
            Tv, Ti, V, I, R, B}) where {
            Tv, Ti<:Integer, V, I, R, B<:KernelAbstractions.Backend}
    factorization.factorization = lu(KAPreconditioners.sparse_matrix(matrix))
    return factorization
end

function transfer_csr_to_backend(
        reference::StaticSparsityMatrixCSR{
            Tv, Ti, V, I, R, B},
        matrix::StaticSparsityMatrixCSR) where {
            Tv, Ti<:Integer, V, I, R, B<:KernelAbstractions.Backend}
    values = similar(reference.nzval, eltype(matrix.nzval), length(matrix.nzval))
    columns = similar(reference.colval, eltype(matrix.colval), length(matrix.colval))
    rows = similar(reference.rowptr, eltype(matrix.rowptr), length(matrix.rowptr))
    copyto!(values, matrix.nzval)
    copyto!(columns, matrix.colval)
    copyto!(rows, matrix.rowptr)
    return StaticSparsityMatrixCSR(
        values, columns, rows, size(matrix, 1), size(matrix, 2),
        reference.backend; nthreads = reference.nthreads,
        minbatch = reference.minbatch, thread_type = reference.thread_type)
end

function LinearAlgebra.ldiv!(output::AbstractVector,
        factorization::HostBackendFactorization,
        right_hand_side::AbstractVector)
    copyto!(factorization.right_hand_side, right_hand_side)
    ldiv!(factorization.solution, factorization.factorization,
        factorization.right_hand_side)
    copyto!(output, factorization.solution)
    return output
end

# Initial implementation keeps the direct factorization on the CPU. Assembly,
# linearization, convergence and variable/property loops stay on the selected
# backend; only the solve vectors and CSR values cross the boundary.
function linear_solve!(sys::LinearizedSystem{<:Any, <:StaticSparsityMatrixCSR},
        ::Nothing, ctx::KernelAbstractionsContext, arg...;
        dx = sys.dx, r = sys.r_buffer, kwarg...)
    A = sys.jac
    nz = Adapt.adapt(Array, nonzeros(A))
    cols = Adapt.adapt(Array, colvals(A))
    rows = Adapt.adapt(Array, A.rowptr)
    At = SparseMatrixCSC(size(A, 2), size(A, 1), rows, cols, nz)
    # Sparse solves can return a SparseVector for small systems. Normalize the
    # result before copying to a device array so copyto! uses bulk transfer.
    host_dx = collect(-(At' \ Adapt.adapt(Array, r)))
    copyto!(dx, host_dx)
    synchronize(ctx)
    return linear_solve_return()
end
