export transfer_to_backend, backend_copyto!, prepare_backend_transfer!

const KASimulationModel = SimulationModel{<:Any, <:Any, <:Any, <:KernelAbstractionsContext}

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
    jac_buffer = _backend_jacobian_buffer(lsys.jac_buffer, jac)
    return LinearizedSystem(jac, r, dx, jac_buffer, r_buffer, dx_buffer, lsys.matrix_layout)
end

function _backend_jacobian_buffer(original_buffer, jac)
    nz = nonzeros(jac)
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

function _backend_vector_alias(original, original_buffer, adapted_buffer)
    buffer = reshape(adapted_buffer, size(original_buffer))
    if eltype(original) == eltype(original_buffer)
        return reshape(buffer, size(original))
    else
        return reinterpret(reshape, eltype(original), buffer)
    end
end

function Adapt.adapt_structure(ctx::KernelAbstractionsContext, lsys::MultiLinearizedSystem)
    r_buffer = Adapt.adapt(ctx, lsys.r_buffer)
    dx_buffer = Adapt.adapt(ctx, lsys.dx_buffer)
    r = _backend_vector_alias(lsys.r, lsys.r_buffer, r_buffer)
    dx = _backend_vector_alias(lsys.dx, lsys.dx_buffer, dx_buffer)

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
            r_i = _backend_vector_alias(old.r, old.r_buffer, r_i_buffer)
            dx_i = _backend_vector_alias(old.dx, old.dx_buffer, dx_i_buffer)
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
    schur_buffer = _adapt_backend_value(ctx, lsys.schur_buffer)
    return MultiLinearizedSystem{typeof(lsys.matrix_layout)}(
        subsystems, r, dx, r_buffer, dx_buffer, lsys.reduction,
        FactorStore(), lsys.matrix_layout, schur_buffer
    )
end

_adapt_backend_value(ctx, x::NamedTuple) = map(v -> _adapt_backend_value(ctx, v), x)
_adapt_backend_value(ctx, x::Tuple) = map(v -> _adapt_backend_value(ctx, v), x)
function _adapt_backend_value(ctx, x::AbstractDict)
    return (; (Symbol(k) => _adapt_backend_value(ctx, v) for (k, v) in pairs(x))...)
end
_adapt_backend_value(ctx, x::JutulStorage) = JutulStorage(_adapt_backend_value(ctx, data(x)))
_adapt_backend_value(ctx, x::AbstractVector{<:JutulStorage}) =
    tuple((_adapt_backend_value(ctx, v) for v in x)...)
_adapt_backend_value(ctx, x) = Adapt.adapt(ctx, x)

function Adapt.adapt_structure(ctx::KernelAbstractionsContext, model::SimulationModel)
    primary = _adapt_backend_value(ctx, model.primary_variables)
    secondary = _adapt_backend_value(ctx, model.secondary_variables)
    parameters = _adapt_backend_value(ctx, model.parameters)
    equations = _adapt_backend_value(ctx, model.equations)
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

function _backend_subcontext(ctx::KernelAbstractionsContext, model::SimulationModel)
    source = model.context
    return KernelAbstractionsContext(ctx.backend;
        float_type = float_type(source),
        index_type = index_type(source),
        matrix_layout = matrix_layout(source),
        workgroupsize = ctx.workgroupsize
    )
end

function _cpu_csr_model(model::SimulationModel)
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

function _cpu_csr_model(model::MultiModel)
    models = (; (key => _cpu_csr_model(submodel)
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
    models = (; (key => Adapt.adapt(_backend_subcontext(ctx, submodel), submodel)
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
prepare_backend_transfer!(storage, model) = storage

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

backend_copyto!(destination, source) = destination

function _state_references(state, definitions)
    return (; (k => state[k] for k in keys(definitions))...)
end

function _adapt_simulation_storage(ctx::KernelAbstractionsContext, storage_cpu,
        model, lsys = nothing)
    state = _adapt_backend_value(ctx, storage_cpu.state)
    state0 = _adapt_backend_value(ctx, storage_cpu.state0)
    equations = _adapt_backend_value(ctx, storage_cpu.equations)
    variable_definitions = _adapt_backend_value(ctx, storage_cpu.variable_definitions)

    converted = OrderedDict{Symbol, Any}()
    for (key, value) in pairs(data(storage_cpu))
        if key in (:state, :state0, :LinearizedSystem, :equations,
                   :variable_definitions, :primary_variables, :parameters, :views)
            continue
        end
        converted[key] = _adapt_backend_value(ctx, value)
    end
    converted[:state] = state
    converted[:state0] = state0
    converted[:equations] = equations
    converted[:variable_definitions] = variable_definitions
    converted[:primary_variables] = _state_references(
        state, variable_definitions.primary_variables)
    converted[:parameters] = _state_references(state, variable_definitions.parameters)
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
arrays. Groups marked [`AssembleOnDevice`](@ref) retain their CPU model and
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
        sim = _set_transfer_group_execution(sim, group_execution)
    end
    model_cpu = sim.model
    model_cpu isa SimulationModel || return _transfer_multimodel_to_backend(sim, ctx)
    storage_cpu = sim.storage
    model = Adapt.adapt(ctx, model_cpu)

    lsys = Adapt.adapt(ctx, storage_cpu.LinearizedSystem)
    storage = _adapt_simulation_storage(ctx, storage_cpu, model, lsys)
    data(storage)[:views] = setup_equations_and_primary_variable_views(
        storage, model, lsys.r_buffer, lsys.dx_buffer
    )
    storage = convert_to_immutable_storage(storage)
    synchronize(ctx)
    return Simulator(sim.executor, model, storage)
end

function _execution_for_submodel(policy, key, model)
    mode = if policy isa Function
        policy(key, model)
    elseif policy isa AbstractDict || policy isa NamedTuple
        get(policy, key, get(policy, :default, SolveFullyOnDevice))
    else
        throw(ArgumentError("group_execution must be a function or keyed collection"))
    end
    mode isa DeviceExecutionMode || throw(ArgumentError(
        "Execution policy for $key must be a DeviceExecutionMode, got $(typeof(mode))"))
    return mode
end

Base.@noinline function _set_transfer_group_execution(sim::Simulator, policy)
    Base.@nospecialize sim policy
    model = sim.model
    model isa MultiModel || throw(ArgumentError(
        "Per-group execution policies require a MultiModel simulator"))
    keys_m = collect(submodels_symbols(model))
    old_groups = isnothing(model.groups) ? ones(Int, length(keys_m)) : model.groups
    modes_by_key = map(keys_m) do key
        _execution_for_submodel(policy, key, model[key])
    end

    # Split existing groups only when their members have different execution
    # policies. Equal (old group, policy) pairs continue to share one block.
    pairs = Tuple{Int, DeviceExecutionMode}[]
    groups = Vector{Int}(undef, length(keys_m))
    for i in eachindex(keys_m)
        pair = (old_groups[i], modes_by_key[i])
        group = findfirst(isequal(pair), pairs)
        if isnothing(group)
            push!(pairs, pair)
            group = length(pairs)
        end
        groups[i] = group
    end
    modes = last.(pairs)
    rebuilt = MultiModel(model.models, multimodel_label(model);
        cross_terms = model.cross_terms,
        groups = groups,
        group_execution = modes,
        context = model.context,
        reduction = model.reduction,
        specialize = false,
        specialize_ad = model.specialize_ad)
    return Simulator(sim.executor, rebuilt, sim.storage)
end

function _transfer_multimodel_to_backend(sim::Simulator,
        ctx::KernelAbstractionsContext)
    model_cpu = sim.model
    model_cpu isa MultiModel || throw(ArgumentError(
        "KernelAbstractions transfer supports SimulationModel and MultiModel simulators"))
    storage_cpu = sim.storage

    modes = model_cpu.group_execution
    all(==(NothingOnDevice), modes) && return sim
    any(==(NothingOnDevice), modes) && throw(ArgumentError(
        "Mixed NothingOnDevice groups are not supported by backend transfer"))
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
    model_setup = _cpu_csr_model(model_cpu)
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
        converted[key] = _adapt_backend_value(ctx, value)
    end

    state = JutulStorage()
    state0 = JutulStorage()
    for key in model_keys
        submodel = model[key]
        subcontext = submodel.context
        substorage = _adapt_simulation_storage(
            subcontext, storage_setup[key], submodel)
        converted[key] = substorage
        state[key] = substorage.state
        state0[key] = substorage.state0
    end
    converted[:state] = state
    converted[:state0] = state0
    converted[:LinearizedSystem] = lsys
    converted[:cross_terms] = [_adapt_backend_value(ctx, ct_s)
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

@kernel function _ka_csr_mul_kernel!(y, nzval, colval, rowptr, x, alpha, beta)
    row = @index(Global)
    value_row = zero(eltype(y))
    @inbounds for pos in rowptr[row]:(rowptr[row + 1] - 1)
        value_row += nzval[pos]*x[colval[pos]]
    end
    @inbounds y[row] = alpha*value_row + beta*y[row]
end

function LinearAlgebra.mul!(y::AbstractVector,
        A::StaticSparsityMatrixCSR{Tv, Ti, V, I, R, Nothing, B},
        x::AbstractVector, alpha::Number, beta::Number) where {Tv, Ti, V, I, R, B<:KernelAbstractions.Backend}
    kernel! = _ka_csr_mul_kernel!(A.backend)
    event = kernel!(y, A.nzval, A.colval, A.rowptr, x, alpha, beta; ndrange = size(A, 1))
    isnothing(event) || wait(event)
    return y
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
