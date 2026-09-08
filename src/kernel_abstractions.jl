export transfer_to_backend

const KASimulationModel = SimulationModel{<:Any, <:Any, <:Any, <:KernelAbstractionsContext}

# Adapt uses the context as the adaptation target. Backend packages define how
# their own backend converts an Array, while Jutul supplies the structural rules.
Adapt.adapt_storage(ctx::KernelAbstractionsContext, a::AbstractArray) = Adapt.adapt(ctx.backend, a)
Adapt.adapt_storage(::KernelAbstractionsContext, a::AbstractArray{Symbol}) = Tuple(a)
transfer(ctx::KernelAbstractionsContext, x) = Adapt.adapt(ctx, x)
backend_allocate(ctx::KernelAbstractionsContext, T, dims...) = KernelAbstractions.allocate(ctx.backend, T, dims...)
backend_to_host(x) = Adapt.adapt(Array, x)

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
    return LinearizedSystem(jac, r, dx, nonzeros(jac), r_buffer, dx_buffer, lsys.matrix_layout)
end

_adapt_backend_value(ctx, x::NamedTuple) = map(v -> _adapt_backend_value(ctx, v), x)
function _adapt_backend_value(ctx, x::AbstractDict)
    return (; (Symbol(k) => _adapt_backend_value(ctx, v) for (k, v) in pairs(x))...)
end
_adapt_backend_value(ctx, x::JutulStorage) = JutulStorage(_adapt_backend_value(ctx, data(x)))
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
    return SimulationModel(
        Adapt.adapt(to, model.domain),
        Adapt.adapt(to, model.system),
        Adapt.adapt(to, model.context),
        Adapt.adapt(to, model.formulation),
        missing,
        Adapt.adapt(to, model.primary_variables),
        Adapt.adapt(to, model.secondary_variables),
        Adapt.adapt(to, model.parameters),
        Adapt.adapt(to, model.equations),
        model.output_variables,
        nothing,
        model.optimization_level
    )
end

function _state_references(state, definitions)
    return (; (k => state[k] for k in keys(definitions))...)
end

"""
    transfer_to_backend(simulator, backend; kwarg...)
    transfer_to_backend(simulator, context::KernelAbstractionsContext)

Adapt a fully initialized, single-model CPU simulator to a
KernelAbstractions backend. The CPU simulator must use `ParallelCSRContext` so
that sparsity discovery and Jacobian alignment finish before the CSR arrays are
moved. Array aliases used by primary variables, parameters, residual views and
the Jacobian buffer are rebuilt against the adapted root arrays.
"""
function transfer_to_backend(sim::Simulator, backend; kwarg...)
    model = sim.model
    model isa SimulationModel || throw(ArgumentError("Only single SimulationModel transfer is supported"))
    ctx = KernelAbstractionsContext(backend;
        float_type = float_type(model.context),
        index_type = index_type(model.context),
        matrix_layout = matrix_layout(model.context),
        kwarg...)
    return transfer_to_backend(sim, ctx)
end

function transfer_to_backend(sim::Simulator, ctx::KernelAbstractionsContext)
    model_cpu = sim.model
    model_cpu isa SimulationModel || throw(ArgumentError("Only single SimulationModel transfer is supported"))
    storage_cpu = sim.storage
    model = Adapt.adapt(ctx, model_cpu)

    state = _adapt_backend_value(ctx, storage_cpu.state)
    state0 = _adapt_backend_value(ctx, storage_cpu.state0)
    lsys = Adapt.adapt(ctx, storage_cpu.LinearizedSystem)
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
    converted[:LinearizedSystem] = lsys
    converted[:equations] = equations
    converted[:variable_definitions] = variable_definitions
    converted[:primary_variables] = _state_references(state, variable_definitions.primary_variables)
    converted[:parameters] = _state_references(state, variable_definitions.parameters)

    storage = JutulStorage(converted)
    converted[:views] = setup_equations_and_primary_variable_views(
        storage, model, lsys.r_buffer, lsys.dx_buffer
    )
    storage = convert_to_immutable_storage(storage)
    synchronize(ctx)
    return Simulator(sim.executor, model, storage)
end

# Secondary variables are processed in dependency order on the host. Each
# property evaluation itself is a backend kernel over its entities.
function update_secondary_variables_state!(state, model::KASimulationModel,
        vars = model.secondary_variables)
    for (symbol, var) in pairs(vars)
        target = state[symbol]
        n = number_of_entities(model, var)
        f(i) = update_secondary_variable!(target, var, model, state, i:i)
        threaded_loop(f, n, model.context)
    end
    return state
end

# The hard-coded TPFA path normally creates a CPU pointer reinterpretation of
# its flux matrix. On a backend, write its components explicitly instead.
function update_half_face_flux!(eq_s::ConservationLawTPFAStorage,
        law::ConservationLaw, storage, model::KASimulationModel, dt)
    flow_disc = law.flow_discretization
    flux = get_entries(eq_s.half_face_flux_cells)
    state = local_ad(storage.state, 1, eltype(flux))
    conn_pos = flow_disc.conn_pos
    conn_data = flow_disc.conn_data
    gmap = global_map(model.domain)
    ncomponents = size(flux, 1)
    nc = length(conn_pos) - 1
    function f(c)
        self = full_cell(c, gmap)
        local_state = new_entity_index(state, self)
        first = @inbounds conn_pos[c]
        last = @inbounds conn_pos[c + 1] - 1
        for i in first:last
            connection = @inbounds conn_data[i]
            value_i = face_flux!(
                zero(flux_vector_type(law, Val(eltype(flux)))),
                connection.self, connection.other, connection.face,
                connection.face_sign, law, local_state, model, dt, flow_disc
            )
            for component in 1:ncomponents
                @inbounds flux[component, i] = value_i[component]
            end
        end
    end
    threaded_loop(f, nc, model.context)
    isnothing(eq_s.half_face_flux_faces) || throw(ArgumentError(
        "KernelAbstractions TPFA currently supports cell primary variables only"
    ))
    return flux
end

function update_accumulation!(eq_s::ConservationLawTPFAStorage,
        law::ConservationLaw, storage, model::KASimulationModel, dt)
    conserved = eq_s.accumulation_symbol
    acc = get_entries(eq_s.accumulation)
    m0, m = state_pair(storage, conserved, model)
    ncomponents, nc = size(acc)
    function f(c)
        for component in 1:ncomponents
            @inbounds acc[component, c] = (m[component, c] - m0[component, c])/dt
        end
    end
    if m isa AbstractVector
        f_scalar(c) = (@inbounds acc[1, c] = (m[c] - m0[c])/dt)
        threaded_loop(f_scalar, nc, model.context)
    else
        threaded_loop(f, nc, model.context)
    end
    return acc
end

# Context-aware primary update avoids scalar iteration by host code.
function update_primary_variable_context!(state, p::JutulVariables, state_symbol,
        model, dx, w, ctx::KernelAbstractionsContext)
    active = active_entities(model.domain, associated_entity(p), for_variables = true)
    values = state[state_symbol]
    abs_max = absolute_increment_limit(p)
    rel_max = relative_increment_limit(p)
    maxval = maximum_value(p)
    minval = minimum_value(p)
    scale = variable_scale(p)
    if values isa AbstractVector
        update_vector(i) = (@inbounds values[active[i]] = update_value(
            values[active[i]], w*dx[i], abs_max, rel_max, minval, maxval, scale))
        threaded_loop(update_vector, length(active), ctx)
    else
        nvalues = size(values, 1)
        function update_matrix(i)
            a = @inbounds active[i]
            for component in 1:nvalues
                @inbounds values[component, a] = update_value(
                    values[component, a], w*dx[component, i],
                    abs_max, rel_max, minval, maxval, scale)
            end
        end
        threaded_loop(update_matrix, length(active), ctx)
    end
    return values
end

function update_primary_variable_context!(state, p::FractionVariables, state_symbol,
        model, dx, w, ctx::KernelAbstractionsContext)
    fractions = state[state_symbol]
    nf, nu = value_dim(model, p)
    abs_max = absolute_increment_limit(p)
    maxval = maximum_value(p)
    minval = minimum_value(p)
    maxval -= nf*minval
    active = active_entities(model.domain, associated_entity(p), for_variables = true)
    if nf == 2
        pair_max = min(1 - minval, maxval)
        pair_min = max(minval, pair_max - 1)
        function update_pair(i)
            @inbounds cell = active[i]
            @inbounds v = value(fractions[1, cell])
            @inbounds dv = dx[i]
            dv = w*choose_increment(v, dv, abs_max, nothing, pair_min, pair_max)
            @inbounds fractions[1, cell] += dv
            @inbounds fractions[2, cell] -= dv
        end
        threaded_loop(update_pair, length(active), ctx)
    elseif unit_update_preserve_direction(p)
        function update_direction(i)
            @inbounds cell = active[i]
            unit_update_direction_local!(
                fractions, i, cell, dx, nf, nu, minval, maxval, abs_max, w)
        end
        threaded_loop(update_direction, length(active), ctx)
    else
        function update_magnitude(i)
            @inbounds cell = active[i]
            unit_update_magnitude_local!(
                fractions, i, cell, dx, nf, nu, minval, maxval, abs_max)
        end
        threaded_loop(update_magnitude, length(active), ctx)
    end
    return fractions
end

@inline _backend_replacement(old::ForwardDiff.Dual, new::Real) = old - value(old) + value(new)
@inline _backend_replacement(old::AbstractFloat, new::ForwardDiff.Dual) = value(new)
@inline _backend_replacement(old, new) = new

function update_values!(dest::AbstractArray, src::AbstractArray,
        ctx::KernelAbstractionsContext)
    f(i) = (@inbounds dest[i] = _backend_replacement(dest[i], src[i]))
    threaded_loop(f, length(dest), ctx)
    return dest
end

function increment_norm(dX, state, model::KASimulationModel, X, pvar)
    T = typeof(value(zero(eltype(dX))))
    out = backend_allocate(model.context, T, 2)
    function reduce_increment(_)
        sum_v = zero(T)
        max_v = zero(T)
        for i in 1:length(dX)
            @inbounds dx_abs = abs(value(dX[i]))
            sum_v += dx_abs
            max_v = max(max_v, dx_abs)
        end
        @inbounds out[1] = sum_v
        @inbounds out[2] = max_v
    end
    threaded_loop(reduce_increment, 1, model.context)
    host = backend_to_host(out)
    scale = @something variable_scale(pvar) one(T)
    return (sum = scale*host[1], max = scale*host[2])
end

function variable_change_report(X::AbstractArray, X0::AbstractArray{T}, pvar,
        ctx::KernelAbstractionsContext) where T<:Real
    out = backend_allocate(ctx, T, 4)
    function reduce_change(_)
        max_dv = max_v = sum_dv = sum_v = zero(T)
        for i in 1:length(X)
            @inbounds x = value(X[i])::T
            @inbounds dx = x - value(X0[i])
            dx_abs = abs(dx)
            max_dv = max(max_dv, dx_abs)
            sum_dv += dx_abs
            x_abs = abs(x)
            max_v = max(max_v, x_abs)
            sum_v += x_abs
        end
        @inbounds out[1] = sum_dv
        @inbounds out[2] = max_dv
        @inbounds out[3] = sum_v
        @inbounds out[4] = max_v
    end
    threaded_loop(reduce_change, 1, ctx)
    host = backend_to_host(out)
    return (dx = (sum = host[1], max = host[2]),
            x = (sum = host[3], max = host[4]), n = length(X))
end

variable_change_report(X, X0, pvar, ::KernelAbstractionsContext) = nothing

function backend_maximum_value(ctx::KernelAbstractionsContext, x)
    T = typeof(value(zero(eltype(x))))
    out = backend_allocate(ctx, T, 1)
    function reduce_maximum(_)
        current = value(x[1])
        for i in 2:length(x)
            @inbounds current = max(current, value(x[i]))
        end
        @inbounds out[1] = current
    end
    threaded_loop(reduce_maximum, 1, ctx)
    return only(backend_to_host(out))
end

function convergence_criterion(model::KASimulationModel, storage,
        eq::JutulEquation, eq_s, r; dt = 1.0, update_report = missing)
    ncomponents = number_of_equations_per_entity(model, eq)
    nentities = size(r, 2)
    out = KernelAbstractions.allocate(model.context.backend, eltype(r), ncomponents)
    function f(component)
        current = zero(eltype(r))
        for entity in 1:nentities
            @inbounds current = max(current, abs(r[component, entity]))
        end
        @inbounds out[component] = current
    end
    threaded_loop(f, ncomponents, model.context)
    errors = vec(Adapt.adapt(Array, out))
    names = ncomponents == 1 ? "R" : map(i -> "R_$i", 1:ncomponents)
    return (AbsMax = (errors = errors, names = names), )
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
    host_dx = -(At' \ Adapt.adapt(Array, r))
    copyto!(dx, host_dx)
    synchronize(ctx)
    return linear_solve_return()
end
