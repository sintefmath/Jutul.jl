export transfer_to_backend

const KASimulationModel = SimulationModel{<:Any, <:Any, <:Any, <:KernelAbstractionsContext}

# Adapt uses the context as the adaptation target. Backend packages define how
# their own backend converts an Array, while Jutul supplies the structural rules.
Adapt.adapt_storage(ctx::KernelAbstractionsContext, a::AbstractArray) = Adapt.adapt(ctx.backend, a)
Adapt.adapt_storage(::KernelAbstractionsContext, a::AbstractArray{Symbol}) = Tuple(a)
transfer(ctx::KernelAbstractionsContext, x) = Adapt.adapt(ctx, x)
backend_to_host(::KernelAbstractionsContext, x) = Adapt.adapt(Array, x)

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
