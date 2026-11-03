# Keep vector and matrix permutations outside the existing natural-order
# kernels, so the block ILU recurrence and triangular dependencies stay intact.
mutable struct MulticolorILUState{S, M, P, V, RP, CV, H, W, C, B} <: AbstractSmootherState
    inner::S
    matrix::M
    permutation::P
    value_positions::V
    rowptr::RP
    colval::CV
    host_rowptr::H
    host_colval::H
    work::W
    residual::W
    permuted_rhs::W
    permuted_x::W
    config::C
    backend::B
    block_size::Int
    n::Int
    apply_cache::Any
end

# Initialize the backend-specific launch cache lazily.
MulticolorILUState(inner, matrix, permutation, value_positions, rowptr, colval,
    host_rowptr, host_colval, work, residual, permuted_rhs, permuted_x, config,
    backend, block_size, n) = MulticolorILUState(inner, matrix, permutation,
        value_positions, rowptr, colval, host_rowptr, host_colval, work, residual,
        permuted_rhs, permuted_x, config, backend, block_size, n, nothing)

Base.size(state::MulticolorILUState) = (state.n, state.n)
Base.eltype(state::MulticolorILUState) = eltype(state.inner)
smoother_damping(state::MulticolorILUState) = smoother_damping(state.inner)

function ilu_undirected_graph(rowptr, colval, n)
    # CSR viewed as CSC is the transposed graph. Include both directions:
    # a one-sided edge must also prevent simultaneous dependent row updates.
    graph = SparseMatrixCSC(n, n, Int.(rowptr), Int.(colval), ones(Int8, length(colval)))
    return graph + transpose(graph)
end

function multicolor_ilu_permutation(rowptr, colval, n)
    graph = ilu_undirected_graph(rowptr, colval, n)
    colors = zeros(Int, n)
    used = zeros(Int, n + 1)
    @inbounds for i in 1:n
        for k in nzrange(graph, i)
            color = colors[rowvals(graph)[k]]
            used[color + 1] = i
        end
        color = 1
        while used[color + 1] == i
            color += 1
        end
        colors[i] = color
    end
    return sortperm(colors)
end

function level_ilu_permutation(rowptr, colval, n)
    graph = ilu_undirected_graph(rowptr, colval, n)
    # Symmetrization also constrains one-sided upper edges. Sorting the
    # resulting DAG levels preserves the direction of every original edge.
    _, rows = level_schedule(graph.colptr, graph.rowval, n)
    return rows
end

ilu_permutation_kind(config::Union{ILU0, DILU}) = config.ordering
ilu_permutation_kind(config) = :multicolor

function multicolor_ilu_pattern(rowptr::Vector{Ti}, colval, permutation) where {Ti}
    n = length(permutation)
    inverse = invperm(permutation)
    permuted_rowptr = Vector{Ti}(undef, n + 1)
    permuted_colval = similar(colval)
    value_positions = similar(colval)
    positions = Ti[]
    cursor = 1
    permuted_rowptr[1] = one(Ti)
    @inbounds for i in 1:n
        row = permutation[i]
        empty!(positions)
        for k in rowptr[row]:(rowptr[row + 1] - one(Ti))
            push!(positions, k)
        end
        sort!(positions; by = k -> inverse[colval[k]])
        for k in positions
            permuted_colval[cursor] = Ti(inverse[colval[k]])
            value_positions[cursor] = k
            cursor += 1
        end
        permuted_rowptr[i + 1] = Ti(cursor)
    end
    return permuted_rowptr, permuted_colval, value_positions
end

@kernel function ilu_gather_kernel!(destination, @Const(source), @Const(positions), n)
    i = @index(Global)
    if i <= n
        @inbounds destination[i] = source[positions[i]]
    end
end

@kernel function ilu_scatter_kernel!(destination, @Const(source), @Const(positions), n)
    i = @index(Global)
    if i <= n
        @inbounds destination[positions[i]] = source[i]
    end
end

function ilu_gather!(destination, source, positions, backend, block_size)
    ilu_gather_kernel!(backend, block_size)(destination, source, positions,
        length(destination); ndrange = length(destination))
    return destination
end

function ilu_scatter!(destination, source, positions, backend, block_size)
    ilu_scatter_kernel!(backend, block_size)(destination, source, positions,
        length(source); ndrange = length(source))
    return destination
end

function ilu_gather!(destination::Vector, source::AbstractVector, positions::Vector,
        ::KernelAbstractions.CPU, block_size)
    @batch for i in eachindex(destination)
        @inbounds destination[i] = source[positions[i]]
    end
    return destination
end

function ilu_scatter!(destination::AbstractVector, source::Vector, positions::Vector,
        ::KernelAbstractions.CPU, block_size)
    @batch for i in eachindex(source)
        @inbounds destination[positions[i]] = source[i]
    end
    return destination
end

function setup_permuted_ilu_factor(matrix, config::Union{ILU0, DILU})
    # This matrix is private to the preconditioner. Gather new coefficients
    # directly into its factor storage on every update. Residuals continue to
    # use the caller's original matrix, which is never overwritten.
    natural = typeof(config)(config.steps, config.damping)
    return setup_smoother(matrix, natural; _in_place = true)
end

setup_permuted_ilu_factor(matrix, config) =
    setup_smoother(matrix, typeof(config)(config.steps, config.damping))

function setup_multicolor_ilu(A, config; reuse = nothing, reallocation_tracker = nothing)
    if reuse isa MulticolorILUState && typeof(config) === typeof(reuse.config) &&
            ilu_permutation_kind(config) == ilu_permutation_kind(reuse.config) &&
            same_backend(reuse.backend, matrix_backend(A)) && same_smoother_pattern(reuse, A)
        reuse.config = config
        reuse.inner.config = typeof(config)(config.steps, config.damping)
        return update_smoother!(reuse, A)
    end
    mark_backend_reallocation!(reallocation_tracker, replaced_ilu_storage_bytes(reuse))
    n = matrix_nrows(A)
    rowptr = host_prefix(A.rowptr, n + 1)
    colval = host_prefix(A.colval, matrix_nonzeros(A))
    permutation = if ilu_permutation_kind(config) == :level
        level_ilu_permutation(rowptr, colval, n)
    else
        multicolor_ilu_permutation(rowptr, colval, n)
    end
    rp, cv, positions = multicolor_ilu_pattern(rowptr, colval, permutation)
    backend, block_size = matrix_backend(A), matrix_batch_size(A)
    device_positions = backend_copy(backend, positions)
    values = similar(A.nzval, matrix_nonzeros(A))
    ilu_gather!(values, A.nzval, device_positions, backend, block_size)
    matrix = csr_matrix(backend_copy(backend, rp), backend_copy(backend, cv), values,
        n, n; backend, block_size)
    inner = setup_permuted_ilu_factor(matrix, config)
    work, residual = allocate_ilu_work(A)
    return MulticolorILUState(inner, matrix,
        backend_copy(backend, eltype(rowptr).(permutation)), device_positions,
        A.rowptr, A.colval, rowptr, colval, work, residual,
        similar(work), similar(work), config, backend, block_size, n)
end

function update_smoother!(state::MulticolorILUState, A::StaticSparsityMatrixCSR)
    require_same_smoother_pattern(state, A)
    ilu_gather!(state.matrix.nzval, A.nzval, state.value_positions,
        state.backend, state.block_size)
    update_smoother!(state.inner, state.matrix)
    return state
end

function update_level_smoother!(state::MulticolorILUState,
        A::StaticSparsityMatrixCSR, options::AMGOptions)
    state.config = options.smoother
    state.inner.config = typeof(state.config)(state.config.steps, state.config.damping)
    return update_smoother!(state, A)
end

function apply!(x::AbstractVector, state::MulticolorILUState, b::AbstractVector)
    require_apply_dimensions(x, state, b)
    same_backend(KernelAbstractions.get_backend(x), state.backend) ||
        throw(ArgumentError("output and smoother must use the same backend"))
    ilu_gather!(state.permuted_rhs, b, state.permutation, state.backend, state.block_size)
    apply_permuted_ilu!(state, state.backend)
    ilu_scatter!(x, state.permuted_x, state.permutation, state.backend, state.block_size)
    return x
end

apply_permuted_ilu!(state, backend) =
    apply!(state.permuted_x, state.inner, state.permuted_rhs)

function apply_correction!(x, state::MulticolorILUState, residual)
    apply!(state.work, state, residual)
    axpy!(x, state.work, one(smoother_damping(state)), state.backend, state.block_size)
    return x
end

function smooth_level!(x, A::StaticSparsityMatrixCSR, b, state::MulticolorILUState,
        steps::Int; residual = nothing, zero_initial::Bool = false)
    steps > 0 || throw(ArgumentError("smoothing steps must be positive"))
    start = 1
    if !isnothing(residual)
        if zero_initial
            apply!(x, state, residual)
        else
            apply_correction!(x, state, residual)
        end
        start = 2
    elseif zero_initial
        fill_backend!(x, zero(eltype(x)), state.backend, state.block_size)
    end
    if start <= steps
        smooth!(x, state, A, b; steps = steps - start + 1)
    end
    return x
end

smooth_result!(x, A::StaticSparsityMatrixCSR, b, state::MulticolorILUState,
    steps::Int; kwargs...) = smooth_level!(x, A, b, state, steps; kwargs...)

function replaced_ilu_storage_bytes(state::MulticolorILUState)
    bytes = replaced_ilu_storage_bytes(state.inner)
    for buffer in (state.matrix.rowptr, state.matrix.colval, state.matrix.nzval,
            state.permutation, state.value_positions, state.work, state.residual,
            state.permuted_rhs, state.permuted_x)
        # The private matrix shares coefficient storage with KA ILU factors.
        shares_values = (state.inner isa ILU0State && buffer === state.inner.factors) ||
            (state.inner isa DILUState && buffer === state.inner.values)
        shares_values || (bytes += backend_buffer_bytes(buffer))
    end
    return bytes
end
