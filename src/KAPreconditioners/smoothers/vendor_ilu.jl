function build_vendor_ilu(A::StaticSparsityMatrixCSR)
    backend = matrix_backend(A)
    throw(
        ArgumentError(
            "VendorILU has no native implementation for backend $(typeof(backend))"
        )
    )
end

function refactor_vendor_ilu!(factor)
    throw(
        ArgumentError(
            "VendorILU cannot update factorization $(typeof(factor)); " *
                "the active backend has no native implementation"
        )
    )
end

vendor_ilu_factor_storage_bytes(factor) = 0

function replaced_ilu_storage_bytes(state::VendorILUState)
    return vendor_ilu_factor_storage_bytes(state.factor) +
        backend_buffer_bytes(state.factor_values) +
        backend_buffer_bytes(state.work) +
        backend_buffer_bytes(state.residual)
end

function reuse_vendor_ilu!(reuse, A, config)
    compatible = reuse isa VendorILUState &&
        same_backend(reuse.backend, matrix_backend(A)) &&
        same_smoother_pattern(reuse, A)
    compatible || return nothing
    reuse.config = config
    return update_smoother!(reuse, A)
end

function setup_smoother(
        A::StaticSparsityMatrixCSR{Tv}, config::VendorILU;
        reuse = nothing, reallocation_tracker = nothing
    ) where {Tv}
    require_square_matrix(A, "VendorILU")
    reused = reuse_vendor_ilu!(reuse, A, config)
    isnothing(reused) || return reused
    mark_backend_reallocation!(
        reallocation_tracker, replaced_ilu_storage_bytes(reuse)
    )
    factor, factor_values = build_vendor_ilu(A)
    work, residual = allocate_ilu_work(A)
    return VendorILUState(
        Tv, factor, factor_values, work, residual,
        A.rowptr, A.colval,
        host_prefix(A.rowptr, matrix_nrows(A) + 1),
        host_prefix(A.colval, matrix_nonzeros(A)),
        config, matrix_backend(A), matrix_batch_size(A), matrix_nrows(A)
    )
end

function update_smoother!(
        state::VendorILUState, A::StaticSparsityMatrixCSR
    )
    require_same_smoother_pattern(state, A)
    copyto!(state.factor_values, A.nzval)
    refactor_vendor_ilu!(state.factor)
    return state
end

vendor_ilu_vector(values::AbstractVector{<:Number}) = values
function vendor_ilu_vector(values::AbstractVector)
    scalar_type = matrix_scalar_type(eltype(values))
    return reinterpret(scalar_type, values)
end

function vendor_ilu_solve!(x, state::VendorILUState, b)
    copyto!(x, b)
    values = vendor_ilu_vector(x)
    ldiv!(UnitLowerTriangular(state.factor), values)
    ldiv!(UpperTriangular(state.factor), values)
    return x
end

function apply!(
        x::AbstractVector, state::VendorILUState, b::AbstractVector
    )
    require_apply_dimensions(x, state, b)
    same_backend(KernelAbstractions.get_backend(x), state.backend) ||
        throw(ArgumentError("output and smoother must use the same backend"))
    return vendor_ilu_solve!(x, state, b)
end

function apply_correction!(x, state::VendorILUState, residual)
    vendor_ilu_solve!(state.work, state, residual)
    axpy!(
        x, state.work, one(smoother_damping(state)), state.backend,
        state.block_size
    )
    return x
end

function smooth_level!(
        x, A::StaticSparsityMatrixCSR, b, state::VendorILUState,
        steps::Int; residual = nothing, zero_initial::Bool = false
    )
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

function smooth_result!(
        x, A::StaticSparsityMatrixCSR, b, state::VendorILUState,
        steps::Int; residual = nothing, zero_initial::Bool = false
    )
    steps > 0 || throw(ArgumentError("smoothing steps must be positive"))
    return smooth_level!(
        x, A, b, state, steps;
        residual = residual, zero_initial = zero_initial
    )
end
