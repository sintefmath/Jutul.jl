# Normalize the partition block to unit diagonal. This both retains signed
# diagonal behavior and implements hypre's zero-diagonal skip without giving
# cuSPARSE a singular triangular matrix. With M=I+w*D^-1*(L_B+U_B),
#   f = lower(M)^-1 * (w*omega*D^-1*r),
#   c = upper(M)^-1 * ((2-w*omega)*f).
# This is the same symmetric correction as the portable implementation.
struct CUDAHybridGSFactor{M, MD, DD, SI, W, I, K}
    matrix::M
    lower_descriptor::MD
    upper_descriptor::MD
    input_descriptor::DD
    output_descriptor::DD
    lower_info::SI
    upper_info::SI
    lower_workspace::W
    upper_workspace::W
    original_indices::I
    rhs_kernel::K
end

struct CUDADiagonalHybridGSFactor{K}
    rhs_kernel::K
end

include("cuda_hybrid_partition.jl")

function KAPreconditioners.hybrid_partition_count(config::HybridGaussSeidel,
        backend::CUDA.CUDABackend, n)
    requested = config.partitions
    if iszero(requested)
        # A single device-wide partition degenerates into a global ordered
        # triangular solve. Use independent partitions as on the threaded
        # host, with one partition per CUDA multiprocessor.
        requested = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    end
    return min(requested, n)
end

@kernel function cuda_hybrid_values_kernel!(values, @Const(rp), @Const(original_indices),
        @Const(original_values), @Const(inverse_diagonal), w, n)
    i = @index(Global)
    if i <= n
        @inbounds for k in rp[i]:(rp[i + 1] - 1)
            q = original_indices[k]
            values[k] = iszero(q) ? one(eltype(values)) :
                w * inverse_diagonal[i] * original_values[q]
        end
    end
end

@kernel function cuda_hybrid_rhs_kernel!(output, @Const(rhs), @Const(inverse_diagonal), alpha, n)
    i = @index(Global)
    if i <= n
        @inbounds output[i] = iszero(inverse_diagonal[i]) ? zero(eltype(output)) :
            alpha * inverse_diagonal[i] * rhs[i]
    end
end

function KAPreconditioners.build_hybrid_native(state, backend::CUDA.CUDABackend)
    A = state.matrix
    T, Ti = eltype(A.nzval), eltype(A.colval)
    T <: CUSPARSEValue && eltype(state.work) === T || return nothing
    # Only retained within-partition edges enter the triangular solves.
    # Inject every diagonal, including rows whose original matrix omits it.
    rp, cv, original_indices = Ti[1], Ti[], Ti[]
    offsets = state.partition_offsets
    for p in 1:(length(offsets) - 1), i in offsets[p]:(offsets[p + 1] - 1)
        columns, indices = Ti[Ti(i)], Ti[0]
        for q in state.host_rowptr[i]:(state.host_rowptr[i + 1] - 1)
            j = state.host_colval[q]
            if j != i && offsets[p] <= j < offsets[p + 1]
                push!(columns, j)
                push!(indices, Ti(q))
            end
        end
        order = sortperm(columns)
        allunique(columns) || return nothing
        append!(cv, columns[order])
        append!(original_indices, indices[order])
        push!(rp, Ti(length(cv) + 1))
    end
    if length(cv) == state.n
        # No within-partition off-diagonal entries: both triangular factors
        # are identity. This is common on coarse levels with automatic
        # partitioning. Avoid sparse descriptors, analysis, and two solves.
        w, omega = KAPreconditioners.smoother_damping(state), KAPreconditioners.hybrid_outer_weight(state)
        kernel = KAPreconditioners.setup_smoother_kernel(cuda_hybrid_rhs_kernel!, backend, state.block_size,
            state.work, state.residual, state.inverse_diagonal, w*omega*(2-w*omega), state.n;
            ndrange = state.n)
        return CUDADiagonalHybridGSFactor(kernel)
    end
    # The CSR storage, descriptors, analysis buffers, and index map are owned
    # by this factor and survive replacement of A's wrapper or value array.
    # Keep the original value-index map, but use the smaller private solve
    # pattern when it fits. cuSPARSE otherwise carries 64-bit row/column
    # indices through every dependency lookup, even for a small coarse block.
    solve_index_type = max(state.n, length(cv)+1) <= typemax(Cint) ? Cint : Ti
    matrix = CuSparseMatrixCSR{T, solve_index_type}(CuArray(solve_index_type.(rp)),
        CuArray(solve_index_type.(cv)), CuVector{T}(undef, length(cv)), size(A))
    indices = CuArray(original_indices)
    partition_factor = cuda_build_hybrid_partition(state, matrix, indices, rp, cv)
    if !isnothing(partition_factor)
        cuda_hybrid_update_values!(partition_factor, state)
        return partition_factor
    end
    # Normalization makes both factors unit triangular; declaring this also
    # avoids a diagonal division for each row in the cuSPARSE solve kernels.
    lower = cuda_set_triangular_attributes!(CUDA.CUSPARSE.CuSparseMatrixDescriptor(matrix, 'O'), 'L', 'U')
    upper = cuda_set_triangular_attributes!(CUDA.CUSPARSE.CuSparseMatrixDescriptor(matrix, 'O'), 'U', 'U')
    input = CUDA.CUSPARSE.CuDenseVectorDescriptor(state.work)
    output = CUDA.CUSPARSE.CuDenseVectorDescriptor(state.forward)
    lower_info, upper_info = CUDA.CUSPARSE.CuSparseSpSVDescriptor(), CUDA.CUSPARSE.CuSparseSpSVDescriptor()
    lower_size = cuda_csr_solve_buffer_size(lower, input, output, lower_info, T)
    upper_size = cuda_csr_solve_buffer_size(upper, input, output, upper_info, T)
    lower_workspace, upper_workspace = CuVector{UInt8}(undef, lower_size), CuVector{UInt8}(undef, upper_size)
    w, omega = KAPreconditioners.smoother_damping(state), KAPreconditioners.hybrid_outer_weight(state)
    rhs_kernel = KAPreconditioners.setup_smoother_kernel(cuda_hybrid_rhs_kernel!, backend, state.block_size,
        state.work, state.residual, state.inverse_diagonal, w*omega, state.n; ndrange = state.n)
    factor = CUDAHybridGSFactor(matrix, lower, upper, input, output, lower_info, upper_info,
        lower_workspace, upper_workspace, indices, rhs_kernel)
    cuda_hybrid_update_values!(factor, state)
    for (descriptor, info, workspace) in (
            (lower, lower_info, lower_workspace), (upper, upper_info, upper_workspace))
        CUDA.CUSPARSE.cusparseSpSV_analysis(CUDA.CUSPARSE.handle(), 'N', Ref{T}(one(T)),
            descriptor, input, output, T, CUDA.CUSPARSE.CUSPARSE_SPSV_ALG_DEFAULT, info, workspace)
    end
    return factor
end

KAPreconditioners.update_hybrid_native!(::CUDADiagonalHybridGSFactor, state) = nothing
KAPreconditioners.hybrid_native_storage_bytes(::CUDADiagonalHybridGSFactor) = 0

function KAPreconditioners.solve_hybrid_native!(factor::CUDADiagonalHybridGSFactor, state, rhs)
    w, omega = KAPreconditioners.smoother_damping(state), KAPreconditioners.hybrid_outer_weight(state)
    KAPreconditioners.launch_smoother_kernel(factor.rhs_kernel, state.work, rhs,
        state.inverse_diagonal, w*omega*(2-w*omega), state.n; ndrange = state.n)
    return state.work
end

function cuda_hybrid_update_values!(factor, state)
    matrix = factor.matrix
    cuda_hybrid_values_kernel!(state.backend, state.block_size)(matrix.nzVal, matrix.rowPtr,
        factor.original_indices, state.matrix.nzval, state.inverse_diagonal,
        KAPreconditioners.smoother_damping(state), state.n; ndrange = state.n)
    return factor
end

function KAPreconditioners.update_hybrid_native!(factor::CUDAHybridGSFactor, state)
    cuda_hybrid_update_values!(factor, state)
    for info in (factor.lower_info, factor.upper_info)
        CUDA.CUSPARSE.cusparseSpSV_updateMatrix(CUDA.CUSPARSE.handle(), info,
            factor.matrix.nzVal, CUDA.CUSPARSE.CUSPARSE_SPSV_UPDATE_GENERAL)
    end
    return factor
end

function KAPreconditioners.hybrid_native_storage_bytes(factor::CUDAHybridGSFactor)
    return sum(sizeof(eltype(array))*length(array) for array in (
        factor.matrix.rowPtr, factor.matrix.colVal, factor.matrix.nzVal, factor.original_indices,
        factor.lower_workspace, factor.upper_workspace))
end

function KAPreconditioners.solve_hybrid_native!(factor::CUDAHybridGSFactor, state, rhs)
    T = eltype(factor.matrix.nzVal)
    w, omega = KAPreconditioners.smoother_damping(state), KAPreconditioners.hybrid_outer_weight(state)
    KAPreconditioners.launch_smoother_kernel(factor.rhs_kernel, state.work, rhs,
        state.inverse_diagonal, w*omega, state.n; ndrange = state.n)
    CUDA.CUSPARSE.cusparseDnVecSetValues(factor.input_descriptor, state.work)
    CUDA.CUSPARSE.cusparseDnVecSetValues(factor.output_descriptor, state.forward)
    CUDA.CUSPARSE.cusparseSpSV_solve(CUDA.CUSPARSE.handle(), 'N', Ref{T}(one(T)),
        factor.lower_descriptor, factor.input_descriptor, factor.output_descriptor, T,
        CUDA.CUSPARSE.CUSPARSE_SPSV_ALG_DEFAULT, factor.lower_info)
    CUDA.CUSPARSE.cusparseDnVecSetValues(factor.input_descriptor, state.forward)
    CUDA.CUSPARSE.cusparseDnVecSetValues(factor.output_descriptor, state.work)
    CUDA.CUSPARSE.cusparseSpSV_solve(CUDA.CUSPARSE.handle(), 'N', Ref{T}(2-w*omega),
        factor.upper_descriptor, factor.input_descriptor, factor.output_descriptor, T,
        CUDA.CUSPARSE.CUSPARSE_SPSV_ALG_DEFAULT, factor.upper_info)
    return state.work
end
