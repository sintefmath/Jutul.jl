# Fixed-pattern partition plan. One block runs both exact triangular sweeps;
# normalized coefficients and the correction stay on-chip between levels.
mutable struct CUDAPartitionHybridGSFactor{M, I, P, K, C}
    matrix::M
    original_indices::I
    diagonal::P
    partitions::P
    lower_parts::P
    lower_offsets::P
    lower_rows::P
    upper_parts::P
    upper_offsets::P
    upper_rows::P
    compiled::K
    call::C
    threads::Int
    blocks::Int
    shared_rows::Int
    shared_values::Int
    shared_bytes::Int
end

function cuda_hybrid_partition_schedule(offsets, rows, partitions)
    np, nl = length(partitions)-1, length(offsets)-1
    counts, row_levels = zeros(Int, np, nl), zeros(Int, length(rows))
    for l in 1:nl, q in offsets[l]:(offsets[l+1]-1)
        i = Int(rows[q])
        p = searchsortedlast(partitions, i)
        counts[p, l] += 1
        row_levels[i] = l
    end
    packed, starts, cursors = Cint[], Cint[1], zeros(Int, np, nl)
    first = 1
    for p in 1:np
        last_level = findlast(>(0), view(counts, p, :))
        for l in 1:last_level
            push!(packed, Cint(first))
            cursors[p, l] = first
            first += counts[p, l]
        end
        push!(packed, Cint(first))
        push!(starts, Cint(length(packed)+1))
    end
    ordered = Vector{Cint}(undef, length(rows))
    for i in eachindex(row_levels)
        p, l = searchsortedlast(partitions, i), row_levels[i]
        ordered[cursors[p, l]] = Cint(i)
        cursors[p, l] += 1
    end
    return CuArray(starts), CuArray(packed), CuArray(ordered)
end

function cuda_hybrid_partition_kernel!(output, rhs, inverse_diagonal, rp, cv, av,
        diagonal, partitions, lp, lo, lr, up, uo, ur, alpha, beta,
        shared_rows, shared_values, add, ::Val{cache_indices}) where {cache_indices}
    t, nt, p = Int(threadIdx().x), Int(blockDim().x), Int(blockIdx().x)
    T = eltype(output)
    correction = CuDynamicSharedArray(T, shared_rows)
    values = CuDynamicSharedArray(T, shared_values, shared_rows*sizeof(T))
    if cache_indices
        index_offset = (shared_rows+shared_values)*sizeof(T)
        columns = CuDynamicSharedArray(Cint, shared_values, index_offset)
        rowptr = CuDynamicSharedArray(Cint, shared_rows+1, index_offset+shared_values*sizeof(Cint))
        diag = CuDynamicSharedArray(Cint, shared_rows, index_offset+(shared_values+shared_rows+1)*sizeof(Cint))
    end
    @inbounds begin
        first, last = partitions[p], partitions[p+1]-1
        first_k = rp[first]
        for k in (first_k+t-1):nt:(rp[last+1]-1)
            values[k-first_k+1] = av[k]
            if cache_indices
                columns[k-first_k+1] = cv[k]-first+1
            end
        end
        if cache_indices
            for i in (first+t-1):nt:last
                rowptr[i-first+1] = rp[i]-first_k+1
                diag[i-first+1] = diagonal[i]-first_k+1
            end
            t == 1 && (rowptr[last-first+2] = rp[last+1]-first_k+1)
        end
        sync_threads()
        for level in lp[p]:(lp[p+1]-2)
            for q in (lo[level]+t-1):nt:(lo[level+1]-1)
                i = lr[q]
                value = iszero(inverse_diagonal[i]) ? zero(T) :
                    alpha*inverse_diagonal[i]*rhs[i]
                if cache_indices
                    for k in rowptr[i-first+1]:(diag[i-first+1]-1)
                        value -= values[k]*correction[columns[k]]
                    end
                else
                    for k in rp[i]:(diagonal[i]-1)
                        value -= values[k-first_k+1]*correction[cv[k]-first+1]
                    end
                end
                correction[i-first+1] = value
            end
            sync_threads()
        end
        for level in up[p]:(up[p+1]-2)
            for q in (uo[level]+t-1):nt:(uo[level+1]-1)
                i = ur[q]
                value = beta*correction[i-first+1]
                if cache_indices
                    for k in (diag[i-first+1]+1):(rowptr[i-first+2]-1)
                        value -= values[k]*correction[columns[k]]
                    end
                else
                    for k in (diagonal[i]+1):(rp[i+1]-1)
                        value -= values[k-first_k+1]*correction[cv[k]-first+1]
                    end
                end
                correction[i-first+1] = value
            end
            sync_threads()
        end
        for i in (first+t-1):nt:last
            output[i] = add ? output[i]+correction[i-first+1] : correction[i-first+1]
        end
    end
    return
end

function cuda_build_hybrid_partition(state, matrix, indices, rp, cv)
    eltype(matrix.rowPtr) === Cint || return nothing
    offsets = state.partition_offsets
    shared_rows = maximum(diff(offsets))
    shared_values = Int(maximum(rp[offsets[p+1]]-rp[offsets[p]] for p in 1:length(offsets)-1))
    shared_bytes = (shared_rows+shared_values)*sizeof(eltype(matrix.nzVal))
    limit = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
    shared_bytes <= limit || return nothing
    # Stage fixed indices too when they fit, removing dependent global index
    # loads from the triangular loops. Otherwise retain the coefficient-only
    # on-chip path with exactly the same partitioning and ordering.
    index_bytes = (shared_values+2shared_rows+1)*sizeof(Cint)
    cache_indices = shared_bytes+index_bytes <= limit
    cache_indices && (shared_bytes += index_bytes)
    diagonal = Cint[rp[i]+searchsortedfirst(view(cv, rp[i]:(rp[i+1]-1)), i)-1 for i in 1:state.n]
    parts, diag = CuArray(Cint.(offsets)), CuArray(diagonal)
    lp, lo, lr = cuda_hybrid_partition_schedule(state.lower_offsets, Array(state.lower_rows), offsets)
    up, uo, ur = cuda_hybrid_partition_schedule(state.upper_offsets, Array(state.upper_rows), offsets)
    threads = shared_rows >= 128 ? 128 : 32
    w, omega = KAPreconditioners.smoother_damping(state), KAPreconditioners.hybrid_outer_weight(state)
    call = CUDA.KernelCall(cuda_hybrid_partition_kernel!, state.work, state.residual,
        state.inverse_diagonal, matrix.rowPtr, matrix.colVal, matrix.nzVal,
        diag, parts, lp, lo, lr, up, uo, ur, w*omega, 2-w*omega, shared_rows, shared_values, false, Val(cache_indices))
    compiled = CUDA.kernel_compile(call; always_inline=state.backend.always_inline, maxthreads=threads)
    attributes = CUDA.attributes(compiled.fun)
    # Compilation can share the same function across hierarchy levels. Never
    # reduce its limit when a later level needs less shared memory.
    if attributes[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] < shared_bytes
        attributes[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shared_bytes
    end
    return CUDAPartitionHybridGSFactor(matrix, indices, diag, parts, lp, lo, lr, up, uo, ur,
        compiled, call, threads, length(offsets)-1, shared_rows, shared_values, shared_bytes)
end

function KAPreconditioners.update_hybrid_native!(factor::CUDAPartitionHybridGSFactor, state)
    return cuda_hybrid_update_values!(factor, state)
end

function KAPreconditioners.solve_hybrid_native!(factor::CUDAPartitionHybridGSFactor, state, rhs)
    cuda_launch_hybrid_partition!(state.work, factor, state, rhs, false)
    return state.work
end

function KAPreconditioners.solve_hybrid_native_to!(x, factor::CUDAPartitionHybridGSFactor, state, rhs, add)
    x isa typeof(state.work) || return false
    cuda_launch_hybrid_partition!(x, factor, state, rhs, add)
    return true
end

function cuda_launch_hybrid_partition!(x, factor, state, rhs, add)
    if !(rhs isa typeof(state.residual))
        copyto!(state.residual, rhs)
        rhs = state.residual
    end
    w, omega = KAPreconditioners.smoother_damping(state), KAPreconditioners.hybrid_outer_weight(state)
    call = factor.call
    # Everything else in the launch belongs to the fixed layout. Numeric
    # reset changes its contents in place; only the RHS/weights can rebind.
    for (i, argument) in ((1, x), (2, rhs), (15, w*omega), (16, 2-w*omega), (19, add))
        call.source.arguments[i] === argument || (call = CUDA.rebind(call, argument, i))
    end
    factor.call = call
    CUDA.kernel_launch(factor.compiled, call; threads=factor.threads,
        blocks=factor.blocks, shmem=factor.shared_bytes)
    return x
end

function KAPreconditioners.hybrid_native_storage_bytes(factor::CUDAPartitionHybridGSFactor)
    return sum(sizeof(eltype(array))*length(array) for array in (
        factor.matrix.rowPtr, factor.matrix.colVal, factor.matrix.nzVal, factor.original_indices,
        factor.diagonal, factor.partitions, factor.lower_parts, factor.lower_offsets,
        factor.lower_rows, factor.upper_parts, factor.upper_offsets, factor.upper_rows))
end
