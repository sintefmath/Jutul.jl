@kernel function copy_kernel!(dst, @Const(src), n)
    i = @index(Global)
    i <= n && (@inbounds dst[i] = src[i])
end

@kernel function fill_kernel!(dst, value, n)
    i = @index(Global)
    i <= n && (@inbounds dst[i] = value)
end

@kernel function spmv_kernel!(y, @Const(rp), @Const(cv), @Const(av), @Const(x), n)
    i = @index(Global)
    if i <= n
        s = zero(eltype(y))
        @inbounds for k in rp[i]:(rp[i + 1] - 1)
            s += av[k] * x[cv[k]]
        end
        @inbounds y[i] = s
    end
end

@kernel function residual_kernel!(r, @Const(b), @Const(x), @Const(rp), @Const(cv), @Const(av), n)
    i = @index(Global)
    if i <= n
        s = zero(eltype(r))
        @inbounds for k in rp[i]:(rp[i + 1] - 1)
            s += av[k] * x[cv[k]]
        end
        @inbounds r[i] = b[i] - s
    end
end

@kernel function strength_kernel!(
        strong, @Const(rp), @Const(cv), @Const(av),
        theta, max_row_sum, strength_mode, n
    )
    i = @index(Global)
    if i <= n
        diag = zero(eltype(av))
        row_sum = zero(eltype(av))
        @inbounds for k in rp[i]:(rp[i + 1] - 1)
            a = av[k]
            row_sum += a
            cv[k] == i && (diag += a)
        end
        usable_diagonal = !iszero(diag)
        weakened = max_row_sum < one(max_row_sum) && usable_diagonal &&
            abs(row_sum) > abs(diag) * max_row_sum
        if weakened
            @inbounds for k in rp[i]:(rp[i + 1] - 1)
                strong[k] = false
            end
        else
            largest_opposite = zero(real(eltype(av)))
            largest_any = zero(real(eltype(av)))
            @inbounds for k in rp[i]:(rp[i + 1] - 1)
                a = av[k]
                if cv[k] != i
                    magnitude = abs(a)
                    largest_any = max(largest_any, magnitude)
                    if real(a * conj(diag)) < 0
                        largest_opposite = max(largest_opposite, magnitude)
                    end
                end
            end
            use_signed = strength_mode == 1 ||
                (strength_mode == 0 && largest_opposite > 0)
            largest = if use_signed
                largest_opposite
            else
                largest_any
            end
            cutoff = theta * largest
            @inbounds for k in rp[i]:(rp[i + 1] - 1)
                a = av[k]
                opposite = real(a * conj(diag)) < 0
                strong[k] = cv[k] != i && largest > 0 &&
                    (!use_signed || opposite) && abs(a) > cutoff
            end
        end
    end
end

@kernel function csr_to_dense_kernel!(dense, @Const(rp), @Const(cv), @Const(av), n)
    i = @index(Global)
    if i <= n
        @inbounds for k in rp[i]:(rp[i + 1] - 1)
            dense[i, cv[k]] = av[k]
        end
    end
end

@kernel function restrict_kernel!(
        bc, @Const(offsets), @Const(rows), @Const(pidx),
        @Const(pval), @Const(r), n
    )
    I = @index(Global)
    if I <= n
        s = zero(eltype(bc))
        @inbounds for k in offsets[I]:(offsets[I + 1] - 1)
            s += conj(pval[pidx[k]]) * r[rows[k]]
        end
        @inbounds bc[I] = s
    end
end

@kernel function prolong_kernel!(x, @Const(rp), @Const(cv), @Const(pv), @Const(xc), n)
    i = @index(Global)
    if i <= n
        s = zero(eltype(x))
        @inbounds for k in rp[i]:(rp[i + 1] - 1)
            s += pv[k] * xc[cv[k]]
        end
        @inbounds x[i] += s
    end
end

@kernel function prolong_to_kernel!(
        dst, @Const(src), @Const(rp), @Const(cv),
        @Const(pv), @Const(xc), n
    )
    i = @index(Global)
    if i <= n
        s = zero(eltype(dst))
        @inbounds for k in rp[i]:(rp[i + 1] - 1)
            s += pv[k] * xc[cv[k]]
        end
        @inbounds dst[i] = src[i] + s
    end
end

@kernel function galerkin_kernel!(
        ac, @Const(a), @Const(p), @Const(offsets),
        @Const(pl), @Const(ai), @Const(pr), n
    )
    k = @index(Global)
    if k <= n
        s = zero(eltype(ac))
        @inbounds for t in offsets[k]:(offsets[k + 1] - 1)
            s += conj(p[pl[t]]) * a[ai[t]] * p[pr[t]]
        end
        @inbounds ac[k] = s
    end
end

# Membership in the complete, untruncated coarse stencil. Numeric updates
# retain P's sparsity but must use the original stencil to compute weights.
@inline function interpolation_candidate(arp, acv, cf, strong, i, target, extended)
    @inbounds for k in arp[i]:(arp[i + 1] - 1)
        strong[k] || continue
        j = acv[k]
        if cf[j] == 1 && j == target
            return true
        elseif extended && cf[j] == -1
            for q in arp[j]:(arp[j + 1] - 1)
                strong[q] && acv[q] == target && cf[target] == 1 && return true
            end
        end
    end
    return false
end

# Return the effective diagonal, the sum of all candidate numerators, and
# the numerator for retained coarse column J. This matches candidate_weights!
# without allocating candidate vectors on the device.
@inline function interpolation_row_terms(arp, acv, av, cf, cmap, strong, i, J, extended)
    T = eltype(av)
    diagonal = zero(T)
    total = zero(T)
    selected = zero(T)
    @inbounds for k in arp[i]:(arp[i + 1] - 1)
        j, aij = acv[k], av[k]
        if j == i
            diagonal += aij
        elseif cf[j] == 1 && interpolation_candidate(arp, acv, cf, strong, i, j, extended)
            total += aij
            cmap[j] == J && (selected += aij)
        elseif strong[k] && cf[j] == -1
            sign_j = 1
            if extended
                diag_j = zero(T)
                for q in arp[j]:(arp[j + 1] - 1)
                    acv[q] == j && (diag_j += av[q])
                end
                sign_j = real(diag_j) < 0 ? -1 : 1
            end
            denominator = zero(T)
            coarse_sum = zero(T)
            coupling = zero(T)
            back = zero(T)
            for q in arp[j]:(arp[j + 1] - 1)
                l, ajl = acv[q], av[q]
                l == j && continue
                extended && sign_j * real(ajl) >= 0 && continue
                if extended && l == i
                    denominator += ajl
                    back += ajl
                elseif cf[l] == 1 &&
                        interpolation_candidate(arp, acv, cf, strong, i, l, extended)
                    denominator += ajl
                    coarse_sum += ajl
                    cmap[l] == J && (coupling += ajl)
                end
            end
            if !iszero(denominator)
                distribute = aij / denominator
                total += distribute * coarse_sum
                selected += distribute * coupling
                diagonal += distribute * back
            else
                diagonal += aij
            end
        else
            diagonal += aij
        end
    end
    return diagonal, total, selected
end

@kernel function update_interpolation_p_kernel!(
        pv, @Const(prp), @Const(pcv),
        @Const(arp), @Const(acv), @Const(av),
        @Const(cf), @Const(cmap), @Const(strong),
        extended, rescale, n
    )
    i = @index(Global)
    if i <= n
        firstp, lastp = prp[i], prp[i + 1] - 1
        if firstp <= lastp
            if cf[i] == 1
                @inbounds pv[firstp] = one(eltype(pv))
            else
                kept_sum = zero(eltype(pv))
                original_sum = zero(eltype(pv))
                @inbounds for pidx in firstp:lastp
                    diagonal, total, selected = interpolation_row_terms(
                        arp, acv, av, cf, cmap, strong, i, pcv[pidx], extended
                    )
                    scale = !iszero(diagonal) ?
                        -inv(diagonal) : one(eltype(pv))
                    weight = scale * selected
                    pv[pidx] = weight
                    kept_sum += weight
                    original_sum = scale * total
                end
                if rescale && !iszero(kept_sum)
                    scale = original_sum / kept_sum
                    @inbounds for pidx in firstp:lastp
                        pv[pidx] *= scale
                    end
                end
            end
        end
    end
end

# GPU extensions return true when a cached vendor product was performed.
vendor_spmv!(y, A, x, alpha, beta) = false

function LinearAlgebra.mul!(y::AbstractVector, A::StaticSparsityMatrixCSR, x::AbstractVector)
    length(y) == matrix_nrows(A) || throw(DimensionMismatch())
    length(x) == matrix_ncols(A) || throw(DimensionMismatch())
    if A.use_vendor_linalg && vendor_spmv!(
            logical_backend_buffer(y), A, logical_backend_buffer(x), 1, 0
        )
        return y
    end
    n = matrix_nrows(A)
    k! = spmv_kernel!(matrix_backend(A), matrix_kernel_block_size(A))
    k!(y, A.rowptr, A.colval, A.nzval, x, n; ndrange = n)
    return y
end

const CPU_THREAD_THRESHOLD = 8_192

@inline function foreach_cpu_row(
        f, n::Int,
        min_batch::Int = CPU_THREAD_THRESHOLD
    )
    number_of_batches = clamp(n ÷ min_batch, 1, Threads.nthreads())
    if number_of_batches > 1
        Threads.@threads :dynamic for batch in 1:number_of_batches
            first_row = fld((batch - 1) * n, number_of_batches) + 1
            last_row = fld(batch * n, number_of_batches)
            @inbounds for i in first_row:last_row
                f(i)
            end
        end
    else
        @inbounds for i in 1:n
            f(i)
        end
    end
    return nothing
end

@inline function csr_row_product(A, x, row, ::Type{T}) where {T}
    value = zero(T)
    @inbounds @simd for k in A.rowptr[row]:(A.rowptr[row + 1] - 1)
        value += A.nzval[k] * x[A.colval[k]]
    end
    return value
end

function LinearAlgebra.mul!(
        y::Vector, A::StaticSparsityMatrixCSR{Tv, Ti, <:Vector, <:Vector, <:Vector},
        x::Vector
    ) where {Tv, Ti}
    length(y) == matrix_nrows(A) || throw(DimensionMismatch())
    length(x) == matrix_ncols(A) || throw(DimensionMismatch())
    foreach_cpu_row(matrix_nrows(A), matrix_batch_size(A)) do i
        value = csr_row_product(A, x, i, eltype(y))
        @inbounds y[i] = value
    end
    return y
end

function copy_matrix_values!(dst::StaticSparsityMatrixCSR, src::StaticSparsityMatrixCSR)
    number_of_nonzeros = matrix_nonzeros(dst)
    number_of_nonzeros == matrix_nonzeros(src) ||
        throw(ArgumentError("matrix patterns differ"))
    k! = copy_kernel!(matrix_backend(dst), matrix_kernel_block_size(dst))
    k!(
        dst.nzval, src.nzval, number_of_nonzeros;
        ndrange = number_of_nonzeros
    )
    return dst
end

function copy_matrix_values!(
        dst::StaticSparsityMatrixCSR{Tv, Ti, <:Vector, <:Vector, <:Vector},
        src::StaticSparsityMatrixCSR{Tv, Ti, <:Vector, <:Vector, <:Vector}
    ) where {Tv, Ti}
    matrix_nonzeros(dst) == matrix_nonzeros(src) || throw(ArgumentError("matrix patterns differ"))
    copyto!(dst.nzval, 1, src.nzval, 1, matrix_nonzeros(dst))
    return dst
end

function fill_backend!(x, value, backend, block_size)
    k! = fill_kernel!(backend, block_size)
    k!(x, value, length(x); ndrange = length(x))
    return x
end

fill_backend!(x::Array, value, ::KernelAbstractions.CPU, block_size) = fill!(x, value)

function update_coarse_solver!(S::CoarseLUState, A::StaticSparsityMatrixCSR)
    @inbounds for k in 1:matrix_nonzeros(A)
        S.matrix.nzval[S.csr_to_csc[k]] = A.nzval[k]
    end
    lu!(S.factors, S.matrix)
    return S
end

function copy_csr_to_dense!(dense, A::StaticSparsityMatrixCSR)
    backend = matrix_backend(A)
    block_size = matrix_kernel_block_size(A)
    n = matrix_nrows(A)
    fill_backend!(dense, zero(eltype(dense)), backend, block_size)
    k! = csr_to_dense_kernel!(backend, block_size)
    k!(dense, A.rowptr, A.colval, A.nzval, n; ndrange = n)
    # Generic `lu!` implementations may access the array from the host, while
    # accelerator implementations enqueue work on their own library stream.
    # Make the KernelAbstractions writes visible before either kind is called.
    synchronize_backend(backend)
    return dense
end

function update_coarse_solver!(S::DenseLUState, A::StaticSparsityMatrixCSR)
    matrix = S.factorization.factors
    copy_csr_to_dense!(matrix, A)
    S.factorization = lu!(matrix)
    return S
end

function update_coarse_solver!(S::HostLUState, A::StaticSparsityMatrixCSR)
    copyto!(S.values, 1, A.nzval, 1, matrix_nonzeros(A))
    matrix = S.factorization.factors
    fill!(matrix, zero(eltype(matrix)))
    @inbounds for i in 1:matrix_nrows(A), k in S.rowptr[i]:(S.rowptr[i + 1] - 1)
        matrix[i, S.colval[k]] = S.values[k]
    end
    S.factorization = lu!(matrix)
    return S
end

function coarse_solve!(x, b, S::CoarseLUState, backend, block_size)
    return ldiv!(x, S.factors, b)
end

logical_backend_buffer(buffer) = buffer

function coarse_solve!(x, b, S::DenseLUState, backend, block_size)
    # Accelerator library methods dispatch on their native vector types rather
    # than AbstractVector. Strip the capacity-retaining wrapper while keeping
    # the logical prefix passed to the dense coarse solve.
    ldiv!(
        logical_backend_buffer(x), S.factorization,
        logical_backend_buffer(b)
    )
    return x
end

function LinearAlgebra.ldiv!(x, S::DenseLUState, b)
    ldiv!(
        logical_backend_buffer(x), S.factorization,
        logical_backend_buffer(b)
    )
    return x
end

function coarse_solve!(x, b, S::HostLUState, backend, block_size)
    copyto!(S.rhs, b)
    ldiv!(S.factorization, S.rhs)
    copyto!(x, S.rhs)
    return x
end

function residual!(r, A::StaticSparsityMatrixCSR, x, b)
    n = matrix_nrows(A)
    k! = residual_kernel!(matrix_backend(A), matrix_kernel_block_size(A))
    k!(r, b, x, A.rowptr, A.colval, A.nzval, n; ndrange = n)
    return r
end


function residual!(
        r::Vector, A::StaticSparsityMatrixCSR{Tv, Ti, <:Vector, <:Vector, <:Vector},
        x::Vector, b::Vector
    ) where {Tv, Ti}
    foreach_cpu_row(matrix_nrows(A), matrix_batch_size(A)) do i
        value = csr_row_product(A, x, i, eltype(r))
        @inbounds r[i] = b[i] - value
    end
    return r
end
