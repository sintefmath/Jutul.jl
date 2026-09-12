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
        @inbounds for k in rp[i]:(rp[i+1]-1)
            s += av[k] * x[cv[k]]
        end
        @inbounds y[i] = s
    end
end

@kernel function residual_kernel!(r, @Const(b), @Const(x), @Const(rp), @Const(cv), @Const(av), n)
    i = @index(Global)
    if i <= n
        s = zero(eltype(r))
        @inbounds for k in rp[i]:(rp[i+1]-1)
            s += av[k] * x[cv[k]]
        end
        @inbounds r[i] = b[i] - s
    end
end

@kernel function strength_kernel!(strong, @Const(rp), @Const(cv), @Const(av),
                                   theta, max_row_sum, n)
    i = @index(Global)
    if i <= n
        diag = zero(eltype(av))
        row_sum = zero(eltype(av))
        @inbounds for k in rp[i]:(rp[i+1]-1)
            a = av[k]
            row_sum += a
            cv[k] == i && (diag += a)
        end
        usable_diagonal = abs(diag) >= eps(real(eltype(av)))
        weakened = max_row_sum < one(max_row_sum) && usable_diagonal &&
                   abs(row_sum) > abs(diag) * max_row_sum
        if weakened
            @inbounds for k in rp[i]:(rp[i+1]-1)
                strong[k] = false
            end
        else
            largest_opposite = zero(real(eltype(av)))
            largest_any = zero(real(eltype(av)))
            @inbounds for k in rp[i]:(rp[i+1]-1)
                a = av[k]
                if cv[k] != i
                    magnitude = abs(a)
                    largest_any = max(largest_any, magnitude)
                    if real(a * conj(diag)) < 0
                        largest_opposite = max(largest_opposite, magnitude)
                    end
                end
            end
            use_signed = largest_opposite > 0
            largest = if use_signed
                largest_opposite
            else
                largest_any
            end
            cutoff = theta * largest
            @inbounds for k in rp[i]:(rp[i+1]-1)
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
        @inbounds for k in rp[i]:(rp[i+1]-1)
            dense[i, cv[k]] = av[k]
        end
    end
end

@kernel function restrict_kernel!(bc, @Const(offsets), @Const(rows), @Const(pidx),
                                   @Const(pval), @Const(r), n)
    I = @index(Global)
    if I <= n
        s = zero(eltype(bc))
        @inbounds for k in offsets[I]:(offsets[I+1]-1)
            s += conj(pval[pidx[k]]) * r[rows[k]]
        end
        @inbounds bc[I] = s
    end
end

@kernel function prolong_kernel!(x, @Const(rp), @Const(cv), @Const(pv), @Const(xc), n)
    i = @index(Global)
    if i <= n
        s = zero(eltype(x))
        @inbounds for k in rp[i]:(rp[i+1]-1)
            s += pv[k] * xc[cv[k]]
        end
        @inbounds x[i] += s
    end
end

@kernel function prolong_to_kernel!(dst, @Const(src), @Const(rp), @Const(cv),
                                     @Const(pv), @Const(xc), n)
    i = @index(Global)
    if i <= n
        s = zero(eltype(dst))
        @inbounds for k in rp[i]:(rp[i+1]-1)
            s += pv[k] * xc[cv[k]]
        end
        @inbounds dst[i] = src[i] + s
    end
end

@kernel function galerkin_kernel!(ac, @Const(a), @Const(p), @Const(offsets),
                                   @Const(pl), @Const(ai), @Const(pr), n)
    k = @index(Global)
    if k <= n
        s = zero(eltype(ac))
        @inbounds for t in offsets[k]:(offsets[k+1]-1)
            s += conj(p[pl[t]]) * a[ai[t]] * p[pr[t]]
        end
        @inbounds ac[k] = s
    end
end

@kernel function update_extended_i_p_kernel!(pv, @Const(prp), @Const(pcv),
                                   @Const(arp), @Const(acv), @Const(av),
                                   @Const(cf), @Const(cmap), @Const(strong),
                                   rescale, n)
    i = @index(Global)
    if i <= n
        firstp, lastp = prp[i], prp[i+1] - 1
        if firstp <= lastp
            if cf[i] == 1
                @inbounds pv[firstp] = one(eltype(pv))
            else
                diag_i = zero(eltype(pv))
                @inbounds for aidx in arp[i]:(arp[i+1]-1)
                    acv[aidx] == i && (diag_i += av[aidx])
                end
                if iszero(diag_i)
                    v = one(eltype(pv)) / (lastp-firstp+1)
                    @inbounds for pidx in firstp:lastp
                        pv[pidx] = v
                    end
                else
                    rowsum = zero(eltype(pv))
                    @inbounds for pidx in firstp:lastp
                        J = pcv[pidx]
                        w = zero(eltype(pv))
                        for aidx in arp[i]:(arp[i+1]-1)
                            j = acv[aidx]
                            if cmap[j] == J
                                w -= av[aidx] / diag_i
                            elseif strong[aidx] && cf[j] != 1
                                diag_j = zero(eltype(pv))
                                a_jJ = zero(eltype(pv))
                                for aj in arp[j]:(arp[j+1]-1)
                                    q = acv[aj]
                                    q == j && (diag_j += av[aj])
                                    cmap[q] == J && (a_jJ += av[aj])
                                end
                                !iszero(diag_j) && (w += (av[aidx] * a_jJ) / (diag_i * diag_j))
                            end
                        end
                        pv[pidx] = w
                        rowsum += w
                    end
                    if rescale && !iszero(rowsum)
                        for pidx in firstp:lastp
                            pv[pidx] /= rowsum
                        end
                    elseif iszero(rowsum)
                        v = one(eltype(pv)) / (lastp-firstp+1)
                        for pidx in firstp:lastp
                            pv[pidx] = v
                        end
                    end
                end
            end
        end
    end
end

@kernel function update_classical_p_kernel!(pv, @Const(prp), @Const(pcv),
                                   @Const(arp), @Const(acv), @Const(av),
                                   @Const(cf), @Const(cmap), @Const(strong),
                                   rescale, n)
    i = @index(Global)
    if i <= n
        firstp, lastp = prp[i], prp[i+1] - 1
        if firstp <= lastp
            if cf[i] == 1
                @inbounds pv[firstp] = one(eltype(pv))
            else
                effective_diagonal = zero(eltype(pv))
                @inbounds for aidx in arp[i]:(arp[i+1]-1)
                    j = acv[aidx]
                    if j == i || !strong[aidx]
                        effective_diagonal += av[aidx]
                    elseif cf[j] == -1
                        denominator = zero(eltype(pv))
                        for aj in arp[j]:(arp[j+1]-1)
                            q = acv[aj]
                            if cf[q] == 1
                                target = cmap[q]
                                included = false
                                for candidate in firstp:lastp
                                    if pcv[candidate] == target
                                        included = true
                                        break
                                    end
                                end
                                included && (denominator += av[aj])
                            end
                        end
                        iszero(denominator) && (effective_diagonal += av[aidx])
                    end
                end
                rowsum = zero(eltype(pv))
                for pidx in firstp:lastp
                    J = pcv[pidx]
                    numerator = zero(eltype(pv))
                    @inbounds for aidx in arp[i]:(arp[i+1]-1)
                        j = acv[aidx]
                        if strong[aidx] && cf[j] == 1 && cmap[j] == J
                            numerator += av[aidx]
                        elseif strong[aidx] && cf[j] == -1
                            denominator = zero(eltype(pv))
                            coupling = zero(eltype(pv))
                            for aj in arp[j]:(arp[j+1]-1)
                                q = acv[aj]
                                if cf[q] == 1
                                    target = cmap[q]
                                    included = false
                                    for candidate in firstp:lastp
                                        if pcv[candidate] == target
                                            included = true
                                            break
                                        end
                                    end
                                    if included
                                        denominator += av[aj]
                                        target == J && (coupling += av[aj])
                                    end
                                end
                            end
                            if !iszero(denominator)
                                numerator += av[aidx] * coupling / denominator
                            end
                        end
                    end
                    weight = if iszero(effective_diagonal)
                        zero(eltype(pv))
                    else
                        -numerator / effective_diagonal
                    end
                    @inbounds pv[pidx] = weight
                    rowsum += weight
                end
                if rescale && !iszero(rowsum)
                    @inbounds for pidx in firstp:lastp
                        pv[pidx] /= rowsum
                    end
                elseif iszero(rowsum)
                    value = one(eltype(pv)) / (lastp-firstp+1)
                    @inbounds for pidx in firstp:lastp
                        pv[pidx] = value
                    end
                end
            end
        end
    end
end

function LinearAlgebra.mul!(y::AbstractVector, A::StaticSparsityMatrixCSR, x::AbstractVector)
    length(y) == matrix_nrows(A) || throw(DimensionMismatch())
    length(x) == matrix_ncols(A) || throw(DimensionMismatch())
    k! = spmv_kernel!(matrix_backend(A), matrix_block_size(A))
    k!(y, A.rowptr, A.colval, A.nzval, x, matrix_nrows(A); ndrange=matrix_nrows(A))
    y
end

const CPU_THREAD_THRESHOLD = 8_192

@inline function foreach_cpu_row(f, n::Int)
    if Threads.nthreads() > 1 && n >= CPU_THREAD_THRESHOLD
        Threads.@threads :dynamic for i in 1:n
            f(i)
        end
    else
        @inbounds for i in 1:n
            f(i)
        end
    end
    nothing
end

function LinearAlgebra.mul!(y::Vector, A::StaticSparsityMatrixCSR{Tv,Ti,<:Vector,<:Vector,<:Vector},
                            x::Vector) where {Tv,Ti}
    length(y) == matrix_nrows(A) || throw(DimensionMismatch())
    length(x) == matrix_ncols(A) || throw(DimensionMismatch())
    foreach_cpu_row(matrix_nrows(A)) do i
        value = zero(eltype(y))
        @inbounds @simd for k in A.rowptr[i]:(A.rowptr[i+1]-1)
            value += A.nzval[k] * x[A.colval[k]]
        end
        @inbounds y[i] = value
    end
    y
end

function copy_matrix_values!(dst::StaticSparsityMatrixCSR, src::StaticSparsityMatrixCSR)
    matrix_nonzeros(dst) == matrix_nonzeros(src) || throw(ArgumentError("matrix patterns differ"))
    k! = copy_kernel!(matrix_backend(dst), matrix_block_size(dst))
    k!(dst.nzval, src.nzval, matrix_nonzeros(dst); ndrange=matrix_nonzeros(dst))
    dst
end

function copy_matrix_values!(dst::StaticSparsityMatrixCSR{Tv,Ti,<:Vector,<:Vector,<:Vector},
                       src::StaticSparsityMatrixCSR{Tv,Ti,<:Vector,<:Vector,<:Vector}) where {Tv,Ti}
    matrix_nonzeros(dst) == matrix_nonzeros(src) || throw(ArgumentError("matrix patterns differ"))
    copyto!(dst.nzval, 1, src.nzval, 1, matrix_nonzeros(dst))
    dst
end

function fill_backend!(x, value, backend, block_size)
    k! = fill_kernel!(backend, block_size)
    k!(x, value, length(x); ndrange=length(x))
    x
end

fill_backend!(x::Vector, value, ::KernelAbstractions.CPU, block_size) = fill!(x, value)

function update_coarse_solver!(S::CoarseLUState, A::StaticSparsityMatrixCSR)
    @inbounds for k in 1:matrix_nonzeros(A)
        S.matrix.nzval[S.csr_to_csc[k]] = A.nzval[k]
    end
    lu!(S.factors, S.matrix)
    S
end

function copy_csr_to_dense!(dense, A::StaticSparsityMatrixCSR)
    fill_backend!(dense, zero(eltype(dense)), matrix_backend(A), matrix_block_size(A))
    k! = csr_to_dense_kernel!(matrix_backend(A), matrix_block_size(A))
    k!(dense, A.rowptr, A.colval, A.nzval, matrix_nrows(A); ndrange=matrix_nrows(A))
    # Generic `lu!` implementations may access the array from the host, while
    # accelerator implementations enqueue work on their own library stream.
    # Make the KernelAbstractions writes visible before either kind is called.
    synchronize_backend(matrix_backend(A))
    dense
end

function update_coarse_solver!(S::DenseLUState, A::StaticSparsityMatrixCSR)
    matrix = S.factorization.factors
    copy_csr_to_dense!(matrix, A)
    S.factorization = lu!(matrix)
    S
end

function update_coarse_solver!(S::HostLUState, A::StaticSparsityMatrixCSR)
    copyto!(S.values, 1, A.nzval, 1, matrix_nonzeros(A))
    matrix = S.factorization.factors
    fill!(matrix, zero(eltype(matrix)))
    @inbounds for i in 1:matrix_nrows(A), k in S.rowptr[i]:(S.rowptr[i+1]-1)
        matrix[i, S.colval[k]] = S.values[k]
    end
    S.factorization = lu!(matrix)
    S
end

function coarse_solve!(x, b, S::CoarseLUState, backend, block_size)
    ldiv!(x, S.factors, b)
end

function coarse_solve!(x, b, S::DenseLUState, backend, block_size)
    ldiv!(x, S.factorization, b)
end

function coarse_solve!(x, b, S::HostLUState, backend, block_size)
    copyto!(S.rhs, b)
    ldiv!(S.factorization, S.rhs)
    copyto!(x, S.rhs)
    x
end

function residual!(r, A::StaticSparsityMatrixCSR, x, b)
    k! = residual_kernel!(matrix_backend(A), matrix_block_size(A))
    k!(r, b, x, A.rowptr, A.colval, A.nzval, matrix_nrows(A); ndrange=matrix_nrows(A))
    r
end


function residual!(r::Vector, A::StaticSparsityMatrixCSR{Tv,Ti,<:Vector,<:Vector,<:Vector},
                    x::Vector, b::Vector) where {Tv,Ti}
    foreach_cpu_row(matrix_nrows(A)) do i
        value = zero(eltype(r))
        @inbounds @simd for k in A.rowptr[i]:(A.rowptr[i+1]-1)
            value += A.nzval[k] * x[A.colval[k]]
        end
        @inbounds r[i] = b[i] - value
    end
    r
end
