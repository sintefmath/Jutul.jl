function setup_preferred_sparse_lu end

"""
The portable sparse LU accepts a fixed CSR pattern. With pivoting enabled it
uses a dense factor buffer so row swaps can introduce arbitrary fill. Set
`pivoting=false` to use the sparse symbolic fill pattern without row swaps.
"""
struct KASparseLUFactor{V, I, B, R, C}
    rowptr::I
    colval::I
    diagonal::I
    source_to_factor::I
    factors::V
    backend::B
    n::Int
    pattern_rowptr::Vector{Int}
    pattern_colval::Vector{Int}
    source_rowptr::R
    source_colval::C
    pivots::I
    status::I
    pivoting::Bool
end

function sparse_lu_pattern(A::StaticSparsityMatrixCSR)
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("sparse LU requires a square matrix"))
    host = host_csr(A)
    rows = [Set{Int}() for _ in 1:n]
    for i in 1:n
        for k in host.rowptr[i]:(host.rowptr[i + 1] - 1)
            push!(rows[i], Int(host.colval[k]))
        end
        i in rows[i] || throw(ArgumentError("sparse LU requires a structural diagonal"))
        lower = sort!(filter(col -> col < i, collect(rows[i])))
        next = 1
        while next <= length(lower)
            j = lower[next]
            for col in rows[j]
                if col > j && !(col in rows[i])
                    push!(rows[i], col)
                    if col < i
                        insert!(lower, searchsortedfirst(lower, col), col)
                    end
                end
            end
            next += 1
        end
    end
    rowptr = Vector{Int}(undef, n + 1)
    rowptr[1] = 1
    colval = Int[]
    diagonal = Vector{Int}(undef, n)
    source_to_factor = Vector{Int}(undef, nnz(A))
    for i in 1:n
        columns = sort!(collect(rows[i]))
        positions = Dict(col => length(colval) + k for (k, col) in enumerate(columns))
        append!(colval, columns)
        diagonal[i] = positions[i]
        for k in host.rowptr[i]:(host.rowptr[i + 1] - 1)
            source_to_factor[k] = positions[Int(host.colval[k])]
        end
        rowptr[i + 1] = length(colval) + 1
    end
    return rowptr, colval, diagonal, source_to_factor,
        Int.(host.rowptr), Int.(host.colval)
end

@kernel function sparse_lu_factor_kernel!(
        factors, source, rowptr, colval, diagonal, source_to_factor,
        status, nnz_source, n
    )
    index = @index(Global)
    if index == 1
        status[1] = zero(eltype(status))
        @inbounds for k in eachindex(factors)
            factors[k] = zero(eltype(factors))
        end
        @inbounds for k in 1:nnz_source
            factors[source_to_factor[k]] = source[k]
        end
        @inbounds for i in 1:n
            for ij in rowptr[i]:(diagonal[i] - 1)
                j = colval[ij]
                pivot = factors[diagonal[j]]
                if iszero(pivot) || !isfinite(pivot)
                    status[1] = j
                    break
                end
                lij = factors[ij] / pivot
                factors[ij] = lij
                for jk in (diagonal[j] + 1):(rowptr[j + 1] - 1)
                    col = colval[jk]
                    for ik in (ij + 1):(rowptr[i + 1] - 1)
                        if colval[ik] == col
                            factors[ik] -= lij * factors[jk]
                            break
                        end
                    end
                end
            end
            if !iszero(status[1])
                break
            end
            pivot = factors[diagonal[i]]
            if iszero(pivot) || !isfinite(pivot)
                status[1] = i
                break
            end
        end
    end
end

@kernel function pivoted_sparse_lu_factor_kernel!(
        factors, pivots, status, source, rowptr, colval, n
    )
    index = @index(Global)
    if index == 1
        status[1] = zero(eltype(status))
        @inbounds for k in eachindex(factors)
            factors[k] = zero(eltype(factors))
        end
        @inbounds for row in 1:n
            for position in rowptr[row]:(rowptr[row + 1] - 1)
                col = colval[position]
                factors[(row - 1)*n + col] = source[position]
            end
        end
        @inbounds for k in 1:n
            pivot_row = k
            pivot_size = abs(factors[(k - 1)*n + k])
            for row in (k + 1):n
                candidate = abs(factors[(row - 1)*n + k])
                if candidate > pivot_size
                    pivot_size = candidate
                    pivot_row = row
                end
            end
            if iszero(pivot_size) || !isfinite(pivot_size)
                status[1] = k
                break
            end
            pivots[k] = pivot_row
            if pivot_row != k
                for col in 1:n
                    first = (k - 1)*n + col
                    second = (pivot_row - 1)*n + col
                    value = factors[first]
                    factors[first] = factors[second]
                    factors[second] = value
                end
            end
            pivot = factors[(k - 1)*n + k]
            for row in (k + 1):n
                lower = (row - 1)*n + k
                multiplier = factors[lower] / pivot
                factors[lower] = multiplier
                for col in (k + 1):n
                    factors[(row - 1)*n + col] -=
                        multiplier*factors[(k - 1)*n + col]
                end
            end
        end
    end
end

@kernel function sparse_lu_solve_kernel!(
        output, rhs, factors, rowptr, colval, diagonal, n
    )
    index = @index(Global)
    if index == 1
        @inbounds for i in 1:n
            value = rhs[i]
            for k in rowptr[i]:(diagonal[i] - 1)
                value -= factors[k] * output[colval[k]]
            end
            output[i] = value
        end
        @inbounds for i in n:-1:1
            value = output[i]
            for k in (diagonal[i] + 1):(rowptr[i + 1] - 1)
                value -= factors[k] * output[colval[k]]
            end
            output[i] = value / factors[diagonal[i]]
        end
    end
end

@kernel function pivoted_sparse_lu_solve_kernel!(output, rhs, factors, pivots, n)
    index = @index(Global)
    if index == 1
        @inbounds for i in 1:n
            output[i] = rhs[i]
        end
        @inbounds for k in 1:n
            pivot_row = pivots[k]
            value = output[k]
            output[k] = output[pivot_row]
            output[pivot_row] = value
        end
        @inbounds for i in 1:n
            value = output[i]
            for j in 1:(i - 1)
                value -= factors[(i - 1)*n + j]*output[j]
            end
            output[i] = value
        end
        @inbounds for i in n:-1:1
            value = output[i]
            for j in (i + 1):n
                value -= factors[(i - 1)*n + j]*output[j]
            end
            output[i] = value / factors[(i - 1)*n + i]
        end
    end
end

function setup_sparse_lu(A::StaticSparsityMatrixCSR; pivoting::Bool = true)
    size(A, 1) == size(A, 2) ||
        throw(DimensionMismatch("sparse LU requires a square matrix"))
    backend = matrix_backend(A)
    if pivoting
        host = host_csr(A)
        rowptr = Int.(host.rowptr)
        colval = Int.(host.colval)
        diagonal = zeros(Int, size(A, 1))
        map = collect(1:nnz(A))
        pattern_rowptr = rowptr
        pattern_colval = colval
        nfactors = Base.checked_mul(size(A, 1), size(A, 1))
    else
        rowptr, colval, diagonal, map, pattern_rowptr, pattern_colval =
            sparse_lu_pattern(A)
        nfactors = length(colval)
    end
    Ti = eltype(A.rowptr)
    factors = KernelAbstractions.allocate(backend, eltype(A.nzval), nfactors)
    pivots = KernelAbstractions.allocate(backend, Ti, size(A, 1))
    status = KernelAbstractions.allocate(backend, Ti, 1)
    state = KASparseLUFactor(
        backend_copy(backend, Ti.(rowptr)), backend_copy(backend, Ti.(colval)),
        backend_copy(backend, Ti.(diagonal)), backend_copy(backend, Ti.(map)),
        factors, backend, size(A, 1),
        pattern_rowptr, pattern_colval, A.rowptr, A.colval,
        pivots, status, pivoting
    )
    return resetup_sparse_lu!(SparseLU(state), A)
end

function sparse_lu_same_pattern(S::SparseLU, A::StaticSparsityMatrixCSR)
    F = S.factorization
    size(A) == (F.n, F.n) && nnz(A) == length(F.pattern_colval) || return false
    if A.rowptr === F.source_rowptr && A.colval === F.source_colval
        return true
    end
    host = host_csr(A)
    return Int.(host.rowptr) == F.pattern_rowptr &&
        Int.(host.colval) == F.pattern_colval
end

function resetup_sparse_lu!(S::SparseLU{<:KASparseLUFactor}, A::StaticSparsityMatrixCSR)
    F = S.factorization
    sparse_lu_same_pattern(S, A) ||
        throw(ArgumentError("sparse LU resetup requires the same CSR pattern"))
    if F.pivoting
        kernel! = pivoted_sparse_lu_factor_kernel!(F.backend, 1)
        event = kernel!(F.factors, F.pivots, F.status,
            A.nzval, F.rowptr, F.colval, F.n; ndrange = 1)
    else
        kernel! = sparse_lu_factor_kernel!(F.backend, 1)
        event = kernel!(F.factors, A.nzval, F.rowptr, F.colval,
            F.diagonal, F.source_to_factor, F.status, nnz(A), F.n;
            ndrange = 1)
    end
    isnothing(event) || wait(event)
    singular = only(Array(F.status))
    iszero(singular) || throw(LinearAlgebra.SingularException(Int(singular)))
    return S
end

function LinearAlgebra.ldiv!(x, S::SparseLU{<:KASparseLUFactor}, b)
    F = S.factorization
    length(x) == F.n && length(b) == F.n || throw(DimensionMismatch())
    if F.pivoting
        kernel! = pivoted_sparse_lu_solve_kernel!(F.backend, 1)
        event = kernel!(x, b, F.factors, F.pivots, F.n; ndrange = 1)
    else
        kernel! = sparse_lu_solve_kernel!(F.backend, 1)
        event = kernel!(x, b, F.factors, F.rowptr, F.colval,
            F.diagonal, F.n; ndrange = 1)
    end
    isnothing(event) || wait(event)
    return x
end

update_coarse_solver!(S::SparseLU, A::StaticSparsityMatrixCSR) =
    resetup_sparse_lu!(S, A)

function coarse_solve!(x, b, S::SparseLU, backend, block_size)
    ldiv!(logical_backend_buffer(x), S, logical_backend_buffer(b))
    return x
end
